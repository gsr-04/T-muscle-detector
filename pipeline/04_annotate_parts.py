"""Part masks for the torso crop: trapezius, clavicle, shoulder, arm, pec, abdomen.

Everything is derived from three inputs: the pose anchors (anatomy.Anchors), the
person mask (pose segmentation) and the luminance image. No learned part
segmentation is involved, which is exactly why it works on a desaturated photo
like 002 where colour cues are useless.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
from scipy import ndimage as ndi

# copy of src/parts.py: anatomy / chest / pose_mp / snap / torso_seg live in
# src/, so put that on the path
import sys as _sys
for _p in (Path(__file__).resolve().parents[1] / "src",
           Path(__file__).resolve().parents[1] / "tools"):
    if str(_p) not in _sys.path:
        _sys.path.insert(0, str(_p))

from anatomy import Anchors, compute_anchors
from chest import find_nipples, fold_below_nipples
from pose_mp import detect, to_mask, to_px
from snap import ridge_map, snap_horizontal, snap_vertical, step_map, valley_map
from torso_seg import (DATA, MODELS, OUT, clean_mask, foreground_mask, load_bgr,
                       outside_gaps,
                       save_bgr)

ROOT = DATA.parent

PALETTE = {
    # BGR, one distinct hue per structure family
    "trapezius_l": (50, 50, 230),      # red
    "trapezius_r": (50, 50, 230),
    "clavicle_l": (0, 220, 255),       # yellow
    "clavicle_r": (0, 220, 255),
    "shoulder_l": (230, 60, 230),      # magenta
    "shoulder_r": (230, 60, 230),
    "arm_l": (60, 200, 60),            # green
    "arm_r": (60, 200, 60),
    "pec_l": (230, 120, 40),           # blue
    "pec_r": (230, 120, 40),
    "rectus": (0, 150, 255),           # orange
    "oblique_l": (230, 230, 0),        # cyan
    "oblique_r": (230, 230, 0),
    "neck": (140, 90, 90),             # grey-blue
}

LEGEND = [("trapezius", "trapezius_l"), ("clavicle", "clavicle_l"),
          ("shoulder", "shoulder_l"), ("arm", "arm_l"),
          ("pec", "pec_l"), ("rectus", "rectus"), ("oblique", "oblique_l"),
          ("neck", "neck")]

# the four classes the user asked for, plus the small neck strip that belongs to
# neither the trapezius nor the torso
SIMPLE = {
    # the clavicle band is drawn over the trapezius (see PRIORITY), so it belongs
    # to that class: without a home of its own it left a strip across the top of
    # the chest with no label at all in simple_<n>.png / simple_<n>.json
    "trapezius": ["trapezius_l", "trapezius_r", "clavicle_l", "clavicle_r"],
    "shoulder": ["shoulder_l", "shoulder_r"],
    "arm": ["arm_l", "arm_r"],
    "torso": ["pec_l", "pec_r", "rectus", "oblique_l", "oblique_r"],
    "neck": ["neck"],
}
SIMPLE_COLORS = {           # BGR
    "arm": (70, 200, 70),
    "shoulder": (235, 70, 235),
    "trapezius": (60, 60, 235),
    "torso": (240, 150, 50),
    "neck": (150, 100, 100),
}
SIMPLE_ORDER = ["neck", "torso", "trapezius", "shoulder", "arm"]

PRIORITY = ["oblique_r", "oblique_l", "rectus", "pec_r", "pec_l", "trapezius_r",
            "trapezius_l", "shoulder_r", "shoulder_l", "neck",
            "clavicle_r", "clavicle_l", "arm_r", "arm_l"]

assert len(set(PRIORITY)) == len(PRIORITY)


# ------------------------------------------------------------------- helpers
def runs_of(row: np.ndarray, min_len: int = 5) -> list[tuple[int, int]]:
    b = np.asarray(row, bool)
    if not b.any():
        return []
    d = np.diff(b.astype(np.int8))
    starts = list(np.flatnonzero(d == 1) + 1)
    ends = list(np.flatnonzero(d == -1) + 1)
    if b[0]:
        starts = [0] + starts
    if b[-1]:
        ends = ends + [len(b)]
    return [(s, e) for s, e in zip(starts, ends) if e - s >= min_len]


def torso_run(runs: list[tuple[int, int]], x_mid: float,
              tol: float) -> tuple[int, int] | None:
    cand = [r for r in runs if r[0] - tol <= x_mid <= r[1] + tol]
    if not cand:
        return None
    return min(cand, key=lambda r: abs(0.5 * (r[0] + r[1]) - x_mid))


def lane_runs(runs, x_mid: float, side: int) -> tuple[int, int] | None:
    """Outermost run strictly on one side of the body centre."""
    pick = [r for r in runs if (r[1] < x_mid if side < 0 else r[0] > x_mid)]
    if not pick:
        return None
    return pick[0] if side < 0 else pick[-1]


def contour_path(mask: np.ndarray, p0, p1) -> np.ndarray:
    """Points of the outer contour walking from p0 to p1 along the short side."""
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    c = max(cnts, key=cv2.contourArea).reshape(-1, 2).astype(float)
    i0 = int(np.argmin(((c - np.asarray(p0, float)) ** 2).sum(1)))
    i1 = int(np.argmin(((c - np.asarray(p1, float)) ** 2).sum(1)))
    if i0 <= i1:
        fwd = c[i0:i1 + 1]
        bwd = np.vstack([c[i1:], c[:i0 + 1]])
    else:
        fwd = c[i1:i0 + 1][::-1]
        bwd = np.vstack([c[:i1 + 1], c[i0:]])[::-1]
    return fwd if len(fwd) <= len(bwd) else bwd


def band_around(p0, p1, half: float, n: int = 40) -> np.ndarray:
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    d = p1 - p0
    L = float(np.linalg.norm(d))
    nrm = np.array([-d[1], d[0]]) / max(L, 1e-6)
    t = np.linspace(0, 1, n)[:, None]
    ctr = p0 + t * d
    return np.vstack([ctr + nrm * half, (ctr - nrm * half)[::-1]])


def fill_poly(shape, pts) -> np.ndarray:
    m = np.zeros(shape[:2], np.uint8)
    cv2.fillPoly(m, [np.round(np.asarray(pts, float)).astype(np.int32)], 255)
    return m


def label_array(p: PartMasks) -> np.ndarray:
    names = [n for n in PRIORITY if n in p.masks]
    lab = np.zeros(next(iter(p.masks.values())).shape, np.int32)
    for i, name in enumerate(names, start=1):
        lab[p.masks[name] > 0] = i
    return lab


def reconcile(p: PartMasks, body: np.ndarray, rows: tuple[int, int]) -> None:
    """Give every unlabelled body pixel inside the ROI to the nearest part.

    Slits between the trapezius and the neck, or slivers left by a snapped
    curve, would otherwise show up as holes in the segmentation.
    """
    names = [n for n in PRIORITY if n in p.masks]
    lab = label_array(p)
    unknown = (lab == 0) & (body > 0)
    unknown[: max(0, rows[0])] = False
    unknown[rows[1]:] = False
    if unknown.any():
        idx = ndi.distance_transform_edt(lab == 0, return_indices=True)[1]
        lab[unknown] = lab[idx[0][unknown], idx[1][unknown]]
    for i, name in enumerate(names, start=1):
        p.masks[name] = ((lab == i).astype(np.uint8)) * 255


def row_band_mask(shape, y_from: int, y_to: int, x_of_y) -> np.ndarray:
    """Rasterise a region given per-row [x_left(y), x_right(y)] extents."""
    m = np.zeros(shape[:2], np.uint8)
    for y in range(max(0, y_from), min(shape[0], y_to)):
        xl, xr = x_of_y(y)
        if xr is None or xr - xl < 2:
            continue
        m[y, int(round(xl)):int(round(xr)) + 1] = 255
    return m


def col_band_mask(shape, x_from: int, x_to: int, y_of_x) -> np.ndarray:
    """Rasterise a region given per-column [y_top(x), y_bottom(x)] extents."""
    m = np.zeros(shape[:2], np.uint8)
    step = 1 if x_to >= x_from else -1
    for x in range(max(0, x_from), min(shape[1], x_to + step), step):
        yt, yb = y_of_x(x)
        if yb is None or yb - yt < 2:
            continue
        m[int(round(yt)):int(round(yb)) + 1, x] = 255
    return m


def trace_fold(V: np.ndarray, x_from: float, y_from: float, x_to: float, side: int,
               step: int, win: int, y_lo: float, y_hi: float,
               w_min: float = 0.12, contrast: float = 1.35
               ) -> tuple[np.ndarray, np.ndarray, float]:
    """Follow a crease outwards in x, keeping y continuous from the previous sample.

    The pectoral fold runs from the sternum laterally and rises towards the
    axilla, so a single horizontal search window cannot find it; walking the
    crease with a small per-step window can.
    """
    h, w = V.shape
    xs = np.arange(int(round(x_from)), int(round(x_to)), side * max(2, step))
    if xs.size == 0:
        return np.zeros(0), np.zeros(0), 0.0
    ys = np.zeros(len(xs))
    hit = np.zeros(len(xs), bool)
    y_prev = float(y_from)
    hi_of_x = y_hi if callable(y_hi) else (lambda _x: float(y_hi))
    for i, x in enumerate(xs):
        xi = int(x)
        if not (0 <= xi < w):
            ys[i] = y_prev
            continue
        y0 = int(np.clip(y_prev - win, 0, h - 1))
        y1 = int(np.clip(y_prev + win + 1, 1, h))
        col = V[y0:y1, xi]
        if col.size and col.max() >= w_min and col.max() >= contrast * np.median(col):
            y_prev = float(y0 + int(np.argmax(col)))
            hit[i] = True
        y_prev = float(np.clip(y_prev, y_lo, hi_of_x(x)))
        ys[i] = y_prev
    k = 5
    pad = k // 2
    ys = np.array([np.median(ys[max(0, i - pad): i + pad + 1]) for i in range(len(ys))])
    return xs.astype(float), ys, float(hit.mean())


# ------------------------------------------------------------------- builder
def _hole_limit(fg: np.ndarray) -> int:
    """Largest hole that still counts as noise rather than real background.

    A hole is a hole only because the mask closes around it; the armpit gap is
    exactly that whenever the arm touches the body below it.  Those are big
    (percent of the person), specks left by the extractor are tiny.
    """
    return int(max(64.0, 0.002 * float((np.asarray(fg) > 0).sum())))


@dataclass
class PartMasks:
    masks: dict[str, np.ndarray] = field(default_factory=dict)
    lines: dict[str, np.ndarray] = field(default_factory=dict)
    points: dict[str, np.ndarray] = field(default_factory=dict)
    roi: tuple[int, int, int, int] = (0, 0, 0, 0)


def build_parts(mask: np.ndarray, a: Anchors, bgr: np.ndarray | None = None,
                ov: dict | None = None) -> PartMasks:
    h, w = mask.shape[:2]
    S = a.scale
    x_mid = float(a.mid["x_mid"][0])
    notch = np.array(a.mid["notch"], float)
    acro = {"l": np.array(a.side["acromion_img_l"], float),
            "r": np.array(a.side["acromion_img_r"], float)}
    y_armpit = int(a.y["armpit"])
    y_pec = int(a.y["pec_low"])
    y_wb = int(a.y["waistband"])
    common = (mask > 0).astype(np.uint8) * 255
    out = PartMasks()

    V = ridge = step = None
    if bgr is not None:
        V = valley_map(bgr, k=int(0.15 * S))
        ridge = ridge_map(bgr, k=int(0.10 * S))
        step = step_map(bgr)

    # ---------------------------------------------------------------- helpers
    def lanes(y: int) -> tuple[tuple[int, int] | None, tuple[int, int] | None,
                               tuple[int, int] | None]:
        rr = runs_of(mask[y] > 0, 6)
        tr = torso_run(rr, x_mid, 0.30 * S)
        if tr is None and rr:
            tr = max(rr, key=lambda r: r[1] - r[0])
        return lane_runs(rr, x_mid, -1), tr, lane_runs(rr, x_mid, +1)

    arm_span: dict[int, dict[int, tuple[int, int]]] = {-1: {}, +1: {}}
    # scanned from the acromion, not from y_armpit: the anchor is a guess and
    # can sit below the row where the arm really leaves the trunk (hands on the
    # hips, a narrow axilla), and in those rows the groove is the arm's own inner
    # edge, not a crease measured inside the arm
    y_span0 = int(min(acro["l"][1], acro["r"][1]))
    for y in range(max(0, y_span0), h):
        lr, _tr, rr_ = lanes(y)
        if lr:
            arm_span[-1][y] = lr
        if rr_:
            arm_span[+1][y] = rr_

    # Where each arm really leaves the trunk: the deltoid ends there and the arm
    # begins.  Per side, because an anchor that only knows the later of the two
    # sides would otherwise label the whole upper arm of the other side deltoid.
    y_lane_from = int(a.mid["mid_shoulder"][1] + 0.05 * S)
    y_apex: dict[int, int] = {}
    for side in (-1, +1):
        rows_lane = sorted(y for y, r in arm_span[side].items()
                           if y >= y_lane_from and r[1] - r[0] >= 0.10 * S)
        y_apex[side] = int(min(int(y_armpit),
                               rows_lane[0] if rows_lane else int(y_armpit)))

    def groove_band(side: int) -> tuple[int, int]:
        """x range where the arm/torso groove can be, for this side."""
        return (int(round(x_mid - 0.52 * S)), int(round(x_mid - 0.16 * S))) \
            if side < 0 else (int(round(x_mid + 0.16 * S)), int(round(x_mid + 0.52 * S)))

    def valley_groove(y: int, side: int, anchor: float | None = None,
                      win: float = 0.07 * S) -> float | None:
        """Arm/torso separation from the dark groove, optionally near an anchor.

        Searching a narrow window around a vertically stable anchor is far more
        reliable than a per-row search over the whole plausible band.
        """
        if V is None:
            return None
        xs = np.flatnonzero(mask[y] > 0)
        if not xs.size:
            return None
        pad = int(0.07 * S)
        lo_lim, hi_lim = groove_band(side)
        lo, hi = max(int(xs[0]) + pad, lo_lim), min(int(xs[-1]) - pad, hi_lim)
        if anchor is not None:
            lo = max(lo, int(round(anchor - win)))
            hi = min(hi, int(round(anchor + win)))
        if hi - lo < 5:
            return None
        row = V[y, lo:hi]
        if row.max() < 0.13:
            return None
        return float(lo + int(np.argmax(row)))

    # anthropometric fallback: the torso's lateral edge at the waist is roughly
    # 1.15x half the hip-joint separation
    hip_half = float(abs(a.side["left_hip"][0] - a.side["right_hip"][0])) / 2.0
    prior_off = float(np.clip(1.15 * hip_half, 0.24 * S, 0.42 * S))

    # one continuous groove curve per side, gap-detected first, crease second
    y_arm_end = int(max(a.side["left_wrist"][1], a.side["right_wrist"][1]))
    groove: dict[int, np.ndarray] = {}
    groove_rows: dict[int, np.ndarray] = {}
    for side in (-1, +1):
        # start above the armpit as well: there the arm and the torso touch, so
        # the deltopectoral groove has to come from the crease rather than from
        # an extrapolated straight line (which used to include the arm root)
        a_s = acro["l" if side < 0 else "r"]
        y_top_g = int(max(0, min(a_s[1], y_armpit - 0.45 * S)))
        ys = np.arange(y_top_g, max(y_top_g + 2,
                                    min(h, y_arm_end + int(0.06 * S) + 1)))
        gx = np.full(len(ys), np.nan)
        for i, y in enumerate(ys):
            if y in arm_span[side]:
                gx[i] = (arm_span[side][y][1] if side < 0 else arm_span[side][y][0])
        # vertical anchor: sum the crease response over the torso's height so a
        # single noisy row cannot decide the groove position
        anchor = None
        if V is not None:
            r0 = int(max(0, y_armpit + 0.05 * S))
            r1 = int(min(h, y_wb))
            lo_lim, hi_lim = groove_band(side)
            lo_lim = max(lo_lim, 0)
            hi_lim = min(hi_lim, w)
            if r1 - r0 > 10 and hi_lim - lo_lim > 10:
                prof = V[r0:r1, lo_lim:hi_lim].sum(axis=0)
                prof = cv2.GaussianBlur(prof.reshape(1, -1).astype(np.float32),
                                        (max(3, int(0.05 * S) | 1), 1), 0).ravel()
                anchor = float(lo_lim + int(np.argmax(prof)))
        for i, y in enumerate(ys):
            if np.isnan(gx[i]):
                v = valley_groove(int(y), side, anchor)
                if v is not None:
                    gx[i] = v
        # rows where the crease is invisible stay on the anchor (a vertical
        # groove) unless the measured arm/torso gap says otherwise
        if anchor is not None:
            gx = np.where(np.isnan(gx), anchor, gx)
        good = ~np.isnan(gx)
        if good.sum() >= 2:
            # throw away candidates that wander off, then interpolate the gaps
            med = float(np.median(gx[good]))
            gx[np.abs(gx - med) > 0.12 * S] = np.nan
            good = ~np.isnan(gx)
        prior_x = x_mid + side * prior_off
        if good.sum() >= 2:
            filled = np.interp(np.arange(len(ys)), np.flatnonzero(good), gx[good])
        elif good.sum() == 1:
            filled = np.full(len(ys), gx[good][0])
        else:
            filled = np.full(len(ys), prior_x)
        if good.sum() < max(2, 0.35 * len(ys)):
            # few measurements: pull only the unmeasured rows towards the
            # anatomical half-width, never the rows that were actually measured
            filled = np.where(good, filled, 0.5 * filled + 0.5 * prior_x)
        gx = filled
        k = 11
        pad = k // 2
        gx = np.array([np.median(gx[max(0, i - pad): i + pad + 1])
                       for i in range(len(gx))])
        # Above the armpit the arm and the trunk touch, so no gap can be
        # measured there.  The arm's inner silhouette is visible below the
        # armpit though, and it continues upward: fit that edge and continue the
        # curve, instead of the straight acromion-to-armpit line (which ran too
        # lateral and let the trunk claim a slice of the arm).
        # The first row where the silhouette really splits is the contact row:
        # there the arm's own run ends exactly on the trunk, so that edge is the
        # boundary.  Above it the two touch and only the curve can say where the
        # boundary is.  The scan is done on the mask itself, because arm_span
        # only starts at the anatomy's armpit row, which can be lower than the
        # real split (016: split 266, anatomy 277).
        scan0 = int(max(0, min(acro["l"][1], acro["r"][1])))
        seen: list[tuple[int, float]] = []
        y_vis = None
        for y in range(scan0, min(h, scan0 + int(2.5 * S))):
            rr_y = runs_of(mask[y] > 0, 6)
            if len(rr_y) < 2:
                continue
            lane = lane_runs(rr_y, x_mid, side)
            if lane is None:
                continue
            if y_vis is None:
                y_vis = y
            if len(seen) and y > y_vis + int(0.35 * S):
                break
            seen.append((y, float(lane[1] if side < 0 else lane[0])))
        if y_vis is None:
            y_vis = min(arm_span[side]) if arm_span[side] else y_armpit
        # the old estimate (straight acromion-to-armpit line, pulled medially
        # with a taper) - kept as the far-field value for the blend below
        old = gx.copy()
        for i, y in enumerate(ys):
            if y < y_vis:
                t = min(1.0, (y_vis - y) / max(1.0, y_vis - ys[0]))
                old[i] -= side * 0.05 * S * t
        if len(seen) >= 8:
            vy = np.array([v[0] for v in seen], float)
            ve = np.array([v[1] for v in seen], float)
            deg = 2 if len(seen) >= 16 else 1
            cf = np.polyfit(vy - y_vis, ve, deg)
            # the curve is only trusted near the contact row: a long quadratic
            # extrapolation up to the chest used to swing too far medial, so the
            # far field fades back to the old estimate over 0.18 S
            fade = max(1.0, 0.18 * S)
            for i, y in enumerate(ys):
                if y >= y_vis:
                    continue
                xs = np.flatnonzero(mask[y] > 0)
                if not xs.size:
                    continue
                bw = min(1.0, (y_vis - y) / fade)
                e = float(np.polyval(cf, y - y_vis))
                gx[i] = float(np.clip((1.0 - bw) * e + bw * old[i],
                                      xs[0] + 2, xs[-1] - 2))
        else:
            # too few visible rows to fit: keep the older estimate
            gx = np.where([y < y_vis for y in ys], old, gx)
        groove[side] = gx
        groove_rows[side] = ys

    def groove_x(y: int, side: int) -> float | None:
        """Medial edge of the arm: measured along the whole torso height."""
        ys0 = int(groove_rows[side][0]) if len(groove_rows[side]) else y_armpit
        arr = groove[side]
        if len(arr) == 0:
            return x_mid + side * 0.30 * S
        if y >= ys0:
            i = int(np.clip(y - ys0, 0, len(arr) - 1))
            return float(arr[i])
        a_s = acro["l" if side < 0 else "r"]
        if y > a_s[1]:
            g0 = float(arr[0])
            t = (y - a_s[1]) / max(1.0, ys0 - a_s[1])
            return float(a_s[0] + t * (g0 - a_s[0]))
        return None

    def torso_edge(y: int, side: int) -> float | None:
        return groove_x(y, side)

    def sil_edge(y: int, side: int) -> float | None:
        xs = np.flatnonzero(mask[y] > 0)
        if not xs.size:
            return None
        return float(xs[0] if side < 0 else xs[-1])

    # ---- arms, extended down to the wrist / end of the lane ----------------
    for name, side in (("arm_l", -1), ("arm_r", +1)):
        m = np.zeros((h, w), np.uint8)
        for y in range(y_apex[side], min(h, y_arm_end + int(0.06 * S) + 1)):
            sil = np.flatnonzero(mask[y] > 0)
            if not sil.size:
                continue
            gx = int(round(groove_x(y, side)))
            x0, x1 = (sil[0], gx) if side < 0 else (gx, sil[-1])
            if x1 - x0 >= 2:
                m[y, x0:x1 + 1] = 255
        out.masks[name] = m & common

    # ---- shoulder / deltoid: acromion down to the armpit -------------------
    for name, side, key in (("shoulder_l", -1, "l"), ("shoulder_r", +1, "r")):
        a_s = acro[key]

        def x_of_y(y, a_s=a_s, side=side):
            """Deltoid: from the silhouette in to the same groove the trunk uses.

            Sharing the boundary with the trunk leaves no unlabelled slit
            between them (that slit used to be filled to the nearest part and
            ended up labelled trunk, inside the arm).
            """
            if y <= a_s[1]:
                return (None, None)
            gx = groove_x(y, side)
            if gx is None:
                return (None, None)
            xs = np.flatnonzero(mask[y] > 0)
            if not xs.size:
                return (None, None)
            return (xs[0], gx) if side < 0 else (gx, xs[-1])

        out.masks[name] = row_band_mask((h, w), int(a_s[1]),
                                        y_apex[side] + 1, x_of_y)

    # ---- clavicle curve: snapped to the bony highlight ---------------------
    clav_curve: dict[str, np.ndarray] = {}
    for key, side in (("l", -1), ("r", +1)):
        a_s = acro[key]
        ts = np.linspace(0, 1, 48)[:, None]
        pts = notch[None, :] + ts * (a_s - notch)[None, :]
        pts[:, 1] -= 0.02 * S
        if ridge is not None:
            pts = snap_vertical(ridge, pts, radius=int(0.07 * S), smooth=11, w_min=0.25)
        clav_curve[key] = pts

    # ---- trapezius: between the shoulder slope and the clavicle curve ------
    for name, key, side in (("trapezius_l", "l", -1), ("trapezius_r", "r", +1)):
        a_s = acro[key]
        surf = clav_curve[key]
        pts = []
        for x, y_surf in surf:
            xi = int(round(x))
            if not (0 <= xi < w):
                continue
            ys = np.flatnonzero(mask[:, xi] > 0)
            if not ys.size:
                continue
            # the trapezius does reach up the neck, but not into the hair, so its
            # top is bounded a fixed distance above the clavicle line
            pts.append((x, max(float(ys[0]), float(y_surf) - 0.55 * S)))
        poly = np.vstack([np.array(pts), surf[::-1]])
        m = fill_poly((h, w), poly)
        x_in = x_mid + side * 0.17 * S
        gate = np.zeros((h, w), np.uint8)
        x0, x1 = (0, int(x_in)) if side < 0 else (int(x_in), w)
        gate[:, x0:x1] = 255
        out.masks[name] = (m & gate) & common

    # ---- clavicle band -----------------------------------------------------
    for name, key in (("clavicle_l", "l"), ("clavicle_r", "r")):
        surf = clav_curve[key]
        half = 0.022 * S
        poly = np.vstack([surf + [0, -half], (surf + [0, half])[::-1]])
        out.masks[name] = fill_poly((h, w), poly) & common

    # ---- waistband curve ---------------------------------------------------
    if step is not None:
        xs = np.linspace(x_mid - 0.50 * S, x_mid + 0.50 * S, 80)
        wb = np.stack([xs, np.full_like(xs, float(y_wb))], 1)
        wb = snap_vertical(step, wb, radius=int(0.05 * S), smooth=9, w_min=0.2)
    else:
        wb = np.array([[x_mid - 0.50 * S, y_wb], [x_mid + 0.50 * S, y_wb]], float)

    # ---- pectoral: clavicle curve + deltopectoral groove + traced pec fold -
    ov = ov or {}
    # nipples anchor the fold: the pectoral border is the first dark edge below
    # them.  An override file can pin the nipples (or the sternal fold height)
    # for images where automatic detection is unsure.
    nipples: dict[int, np.ndarray] = {}
    for key, side in (("l", -1), ("r", +1)):
        pt = ov.get("nipples", {}).get(key)
        if pt:
            nipples[side] = np.array([float(pt[0]), float(pt[1])])
    if len(nipples) < 2:
        nipples = find_nipples(bgr, mask, x_mid, S,
                               a.y["neck_top"] + 0.30 * (a.y["waistband"] - a.y["neck_top"]),
                               a.y["pec_low"] + 0.10 * S) if bgr is not None else {}
    folds: dict[int, np.ndarray] = {}
    if len(nipples) == 2 and V is not None:
        x_lat_all = {}
        for key, side in (("l", -1), ("r", +1)):
            lat_y = np.array([(groove_x(y, side), y)
                              for y in range(int(acro[key][1]), y_armpit + 1)
                              if groove_x(y, side) is not None], float)
            x_lat_all[side] = float(lat_y[-1, 0]) if len(lat_y) else \
                x_mid + side * 0.30 * S
        folds = fold_below_nipples(V, step, mask, x_mid, S, nipples, x_lat_all,
                                   float(y_armpit + 0.06 * S))

    # fall-back anchor: the medial end of the fold at the sternum
    y_pec_med = float(y_pec)
    if V is not None:
        r0 = max(0, int(y_pec - 0.16 * S))
        r1 = min(h, int(y_pec + 0.16 * S) + 1)
        c0, c1 = int(max(0, x_mid - 0.10 * S)), int(min(w, x_mid + 0.10 * S) + 1)
        if r1 > r0 and c1 > c0:
            band = V[r0:r1, c0:c1].mean(axis=1)
            if band.size and band.max() > 0.12:
                y_pec_med = float(r0 + int(np.argmax(band)))
    fold_curve: dict[int, np.ndarray] = {}
    for name, key, side in (("pec_l", "l", -1), ("pec_r", "r", +1)):
        a_s = acro[key]
        surf = clav_curve[key]
        lat = np.array([(groove_x(y, side), y)
                        for y in range(int(a_s[1]), y_armpit + 1)
                        if groove_x(y, side) is not None],
                       float).reshape(-1, 2)
        x_lat = float(lat[-1, 0]) if len(lat) else x_mid + side * 0.30 * S
        span = max(1e-6, abs(x_lat - x_mid))
        y_arm = float(y_armpit + 0.06 * S)

        def y_hi_at(x, span=span, x_lat=x_lat, y_arm=y_arm, y_med=y_pec_med):
            """Anatomical ceiling for the fold: it rises monotonically from the
            sternum attachment towards the anterior axillary fold."""
            u = min(1.0, abs(x - x_mid) / span)
            return y_med + (y_arm - y_med) * u

        if side in folds and len(folds[side]) > 3:
            fx, fy = folds[side][:, 0], folds[side][:, 1]
        elif V is not None:
            fx, fy, hit = trace_fold(V, x_mid + side * 0.10 * S, y_pec_med, x_lat, side,
                                     step=max(2, int(0.020 * S)),
                                     win=max(3, int(0.05 * S)),
                                     y_lo=y_pec_med - 0.04 * S, y_hi=y_hi_at)
            if fx.size < 4 or hit < 0.25:
                t = np.linspace(0, 1, 40) ** 1.6
                fx = x_mid + side * t * abs(x_lat - x_mid)
                fy = y_pec_med * (1 - t) + y_arm * t
        else:
            t = np.linspace(0, 1, 40) ** 1.6
            fx = x_mid + side * t * abs(x_lat - x_mid)
            fy = y_pec_med * (1 - t) + y_arm * t
        fold = np.stack([fx, fy], 1)
        # manual control from overrides.json:
        #   fold_y     -> put the sternal end of the fold at this y
        #   fold_shift -> shift the whole fold by this many pixels (negative = up)
        if "fold_y" in ov:
            fold[:, 1] += float(ov["fold_y"]) - fold[0, 1]
        if "fold_shift" in ov:
            fold[:, 1] += float(ov["fold_shift"])
        # make the fold start exactly on the midline so the pec and abdomen
        # polygons tile without a slit down the centre of the torso
        fold = np.vstack([[x_mid, fold[0, 1]], fold])
        fold_curve[side] = fold
        poly = np.vstack([surf, lat, fold[::-1]])
        m = fill_poly((h, w), poly)
        gate = np.zeros((h, w), np.uint8)
        x0, x1 = (0, int(x_mid)) if side < 0 else (int(x_mid), w)
        gate[:, x0:x1] = 255
        out.masks[name] = (m & gate) & common
        out.lines[f"pec_fold_{key}"] = fold
        if side in nipples:
            out.points[f"nipple_{key}"] = nipples[side]

    # ---- abdomen: the traced pec fold down to the waistband curve ----------
    abd = {}
    for key, side in (("l", -1), ("r", +1)):
        fold = fold_curve[side]
        lat = np.array([(groove_x(y, side), y) for y in range(y_armpit, y_wb + 1)
                        if groove_x(y, side) is not None], float).reshape(-1, 2)
        sel = wb[:, 0] < x_mid if side < 0 else wb[:, 0] > x_mid
        wb_side = wb[sel]
        wb_side = wb_side[np.argsort(wb_side[:, 0])]
        if side > 0:
            wb_side = wb_side[::-1]        # both sides run outer -> midline
        poly = np.vstack([fold, lat, wb_side]) if len(lat) else np.vstack([fold, wb_side])
        abd[side] = fill_poly((h, w), poly) & common
    abdomen = abd[-1] | abd[+1]

    # ---- linea alba and the rectus borders --------------------------------
    split_rows = np.arange(y_armpit, min(h, y_wb + 1))
    half = 0.215 * S
    alba = np.full(len(split_rows), float(x_mid))
    lo_b = np.full(len(split_rows), x_mid - half)
    hi_b = np.full(len(split_rows), x_mid + half)
    if V is not None and len(split_rows):
        p = np.stack([np.full_like(split_rows, float(x_mid), dtype=float), split_rows], 1)
        alba = snap_horizontal(V, p, radius=int(0.07 * S), smooth=15, w_min=0.12)[:, 0]
        alba = np.clip(alba, x_mid - 0.07 * S, x_mid + 0.07 * S)
        for arr, off in ((lo_b, -half), (hi_b, half)):
            p = np.stack([np.full_like(split_rows, x_mid + off, dtype=float),
                          split_rows], 1)
            arr[:] = snap_horizontal(V, p, radius=int(0.05 * S), smooth=15,
                                     w_min=0.12)[:, 0]
    out.lines["linea_alba"] = np.stack([alba, split_rows], 1).astype(float)

    rectus = np.zeros((h, w), np.uint8)
    for i, y in enumerate(split_rows):
        yi = int(y)
        if yi >= h:
            break
        x0, x1 = int(round(lo_b[i])), int(round(hi_b[i]))
        rectus[yi, max(0, x0):min(w, x1)] = 255
    not_rectus = np.where(rectus > 0, 0, 255).astype(np.uint8)
    out.masks["rectus"] = rectus & abdomen
    out.masks["oblique_l"] = abd[-1] & not_rectus
    out.masks["oblique_r"] = abd[+1] & not_rectus

    # tendinous intersections dividing the rectus blocks
    if V is not None and (out.masks["rectus"] > 0).any():
        prof = np.array([float(V[int(split_rows[i]),
                                 max(0, int(lo_b[i])):int(hi_b[i])].mean())
                         for i in range(len(split_rows))])
        peaks = [int(split_rows[i]) for i in range(1, len(split_rows) - 1)
                 if prof[i] > 0.15 and prof[i] >= prof[i - 1] and prof[i] >= prof[i + 1]]
        merged: list[int] = []
        for pk in peaks:
            if all(abs(pk - q) > 0.12 * S for q in merged):
                merged.append(pk)
        for j, pk in enumerate(sorted(merged)[:3]):
            out.lines[f"intersection_{j + 1}"] = np.array(
                [[x_mid - half * 0.95, float(pk)], [x_mid + half * 0.95, float(pk)]])

    # ---- neck: the gap between the shoulders stays uncovered otherwise ----
    # only the strip of neck between the two trapezius muscles: the head itself
    # is not one of the requested parts
    y_clav_top = float(min(clav_curve["l"][:, 1].min(), clav_curve["r"][:, 1].min(),
                           notch[1]))

    def neck_y(x: int) -> tuple[float, float | None]:
        ys = np.flatnonzero(mask[:, x] > 0)
        if not ys.size:
            return (0.0, None)
        return (max(float(ys[0]), y_clav_top - 0.45 * S), y_clav_top)

    m = col_band_mask((h, w), int(x_mid - 0.135 * S), int(x_mid + 0.135 * S), neck_y)
    out.masks["neck"] = m & common

    ys, xs = np.nonzero(np.any([m > 0 for m in out.masks.values()], axis=0))
    out.roi = (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))
    reconcile(out, common, (int(a.y["neck_top"]), int(y_wb)))
    return out


# ------------------------------------------------------------------- render
def label_map(p: PartMasks) -> np.ndarray:
    names = list(p.masks)
    label = np.zeros(p.masks[names[0]].shape, np.int32)
    for i, name in enumerate(PRIORITY, start=1):
        if name in p.masks:
            label[p.masks[name] > 0] = i
    return label


def render(bgr: np.ndarray, p: PartMasks, alpha: float = 0.45) -> tuple[np.ndarray, np.ndarray]:
    lab = label_map(p)
    color = bgr.copy()
    names = [n for n in PRIORITY if n in p.masks]
    for i, name in enumerate(names, start=1):
        color[lab == i] = PALETTE[name]
    vis = cv2.addWeighted(bgr, 1 - alpha, color, alpha, 0)
    for i, name in enumerate(names, start=1):
        m = np.where(lab == i, 255, 0).astype(np.uint8)
        cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(vis, cnts, -1, (20, 20, 20), 1, cv2.LINE_AA)
    for lname, pts in p.lines.items():
        if len(pts) < 2:
            continue
        col = (255, 255, 255) if lname.startswith(("pec_fold", "linea")) else (0, 255, 255)
        cv2.polylines(vis, [np.round(pts).astype(np.int32)], False, col, 2, cv2.LINE_AA)
    for pname, pt in p.points.items():
        cv2.drawMarker(vis, (int(pt[0]), int(pt[1])), (0, 0, 255),
                       cv2.MARKER_TILTED_CROSS, 16, 2)
    vis = add_legend(vis)
    return vis, lab


def add_legend(vis: np.ndarray) -> np.ndarray:
    """Append a swatch legend so the result is self-describing."""
    h, w = vis.shape[:2]
    bar = 22
    out = np.full((h + bar, w, 3), 245, np.uint8)
    out[:h] = vis
    x = 6
    for label, key in LEGEND:
        cv2.rectangle(out, (x, h + 5), (x + 14, h + 17), PALETTE[key], -1)
        cv2.rectangle(out, (x, h + 5), (x + 14, h + 17), (40, 40, 40), 1)
        cv2.putText(out, label, (x + 18, h + 17), cv2.FONT_HERSHEY_SIMPLEX,
                    0.42, (30, 30, 30), 1, cv2.LINE_AA)
        x += 22 + int(cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)[0][0])
    return out


def merge_simple(p: PartMasks) -> dict[str, np.ndarray]:
    """Collapse the detailed parts into the four requested classes."""
    out: dict[str, np.ndarray] = {}
    for cls, members in SIMPLE.items():
        acc = None
        for name in members:
            m = p.masks.get(name)
            if m is None:
                continue
            acc = m.copy() if acc is None else cv2.bitwise_or(acc, m)
        if acc is not None:
            out[cls] = acc
    return out


def render_simple(bgr: np.ndarray, p: PartMasks, alpha: np.ndarray | None,
                  alpha_blend: float = 0.45) -> np.ndarray:
    """Paint the four classes over the cut-out person on a checkerboard."""
    h, w = bgr.shape[:2]
    masks = merge_simple(p)
    lab = np.zeros((h, w), np.int32)
    names = [n for n in SIMPLE_ORDER if n in masks]
    for i, name in enumerate(names, start=1):
        lab[masks[name] > 0] = i
    over = bgr.copy()
    for i, name in enumerate(names, start=1):
        over[lab == i] = SIMPLE_COLORS[name]
    vis = cv2.addWeighted(bgr, 1 - alpha_blend, over, alpha_blend, 0)
    for i, name in enumerate(names, start=1):
        m = np.where(lab == i, 255, 0).astype(np.uint8)
        cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(vis, cnts, -1, (15, 15, 15), 2, cv2.LINE_AA)
    # keep the pectoral fold and the abdominal lines visible inside the torso
    for lname, pts in p.lines.items():
        if len(pts) < 2:
            continue
        cv2.polylines(vis, [np.round(pts).astype(np.int32)], False,
                      (255, 255, 255), 2, cv2.LINE_AA)
    # composite onto a checkerboard so the extracted silhouette is obvious
    yy, xx = np.mgrid[0:h, 0:w]
    board = (((yy // 16) + (xx // 16)) % 2 * 40 + 190).astype(np.uint8)
    board = cv2.cvtColor(board, cv2.COLOR_GRAY2BGR)
    if alpha is None:
        a = (p.masks[names[0]] >= 0).astype(np.float32)
        a = np.ones((h, w), np.float32)
    else:
        a = (alpha.astype(np.float32) / 255.0)
    a = a[:, :, None]
    out = (vis.astype(np.float32) * a + board.astype(np.float32) * (1 - a))
    out = cv2.copyMakeBorder(out.astype(np.uint8), 0, 26, 0, 0,
                             cv2.BORDER_CONSTANT, value=(245, 245, 245))
    x = 8
    for name in names:
        cv2.rectangle(out, (x, h + 5), (x + 16, h + 20), SIMPLE_COLORS[name], -1)
        cv2.rectangle(out, (x, h + 5), (x + 16, h + 20), (40, 40, 40), 1)
        cv2.putText(out, name, (x + 21, h + 19), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (25, 25, 25), 1, cv2.LINE_AA)
        x += 30 + int(cv2.getTextSize(name, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)[0][0])
    return out


def to_json(p: PartMasks, stem: str) -> dict:
    out: dict[str, list] = {"image": stem, "parts": {}}
    for name, m in p.masks.items():
        cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        polys = []
        for c in cnts:
            if len(c) < 3:
                continue
            polys.append([[float(x), float(y)] for x, y in c.reshape(-1, 2)])
        out["parts"][name] = polys
    return out


def simple_to_json(bgr: np.ndarray, p: PartMasks, stem: str, alpha: np.ndarray | None
                   ) -> dict:
    """Contours of the four classes, in pixels, plus the source image size."""
    h, w = bgr.shape[:2]
    doc: dict = {"image": stem, "size": [w, h], "classes": {}}
    masks = merge_simple(p)
    # pixels are counted on a disjoint label map so the classes tile the person
    lab = np.zeros((h, w), np.int32)
    names = [n for n in SIMPLE_ORDER if n in masks]
    for i, name in enumerate(names, start=1):
        lab[masks[name] > 0] = i
    for i, name in enumerate(names, start=1):
        m = np.where(lab == i, 255, 0).astype(np.uint8)
        cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        polys = [[[float(x), float(y)] for x, y in c.reshape(-1, 2)]
                 for c in cnts if len(c) >= 3]
        doc["classes"][name] = {
            "pixels": int((lab == i).sum()),
            "polygons": polys,
        }
    if alpha is not None:
        doc["person_pixels"] = int((alpha > 127).sum())
    # internal lines of the torso, so the pec/abs structure is not lost
    doc["lines"] = {k: [[float(x), float(y)] for x, y in v]
                    for k, v in p.lines.items() if len(v) >= 2}
    return doc


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=DATA)
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--model", type=Path, default=MODELS / "pose_landmarker_full.task")
    ap.add_argument("--overrides", type=Path, default=ROOT / "overrides.json",
                    help="per-image manual anchors, see overrides.json")
    ap.add_argument("--mask", choices=("auto", "pose", "extract"), default="auto",
                    help="person mask source: the extracted cutout when available")
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    overrides: dict = {}
    if args.overrides.exists():
        overrides = json.loads(args.overrides.read_text(encoding="utf-8"))
    for path in sorted(args.data.glob("*.png")):
        bgr = load_bgr(path)
        res = detect(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), args.model)
        pose = to_px(res, bgr.shape[1], bgr.shape[0])
        pose_fg = to_mask(res, (bgr.shape[0], bgr.shape[1]))
        # The person mask comes from the extraction step when it is available
        # (it keeps the arms that the pose head misses); grabCut is the fallback.
        # Only small holes are filled: the big one between an arm and the trunk
        # (closed below when the hand touches the hip) is background, and filling
        # it used to put a slab of it into the arm lanes of every row.
        mask = clean_mask(pose_fg)
        cut = args.out / f"fgmask_{path.stem}.png"
        if args.mask in ("auto", "extract") and cut.exists():
            ext = cv2.imdecode(np.fromfile(str(cut), dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
            if ext is not None and (ext > 0).mean() > 0.05:
                mask = clean_mask(ext, max_hole=_hole_limit(ext))
        elif args.mask == "extract" and not cut.exists():
            print(f"    (no {cut.name}, falling back to the pose mask)")
        if (mask > 0).mean() < 0.05:
            raw_fg, _ = foreground_mask(bgr)
            mask = clean_mask(raw_fg, max_hole=_hole_limit(raw_fg))
        ov_img = overrides.get(path.stem) or {}
        # The extractor sometimes cuts real skin away (013 loses ~140 px per side
        # at the top of the trapezius, 012 and 015 as well) and the pose head
        # still has it.  Bring the head back in, but never inside a gap of the
        # extract mask: that is where it bridges the arm/trunk space (or the
        # crotch), and that region is background even though the head paints it.
        add = cv2.bitwise_and(pose_fg, outside_gaps(mask))
        mask = clean_mask(cv2.bitwise_or(mask, add), max_hole=_hole_limit(mask))
        print(f"    silhouette {int((mask > 0).sum())} px, holes up to "
              f"{_hole_limit(mask)} px filled")
        a = compute_anchors(bgr, mask, mask, pose, ov_img)
        p = build_parts(mask, a, bgr, ov_img)
        vis, lab = render(bgr, p)
        save_bgr(args.out / f"parts_{path.stem}.png", vis)
        alpha_path = args.out / f"alpha_{path.stem}.png"
        alpha = None
        if alpha_path.exists():
            alpha = cv2.imdecode(np.fromfile(str(alpha_path), dtype=np.uint8),
                                 cv2.IMREAD_GRAYSCALE)
        save_bgr(args.out / f"simple_{path.stem}.png",
                 render_simple(bgr, p, alpha))
        (args.out / f"simple_{path.stem}.json").write_text(
            json.dumps(simple_to_json(bgr, p, path.stem, alpha), indent=1),
            encoding="utf-8")
        save_bgr(args.out / f"labels_{path.stem}.png",
                 cv2.applyColorMap((lab * 18).astype(np.uint8), cv2.COLORMAP_TURBO))
        (args.out / f"parts_{path.stem}.json").write_text(
            json.dumps(to_json(p, path.stem), indent=1), encoding="utf-8")
        print(f"[{path.stem}] roi={p.roi}")
        simple = merge_simple(p)
        tot = float((alpha > 127).sum()) if alpha is not None else 1.0
        lab_s = np.zeros(bgr.shape[:2], np.int32)
        for i, k in enumerate([n for n in SIMPLE_ORDER if n in simple], start=1):
            lab_s[simple[k] > 0] = i
        print("    4 classes: " + ", ".join(
            f"{k}={100.0 * int((lab_s == i).sum()) / max(1.0, tot):.1f}% of person"
            for i, k in enumerate([n for n in SIMPLE_ORDER if n in simple], start=1)))
        if p.points:
            print("    anchors: " + ", ".join(
                f"{k}=({int(v[0])},{int(v[1])})" for k, v in p.points.items()))
        for name in PRIORITY:
            if name in p.masks:
                print(f"    {name:<14} {100.0 * (p.masks[name] > 0).mean():5.2f}% of frame")


if __name__ == "__main__":
    main()
