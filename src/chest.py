"""Nipple-anchored pectoral fold.

The nipple/areola is a compact dark blob on an otherwise smooth chest, which
makes it a stable anchor. The inferior border of pectoralis major is then the
*first* dark ridge below the nipple, so the fold is found by scanning down each
column from just under the nipple instead of guessing a global height.
"""

from __future__ import annotations

import cv2
import numpy as np


def _norm(a: np.ndarray) -> np.ndarray:
    a = a.astype(np.float32)
    lo, hi = float(a.min()), float(a.max())
    return (a - lo) / (hi - lo) if hi > lo else np.zeros_like(a)


def nipple_map(bgr: np.ndarray, k: int) -> np.ndarray:
    """Small dark blobs, with a colour term when the image has real colour."""
    gray = cv2.GaussianBlur(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY), (0, 0), 1.4)
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k | 1, k | 1))
    bh = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, ker).astype(np.float32)
    bg = cv2.GaussianBlur(gray, (0, 0), max(2.0, k * 0.9))
    dark = np.clip(bg - gray, 0, None)
    resp = 0.55 * _norm(bh) + 0.45 * _norm(dark)

    b, g, r = (bgr[:, :, i].astype(np.float32) for i in range(3))
    chroma = (r - b)
    if float(np.percentile(np.abs(chroma), 95)) > 12.0:
        # areola is usually redder than the surrounding skin
        col = cv2.GaussianBlur(chroma, (0, 0), 1.4)
        bgc = cv2.GaussianBlur(col, (0, 0), max(2.0, k * 0.9))
        resp = 0.5 * resp + 0.5 * _norm(np.clip(col - bgc, 0, None))
    return resp


def blob_nipples(bgr: np.ndarray, mask: np.ndarray, s: float,
                 y0: float, y1: float) -> list[tuple[float, float, float]]:
    """Round dark blobs of nipple size; rejects the elongated pectoral shadow."""
    h, w = mask.shape[:2]
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (0, 0), max(1.0, 0.012 * s))
    inv = cv2.bitwise_not(gray)
    inv[mask == 0] = 0

    d = 0.048 * s
    params = cv2.SimpleBlobDetector_Params()
    params.filterByColor = True
    params.blobColor = 255
    params.filterByArea = True
    params.minArea = float(np.pi * (0.35 * d) ** 2)
    params.maxArea = float(np.pi * (1.9 * d) ** 2)
    params.filterByCircularity = True
    params.minCircularity = 0.55
    params.filterByInertia = True
    params.minInertiaRatio = 0.45
    params.filterByConvexity = True
    params.minConvexity = 0.7
    params.minDistBetweenBlobs = float(max(6.0, 0.5 * d))
    det = cv2.SimpleBlobDetector_create(params)

    blobs = []
    for kp in det.detect(inv):
        x, y = kp.pt
        if not (y0 <= y <= y1):
            continue
        if mask[int(min(h - 1, max(0, y))), int(min(w - 1, max(0, x)))] == 0:
            continue
        blobs.append((float(x), float(y), float(kp.size)))
    return blobs


def symmetric_dark_pair(bgr: np.ndarray, mask: np.ndarray, x_mid: float, s: float,
                        y_lo: float, y_hi: float
                        ) -> dict[int, np.ndarray]:
    """Search for the mirror-symmetric dark pair that a pair of nipples forms.

    Horizontally elongated structures (the pectoral shadow under the muscle) are
    suppressed by subtracting a wide horizontally blurred copy, so what remains
    is compact blobs -- nipples, navel, small marks.
    """
    h, w = mask.shape[:2]
    gray = cv2.GaussianBlur(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY), (0, 0), 1.4)
    local = cv2.GaussianBlur(gray, (0, 0), max(1.5, 0.055 * s))
    spot = np.clip(local - gray, 0, None).astype(np.float32)
    wide = cv2.GaussianBlur(spot, (max(3, int(0.22 * s) | 1), 1), 0)
    spot = np.clip(spot - wide, 0, None)
    spot = cv2.GaussianBlur(spot, (0, 0), max(1.0, 0.018 * s))
    spot[mask == 0] = 0

    y0, y1 = int(max(0, y_lo)), int(min(h, y_hi))
    off_lo, off_hi = int(0.14 * s), int(0.40 * s)
    best, best_score = None, -1e9
    for y in range(y0, y1):
        for off in range(off_lo, off_hi + 1):
            xl, xr = int(round(x_mid)) - off, int(round(x_mid)) + off
            if not (0 <= xl < w and 0 <= xr < w):
                continue
            score = float(spot[y, xl] + spot[y, xr])
            if score > best_score:
                best, best_score = (xl, xr, y), score
    if best is None or best_score <= 0:
        return {}

    xl, xr, y = best
    out: dict[int, np.ndarray] = {}
    for side, x0 in ((-1, xl), (+1, xr)):
        win = max(2, int(0.05 * s))
        sub = spot[max(0, y - win): y + win + 1, max(0, x0 - win): x0 + win + 1]
        if sub.size == 0 or sub.max() <= 0:
            continue
        iy, ix = np.unravel_index(int(np.argmax(sub)), sub.shape)
        out[side] = np.array([float(max(0, x0 - win) + ix),
                              float(max(0, y - win) + iy)])
    return out if len(out) == 2 else {}


def find_nipples(bgr: np.ndarray, mask: np.ndarray, x_mid: float, s: float,
                 y_lo: float, y_hi: float) -> dict[int, np.ndarray]:
    """Locate one nipple per side, using a symmetric, prior-consistent pair."""
    h, w = mask.shape[:2]
    resp = nipple_map(bgr, k=max(5, int(0.085 * s)))
    blobs = blob_nipples(bgr, mask, s, y_lo, y_hi)
    d = max(3.0, 0.048 * s)

    out: dict[int, np.ndarray] = {}
    cands: dict[int, list[tuple[float, float, float]]] = {-1: [], +1: []}
    for x, y, size in blobs:
        dx = x - x_mid
        if not (0.12 * s <= abs(dx) <= 0.40 * s):
            continue
        if not (y_lo <= y <= y_hi):
            continue
        side = -1 if dx < 0 else +1
        r = float(resp[int(y), int(x)]) + 0.6 * max(0.0, 1.0 - abs(size - d) / d)
        cands[side].append((x, y, r))

    if not cands[-1] or not cands[+1]:
        # fall back to the compact mirror-symmetric dark pair
        return symmetric_dark_pair(bgr, mask, x_mid, s, y_lo, y_hi)

    y_expect = 0.5 * (y_lo + y_hi)
    best, best_score = None, -1e9
    for xl, yl, vl in cands[-1]:
        for xr, yr, vr in cands[+1]:
            dy = abs(yl - yr)
            dx = abs((x_mid - xl) - (xr - x_mid))
            score = (vl + vr) - 1.2 * dy / max(1.0, s) - 0.8 * dx / max(1.0, s) \
                - 0.8 * abs(0.5 * (yl + yr) - y_expect) / max(1.0, s)
            if score > best_score:
                best, best_score = ((-1, np.array([xl, yl])), (+1, np.array([xr, yr]))), score
    if best is not None:
        for side, pt in best:
            out[side] = pt
    return out


def first_edge_below(v: np.ndarray, e: np.ndarray, x: int, y_from: int, y_to: int,
                     w_min: float = 0.16) -> float | None:
    """First strong dark ridge at or below ``y_from`` in column ``x``."""
    y0, y1 = int(max(0, y_from)), int(min(v.shape[0], y_to))
    if y1 - y0 < 3:
        return None
    col = v[y0:y1, x]
    if col.max() < w_min:
        return None
    top = float(col.max())
    for i in range(1, len(col) - 1):
        if col[i] >= w_min and col[i] >= 0.45 * top \
           and col[i] >= col[i - 1] and col[i] >= col[i + 1]:
            # refine on the luminance step, which is sharper than the ridge map
            lo, hi = y0 + i - 3, y0 + i + 4
            if 0 <= lo < hi <= v.shape[0]:
                seg = e[lo:hi, x]
                if seg.size:
                    return float(lo + int(np.argmax(seg)))
            return float(y0 + i)
    return None


def fold_below_nipples(v: np.ndarray, e: np.ndarray, mask: np.ndarray,
                       x_mid: float, s: float, nipples: dict[int, np.ndarray],
                       x_lat: dict[int, float], y_arm: float) -> dict[int, np.ndarray]:
    """Trace the pectoral fold laterally at the first edge below each nipple."""
    h, w = mask.shape[:2]
    step = max(2, int(0.020 * s))
    curves: dict[int, np.ndarray] = {}
    for side in (-1, +1):
        if side not in nipples:
            continue
        nx, ny = float(nipples[side][0]), float(nipples[side][1])
        y_start, y_end = ny + 0.06 * s, ny + 0.34 * s
        xs = np.arange(int(round(x_mid)), int(round(x_lat[side])), side * step)
        if xs.size < 3:
            continue
        ys = np.full(len(xs), np.nan)
        # the medial stretch between the sternum and the nipple keeps the value
        # found under the nipple unless a crease pulls it up
        for i, x in enumerate(xs):
            if not (0 <= x < w):
                continue
            near_nipple = abs(x - nx) <= 0.30 * s
            y_from = y_start if near_nipple else ny + 0.02 * s
            y_to = y_end
            y = first_edge_below(v, e, int(x), int(y_from), int(y_to))
            if y is None:
                y = ys[i - 1] if i and not np.isnan(ys[i - 1]) else ny + 0.10 * s
            ys[i] = y
        # anatomical ceiling: monotone rise from the sternum end to the axilla
        y_med = float(ys[0]) if not np.isnan(ys[0]) else ny + 0.10 * s
        span = max(1e-6, abs(x_lat[side] - x_mid))
        ceil = y_med + (y_arm - y_med) * (np.abs(xs - x_mid) / span)
        ys = np.clip(ys, None, ceil)
        good = ~np.isnan(ys)
        if good.sum() >= 2:
            ys = np.interp(np.arange(len(ys)), np.flatnonzero(good), ys[good])
        else:
            continue
        k = 7
        pad = k // 2
        ys = np.array([np.median(ys[max(0, i - pad): i + pad + 1]) for i in range(len(ys))])
        curves[side] = np.stack([xs.astype(float), ys], 1)
    return curves


def find_navel(bgr: np.ndarray, mask: np.ndarray, x_mid: float, s: float,
               y_top: float, y_bot: float) -> np.ndarray | None:
    """The navel: the most compact dark spot on the midline of the abdomen.

    The linea alba is a vertical groove and the waistband a horizontal band, so
    both are suppressed by removing a wide horizontally blurred copy before
    looking for a small round blob.
    """
    h, w = mask.shape[:2]
    gray = cv2.GaussianBlur(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY), (0, 0), 1.4)
    local = cv2.GaussianBlur(gray, (0, 0), max(1.5, 0.055 * s))
    spot = np.clip(local - gray, 0, None).astype(np.float32)
    wide = cv2.GaussianBlur(spot, (max(3, int(0.20 * s) | 1), 1), 0)
    spot = np.clip(spot - wide, 0, None)
    spot = cv2.GaussianBlur(spot, (0, 0), max(1.0, 0.020 * s))

    band = np.zeros((h, w), np.float32)
    y0, y1 = int(max(0, y_top)), int(min(h, y_bot))
    x0 = int(max(0, x_mid - 0.10 * s))
    x1 = int(min(w, x_mid + 0.10 * s))
    if y1 - y0 < 4 or x1 - x0 < 4:
        return None
    band[y0:y1, x0:x1] = spot[y0:y1, x0:x1]
    band[mask == 0] = 0
    if band.max() <= 0:
        return None
    iy, ix = np.unravel_index(int(np.argmax(band)), band.shape)
    return np.array([float(ix), float(iy)])
