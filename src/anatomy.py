"""Anatomical anchors derived from pose landmarks plus silhouette geometry."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from pose_mp import detect, to_mask, to_px
from torso_seg import (DATA, MODELS, OUT, clean_mask, foreground_mask, load_bgr,
                       save_bgr, skin_mask)

MODEL = MODELS / "pose_landmarker_full.task"


def runs_of(row: np.ndarray, min_len: int = 5) -> list[tuple[int, int]]:
    """Inclusive-exclusive index runs of True values longer than ``min_len``."""
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


@dataclass
class Anchors:
    w: int
    h: int
    side: dict[str, tuple[float, float]] = field(default_factory=dict)
    mid: dict[str, tuple[float, float]] = field(default_factory=dict)
    y: dict[str, float] = field(default_factory=dict)
    scale: float = 0.0            # shoulder width, everything is normalised by it
    peaks: list = field(default_factory=list)   # (y, strength) horizontal edges

    def pt(self, name: str) -> np.ndarray:
        if name in self.side:
            return np.array(self.side[name], float)
        return np.array(self.mid[name], float)


def compute_anchors(bgr: np.ndarray, raw_fg: np.ndarray, mask: np.ndarray,
                    pose: dict[str, tuple[float, float, float]],
                    ov: dict | None = None) -> Anchors:
    h, w = mask.shape[:2]
    a = Anchors(w=w, h=h)
    req = ["left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
           "left_wrist", "right_wrist", "left_hip", "right_hip"]
    miss = [k for k in req if k not in pose]
    if miss:
        raise RuntimeError(f"pose landmarks missing: {miss}")

    for k in req:
        x, y, _ = pose[k]
        a.side[k] = (float(x), float(y))

    sh = {s: a.pt(f"{s}_shoulder") for s in ("left", "right")}
    hp = {s: a.pt(f"{s}_hip") for s in ("left", "right")}
    el = {s: a.pt(f"{s}_elbow") for s in ("left", "right")}
    wr = {s: a.pt(f"{s}_wrist") for s in ("left", "right")}

    mid_sh = (sh["left"] + sh["right"]) / 2.0
    mid_hp = (hp["left"] + hp["right"]) / 2.0
    a.scale = float(np.linalg.norm(sh["left"] - sh["right"]))
    S = a.scale

    # image-left / image-right naming: pose 'left' is the subject's left hand
    # side which appears on the image right when facing the camera.
    img_l = "right"      # subject's right shoulder is on the image left
    img_r = "left"
    a.mid["mid_shoulder"] = tuple(mid_sh)
    a.mid["mid_hip"] = tuple(mid_hp)
    a.mid["x_mid"] = (float(mid_sh[0]), float(mid_hp[1]))

    # sternal notch / front of the neck base
    a.mid["notch"] = (float(mid_sh[0]), float(mid_sh[1] - 0.06 * S))
    a.mid["neck"] = (float(mid_sh[0]), float(mid_sh[1] - 0.30 * S))

    # silhouette corners: acromion = top-outer corner of the shoulder
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cnt = max(cnts, key=cv2.contourArea).reshape(-1, 2).astype(np.float64)
    # The corner is searched only in a vertical band around the shoulder joint:
    # when the head is inside the crop its silhouette corner is more "top-left"
    # than the acromion and used to be picked by mistake.
    lo = mid_sh[1] - 0.30 * S
    hi = mid_sh[1] + 0.35 * S
    band = (cnt[:, 1] > max(0.0, lo)) & (cnt[:, 1] < hi)
    if band.sum() < 5:                      # fallback: the old whole-upper-body band
        band = (cnt[:, 1] > mask.shape[0] * 0.02) & (cnt[:, 1] < mid_sh[1] + 0.30 * S)
    # acromion from the pose shoulder joint (reliable even when the head is in
    # frame, where a pure silhouette corner search lands on the hair): the
    # acromion sits ~0.11S lateral and ~0.04S above the joint, then snaps to the
    # silhouette edge of that row.
    def corner(side_key: str, side: int) -> np.ndarray:
        sh = np.asarray(pose[f"{side_key}_shoulder"][:2], float)
        ax = sh[0] + side * 0.11 * S
        ay = sh[1] - 0.04 * S
        y = int(np.clip(round(ay), 0, mask.shape[0] - 1))
        xs = np.flatnonzero(mask[y] > 0)
        if xs.size:
            edge = float(xs[0] if side < 0 else xs[-1])
            if (edge - ax) * side >= 0 or abs(edge - ax) < 0.12 * S:
                ax = edge - side * 0.01 * S
        # the acromion lies on the body, so it cannot sit above the silhouette of
        # its own column
        col = int(np.clip(round(ax), 0, mask.shape[1] - 1))
        ys = np.flatnonzero(mask[:, col] > 0)
        if ys.size:
            ay = max(ay, float(ys[0]) + 0.01 * S)
        return np.array([ax, ay], float)

    left_corner = corner("right", -1)      # image-left is the subject's right
    right_corner = corner("left", +1)
    a.side["acromion_img_l"] = (float(left_corner[0]), float(left_corner[1]))
    a.side["acromion_img_r"] = (float(right_corner[0]), float(right_corner[1]))

    # armpit: first row below the shoulder where the body splits into >=3 runs
    y_from = int(mid_sh[1] + 0.05 * S)
    armpit_y = None
    for y in range(y_from, min(h, int(mid_hp[1]))):
        if len(runs_of(raw_fg[y] > 0, min_len=6)) >= 3:
            armpit_y = y
            break
    if armpit_y is None:
        for y in range(y_from, min(h, int(mid_hp[1]))):
            if len(runs_of(mask[y] > 0, min_len=6)) >= 3:
                armpit_y = y
                break
    if armpit_y is None:
        # no visible gap between arm and torso (arms close to the body, hands on
        # hips, ...): fall back to anthropometry, axilla apex sits ~1/3 of the
        # way down the upper arm
        el_mean = float(np.mean([p[1] for p in (el["left"], el["right"])]))
        armpit_y = int(round(mid_sh[1] + 0.34 * (el_mean - mid_sh[1])))
    a.y["armpit"] = float(armpit_y)

    # pelvis / leg split: first row below the hips where the torso splits in two
    split_y = None
    for y in range(int(mid_hp[1] - 0.10 * S), h):
        r = runs_of(mask[y] > 0, min_len=6)
        core = [(s, e) for s, e in r
                if abs(0.5 * (s + e) - mid_sh[0]) < 0.45 * S]
        if len(core) >= 2:
            split_y = y
            break
    a.y["leg_split"] = float(split_y if split_y is not None else h - 1)

    # Waistband = top edge of the garment. The garment is dark on both samples
    # and one of them is almost desaturated, so this uses luminance steps in the
    # central band, optionally confirmed by the skin prior.
    xs = np.arange(w)
    central = np.abs(xs - mid_sh[0]) < 0.45 * S
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float64)
    gp = cv2.GaussianBlur(gray, (0, 0), 2.0)[:, central].mean(axis=1)
    gp = np.convolve(gp, np.ones(9) / 9, mode="same")
    skin = (skin_mask(bgr) > 0)[:, central].mean(axis=1)
    skin = np.convolve(skin, np.ones(9) / 9, mode="same")

    lo = int(mid_sh[1] + 0.35 * (mid_hp[1] - mid_sh[1]))
    hi = int(min(h - 12, max(a.y["leg_split"], mid_hp[1])))
    k = 12
    drop = np.zeros(max(0, hi - lo))
    for i, y in enumerate(range(lo, hi)):
        drop[i] = gp[y] - gp[min(h - 1, y + k)]
    best = int(np.argmax(drop)) if drop.size else 0
    y_step = lo + best
    # refined edge: halfway through the step
    target = 0.5 * (gp[y_step] + gp[min(h - 1, y_step + k)])
    y_ref = y_step
    for y in range(y_step, min(h, y_step + k + 6)):
        if gp[y] <= target:
            y_ref = y
            break
    a.y["waistband"] = float(y_ref)
    a.y["waistband_step"] = float(y_step)
    a.peaks = [(int(y), round(float(gp[y]), 1)) for y in range(lo, hi, max(1, (hi - lo) // 12))]

    # anthropometric fall-backs / cross-checks
    a.y["pec_low"] = float(mid_sh[1] + 0.66 * (mid_hp[1] - mid_sh[1]))
    a.y["navel"] = float(mid_sh[1] + 0.88 * (mid_hp[1] - mid_sh[1]))
    a.y["neck_top"] = float(max(0.0, a.mid["neck"][1] - 0.25 * S))
    # sanity clamp: the garment edge lies within a shoulder width of the hip
    # joints and the abdomen keeps room below the pectoral fold
    y_lo = max(mid_hp[1] - 0.35 * S,
               a.y["pec_low"] + 0.20 * (mid_hp[1] - a.y["pec_low"]))
    a.y["waistband"] = float(np.clip(a.y["waistband"], y_lo, mid_hp[1] + 0.45 * S))
    # manual anchors from overrides.json win over anything computed above
    ov = ov or {}
    for key, name in (("acromion_l", "acromion_img_l"), ("acromion_r", "acromion_img_r")):
        pt = ov.get(key)
        if pt:
            a.side[name] = (float(pt[0]), float(pt[1]))
    if ov.get("armpit"):
        a.y["armpit"] = float(ov["armpit"])
    if ov.get("waistband"):
        a.y["waistband"] = float(ov["waistband"])
    return a


def draw_anchors(bgr: np.ndarray, a: Anchors) -> np.ndarray:
    vis = bgr.copy()
    for name, (x, y) in {**a.side, **a.mid}.items():
        cv2.drawMarker(vis, (int(x), int(y)), (0, 0, 255), cv2.MARKER_CROSS, 14, 2)
        cv2.putText(vis, name, (int(x) + 6, int(y) - 6), cv2.FONT_HERSHEY_SIMPLEX,
                    0.42, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(vis, name, (int(x) + 6, int(y) - 6), cv2.FONT_HERSHEY_SIMPLEX,
                    0.42, (0, 255, 255), 1, cv2.LINE_AA)
    for i, (name, yy) in enumerate(a.y.items()):
        col = [(255, 0, 255), (0, 255, 0), (255, 128, 0), (0, 200, 255),
               (128, 255, 128)][i % 5]
        cv2.line(vis, (0, int(yy)), (a.w - 1, int(yy)), col, 1)
        cv2.putText(vis, f"{name} y={int(yy)}", (3, int(yy) - 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, col, 1, cv2.LINE_AA)
    return vis


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=DATA)
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--model", type=Path, default=MODEL)
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    for path in sorted(args.data.glob("*.png")):
        bgr = load_bgr(path)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        res = detect(rgb, args.model)
        pose = to_px(res, bgr.shape[1], bgr.shape[0])
        raw_fg, _ = foreground_mask(bgr)
        pose_fg = to_mask(res, mask_shape := (bgr.shape[0], bgr.shape[1]))
        mask = clean_mask(cv2.bitwise_or(pose_fg, clean_mask(raw_fg)))
        a = compute_anchors(bgr, pose_fg, mask, pose)
        save_bgr(args.out / f"anchor_{path.stem}.png", draw_anchors(bgr, a))
        save_bgr(args.out / f"fg_{path.stem}.png", mask)
        print(f"[{path.stem}] S(shoulder width)={a.scale:.1f}px")
        print("   side:", {k: (round(v[0], 1), round(v[1], 1)) for k, v in a.side.items()})
        print("   mid :", {k: (round(v[0], 1), round(v[1], 1)) for k, v in a.mid.items()})
        print("   y   :", {k: round(v, 1) for k, v in a.y.items()})
        print("   edges:", a.peaks)


if __name__ == "__main__":
    main()
