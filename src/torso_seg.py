"""Stage 1: foreground extraction, torso ROI and landmark geometry.

Only cv2 / numpy / scipy are used so the pipeline runs on the CPU-only
``friday`` environment without any extra download.
"""

from __future__ import annotations

import argparse
import shutil
import tempfile
from dataclasses import dataclass, asdict
from pathlib import Path

import cv2
import numpy as np
from scipy import ndimage as ndi

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = ROOT / "out"
MODELS = ROOT / "models"


# --------------------------------------------------------------------------- io
def load_bgr(path: Path) -> np.ndarray:
    """cv2.imread breaks on non-ascii paths, imdecode does not."""
    buf = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"cannot decode {path}")
    return img


def save_bgr(path: Path, img: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, buf = cv2.imencode(path.suffix or ".png", img)
    if not ok:
        raise RuntimeError(f"cannot encode {path}")
    buf.tofile(str(path))


def ascii_path(path: Path) -> Path:
    """MediaPipe/onnxruntime C++ loaders cannot open non-ascii paths on Windows.

    Returns the original path when it is already ascii, otherwise a cached copy
    under the system temp directory whose path is ascii.
    """
    if str(path).isascii():
        return path
    cache = Path(tempfile.gettempdir()) / "torso_seg_models"
    cache.mkdir(parents=True, exist_ok=True)
    dst = cache / path.name
    if not dst.exists() or dst.stat().st_size != path.stat().st_size:
        shutil.copy2(path, dst)
    return dst


# ------------------------------------------------------------------ foreground
def skin_mask(bgr: np.ndarray) -> np.ndarray:
    """Classic YCrCb skin rule, works well on hard-lit bare skin."""
    ycrcb = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = cv2.split(ycrcb)
    m = (cr > 133) & (cr < 180) & (cb > 77) & (cb < 130) & (y > 40)
    return (m.astype(np.uint8)) * 255


def foreground_mask(bgr: np.ndarray, iters: int = 5) -> tuple[np.ndarray, np.ndarray]:
    """GrabCut seeded with a border ring, a centre ellipse and the skin prior."""
    h, w = bgr.shape[:2]
    gc = np.full((h, w), cv2.GC_PR_BGD, np.uint8)

    ring = max(2, int(round(min(h, w) * 0.01)))
    gc[:ring] = cv2.GC_BGD
    gc[-ring:] = cv2.GC_BGD
    gc[:, :ring] = cv2.GC_BGD
    gc[:, -ring:] = cv2.GC_BGD

    cv2.ellipse(
        gc,
        (w // 2, int(h * 0.55)),
        (int(w * 0.42), int(h * 0.45)),
        0, 0, 360, cv2.GC_PR_FGD, -1,
    )
    sk = skin_mask(bgr)
    sk = cv2.morphologyEx(sk, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    gc[sk > 0] = cv2.GC_PR_FGD

    bgd, fgd = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)
    cv2.grabCut(bgr, gc, None, bgd, fgd, iters, cv2.GC_INIT_WITH_MASK)
    fg = np.where((gc == cv2.GC_FGD) | (gc == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
    return fg, sk


def clean_mask(fg: np.ndarray, fill_holes: bool = True,
               max_hole: int | None = None) -> np.ndarray:
    """Largest connected component + hole filling + light morphological repair.

    max_hole fills only the holes up to that area in pixels and leaves the big
    ones open.  The silhouette between an arm and the trunk is a hole as soon as
    the arm touches the body somewhere below it (a hand on the hip, a forearm on
    the thigh), and filling it puts a slab of background inside the person, in
    the lanes of every row and in the part masks.  Small holes are still noise.
    """
    m = fg > 0
    lab, n = ndi.label(m)
    if n > 1:
        sizes = ndi.sum(m, lab, range(1, n + 1))
        m = lab == (int(np.argmax(sizes)) + 1)
    if fill_holes and max_hole is None:
        m = ndi.binary_fill_holes(m)
    elif fill_holes:
        holes = ndi.binary_fill_holes(m) & ~m
        lab_h, n_h = ndi.label(holes)
        for i in range(1, n_h + 1):
            h = lab_h == i
            if int(h.sum()) <= max_hole:
                m = m | h
    m = ndi.binary_closing(m, np.ones((5, 5), bool))
    m = ndi.binary_opening(m, np.ones((3, 3), bool))
    return (m.astype(np.uint8)) * 255


def outside_gaps(mask: np.ndarray) -> np.ndarray:
    """255 outside the gaps of a mask: not between two runs of a row, and not
    inside one of its holes.

    A second, blobbier person mask (the pose head) may be brought in to repair a
    silhouette, but never inside those gaps: that is exactly where it bridges
    the space between an arm and the trunk, or between the legs.
    """
    b = np.asarray(mask) > 0
    keep = np.zeros_like(b)
    for y in range(b.shape[0]):
        idx = np.flatnonzero(b[y])
        if idx.size < 2:
            keep[y] = True
            continue
        keep[y, :idx[0]] = True          # outside the body on the left
        keep[y, idx[-1] + 1:] = True     # ... and on the right
        keep[y, idx] = True              # and the body itself
    return (keep.astype(np.uint8)) * 255


# ------------------------------------------------------------------- geometry
@dataclass
class Landmarks:
    y_top: int
    y_shoulder: int
    y_armpit: int
    y_waist: int
    y_bottom: int
    x_center: int
    x_left_shoulder: int
    x_right_shoulder: int
    shoulder_width: int
    neck_width: int
    roi: tuple[int, int, int, int]      # x0, y0, x1, y1


def row_extents(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per row: left edge, right edge and width of the silhouette."""
    h, w = mask.shape
    left = np.full(h, -1, np.int32)
    right = np.full(h, -1, np.int32)
    for y in range(h):
        xs = np.flatnonzero(mask[y] > 0)
        if xs.size:
            left[y], right[y] = int(xs[0]), int(xs[-1])
    width = np.where(left >= 0, right - left + 1, 0)
    return left, right, width


def smooth(v: np.ndarray, k: int = 15) -> np.ndarray:
    k = max(3, k | 1)
    ker = np.ones(k) / k
    pad = k // 2
    return np.convolve(np.pad(v, pad, mode="edge"), ker, mode="valid")


def find_landmarks(mask: np.ndarray) -> Landmarks:
    h, w = mask.shape
    left, right, width = row_extents(mask)
    rows = np.flatnonzero(width > 0)
    y_top, y_bottom = int(rows[0]), int(rows[-1])
    span = y_bottom - y_top
    wsm = smooth(width[y_top:y_bottom + 1].astype(np.float64), 15)

    # Shoulder line = widest row of the upper third of the body.
    upper = int(len(wsm) * 0.33)
    y_shoulder = y_top + int(np.argmax(wsm[:upper]))
    shoulder_width = int(wsm[y_shoulder - y_top])

    # Neck/traps apex = narrowest row above the shoulder line.
    y_neck = y_top + int(np.argmin(wsm[: max(3, y_shoulder - y_top)]))
    neck_width = int(wsm[y_neck - y_top])

    # Armpit: first row below the shoulders where the width drops clearly and
    # the silhouette stops growing (arms hanging alongside the torso).
    y_armpit = y_shoulder
    for y in range(y_shoulder, min(y_shoulder + int(span * 0.35), y_bottom)):
        if wsm[y - y_top] < 0.93 * shoulder_width:
            y_armpit = y
            break

    # Waist = narrowest row between armpit and the bottom of the visible body.
    lo = min(y_armpit + int(span * 0.15), y_bottom - 1)
    seg = wsm[lo - y_top: y_bottom - y_top + 1]
    y_waist = lo + int(np.argmin(seg))

    xl, xr = int(left[y_shoulder]), int(right[y_shoulder])
    x_center = (xl + xr) // 2

    y0 = max(0, y_neck - int(0.02 * span))
    y1 = min(h - 1, y_waist + int(0.06 * span))
    x0 = max(0, int(np.min(left[left >= 0])) - 4)
    x1 = min(w - 1, int(np.max(right[right >= 0])) + 4)

    return Landmarks(
        y_top=y_top, y_shoulder=y_shoulder, y_armpit=y_armpit, y_waist=y_waist,
        y_bottom=y_bottom, x_center=x_center, x_left_shoulder=xl,
        x_right_shoulder=xr, shoulder_width=shoulder_width, neck_width=neck_width,
        roi=(x0, y0, x1, y1),
    )


# ---------------------------------------------------------------------- render
def draw_debug(bgr: np.ndarray, mask: np.ndarray, lm: Landmarks) -> np.ndarray:
    vis = bgr.copy()
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(vis, cnts, -1, (0, 255, 0), 1)

    x0, y0, x1, y1 = lm.roi
    cv2.rectangle(vis, (x0, y0), (x1, y1), (255, 160, 0), 1)

    def hline(y, color, label):
        cv2.line(vis, (0, y), (vis.shape[1] - 1, y), color, 1)
        cv2.putText(vis, label, (4, max(12, y - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1, cv2.LINE_AA)

    hline(lm.y_shoulder, (0, 0, 255), "shoulder")
    hline(lm.y_armpit, (0, 210, 255), "armpit")
    hline(lm.y_waist, (255, 0, 255), "waist")
    cv2.line(vis, (lm.x_center, y0), (lm.x_center, y1), (0, 255, 255), 1)
    cv2.circle(vis, (lm.x_left_shoulder, lm.y_shoulder), 3, (0, 0, 255), -1)
    cv2.circle(vis, (lm.x_right_shoulder, lm.y_shoulder), 3, (0, 0, 255), -1)
    return vis


# ------------------------------------------------------------------- creases
def crease_map(bgr: np.ndarray, k: int = 23, blur: float = 1.2) -> np.ndarray:
    """Black top-hat: responds to dark valleys narrower than ``k`` pixels.

    On a hard-lit bare torso the anatomical boundaries (clavicle shadow,
    deltopectoral groove, pectoral fold, linea alba, tendinous intersections,
    waistband) are all dark creases, so this is the signal that carries the
    real contours.
    """
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (0, 0), blur)
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    bh = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, ker)
    return cv2.normalize(bh, None, 0, 255, cv2.NORM_MINMAX)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=DATA)
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--crease-k", type=int, default=23)
    ap.add_argument("--crease-k2", type=int, default=0,
                    help="optional second crease scale; writes creasegray<k2>_*.png")
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    for path in sorted(args.data.glob("*.png")):
        bgr = load_bgr(path)
        fg, sk = foreground_mask(bgr)
        mask = clean_mask(fg)
        lm = find_landmarks(mask)

        stem = path.stem
        save_bgr(args.out / f"mask_{stem}.png", mask)
        save_bgr(args.out / f"debug_{stem}.png", draw_debug(bgr, mask, lm))
        cm = crease_map(bgr, args.crease_k)
        save_bgr(args.out / f"crease_{stem}.png", cv2.applyColorMap(cm, cv2.COLORMAP_INFERNO))
        save_bgr(args.out / f"creasegray_{stem}.png", cm)
        if args.crease_k2:
            cm2 = crease_map(bgr, args.crease_k2)
            save_bgr(args.out / f"creasegray{args.crease_k2}_{stem}.png", cm2)
            save_bgr(args.out / f"crease{args.crease_k2}_{stem}.png",
                     cv2.applyColorMap(cm2, cv2.COLORMAP_INFERNO))

        h, w = mask.shape[:2]
        print(f"[{stem}] {w}x{h} fg={100.0 * (mask > 0).mean():.1f}% "
              f"skin={(sk > 0).mean() * 100:.1f}%")
        print("   " + ", ".join(f"{k}={v}" for k, v in asdict(lm).items()))


if __name__ == "__main__":
    main()
