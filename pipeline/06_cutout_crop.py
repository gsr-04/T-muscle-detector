"""Crop each RGBA cutout down to the person's bounding box.

    python tools/cutout_crop.py [--margin 0] [--square]

Writes out/crop_<name>.png (RGBA) plus out/crops.json holding the boxes so any
coordinate can be mapped back to the original frame.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def load_rgba(path: Path) -> np.ndarray:
    img = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise RuntimeError(f"cannot decode {path}")
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGRA)
    if img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
    return img


def save_rgba(path: Path, img: np.ndarray) -> None:
    ok, buf = cv2.imencode(path.suffix or ".png", img)
    if not ok:
        raise RuntimeError(f"cannot encode {path}")
    buf.tofile(str(path))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=ROOT / "out")
    ap.add_argument("--margin", type=int, default=0,
                    help="extra pixels kept around the subject")
    ap.add_argument("--square", action="store_true",
                    help="pad the crop to a square (centred on the subject)")
    ap.add_argument("--alpha-thr", type=int, default=127)
    args = ap.parse_args()

    boxes: dict[str, list[int]] = {}
    print(f"{'image':8s} {'cutout':>13s} {'crop':>13s}  box (x0,y0,x1,y1)  "
          f"kept%")
    for path in sorted(args.out.glob("cutout_*.png")):
        img = load_rgba(path)
        h, w = img.shape[:2]
        ys, xs = np.nonzero(img[:, :, 3] > args.alpha_thr)
        if not ys.size:
            print(f"{path.stem:8s} empty alpha, skipped")
            continue
        x0, x1 = int(xs.min()), int(xs.max()) + 1
        y0, y1 = int(ys.min()), int(ys.max()) + 1
        x0 = max(0, x0 - args.margin)
        y0 = max(0, y0 - args.margin)
        x1 = min(w, x1 + args.margin)
        y1 = min(h, y1 + args.margin)
        if args.square:
            side = max(x1 - x0, y1 - y0)
            cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
            x0 = max(0, min(w - side, cx - side // 2))
            y0 = max(0, min(h - side, cy - side // 2))
            x1, y1 = min(w, x0 + side), min(h, y0 + side)
        crop = img[y0:y1, x0:x1]
        name = path.stem.replace("cutout_", "")
        save_rgba(args.out / f"crop_{name}.png", crop)
        # carry the annotations into the same crop frame when present
        for src, dst in ((f"ckstyle_{name}.png", f"cropck_{name}.png"),
                         (f"simple_{name}.png", f"cropsimple_{name}.png")):
            p = args.out / src
            if p.exists():
                v = cv2.imdecode(np.fromfile(str(p), np.uint8), cv2.IMREAD_UNCHANGED)
                save_rgba(args.out / dst, v[y0:y1, x0:x1])
        boxes[name] = [x0, y0, x1, y1]
        kept = 100.0 * crop.shape[0] * crop.shape[1] / (h * w)
        print(f"{name:8s} {w:5d}x{h:<7d} {crop.shape[1]:5d}x{crop.shape[0]:<7d}  "
              f"({x0},{y0},{x1},{y1})  {kept:5.1f}%")

    (args.out / "crops.json").write_text(json.dumps(boxes, indent=1), encoding="utf-8")
    print(f"\nwrote {len(boxes)} crops + out/crops.json "
          f"(x0,y0,x1,y1 in the original frame)")


if __name__ == "__main__":
    main()
