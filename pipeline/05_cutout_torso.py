"""Keep only the torso in the cutout: drop arms, legs, head (and neck).

    python tools/cutout_torso.py                     # torso only (default)
    python tools/cutout_torso.py --keep torso,shoulder,trapezius

Writes out/torso_<name>.png (original frame) and out/crop_torso_<name>.png
(tight bbox), both RGBA with everything else fully transparent.
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


def keep_mask(classes: dict, keep: list[str], shape: tuple[int, int]) -> np.ndarray:
    m = np.zeros(shape, np.uint8)
    for cls in keep:
        info = classes.get(cls)
        if not info:
            continue
        for poly in info["polygons"]:
            cv2.fillPoly(m, [np.round(np.array(poly)).astype(np.int32)], 255)
    return m


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=ROOT / "out")
    ap.add_argument("--keep", default="torso",
                    help="comma separated classes: torso, shoulder, trapezius, neck")
    ap.add_argument("--suffix", default="torso", help="output name suffix")
    ap.add_argument("--margin", type=int, default=0)
    ap.add_argument("--feather", type=int, default=2)
    args = ap.parse_args()

    keep = [c.strip() for c in args.keep.split(",") if c.strip()]
    boxes: dict[str, list[int]] = {}
    print(f"keeping: {', '.join(keep)}")
    print(f"{'image':8s} {'cutout':>11s} {'torso':>11s}  kept%   crop box")
    for path in sorted(args.out.glob("cutout_*.png")):
        name = path.stem.replace("cutout_", "")
        doc_path = args.out / f"simple_{name}.json"
        if not doc_path.exists():
            print(f"{name:8s} no simple_{name}.json, skipped")
            continue
        doc = json.loads(doc_path.read_text(encoding="utf-8"))
        img = load_rgba(path)
        h, w = img.shape[:2]
        km = keep_mask(doc["classes"], keep, (h, w))
        # the region polygons come from row-wise rasterisation, so their edges
        # are stair-stepped: close the notches and smooth before feathering
        km = cv2.morphologyEx(km, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        sig = max(1.2, 0.004 * min(h, w))
        km = cv2.GaussianBlur(km.astype(np.float32), (0, 0), sig)
        km = np.where(km >= 127, 255, 0).astype(np.uint8)
        if args.feather > 0:
            k = max(3, (args.feather * 2 + 1) | 1)
            km = cv2.GaussianBlur(km, (k, k), 0)
        alpha = (img[:, :, 3].astype(np.float32) * (km.astype(np.float32) / 255.0))
        out = img.copy()
        out[:, :, 3] = np.clip(alpha, 0, 255).astype(np.uint8)
        save_rgba(args.out / f"{args.suffix}_{name}.png", out)

        ys, xs = np.nonzero(out[:, :, 3] > 127)
        if not ys.size:
            print(f"{name:8s} empty after masking")
            continue
        x0, x1 = max(0, int(xs.min()) - args.margin), min(w, int(xs.max()) + 1 + args.margin)
        y0, y1 = max(0, int(ys.min()) - args.margin), min(h, int(ys.max()) + 1 + args.margin)
        crop = out[y0:y1, x0:x1]
        save_rgba(args.out / f"crop_{args.suffix}_{name}.png", crop)
        boxes[name] = [x0, y0, x1, y1]
        kept = 100.0 * float((out[:, :, 3] > 127).sum()) / max(
            1.0, float((img[:, :, 3] > 127).sum()))
        print(f"{name:8s} {w:4d}x{h:<6d} {crop.shape[1]:4d}x{crop.shape[0]:<6d}  "
              f"{kept:5.1f}%  ({x0},{y0},{x1},{y1})")

    tag = "crop_" + args.suffix if args.suffix != "torso" else "crop_torso"
    (args.out / f"{tag}_boxes.json").write_text(json.dumps(boxes, indent=1),
                                                encoding="utf-8")
    print(f"\nwrote {args.suffix}_*.png, crop_{args.suffix}_*.png and "
          f"{tag}_boxes.json")


if __name__ == "__main__":
    main()
