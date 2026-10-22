"""Pool every torso crop down to a fixed horizontal resolution (default 30 px).

The same rule as before: each output pixel takes the colour of the *darkest*
pixel in its source block (a real pixel colour, not a per-channel minimum).  The
block size is fractional, so any image can reach exactly 30 px wide; the height
scales with it to keep the aspect ratio.

    python tools/pool_to_width.py                  # all crops -> 30 px wide
    python tools/pool_to_width.py --target 30 --mode darkest
    python tools/pool_to_width.py --mode max --keep-height
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def pool_region(block: np.ndarray, mode: str) -> np.ndarray:
    """One output pixel from a block, following the pooling rule."""
    if block.size == 0:
        return np.zeros(block.shape[-1], block.dtype)
    if mode == "mean":
        return block.reshape(-1, block.shape[-1]).mean(0).round().astype(np.uint8)
    c = block.shape[-1]
    if c >= 3:
        lum = (0.114 * block[:, :, 0].astype(np.float32)
               + 0.587 * block[:, :, 1].astype(np.float32)
               + 0.299 * block[:, :, 2].astype(np.float32))
    else:
        lum = block[:, :, 0].astype(np.float32)
    if mode == "max":
        idx = np.unravel_index(int(lum.argmax()), lum.shape)
    else:                                   # darkest
        idx = np.unravel_index(int(lum.argmin()), lum.shape)
    return block[idx[0], idx[1]]


def pool_to_width(img: np.ndarray, target: int, mode: str,
                  keep_height: bool = False) -> np.ndarray:
    h, w = img.shape[:2]
    out_h = h if keep_height else max(1, int(round(h * target / w)))
    c = img.shape[2]
    out = np.zeros((out_h, target, c), img.dtype)
    for i in range(out_h):
        y0 = int(round(i * h / out_h))
        y1 = max(y0 + 1, int(round((i + 1) * h / out_h)))
        for j in range(target):
            x0 = int(round(j * w / target))
            x1 = max(x0 + 1, int(round((j + 1) * w / target)))
            blk = img[y0:y1, x0:x1]
            out[i, j] = pool_region(blk, mode)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=ROOT / "out")
    ap.add_argument("--skip", default="")
    ap.add_argument("--target", type=int, default=30)
    ap.add_argument("--mode", choices=("darkest", "max", "mean"), default="darkest")
    ap.add_argument("--keep-height", action="store_true",
                    help="only change the width, leave the height untouched")
    ap.add_argument("--prefix", default=None)
    args = ap.parse_args()
    prefix = args.prefix or f"w{args.target}{'' if args.mode == 'darkest' else args.mode}"

    skip = {s for s in args.skip.split(",") if s}
    print(f"pooling to width={args.target}, rule={args.mode}, "
          f"height={'kept' if args.keep_height else 'scaled'}")
    print(f"{'img':6s} {'from':>11s} {'to':>9s}  aspect kept")
    for p in sorted(args.out.glob("crop_torso_*.png")):
        name = p.stem.replace("crop_torso_", "")
        if name in skip:
            continue
        img = cv2.imdecode(np.fromfile(str(p), np.uint8), cv2.IMREAD_UNCHANGED)
        if img is None:
            continue
        h, w = img.shape[:2]
        small = pool_to_width(img, args.target, args.mode, args.keep_height)
        alpha = pool_to_width(img[:, :, 3:4], args.target, "max",
                              args.keep_height)[:, :, 0]
        small[:, :, 3] = alpha
        ok, buf = cv2.imencode(".png", small)
        buf.tofile(str(args.out / f"{prefix}_{name}.png"))
        ratio = (small.shape[0] / small.shape[1]) / (h / w) if small.shape[1] else 0
        print(f"{name:6s} {w:4d}x{h:<6d} {small.shape[1]:4d}x{small.shape[0]:<5d} "
              f"{ratio:.2f}")
    print(f"\nwrote {prefix}_*.png")


if __name__ == "__main__":
    main()
