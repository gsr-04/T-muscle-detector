"""Denoise the 100-px pooled images, then redo the peak annotation on them.

    python tools/denoise_peaks_w100.py                     # 3x3 median
    python tools/denoise_peaks_w100.py --method bilateral --d 5
    python tools/denoise_peaks_w100.py --min-contrast 4

A median filter removes single-pixel ripples while keeping the edges of the dark
bands, which is what the spurious peaks come from.  Counts before and after show
how much noise the filter removed.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "out"


def peaks_1d(a: np.ndarray, valid: np.ndarray, contrast: float,
             axis: int) -> np.ndarray:
    if axis == 0:
        l, m, r = a[:-2, :], a[1:-1, :], a[2:, :]
        v = valid[:-2, :] & valid[1:-1, :] & valid[2:, :]
        out = np.zeros_like(a, bool)
        out[1:-1, :] = v & (m > l + contrast) & (m > r + contrast)
    else:
        l, m, r = a[:, :-2], a[:, 1:-1], a[:, 2:]
        v = valid[:, :-2] & valid[:, 1:-1] & valid[:, 2:]
        out = np.zeros_like(a, bool)
        out[:, 1:-1] = v & (m > l + contrast) & (m > r + contrast)
    return out


def denoise(img: np.ndarray, method: str, d: int) -> np.ndarray:
    if method == "median":
        return cv2.medianBlur(img, d if d % 2 else d + 1)
    if method == "gaussian":
        return cv2.GaussianBlur(img, (0, 0), max(0.6, d / 2))
    if method == "bilateral":
        return cv2.bilateralFilter(img, d, 25, 25)
    return img


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--prefix", default="w100")
    ap.add_argument("--method", choices=("none", "median", "gaussian", "bilateral"),
                    default="median",
                    help="'none' marks the raw (un-denoised) pooled image")
    ap.add_argument("--d", type=int, default=3)
    ap.add_argument("--min-contrast", type=float, default=0.0)
    ap.add_argument("--scale", type=int, default=6)
    ap.add_argument("--paint", choices=("split", "union", "vrh"), default="split",
                    help="'union' draws every maximum in one colour instead of "
                         "splitting vertical / horizontal / shared; 'vrh' uses "
                         "vertical=red, horizontal=blue, shared=white")
    ap.add_argument("--radius", type=int, default=0,
                    help="marker radius in preview pixels (0 = automatic)")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--names", default="",
                    help="comma separated image names; empty = every image")
    args = ap.parse_args()
    tag = args.tag or f"peaks2d{args.prefix}_{args.method}{args.d}"

    print(f"denoise = {args.method} (d={args.d}), min contrast = "
          f"{args.min_contrast:g}")
    print(f"{'img':6s} {'size':>9s} {'before':>15s} {'after':>15s} {'removed':>9s}")
    want = {n for n in args.names.split(",") if n}
    for p in sorted(args.out.glob(f"{args.prefix}_*.png")):
        if want and p.stem.replace(f"{args.prefix}_", "") not in want:
            continue
        name = p.stem.replace(f"{args.prefix}_", "")
        img = cv2.imdecode(np.fromfile(str(p), np.uint8), cv2.IMREAD_UNCHANGED)
        gray = cv2.cvtColor(img[:, :, :3], cv2.COLOR_BGR2GRAY)
        valid = img[:, :, 3] > 0
        dark_raw = (255.0 - gray).astype(np.float32)
        den = denoise(gray, args.method, args.d)
        dark = (255.0 - den).astype(np.float32)
        # keep the subject edge unchanged by the filter
        dark[~valid] = np.nan
        dark_raw[~valid] = np.nan

        def counts(a):
            v = ~np.isnan(a)
            af = np.nan_to_num(a)
            vert = peaks_1d(af, v, args.min_contrast, 0)
            horz = peaks_1d(af, v, args.min_contrast, 1)
            return int(vert.sum()), int(horz.sum()), vert, horz, vert & horz

        bv, bh, _bvm, _bhm, bboth = counts(dark_raw)
        av, ah, av_mask, ah_mask, aboth = counts(dark)
        h, w = gray.shape
        s = args.scale
        big = cv2.resize(img[:, :, :3], (w * s, h * s), interpolation=cv2.INTER_NEAREST)
        # draw on a BGR canvas: OpenCV zeroes the 4th channel when a 3-element
        # colour is used on a 4-channel image, which made every marker invisible
        vis = big.copy()
        for i in range(h + 1):
            cv2.line(vis, (0, i * s), (w * s, i * s), (80, 80, 80), 1)
        for j in range(w + 1):
            cv2.line(vis, (j * s, 0), (j * s, h * s), (80, 80, 80), 1)

        def paint(mask, col, rad):
            for y, x in zip(*np.nonzero(mask)):
                c = (x * s + s // 2, y * s + s // 2)
                if args.paint in ("union", "vrh"):
                    cv2.circle(vis, c, rad + 1, (0, 0, 0), 1)   # keep it visible
                cv2.circle(vis, c, rad, col, -1)

        rad = args.radius or max(3, s // 2)
        if args.paint == "union":
            paint(av_mask | ah_mask, (0, 0, 255), rad)
        elif args.paint == "vrh":
            # the three families must stay visually distinct: different colour
            # and the shared points noticeably larger, drawn last so they sit on
            # top of the others
            for y, x in zip(*np.nonzero(av_mask & ~aboth)):
                c = (x * s + s // 2, y * s + s // 2)
                cv2.circle(vis, c, rad + 1, (0, 0, 0), 1)
                cv2.circle(vis, c, rad, (0, 0, 255), -1)        # vertical: red
            for y, x in zip(*np.nonzero(ah_mask & ~aboth)):
                c = (x * s + s // 2, y * s + s // 2)
                cv2.circle(vis, c, rad + 1, (0, 0, 0), 1)
                cv2.circle(vis, c, rad, (255, 0, 0), -1)        # horizontal: blue
            for y, x in zip(*np.nonzero(aboth)):
                c = (x * s + s // 2, y * s + s // 2)
                cv2.circle(vis, c, rad + 4, (0, 0, 0), 2)
                cv2.circle(vis, c, rad + 2, (255, 255, 255), -1)  # shared: white
        else:
            paint(av_mask & ~aboth, (255, 150, 0), 2)
            paint(ah_mask & ~aboth, (0, 200, 0), 2)
            paint(aboth, (0, 0, 255), 3)
        out_img = cv2.cvtColor(vis, cv2.COLOR_BGR2BGRA)
        out_img[:, :, 3] = 255
        ok, buf = cv2.imencode(".png", out_img)
        buf.tofile(str(args.out / f"{tag}_{name}.png"))
        def pts(mask):
            return [[int(x), int(y)] for y, x in zip(*np.nonzero(mask))]

        dark_den = dark                                # darkness of the denoised image
        (args.out / f"{tag}_{name}.json").write_text(json.dumps(
            {"image": name, "size": [w, h], "method": args.method, "d": args.d,
             "min_contrast": args.min_contrast,
             "vertical_after": av, "horizontal_after": ah,
             "both_after": int(aboth.sum()),
             "vertical_before": bv, "horizontal_before": bh,
             "both_before": int(bboth.sum()),
             "note": "vertical = local maximum of its column, horizontal = of its "
                     "row, both = in both; all coordinates are x,y",
             "vertical_points": pts(av_mask),
             "horizontal_points": pts(ah_mask),
             "both_points": pts(aboth)}, indent=1),
            encoding="utf-8")
        with (args.out / f"{tag}_{name}.csv").open("w", newline="",
                                                   encoding="utf-8") as fh:
            wr = csv.writer(fh)
            wr.writerow(["type", "x", "y", "darkness"])
            for tag2, mask in (("both", aboth), ("vertical_only", av_mask & ~aboth),
                               ("horizontal_only", ah_mask & ~aboth)):
                for y, x in zip(*np.nonzero(mask)):
                    wr.writerow([tag2, int(x), int(y),
                                 round(float(dark_den[y, x]), 1)])
        red = 100.0 * (int(aboth.sum()) - int(bboth.sum())) / max(1, int(bboth.sum()))
        print(f"{name:6s} {w:4d}x{h:<4d} {bv:6d}/{bh:<7d} {av:6d}/{ah:<7d} "
              f"{red:+8.0f}%")
    print(f"\nwrote {tag}_*.png / .json / .csv  "
          f"(vertical/horizontal counts before vs after, and the change in the "
          f"shared points)")


if __name__ == "__main__":
    main()
