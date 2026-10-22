"""Belly's narrowest line from the shoulder-midpoint vertical split.

1. left/right shoulder midpoint -> a vertical divider;
2. from the lowest shoulder downward, find the first green body-outline point
   on the left (searching left) and on the right (searching right);
3. the width between those two green points is the row's width; the narrowest
   row is the belly line.

    python tools/waist_shoulder_green.py --names 001,008,015,029,034
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def imread(path, flag=cv2.IMREAD_COLOR):
    img = cv2.imdecode(np.fromfile(str(path), np.uint8), flag)
    if img is None:
        raise RuntimeError(f"cannot read {path}")
    return img


def green_mask(bgr):
    b, g, r = cv2.split(bgr.astype(np.int16))
    return (g > 140) & (g > r + 45) & (g > b + 45)


def median5(a):
    return np.array([np.median(a[max(0, i - 2):i + 3])
                     for i in range(len(a))], float)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--names", default="001,008,015,029,034")
    ap.add_argument("--out", type=Path, default=ROOT / "out3")
    args = ap.parse_args()
    print(f"{'img':5s} {'shoulder y':>10s} {'divider x':>9s} "
          f"{'row':>6s} {'left':>5s} {'right':>5s} {'width':>6s}")
    for name in [n for n in args.names.split(",") if n]:
        img = imread(ROOT / "data" / f"{name}.png")
        seg = imread(ROOT / "out2" / f"seg_{name}.png")
        h, w = img.shape[:2]
        pose = json.loads((ROOT / "out3" / f"pose_{name}.json").read_text(
            encoding="utf-8"))["landmarks"]
        lm = {q["name"]: q for q in pose}
        sl, sr = lm["left_shoulder"], lm["right_shoulder"]
        x_mid = int(round(0.5 * (sl["x"] + sr["x"])))
        y0 = int(round(max(sl["y"], sr["y"])))
        torso = imread(ROOT / "out2" / f"torsomask_{name}.png",
                       cv2.IMREAD_GRAYSCALE) > 127
        tys = np.flatnonzero(torso.any(1))
        torso_bottom = int(tys.max()) if tys.size else h - 1
        y_end = int(round(torso_bottom - 0.10 * (torso_bottom - y0)))
        y_end = min(h - 1, max(y0, y_end))
        gm = green_mask(seg)
        rows, lefts, rights, widths = [], [], [], []
        for y in range(max(0, y0), y_end + 1):
            left = np.flatnonzero(gm[y, :x_mid])
            right = np.flatnonzero(gm[y, x_mid + 1:]) + x_mid + 1
            if not left.size or not right.size:
                continue
            lx, rx = int(left[-1]), int(right[0])
            rows.append(y)
            lefts.append(lx)
            rights.append(rx)
            widths.append(rx - lx + 1)
        if not widths:
            print(f"{name}: no green pair")
            continue
        widths = np.array(widths, float)
        sm = median5(widths)
        i = int(np.argmin(sm))
        row, lx, rx, wd = rows[i], lefts[i], rights[i], int(widths[i])
        vis = img.copy()
        cv2.line(vis, (x_mid, 0), (x_mid, h - 1), (255, 0, 255), 2,
                 cv2.LINE_AA)
        cv2.line(vis, (int(sl["x"]), int(sl["y"])),
                 (int(sr["x"]), int(sr["y"])), (0, 255, 255), 2,
                 cv2.LINE_AA)
        cv2.line(vis, (lx, row), (rx, row), (0, 0, 255), 2, cv2.LINE_AA)
        cv2.circle(vis, (lx, row), 5, (255, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(vis, (rx, row), 5, (255, 255, 255), -1, cv2.LINE_AA)
        cv2.putText(vis, f"{row}  width {wd}", (6, max(18, row - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2,
                    cv2.LINE_AA)
        cv2.imencode(".png", vis)[1].tofile(
            str(args.out / f"greensplit_{name}.png"))
        (args.out / f"greensplit_{name}.json").write_text(json.dumps(
            {"image": name, "size": [w, h],
             "shoulder_left": [round(float(sl["x"]), 2),
                               round(float(sl["y"]), 2)],
             "shoulder_right": [round(float(sr["x"]), 2),
                                round(float(sr["y"]), 2)],
             "divider_x": x_mid, "start_row": y0,
             "torso_bottom": torso_bottom, "search_end": y_end,
             "min_row": row, "left_green": lx, "right_green": rx,
             "width": wd,
             "profile_rows": rows, "profile_widths": [int(v) for v in widths]},
            indent=1), encoding="utf-8")
        print(f"{name:5s} {y0:10d} {x_mid:9d} {row:6d} {lx:5d} {rx:5d} "
              f"{wd:6d}")
    print(f"\nwrote {args.out}/greensplit_<n>.png/.json")


if __name__ == "__main__":
    main()
