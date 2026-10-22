"""First local minimum of the trunk-span width (the row level of the belly).

The function is the per-row width of out2/trunkspanmask_<n>.png, i.e. the part
between the middle two green points on the six-cell rows.  008/009 excluded.
"""

from __future__ import annotations

import csv
import json
import argparse
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT2 = ROOT / "out2"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--allow-skip", action="store_true",
                    help="also handle 008/009 (one-off tests)")
    args = ap.parse_args()
    skip = set() if args.allow_skip else {"008", "009"}
    names = sorted(p.stem.replace("trunkspanmask_", "") for p in
                   OUT2.glob("trunkspanmask_[0-9]*.png")
                   if p.stem.replace("trunkspanmask_", "") not in skip)
    info = {}
    print(f"{'img':5s} {'band':>12s} {'min row':>8s} {'width':>6s} "
          f"{'prev row':>9s} {'prev width':>10s}")
    for name in names:
        mask = cv2.imdecode(np.fromfile(str(OUT2 / f"trunkspanmask_{name}.png"),
                                        np.uint8), cv2.IMREAD_GRAYSCALE) > 127
        rows = np.flatnonzero(mask.any(1))
        r0, r1 = int(rows.min()), int(rows.max())
        width = np.array([int(mask[y].sum()) for y in range(r0, r1 + 1)])
        # light smoothing so single pixel teeth do not create minima
        k = 5
        sm = np.array([np.median(width[max(0, i - k // 2):i + k // 2 + 1])
                       for i in range(width.size)])
        # the narrowest row of the band: the minimum of the width function
        i = int(np.argmin(sm))
        rmin, wmin = r0 + i, int(width[i])
        info[name] = {"band": [r0, r1], "min_row": rmin, "width_at_min": wmin}
        vis = cv2.imdecode(np.fromfile(str(OUT2 / f"seg_{name}.png"), np.uint8),
                           cv2.IMREAD_COLOR)
        if rmin is not None:
            cv2.line(vis, (0, rmin), (vis.shape[1] - 1, rmin), (0, 0, 255), 2,
                     cv2.LINE_AA)
            cv2.putText(vis, f"min {rmin}", (6, max(14, rmin - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2,
                        cv2.LINE_AA)
        ok, buf = cv2.imencode(".png", vis)
        buf.tofile(str(OUT2 / f"trunkspanmin_{name}.png"))
        with (OUT2 / f"trunkspanwidth_{name}.csv").open("w", newline="",
                                                        encoding="utf-8") as fh:
            wr = csv.writer(fh)
            wr.writerow(["row", "width_px"])
            for i, wv in enumerate(width):
                wr.writerow([r0 + i, int(wv)])
        print(f"{name:5s} {f'{r0}-{r1}':>12s} "
              f"{rmin if rmin is not None else -1:8d} "
              f"{wmin if wmin is not None else -1:6d} "
              f"{'-':>9s} {'-':>10s}")
    (OUT2 / "trunkspan_min.json").write_text(json.dumps(info, indent=1),
                                             encoding="utf-8")
    print("\nwrote out2/trunkspanmin_<n>.png (red = the first local minimum), "
          "out2/trunkspanwidth_<n>.csv (row, width) and "
          "out2/trunkspan_min.json")


if __name__ == "__main__":
    main()
