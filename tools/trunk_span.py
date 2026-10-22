"""The trunk between the middle two green points, on the 6-cell rows.

Rows whose silhouette crosses the green border six times are arm | gap | trunk |
gap | arm; the middle two crossings are the trunk's own edges.  If such rows
come in several bands the upper one is kept.  008/009 are not measured.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT2 = ROOT / "out2"


def runs_of(row: np.ndarray, min_len: int = 8) -> list[tuple[int, int]]:
    idx = np.flatnonzero(row)
    out = []
    if not idx.size:
        return out
    s = p = int(idx[0])
    for v in idx[1:]:
        v = int(v)
        if v == p + 1:
            p = v
            continue
        out.append((s, p))
        s = p = v
    out.append((s, p))
    return [r for r in out if r[1] - r[0] + 1 >= min_len]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--allow-skip", action="store_true",
                    help="also handle 008/009 (one-off tests)")
    args = ap.parse_args()
    skip = set() if args.allow_skip else {"008", "009"}
    names = sorted(p.stem.replace("seg_", "") for p in OUT2.glob("seg_[0-9]*.png")
                   if p.stem.replace("seg_", "") not in skip)
    info = {}
    print(f"{'img':5s} {'band':>12s} {'rows':>5s} {'trunk px':>9s} "
          f"{'width top/bot':>14s}")
    for name in names:
        bgr = cv2.imdecode(np.fromfile(str(DATA / f"{name}.png"), np.uint8),
                           cv2.IMREAD_COLOR)
        h, w = bgr.shape[:2]
        body = cv2.imdecode(np.fromfile(str(OUT2 / f"bodymask_{name}.png"),
                                        np.uint8), cv2.IMREAD_GRAYSCALE) > 127
        mask = np.zeros((h, w), np.uint8)
        rows6, mids = [], []
        for y in range(h):
            rr = runs_of(body[y])
            if len(rr) == 3:
                rows6.append(y)
                a, b = rr[1]                 # the middle span: the trunk
                mask[y, a:b + 1] = 1
                mids.append((y, a, b))
        if not rows6:
            print(f"{name:5s} {'-':>12s} {0:5d} {0:9d} {'-':>14s}")
            info[name] = None
            continue
        rows6 = np.array(rows6)
        # several bands -> keep the upper one
        breaks = np.flatnonzero(np.diff(rows6) > 1)
        starts = np.r_[0, breaks + 1]
        ends = np.r_[breaks, len(rows6) - 1]
        bands = [(int(rows6[a]), int(rows6[b])) for a, b in zip(starts, ends)]
        rows_body = np.flatnonzero(body.any(1))
        need = max(8, int(0.02 * (int(rows_body.max())
                                  - int(rows_body.min()) + 1)))
        solid = [b for b in bands if b[1] - b[0] + 1 >= need] or bands
        lo, hi = solid[0]                 # the upper *substantial* band
        keep = np.zeros((h, w), bool)
        for y, a, b in mids:
            if lo <= y <= hi:
                keep[y, a:b + 1] = True
        vis = np.full((h, w, 3), 236, np.uint8)
        vis[body] = bgr[body]
        vis[keep] = (0.55 * vis[keep] + 0.45 * np.array([60, 150, 240])).astype(
            np.uint8)
        ok, buf = cv2.imencode(".png", vis)
        buf.tofile(str(OUT2 / f"trunkspan_{name}.png"))
        ok, buf = cv2.imencode(".png", (keep * 255).astype(np.uint8))
        buf.tofile(str(OUT2 / f"trunkspanmask_{name}.png"))
        wt = int(keep[lo].sum())
        wb = int(keep[hi].sum())
        info[name] = {"bands": bands, "kept_band": [lo, hi],
                      "rows": int(hi - lo + 1), "trunk_px": int(keep.sum()),
                      "width_top": wt, "width_bottom": wb}
        print(f"{name:5s} {f'{lo}-{hi}':>12s} {hi - lo + 1:5d} "
              f"{int(keep.sum()):9d} {f'{wt}/{wb}':>14s}"
              + (f"   bands {bands}" if len(bands) > 1 else ""))
    (OUT2 / "trunkspan.json").write_text(json.dumps(info, indent=1),
                                         encoding="utf-8")
    print("\nwrote out2/trunkspan_<n>.png (blue = the trunk between the middle "
          "two green points), out2/trunkspanmask_<n>.png and "
          "out2/trunkspan.json")


if __name__ == "__main__":
    main()
