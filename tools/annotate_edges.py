"""BiRefNet body edge + DexiNed muscle lines -> one annotated picture per image.

Standalone: it does not touch the measuring chain (no w100, no matrix, no score).

  * body outline  = contour of the BiRefNet mask (models/birefnet_lite.onnx)
  * muscle lines  = DexiNed edges inside the body, with the silhouette band cut
                    away (models/dexined_2024sep.onnx, cv.dnn)

    python tools/annotate_edges.py                      # every data/*.png
    python tools/annotate_edges.py --names 013,016
    python tools/annotate_edges.py --thr 0.4 --dilate 1
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from torso_seg import clean_mask, load_bgr             # noqa: E402

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], np.float32)
BIPED_MEAN = (103.5, 116.2, 123.6)


def birefnet_mask(bgr: np.ndarray, model: Path, size: int = 1024,
                  thr: float = 0.5) -> np.ndarray:
    import onnxruntime as ort

    h, w = bgr.shape[:2]
    scale = size / max(h, w)
    nh, nw = max(1, int(round(h * scale))), max(1, int(round(w * scale)))
    interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC
    canvas = np.zeros((size, size, 3), np.uint8)
    canvas[:nh, :nw] = cv2.resize(bgr, (nw, nh), interpolation=interp)
    x = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    x = ((x - IMAGENET_MEAN) / IMAGENET_STD).transpose(2, 0, 1)[None]
    so = ort.SessionOptions()
    so.log_severity_level = 3
    sess = ort.InferenceSession(str(model), so, providers=["CPUExecutionProvider"])
    out = np.asarray(sess.run(None, {sess.get_inputs()[0].name: x})[0])[0, 0]
    prob = out if 0.0 <= out.min() and out.max() <= 1.0 else 1 / (1 + np.exp(-out))
    prob = cv2.resize(prob[:nh, :nw].astype(np.float32), (w, h),
                      interpolation=cv2.INTER_LINEAR)
    return prob >= thr


_NET: dict[str, object] = {}


def dexined_edges(crop: np.ndarray, model: Path,
                  max_side: int = 960) -> np.ndarray:
    """DexiNed edge probability (0..1) at the crop's resolution."""
    if "net" not in _NET:
        # cv.dnn wants bytes: the model path itself breaks on a Chinese path
        _NET["net"] = cv2.dnn.readNetFromONNX(np.fromfile(str(model), np.uint8))
    net = _NET["net"]
    h, w = crop.shape[:2]
    # the export needs multiples of 32 (its dense blocks concat); keep the shape
    def fit(side: int) -> tuple[int, int]:
        sc = min(1.0, side / max(h, w))
        return (max(32, int(round(w * sc / 32)) * 32),
                max(32, int(round(h * sc / 32)) * 32))
    for side in (max_side, 640, 480):
        iw, ih = fit(side)
        if abs(iw / ih - w / h) > 0.25:
            continue
        sc = min(iw / w, ih / h)
        nw, nh = max(32, int(round(w * sc))), max(32, int(round(h * sc)))
        canvas = np.zeros((ih, iw, 3), np.uint8)
        canvas[:, :] = BIPED_MEAN
        canvas[:nh, :nw] = cv2.resize(crop, (nw, nh),
                                      interpolation=cv2.INTER_AREA)
        try:
            net.setInput(cv2.dnn.blobFromImage(canvas, 1.0, (iw, ih), BIPED_MEAN,
                                               swapRB=False, crop=False))
            out = net.forward()
        except cv2.error:
            continue
        outs = out if isinstance(out, (list, tuple)) else [out]
        fused = 1.0 / (1.0 + np.exp(-np.squeeze(np.asarray(outs[-1]))))
        fused = fused[:nh, :nw]
        return cv2.resize(fused.astype(np.float32), (w, h),
                          interpolation=cv2.INTER_LINEAR)
    raise RuntimeError("DexiNed refused every input size tried")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--names", default="")
    ap.add_argument("--data", type=Path, default=ROOT / "data")
    ap.add_argument("--out", type=Path, default=ROOT / "out2")
    ap.add_argument("--birefnet", type=Path,
                    default=ROOT / "models" / "birefnet_lite.onnx")
    ap.add_argument("--dexined", type=Path,
                    default=ROOT / "models" / "dexined_2024sep.onnx")
    ap.add_argument("--cover", type=float, default=0.04,
                    help="fraction of the body kept as muscle lines; the "
                         "threshold is the matching percentile of the DexiNed "
                         "response inside the body")
    ap.add_argument("--thr", type=float, default=-1.0,
                    help="use this DexiNed probability instead of --cover")
    ap.add_argument("--band", type=float, default=0.02,
                    help="silhouette band removed, as a fraction of the short "
                         "side")
    ap.add_argument("--dilate", type=int, default=1,
                    help="thicken the lines by this many pixels")
    ap.add_argument("--parts", action="store_true",
                    help="also write the mask and the raw edge map")
    ap.add_argument("--reuse", action="store_true",
                    help="reuse out2/bodymask_<n>.png instead of running "
                         "BiRefNet again")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    want = [n for n in args.names.split(",") if n]
    names = want or sorted(p.stem for p in args.data.glob("*.png"))
    print(f"{'img':5s} {'size':>11s} {'body px':>9s} {'line px':>9s} "
          f"{'birefnet':>9s} {'dexined':>8s}")
    for name in names:
        bgr = load_bgr(args.data / f"{name}.png")
        h, w = bgr.shape[:2]
        t0 = time.time()
        mask_path = args.out / f"bodymask_{name}.png"
        if args.reuse and mask_path.exists():
            body = cv2.imdecode(np.fromfile(str(mask_path), np.uint8),
                                cv2.IMREAD_GRAYSCALE) > 127
        else:
            body = birefnet_mask(bgr, args.birefnet)
            body = clean_mask((body.astype(np.uint8)) * 255,
                              max_hole=int(max(64.0, 0.002 * body.sum()))) > 0
            ok, buf = cv2.imencode(".png", (body * 255).astype(np.uint8))
            buf.tofile(str(mask_path))
        t1 = time.time()
        ys, xs = np.flatnonzero(body.any(1)), np.flatnonzero(body.any(0))
        pad = int(0.05 * max(h, w))
        y0, y1 = max(0, int(ys.min()) - pad), min(h, int(ys.max()) + pad + 1)
        x0, x1 = max(0, int(xs.min()) - pad), min(w, int(xs.max()) + pad + 1)
        edge = dexined_edges(bgr[y0:y1, x0:x1], args.dexined)
        full = np.zeros((h, w), np.float32)
        full[y0:y1, x0:x1] = edge
        t2 = time.time()
        r = max(2, int(round(args.band * min(h, w))))
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
        inner = cv2.erode(body.astype(np.uint8), k) > 0
        vals = full[inner]
        thr = (args.thr if args.thr > 0 else
               float(np.percentile(vals, 100.0 * (1.0 - args.cover)))
               if vals.size else 1.0)
        lines = (full >= thr) & inner
        if hasattr(cv2, "ximgproc") and lines.any():
            thinned = cv2.ximgproc.thinning((lines * 255).astype(np.uint8))
            if thinned.any():
                lines = thinned > 0
        if args.dilate > 0:
            kk = np.ones((2 * args.dilate + 1, 2 * args.dilate + 1), np.uint8)
            lines = cv2.dilate(lines.astype(np.uint8), kk) > 0
        vis = bgr.copy()
        vis[lines] = (0.35 * vis[lines] + 0.65 * np.array([0, 0, 255])).astype(
            np.uint8)                                  # muscle lines, red
        # RETR_CCOMP, not RETR_EXTERNAL: the space between an arm and the trunk
        # is a *hole* as soon as the two touch above and below (axilla and hand
        # on the hip) and an outer-only contour would leave that inner boundary
        # - the inner side of the arm and the two sides of the trunk - unmarked
        cnts, _ = cv2.findContours(body.astype(np.uint8), cv2.RETR_CCOMP,
                                   cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(vis, cnts, -1, (0, 255, 0), 2, cv2.LINE_AA)
        ok, buf = cv2.imencode(".png", vis)
        buf.tofile(str(args.out / f"annot_{name}.png"))
        if args.parts:
            ok, buf = cv2.imencode(".png",
                                   (np.clip(full, 0, 1) * 255).astype(np.uint8))
            buf.tofile(str(args.out / f"dexined_map_{name}.png"))
        print(f"{name:5s} {str(w) + 'x' + str(h):>11s} {int(body.sum()):9d} "
              f"{int(lines.sum()):9d} {t1 - t0:8.1f}s {t2 - t1:7.1f}s   "
              f"thr {thr:.3f}, {100.0 * lines.sum() / max(1, inner.sum()):.1f}% "
              f"of the body")
    print(f"\nwrote {args.out}/annot_<n>.png  "
          f"(green = BiRefNet body outline, red = DexiNed muscle lines)")


if __name__ == "__main__":
    main()
