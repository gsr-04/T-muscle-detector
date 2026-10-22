"""What the MediaPipe selfie multiclass model says, next to the masks in use.

Categories of selfie_multiclass.tflite: 0 background, 1 hair, 2 body-skin,
3 face-skin, 4 clothes, 5 others (accessories).

    python tools/multiclass_probe.py --names 013,006,014
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from pose_mp import detect, to_mask              # noqa: E402
from torso_seg import ascii_path, clean_mask, load_bgr, outside_gaps  # noqa: E402
NAMES = ["background", "hair", "body-skin", "face-skin", "clothes", "others"]


def imread(path: Path, flag=cv2.IMREAD_UNCHANGED) -> np.ndarray:
    img = cv2.imdecode(np.fromfile(str(path), np.uint8), flag)
    if img is None:
        raise RuntimeError(f"cannot decode {path}")
    return img


def multiclass(bgr: np.ndarray, model: Path) -> np.ndarray:
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision

    opts = vision.ImageSegmenterOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(ascii_path(model))),
        running_mode=vision.RunningMode.IMAGE,
        output_confidence_masks=True, output_category_mask=True)
    img = mp.Image(image_format=mp.ImageFormat.SRGB,
                   data=np.ascontiguousarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)))
    with vision.ImageSegmenter.create_from_options(opts) as seg:
        res = seg.segment(img)
    conf = np.stack([np.asarray(m.numpy_view()) for m in res.confidence_masks])
    if conf.ndim == 4:
        conf = conf[..., 0]
    if conf.shape[1:3] != bgr.shape[:2]:
        conf = np.stack([cv2.resize(c, (bgr.shape[1], bgr.shape[0]),
                                    interpolation=cv2.INTER_LINEAR)
                         for c in conf])
    return conf.astype(np.float32)


def runs(row: np.ndarray, min_len: int = 3) -> str:
    idx = np.flatnonzero(row)
    if not idx.size:
        return "-"
    out, s, p = [], int(idx[0]), int(idx[0])
    for v in idx[1:]:
        v = int(v)
        if v == p + 1:
            p = v
            continue
        out.append((s, p))
        s = p = v
    out.append((s, p))
    return " ".join(f"{a}-{b}" for a, b in out if b - a + 1 >= min_len)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--names", default="013")
    ap.add_argument("--data", type=Path, default=ROOT / "data")
    ap.add_argument("--out", type=Path, default=ROOT / "out")
    ap.add_argument("--model", type=Path,
                    default=ROOT / "models" / "selfie_multiclass.tflite")
    ap.add_argument("--rows", default="")
    args = ap.parse_args()
    for name in [n for n in args.names.split(",") if n]:
        bgr = load_bgr(args.data / f"{name}.png")
        h, w = bgr.shape[:2]
        conf = multiclass(bgr, args.model)
        cat = conf.argmax(0)
        person = (cat > 0)
        skin = (cat == 2) | (cat == 3)
        fog = imread(args.out / f"fgmask_{name}.png", cv2.IMREAD_GRAYSCALE)
        ext = clean_mask(fog, max_hole=int(max(64.0, 0.002 * float(
            (fog > 0).sum())))) > 0
        head = to_mask(detect(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB),
                              ROOT / "models" / "pose_landmarker_full.task"),
                       (h, w)) > 0
        body = clean_mask(cv2.bitwise_or(
            fog, cv2.bitwise_and(head.astype(np.uint8) * 255,
                                 outside_gaps(fog))),
            max_hole=int(max(64.0, 0.002 * float((fog > 0).sum())))) > 0
        print(f"\n=== {name} {w}x{h}")
        for i, n in enumerate(NAMES):
            print(f"  {n:11s} {int((cat == i).sum()):9d} px")
        print(f"  multiclass person {int(person.sum())} px   "
              f"extract {int(ext.sum())}   extract+head {int(body.sum())}")
        print("  multiclass only (not in extract+head): "
              f"{int((person & ~body).sum())} px;   "
              "extract+head only (not in multiclass): "
              f"{int((body & ~person).sum())} px")
        rows = ([int(x) for x in args.rows.split(",")] if args.rows else
                [int(v) for v in np.linspace(int(h * 0.4), int(h * 0.5), 4)])
        for y in rows:
            print(f"  row {y:5d} person {runs(person[y]):>28s} "
                  f"skin {runs(skin[y]):>28s}")
            print(f"            extract {runs(ext[y]):>28s} "
                  f"extract+head {runs(body[y]):>28s}")


if __name__ == "__main__":
    main()
