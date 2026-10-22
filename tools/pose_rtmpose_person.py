"""RTMPose-m (COCO 17) pose annotation on the *original* picture.

    python tools/pose_rtmpose_person.py --names 020

RTMPose is a top-down model: it needs a person box.  The box is taken from the
MediaPipe landmarks of out1/pose_<n>.json (all joints with enough visibility),
or from out2/bodymask_<n>.png when that pose is missing/inconsistent.  The box
is expanded to the model's 3:4 aspect (192x256) with mmpose's 1.25 padding,
then the 192x256 crop is normalized the way the export expects
(RGB, (pixel-mean)/std in 0..255) and the SimCC peaks (argmax / 2) are mapped
back through the inverse affine.

Writes out3/pose_<n>.json, out3/pose_<n>.png and out3/poseonly_<n>.png; the
background of both pictures is the original frame.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

ROOT = Path(__file__).resolve().parents[1]
COCO = ["nose", "left_eye", "right_eye", "left_ear", "right_ear",
        "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
        "left_wrist", "right_wrist", "left_hip", "right_hip",
        "left_knee", "right_knee", "left_ankle", "right_ankle"]
SKELETON = [(5, 6), (5, 7), (7, 9), (6, 8), (8, 10), (5, 11), (6, 12),
            (11, 12), (11, 13), (13, 15), (12, 14), (14, 16), (0, 5), (0, 6)]
MEAN = np.array([123.675, 116.28, 103.53], np.float32)
STD = np.array([58.395, 57.12, 57.375], np.float32)
MODEL = (ROOT / "models" / "RTMPose-ONNX" / "rtmpose-m.onnx")
W, H = 192, 256


def imread(path: Path, flag=cv2.IMREAD_COLOR) -> np.ndarray:
    img = cv2.imdecode(np.fromfile(str(path), np.uint8), flag)
    if img is None:
        raise RuntimeError(f"cannot decode {path}")
    return img


def save(path: Path, img: np.ndarray) -> None:
    ok, buf = cv2.imencode(".png", img)
    if not ok:
        raise RuntimeError(f"cannot encode {path}")
    buf.tofile(str(path))


def landmark_box(name: str, iw: int, ih: int):
    """Person box from the out1 MediaPipe landmarks."""
    path = ROOT / "out1" / f"pose_{name}.json"
    if not path.exists():
        return None
    doc = json.loads(path.read_text(encoding="utf-8"))
    keep = [q for q in doc.get("landmarks", [])
            if float(q.get("visibility", 1.0)) > 0.3
            and 0 <= q["x"] < iw and 0 <= q["y"] < ih]
    if len(keep) < 4:
        return None
    xs = [q["x"] for q in keep]
    ys = [q["y"] for q in keep]
    bw, bh = max(xs) - min(xs), max(ys) - min(ys)
    if bw < 0.12 * iw or bh < 0.18 * ih:
        return None
    return min(xs), min(ys), max(xs), max(ys)


def mask_box(name: str):
    """Fallback person box from the BiRefNet body mask."""
    path = ROOT / "out2" / f"bodymask_{name}.png"
    if not path.exists():
        return None
    mask = imread(path, cv2.IMREAD_GRAYSCALE) > 127
    ys, xs = np.flatnonzero(mask.any(1)), np.flatnonzero(mask.any(0))
    if not xs.size:
        return None
    return float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())


def expand(box, aspect: float, pad: float):
    x0, y0, x1, y1 = box
    cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
    bw, bh = max(1.0, x1 - x0), max(1.0, y1 - y0)
    if bw / bh < aspect:
        bw = bh * aspect
    else:
        bh = bw / aspect
    bw *= pad
    bh *= pad
    x0, y0 = cx - bw / 2.0, cy - bh / 2.0
    M = np.array([[W / bw, 0, -x0 * W / bw],
                  [0, H / bh, -y0 * H / bh]], np.float32)
    return M


def run_model(sess, img: np.ndarray, M: np.ndarray):
    crop = cv2.warpAffine(img, M, (W, H), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
    x = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB).astype(np.float32)
    x = (x - MEAN) / STD
    x = x.transpose(2, 0, 1)[None]
    sx, sy = sess.run(None, {sess.get_inputs()[0].name: x})
    sx, sy = np.asarray(sx)[0], np.asarray(sy)[0]
    inv = cv2.invertAffineTransform(M)
    pts = []
    for i, nm in enumerate(COCO):
        u, v = float(np.argmax(sx[i])) / 2.0, float(np.argmax(sy[i])) / 2.0
        px = inv[0, 0] * u + inv[0, 1] * v + inv[0, 2]
        py = inv[1, 0] * u + inv[1, 1] * v + inv[1, 2]
        conf = float(min(sx[i].max(), sy[i].max()))
        pts.append({"index": i, "name": nm, "x": round(float(px), 2),
                    "y": round(float(py), 2),
                    "visibility": round(conf, 4), "presence": round(conf, 4)})
    return pts, crop


def draw(img: np.ndarray, pts: list[dict], min_conf: float = 0.15) -> np.ndarray:
    vis = img.copy()
    h, w = vis.shape[:2]
    for a, b in SKELETON:
        pa, pb = pts[a], pts[b]
        if min(pa["visibility"], pb["visibility"]) < min_conf:
            continue
        if not (0 <= pa["x"] < w and 0 <= pa["y"] < h
                and 0 <= pb["x"] < w and 0 <= pb["y"] < h):
            continue
        cv2.line(vis, (int(round(pa["x"])), int(round(pa["y"]))),
                 (int(round(pb["x"])), int(round(pb["y"]))),
                 (0, 255, 0), 2, cv2.LINE_AA)
    for q in pts:
        if q["visibility"] < min_conf:
            continue
        x, y = int(round(q["x"])), int(round(q["y"]))
        if not (0 <= x < w and 0 <= y < h):
            continue
        col = (0, 0, 255) if q["name"] in (
            "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
            "left_wrist", "right_wrist", "left_hip", "right_hip") else (255, 0, 0)
        cv2.circle(vis, (x, y), 4, col, -1, cv2.LINE_AA)
    return vis


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--names", required=True)
    ap.add_argument("--data", type=Path, default=ROOT / "data")
    ap.add_argument("--out3", type=Path, default=ROOT / "out3")
    ap.add_argument("--model", type=Path, default=MODEL)
    ap.add_argument("--pad", type=float, default=1.25)
    ap.add_argument("--refine", action="store_true",
                    help="second pass using RTMPose's own visible joints as box")
    args = ap.parse_args()
    so = ort.SessionOptions()
    so.log_severity_level = 3
    sess = ort.InferenceSession(str(args.model), so,
                                providers=["CPUExecutionProvider"])
    print(f"{'img':5s} {'box source':10s} {'crop size':>16s} "
          f"{'nose y':>8s} {'shoulder y':>10s} {'hip y':>8s}")
    for name in [n for n in args.names.split(",") if n]:
        img = imread(args.data / f"{name}.png")
        ih, iw = img.shape[:2]
        box = landmark_box(name, iw, ih)
        source = "out1 pose"
        if box is None:
            box = mask_box(name)
            source = "body mask"
        if box is None:
            box = (0.05 * iw, 0.02 * ih, 0.95 * iw, 0.98 * ih)
            source = "frame"
        M = expand(box, W / H, args.pad)
        pts, _ = run_model(sess, img, M)
        if args.refine:
            good = [q for q in pts if q["visibility"] > 0.3
                    and -0.1 * iw < q["x"] < 1.1 * iw
                    and -0.1 * ih < q["y"] < 1.1 * ih]
            if len(good) >= 4:
                xs = [q["x"] for q in good]
                ys = [q["y"] for q in good]
                M = expand((min(xs), min(ys), max(xs), max(ys)),
                           W / H, args.pad)
                pts, _ = run_model(sess, img, M)
                source += "+refine"
        d = {q["name"]: q for q in pts}
        doc = {"image": name, "size": [iw, ih],
               "model": args.model.name,
               "source": "RTMPose-m ONNX (COCO 17), top-down",
               "landmarks": pts,
               "note": "top-down RTMPose-m on the original picture; person box "
                       "from out1 MediaPipe landmarks / body mask, 3:4 crop, "
                       "padding 1.25, SimCC argmax/2 mapped back"}
        (args.out3 / f"pose_{name}.json").write_text(
            json.dumps(doc, indent=1), encoding="utf-8")
        vis = draw(img, pts)
        save(args.out3 / f"pose_{name}.png", vis)
        save(args.out3 / f"poseonly_{name}.png", vis.copy())
        sh = (d["left_shoulder"]["y"] + d["right_shoulder"]["y"]) / 2
        hp = (d["left_hip"]["y"] + d["right_hip"]["y"]) / 2
        print(f"{name:5s} {source:10s} {M[0,0]:8.3f}x{M[1,1]:<6.3f} "
              f"{d['nose']['y']:8.1f} {sh:10.1f} {hp:8.1f}")
        print(f"  shoulders ({d['left_shoulder']['x']:.0f},"
              f"{d['left_shoulder']['y']:.0f}) / ({d['right_shoulder']['x']:.0f},"
              f"{d['right_shoulder']['y']:.0f});  elbows "
              f"({d['left_elbow']['x']:.0f},{d['left_elbow']['y']:.0f}) / "
              f"({d['right_elbow']['x']:.0f},{d['right_elbow']['y']:.0f});  "
              f"wrists ({d['left_wrist']['x']:.0f},{d['left_wrist']['y']:.0f}) / "
              f"({d['right_wrist']['x']:.0f},{d['right_wrist']['y']:.0f})")
    print(f"\nwrote {args.out3}/pose_<n>.json, pose_<n>.png, poseonly_<n>.png")


if __name__ == "__main__":
    main()
