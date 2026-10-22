"""MediaPipe pose landmarks + person mask, with the crops handled leniently."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from torso_seg import ascii_path, load_bgr, save_bgr

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = ROOT / "out"
MODELS = ROOT / "models"

NAMES = [
    "nose", "left_eye_inner", "left_eye", "left_eye_outer", "right_eye_inner",
    "right_eye", "right_eye_outer", "left_ear", "right_ear", "mouth_left",
    "mouth_right", "left_shoulder", "right_shoulder", "left_elbow",
    "right_elbow", "left_wrist", "right_wrist", "left_pinky", "right_pinky",
    "left_index", "right_index", "left_thumb", "right_thumb", "left_hip",
    "right_hip", "left_knee", "right_knee", "left_ankle", "right_ankle",
    "left_heel", "right_heel", "left_foot_index", "right_foot_index",
]

# Only the upper-body joints matter for the torso crop.
KEEP = [11, 12, 13, 14, 15, 16, 23, 24]

EDGES = [(11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
         (11, 23), (12, 24), (23, 24)]

# The full 33-landmark skeleton, drawn when --full is used.
SKELETON = [
    (0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8),
    (9, 10), (11, 12), (11, 13), (13, 15), (15, 17), (15, 19), (15, 21),
    (17, 19), (12, 14), (14, 16), (16, 18), (16, 20), (16, 22), (18, 20),
    (11, 23), (12, 24), (23, 24), (23, 25), (24, 26), (25, 27), (26, 28),
    (27, 29), (28, 30), (29, 31), (30, 32), (27, 31), (28, 32),
]


def detect(rgb: np.ndarray, model: Path, det_conf: float = 0.2):
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision

    opts = vision.PoseLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(ascii_path(model))),
        running_mode=vision.RunningMode.IMAGE,
        num_poses=1,
        min_pose_detection_confidence=det_conf,
        min_pose_presence_confidence=0.2,
        min_tracking_confidence=0.2,
        output_segmentation_masks=True,
    )
    img = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
    with vision.PoseLandmarker.create_from_options(opts) as lm:
        return lm.detect(img)


def to_px(res, w: int, h: int) -> dict[str, tuple[float, float, float]]:
    out: dict[str, tuple[float, float, float]] = {}
    if not res.pose_landmarks:
        return out
    for i, p in enumerate(res.pose_landmarks[0]):
        out[NAMES[i]] = (p.x * w, p.y * h, p.visibility)
    return out


def to_mask(res, shape: tuple[int, int], thr: float = 0.5) -> np.ndarray:
    """Boolean person mask from the pose landmarker segmentation head."""
    if not res.segmentation_masks:
        raise RuntimeError("pose result has no segmentation mask")
    m = np.asarray(res.segmentation_masks[0].numpy_view())
    if m.ndim == 3:          # mediapipe returns (H, W, 1)
        m = m[..., 0]
    if m.shape[:2] != shape:
        m = cv2.resize(m, (shape[1], shape[0]), interpolation=cv2.INTER_LINEAR)
    return (m > thr).astype(np.uint8) * 255


def draw(vis: np.ndarray, pts: dict[str, tuple[float, float, float]],
         lines: bool = True) -> np.ndarray:
    vis = vis.copy()
    if lines:
        for a, b in SKELETON:
            if NAMES[a] in pts and NAMES[b] in pts:
                pa, pb = pts[NAMES[a]], pts[NAMES[b]]
                cv2.line(vis, (int(pa[0]), int(pa[1])), (int(pb[0]), int(pb[1])),
                         (0, 255, 0), 2, cv2.LINE_AA)
    for name, (x, y, v) in pts.items():
        col = (0, 0, 255) if v > 0.5 else (255, 0, 0)
        cv2.circle(vis, (int(x), int(y)), 3, col, -1)
    return vis


def draw_labelled(vis: np.ndarray,
                  pts: dict[str, tuple[float, float, float]],
                  lines: bool = True, labels: bool = True) -> np.ndarray:
    """Same skeleton plus the joint names and confidences.

    The labels go on before the joints, so a joint dot is never hidden by the
    text of a neighbour.
    """
    out = vis.copy()
    if lines:
        for a, b in SKELETON:
            if NAMES[a] in pts and NAMES[b] in pts:
                pa, pb = pts[NAMES[a]], pts[NAMES[b]]
                cv2.line(out, (int(pa[0]), int(pa[1])), (int(pb[0]), int(pb[1])),
                         (0, 255, 0), 2, cv2.LINE_AA)
    if labels:
        for name, (x, y, v) in pts.items():
            cv2.putText(out, f"{name[:11]}|{v:.2f}", (int(x) + 4, int(y) - 3),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.32, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(out, f"{name[:11]}|{v:.2f}", (int(x) + 4, int(y) - 3),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.32, (0, 255, 255), 1, cv2.LINE_AA)
    for name, (x, y, v) in pts.items():
        col = (0, 0, 255) if v > 0.5 else (255, 0, 0)
        cv2.circle(out, (int(x), int(y)), 3, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.circle(out, (int(x), int(y)), 3, col, -1)
    return out


def all_landmarks(res, w: int, h: int) -> list[dict]:
    """Every landmark of the first pose, in pixels, with its confidences."""
    out: list[dict] = []
    if not res.pose_landmarks:
        return out
    for i, p in enumerate(res.pose_landmarks[0]):
        out.append({"index": i, "name": NAMES[i],
                    "x": round(float(p.x) * w, 3), "y": round(float(p.y) * h, 3),
                    "z": round(float(p.z), 5),
                    "visibility": round(float(p.visibility), 4),
                    "presence": round(float(getattr(p, "presence", 0.0)), 4)})
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=Path, default=MODELS / "pose_landmarker_full.task")
    ap.add_argument("--data", type=Path, default=DATA)
    ap.add_argument("--det-conf", type=float, default=0.2)
    ap.add_argument("--out", type=Path, default=OUT,
                    help="where the annotations are written")
    ap.add_argument("--names", default="",
                    help="comma separated image stems; default = every png")
    ap.add_argument("--lines", action="store_true", default=False,
                    help="draw the green skeleton lines between the joints; off "
                         "by default so the overlay is only the joint markers")
    ap.add_argument("--no-labels", action="store_true",
                    help="joints only, without the name and confidence labels")
    ap.add_argument("--no-mask", action="store_true")
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    want = {n for n in args.names.split(",") if n}
    paths = [p for p in sorted(args.data.glob("*.png"))
             if not want or p.stem in want]
    import csv as _csv
    rows: list[list] = []
    print(f"pose annotation of {len(paths)} image(s) -> {args.out}")
    print(f"{'image':8s} {'size':>10s} {'poses':>5s} {'landmarks':>9s} "
          f"{'mean vis':>8s}  note")
    for path in paths:
        bgr = load_bgr(path)
        h, w = bgr.shape[:2]
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        try:
            res = detect(rgb, args.model, args.det_conf)
            err = ""
        except Exception as exc:                     # a cutout may break the model
            res, err = None, f"{type(exc).__name__}: {exc}"
        if res is None:
            (args.out / f"pose_{path.stem}.json").write_text(
                json.dumps(
                    {"image": path.stem, "size": [w, h], "poses": 0,
                     "landmarks": [], "error": err}, indent=1), encoding="utf-8")
            print(f"{path.stem:8s} {str(w) + 'x' + str(h):>10s} {0:5d} {0:9d} "
                  f"{float('nan'):8.2f}  {err[:60]}")
            rows.append([path.stem, w, h, 0, 0, ""])
            continue
        pts = to_px(res, w, h)
        lms = all_landmarks(res, w, h)
        vis_mean = float(np.mean([q["visibility"] for q in lms])) if lms else 0.0
        n_pose = len(res.pose_landmarks) if res.pose_landmarks else 0
        (args.out / f"pose_{path.stem}.json").write_text(
            json.dumps(
                {"image": path.stem, "size": [w, h], "model":
                 args.model.name, "det_conf": args.det_conf,
                 "poses": n_pose, "landmarks": lms,
                 "landmark_names": NAMES,
                 "skeleton": [list(e) for e in SKELETON],
                 "lines_drawn": bool(args.lines),
                 "labels_drawn": not args.no_labels,
                 "mean_visibility": round(vis_mean, 4),
                 "note": "landmarks in image pixels; z is the model's relative "
                         "depth; visibility/presence are the model's confidences"},
            indent=1), encoding="utf-8")
        if res.segmentation_masks and not args.no_mask:
            m = (res.segmentation_masks[0].numpy_view() * 255).astype(np.uint8)
            save_bgr(args.out / f"posemask_{path.stem}.png", m)
        save_bgr(args.out / f"pose_{path.stem}.png",
                 draw_labelled(bgr, pts, lines=args.lines,
                               labels=not args.no_labels))
        miss = sum(1 for i in range(len(NAMES)) if NAMES[i] not in pts)
        by_name = {q["name"]: q for q in lms}
        rows.append([path.stem, w, h, n_pose, len(lms), f"{vis_mean:.3f}"]
                    + [f"{by_name[n]['x']:.1f},{by_name[n]['y']:.1f}"
                       if n in by_name else "" for n in
                       ("left_shoulder", "right_shoulder", "left_hip",
                        "right_hip", "left_elbow", "right_elbow",
                        "left_wrist", "right_wrist")])
        print(f"{path.stem:8s} {str(w) + 'x' + str(h):>10s} {n_pose:5d} "
              f"{len(lms):9d} {vis_mean:8.3f}  "
              f"{'all 33 landmarks' if not miss else str(miss) + ' missing'}")
    with (args.out / "pose_summary.csv").open("w", newline="",
                                              encoding="utf-8") as fh:
        wr = _csv.writer(fh)
        wr.writerow(["image", "width", "height", "poses", "landmarks",
                     "mean_visibility", "left_shoulder", "right_shoulder",
                     "left_hip", "right_hip", "left_elbow", "right_elbow",
                     "left_wrist", "right_wrist"])
        wr.writerows(rows)
    print(f"\nwrote pose_*.json / pose_*.png / posemask_*.png and "
          f"pose_summary.csv to {args.out}")


if __name__ == "__main__":
    main()
