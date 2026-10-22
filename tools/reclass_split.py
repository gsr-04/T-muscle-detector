"""Re-do the split on the classified person: arms, torso, pooled torso.

Reads the classification in out2 (armmask/torsomask/bodymask), builds the person
on a plain field, separates the arms from the trunk and writes everything the
scoring stages need into out3/.  The pose is taken from an existing RTMPose
annotation (out3/pose_<n>.json, source contains "RTMPose"); MediaPipe on the
classified person is only the fallback when that annotation is absent.

    python tools/reclass_split.py --names 016,002
    python tools/reclass_split.py                     # all 16
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from pose_mp import detect, draw_labelled, to_px        # noqa: E402
from torso_seg import clean_mask, load_bgr              # noqa: E402


def rtm_pose(name: str, out_dir: Path = ROOT / "out3"):
    """The RTMPose annotation already in out3, if it is usable."""
    path = out_dir / f"pose_{name}.json"
    if not path.exists():
        return None
    doc = json.loads(path.read_text(encoding="utf-8"))
    if "RTMPose" not in str(doc.get("source", "")):
        return None
    lm = {q["name"]: q for q in doc.get("landmarks", [])}
    need = ("left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
            "left_wrist", "right_wrist")
    if not all(k in lm for k in need):
        return None
    return {k: (float(q["x"]), float(q["y"]),
                float(q.get("visibility", 1.0))) for k, q in lm.items()}


def imread(path: Path, flag=cv2.IMREAD_UNCHANGED) -> np.ndarray:
    img = cv2.imdecode(np.fromfile(str(path), np.uint8), flag)
    if img is None:
        raise RuntimeError(f"cannot decode {path}")
    return img


def save(path: Path, img: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, buf = cv2.imencode(".png", img)
    buf.tofile(str(path))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--names", default="")
    ap.add_argument("--data", type=Path, default=ROOT / "data")
    ap.add_argument("--out2", type=Path, default=ROOT / "out2")
    ap.add_argument("--out3", type=Path, default=ROOT / "out3")
    ap.add_argument("--pose", type=Path,
                    default=ROOT / "models" / "pose_landmarker_full.task")
    ap.add_argument("--margin", type=float, default=0.06,
                    help="crop margin around the trunk, as a fraction of its size")
    args = ap.parse_args()
    want = [n for n in args.names.split(",") if n]
    names = want or sorted(p.stem for p in args.data.glob("*.png"))
    args.out3.mkdir(parents=True, exist_ok=True)
    box_path = args.out3 / "crop_torso_boxes.json"
    boxes: dict[str, list[int]] = (
        json.loads(box_path.read_text(encoding="utf-8"))
        if box_path.exists() else {})
    print(f"{'img':5s} {'body':>8s} {'arm':>8s} {'torso':>8s} {'pool':>7s} "
          f"{'trunk rows':>12s}")
    for name in names:
        bgr = load_bgr(args.data / f"{name}.png")
        h, w = bgr.shape[:2]
        body = imread(args.out2 / f"bodymask_{name}.png", cv2.IMREAD_GRAYSCALE) > 127
        arm = imread(args.out2 / f"armmask_{name}.png", cv2.IMREAD_GRAYSCALE) > 127
        torso = imread(args.out2 / f"torsomask_{name}.png", cv2.IMREAD_GRAYSCALE) > 127
        body = clean_mask((body.astype(np.uint8)) * 255,
                          max_hole=int(max(64.0, 0.002 * body.sum()))) > 0
        arm &= body
        torso = (torso & body) & ~arm

        # 1. the classified person on a plain field: the pose model reads that
        field = np.full((h, w, 3), 236, np.uint8)
        person = field.copy()
        person[body] = bgr[body]
        save(args.out2 / f"clsperson_{name}.png", person)

        # 2. pose: use the RTMPose annotation when it exists; only fall back to
        # MediaPipe when the new model has not been run for this picture
        pts = rtm_pose(name, args.out3)
        pose_model = "RTMPose" if pts is not None else "MediaPipe"
        if pts is None:
            res = detect(cv2.cvtColor(person, cv2.COLOR_BGR2RGB), args.pose)
            pts = to_px(res, w, h)
            old_path = ROOT / "out1" / f"pose_{name}.json"
            doc = (json.loads(old_path.read_text(encoding="utf-8"))
                   if old_path.exists() else {})
            lm = []
            for i, p in enumerate(res.pose_landmarks[0]
                                  if res.pose_landmarks else []):
                lm.append({"index": i,
                           "name": doc.get("landmark_names", [None] * 33)[i]
                           if doc.get("landmark_names") else f"lm{i}",
                           "x": float(p.x * w), "y": float(p.y * h),
                           "z": float(p.z),
                           "visibility": float(p.visibility),
                           "presence": float(getattr(p, "presence", 0.0))})
            doc.update({"image": name, "size": [w, h],
                        "model": str(args.pose.name), "poses": len(lm) // 33,
                        "landmarks": lm,
                        "note": "MediaPipe pose on the *classified* person "
                                "(out2/clsperson_<n>.png); fallback because no "
                                "RTMPose annotation was available"})
            (args.out3 / f"pose_{name}.json").write_text(
                json.dumps(doc, indent=1), encoding="utf-8")
            save(args.out3 / f"pose_{name}.png",
                 draw_labelled(person, pts, lines=True, labels=False))

        # 3. arms / trunk separated, plus the 1/2 label image stage 18 wants
        arms_only = field.copy()
        arms_only[arm] = bgr[arm]
        trunk_only = field.copy()
        trunk_only[torso] = bgr[torso]
        save(args.out3 / f"armsplit_{name}.png", arms_only)
        save(args.out3 / f"torsosplit_{name}.png", trunk_only)
        label = np.zeros((h, w), np.uint8)
        lx = pts.get("left_wrist", (0, 0, 0))[0]
        rx = pts.get("right_wrist", (w, 0, 0))[0]
        left_is_image_left = lx < rx
        for side, val in (("left", 1), ("right", 2)):
            on_left = (side == "left") == left_is_image_left
            sub = arm.copy()
            half = np.zeros((h, w), bool)
            half[:, : w // 2] = on_left
            half[:, w // 2:] = not on_left
            sub &= half
            # whatever is on the wrong side belongs to the other arm
            label[sub] = val
            arm_rest = arm & ~sub
            if val == 1:
                other = arm_rest
            else:
                label[arm_rest] = 1
        save(args.out3 / f"armsep_label_{name}.png", label)

        # 4. the trunk crop for the pooling stage, and its box
        # the crop stays the *classification's* trunk (user's rule); the kernel
        # centre is mapped into w100 by its fractional position inside this crop
        ys, xs = np.flatnonzero(torso.any(1)), np.flatnonzero(torso.any(0))
        my = int(args.margin * (ys.max() - ys.min() + 1))
        mx = int(args.margin * (xs.max() - xs.min() + 1))
        y0, y1 = max(0, int(ys.min()) - my), min(h, int(ys.max()) + my + 1)
        x0, x1 = max(0, int(xs.min()) - mx), min(w, int(xs.max()) + mx + 1)
        rgba = np.zeros((y1 - y0, x1 - x0, 4), np.uint8)
        rgba[:, :, :3] = bgr[y0:y1, x0:x1]
        alpha = np.zeros((y1 - y0, x1 - x0), np.uint8)
        alpha[torso[y0:y1, x0:x1]] = 255
        rgba[:, :, 3] = alpha
        save(args.out3 / f"crop_torso_pose_{name}.png", rgba)
        save(args.out3 / f"crop_torso_weight_{name}.png", alpha)
        boxes[name] = [x0, y0, x1, y1]
        print(f"{name:5s} {int(body.sum()):8d} {int(arm.sum()):8d} "
              f"{int(torso.sum()):8d} {x1 - x0:3d}x{y1 - y0:<3d} "
              f"{int(ys.min())}-{int(ys.max())}  pose={pose_model}")
    box_path.write_text(
        json.dumps(boxes, indent=1), encoding="utf-8")
    print(f"\nwrote {args.out3}/pose_*, armsplit_*, torsosplit_*, "
          f"armsep_label_*, crop_torso_pose_* and the box file")


if __name__ == "__main__":
    main()
