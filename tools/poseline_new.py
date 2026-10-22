"""New shoulder line + belly line annotation.

Shoulder line:
  * u = elbow - shoulder (positive direction)
  * image-left arm uses n_left = (-u_y, u_x)
  * image-right arm uses n_right = (u_y, -u_x)
  * follow the normal from the shoulder to the first green body-outline point
  * path = left hit -> left shoulder -> right shoulder -> right hit

Belly line: the green-split JSON from waist_shoulder_green.py when available.
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def imread(path, flag=cv2.IMREAD_UNCHANGED):
    img = cv2.imdecode(np.fromfile(str(path), np.uint8), flag)
    if img is None:
        raise RuntimeError(f"cannot read {path}")
    return img


def walk_out(mask, p, n, step=0.25, reach=3000):
    h, w = mask.shape
    t = 0.0
    while t <= reach:
        x, y = int(round(p[0] + t * n[0])), int(round(p[1] + t * n[1]))
        if not (0 <= x < w and 0 <= y < h) or not mask[y, x]:
            return np.array([p[0] + t * n[0], p[1] + t * n[1]])
        t += step
    return np.array([p[0] + reach * n[0], p[1] + reach * n[1]])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--names", required=True)
    ap.add_argument("--out2", type=Path, default=ROOT / "out2")
    ap.add_argument("--out3", type=Path, default=ROOT / "out3")
    args = ap.parse_args()
    print(f"{'img':5s} {'waist':>6s} {'L hit':>9s} {'R hit':>9s} "
          f"{'3-seg':>8s}")
    for name in [n for n in args.names.split(",") if n]:
        src = args.out3 / f"poseonly_{name}.png"
        if not src.exists():
            src = args.out3 / f"pose_{name}.png"
        img = imread(src, cv2.IMREAD_COLOR)
        h, w = img.shape[:2]
        body = imread(args.out2 / f"bodymask_{name}.png",
                      cv2.IMREAD_GRAYSCALE) > 127
        pose = {q["name"]: q for q in json.loads(
            (args.out3 / f"pose_{name}.json").read_text(
                encoding="utf-8"))["landmarks"]}
        s_l = np.array([pose["left_shoulder"]["x"],
                        pose["left_shoulder"]["y"]], float)
        s_r = np.array([pose["right_shoulder"]["x"],
                        pose["right_shoulder"]["y"]], float)
        mid = 0.5 * (s_l + s_r)
        vis = img.copy()
        # belly line from the green split
        gs_path = args.out3 / f"greensplit_{name}.json"
        waist = 0
        if gs_path.exists():
            gs = json.loads(gs_path.read_text(encoding="utf-8"))
            waist = int(gs["min_row"])
            gl, gr = int(gs["left_green"]), int(gs["right_green"])
            cv2.line(vis, (gl, waist), (gr, waist), (0, 0, 255), 2,
                     cv2.LINE_AA)
            cv2.line(vis, (0, waist), (w - 1, waist), (0, 0, 170), 1,
                     cv2.LINE_AA)
            cv2.putText(vis, f"belly {waist}", (6, max(14, waist - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2,
                        cv2.LINE_AA)
        hits = {}
        for side in ("left", "right"):
            s = np.array([pose[f"{side}_shoulder"]["x"],
                          pose[f"{side}_shoulder"]["y"]], float)
            e = np.array([pose[f"{side}_elbow"]["x"],
                          pose[f"{side}_elbow"]["y"]], float)
            u = (e - s) / max(1e-9, np.hypot(*(e - s)))
            n_left = np.array([-u[1], u[0]])
            n_right = np.array([u[1], -u[0]])
            image_left = s[0] < mid[0]
            n = n_left if image_left else n_right
            hit = walk_out(body, s, n)
            hits[side] = (s, hit)
            cv2.line(vis, tuple(np.round(s).astype(int)),
                     tuple(np.round(hit).astype(int)), (255, 0, 0), 2,
                     cv2.LINE_AA)
            cv2.circle(vis, tuple(np.round(hit).astype(int)), 5,
                       (255, 255, 255), -1, cv2.LINE_AA)
            cv2.circle(vis, tuple(np.round(s).astype(int)), 4,
                       (0, 0, 255), -1, cv2.LINE_AA)
        image_left_side = "left" if s_l[0] < s_r[0] else "right"
        image_right_side = "right" if image_left_side == "left" else "left"
        s_l0, b_l = hits[image_left_side]
        s_r0, b_r = hits[image_right_side]
        l1 = float(np.hypot(*(b_l - s_l0)))
        l2 = float(np.hypot(*(s_r0 - s_l0)))
        l3 = float(np.hypot(*(b_r - s_r0)))
        total = l1 + l2 + l3
        cv2.polylines(vis, [np.array([b_l, s_l0, s_r0, b_r], np.int32)],
                      False, (0, 255, 255), 2, cv2.LINE_AA)
        txt = f"shoulder line {total:.1f} px"
        cv2.putText(vis, txt, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(vis, txt, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (0, 255, 255), 1, cv2.LINE_AA)
        (args.out3 / f"shoulderline_{name}.json").write_text(json.dumps(
            {"image": name,
             "image_left_side": image_left_side,
             "image_right_side": image_right_side,
             "left_boundary": b_l.round(2).tolist(),
             "left_shoulder": s_l0.round(2).tolist(),
             "right_shoulder": s_r0.round(2).tolist(),
             "right_boundary": b_r.round(2).tolist(),
             "left_segment_px": round(l1, 2),
             "shoulder_segment_px": round(l2, 2),
             "right_segment_px": round(l3, 2),
             "three_segments": round(total, 2),
             "waist_row": waist,
             "note": "shoulder->elbow is positive; image-left arm uses "
                     "n_left, image-right arm uses n_right; first green "
                     "outline hit"}, indent=1), encoding="utf-8")
        cv2.imencode(".png", vis)[1].tofile(
            str(args.out3 / f"poseline_{name}.png"))
        cv2.imencode(".png", vis)[1].tofile(
            str(args.out3 / f"pose_{name}.png"))
        print(f"{name:5s} {waist:6d} "
              f"{str(tuple(b_l.round(0).astype(int))):>9s} "
              f"{str(tuple(b_r.round(0).astype(int))):>9s} "
              f"{total:8.1f}")
    print(f"\nwrote {args.out3}/poseline_<n>.png and shoulderline_<n>.json")


if __name__ == "__main__":
    main()
