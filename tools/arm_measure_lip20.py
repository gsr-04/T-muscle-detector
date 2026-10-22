"""Arm measurement with LIP-20 skin, body-outline perpendiculars, and the
pose-based usable-arm choice.

At M1 (shoulder-elbow midpoint) and M2 (elbow-wrist midpoint) the two
perpendicular directions are followed to the green body outline.  If a piece
is longer than the wrist-elbow line, it is re-cut at the LIP-20 arm-skin edge.
The kept line is M1 -> elbow -> M2; ratio = kept / four pieces.
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "src"))
from torso_muscle_annot import schp                       # noqa: E402
from torso_seg import load_bgr                            # noqa: E402


def imread(path, flag=cv2.IMREAD_UNCHANGED):
    img = cv2.imdecode(np.fromfile(str(path), np.uint8), flag)
    if img is None:
        raise RuntimeError(f"cannot read {path}")
    return img


def exit_dist(mask, p, n, reach, step=0.25):
    """Distance from p along unit n until the mask becomes False."""
    h, w = mask.shape
    t = 0.0
    while t <= reach:
        x = int(round(p[0] + t * n[0]))
        y = int(round(p[1] + t * n[1]))
        if not (0 <= x < w and 0 <= y < h) or not mask[y, x]:
            return t
        t += step
    return reach


def first_inside(m, target, body, lip_any):
    """Move m toward target until it lies in the body or LIP arm mask."""
    if body[int(round(m[1])), int(round(m[0]))] or \
            lip_any[int(round(m[1])), int(round(m[0]))]:
        return m, 0.0
    vec = target - m
    length = float(np.hypot(*vec))
    if length < 1e-6:
        return m, 0.0
    unit = vec / length
    for t in np.arange(1.0, min(0.45 * length, 120.0) + 1.0, 1.0):
        q = m + t * unit
        x, y = int(round(q[0])), int(round(q[1]))
        if 0 <= y < body.shape[0] and 0 <= x < body.shape[1] and \
                (body[y, x] or lip_any[y, x]):
            return q, float(t)
    return m, 0.0


def runs_of(row, min_len=5):
    idx = np.flatnonzero(row)
    if not idx.size:
        return []
    out, s, p = [], int(idx[0]), int(idx[0])
    for v in idx[1:]:
        v = int(v)
        if v == p + 1:
            p = v
        else:
            if p - s + 1 >= min_len:
                out.append((s, p))
            s = p = v
    if p - s + 1 >= min_len:
        out.append((s, p))
    return out


def snap_to_arm_run(m, wrist_x, side_arm, legacy, body):
    """Use the arm run on this row that contains/nearest the wrist x."""
    y = int(round(m[1]))
    for mask in (side_arm, legacy, body):
        rr = runs_of(mask[y])
        if not rr:
            continue
        containing = [r for r in rr if r[0] - 2 <= wrist_x <= r[1] + 2]
        r = (containing[0] if containing else
             min(rr, key=lambda q: abs(0.5 * (q[0] + q[1]) - wrist_x)))
        cx = 0.5 * (r[0] + r[1])
        return np.array([cx, float(y)]), abs(cx - m[0])
    return m, 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--names", default="018,020,025,028,029")
    ap.add_argument("--out", type=Path, default=ROOT / "out3")
    args = ap.parse_args()
    print(f"{'img':5s} {'use':>5s} {'side':>5s} {'four':>7s} "
          f"{'kept':>7s} {'ratio':>6s} {'score':>6s}")
    for name in [n for n in args.names.split(",") if n]:
        bgr = load_bgr(ROOT / "data" / f"{name}.png")
        h, w = bgr.shape[:2]
        body = imread(ROOT / "out2" / f"bodymask_{name}.png",
                      cv2.IMREAD_GRAYSCALE) > 127
        legacy = imread(ROOT / "out2" / f"armmask_{name}.png",
                        cv2.IMREAD_GRAYSCALE) > 127
        cls = schp("lip", bgr, ROOT / "models")
        lip = {14: cls == 14, 15: cls == 15}
        lip_any = (cls == 14) | (cls == 15)
        pose = {q["name"]: q for q in json.loads(
            (ROOT / "out3" / f"pose_{name}.json").read_text(
                encoding="utf-8"))["landmarks"]}
        sel = json.loads((ROOT / "out3" / f"armselect_{name}.json").read_text(
            encoding="utf-8"))
        selected = sel["selected"]
        sides = {}
        for side, lip_val in (("left", 14), ("right", 15)):
            s = np.array([pose[f"{side}_shoulder"]["x"],
                          pose[f"{side}_shoulder"]["y"]], float)
            e = np.array([pose[f"{side}_elbow"]["x"],
                          pose[f"{side}_elbow"]["y"]], float)
            wr = np.array([pose[f"{side}_wrist"]["x"],
                           pose[f"{side}_wrist"]["y"]], float)
            side_mask = lip[lip_val]
            m1, m2 = 0.5 * (s + e), 0.5 * (e + wr)
            b1, sn1 = snap_to_arm_run(m1, wr[0], side_mask, legacy, body)
            b2, sn2 = snap_to_arm_run(m2, wr[0], side_mask, legacy, body)
            if not (body[int(round(b1[1])), int(round(b1[0]))] or
                    lip_any[int(round(b1[1])), int(round(b1[0]))]):
                b1, sh1 = first_inside(b1, e, body, lip_any)
            else:
                sh1 = 0.0
            if not (body[int(round(b2[1])), int(round(b2[0]))] or
                    lip_any[int(round(b2[1])), int(round(b2[0]))]):
                b2, sh2 = first_inside(b2, wr, body, lip_any)
            else:
                sh2 = 0.0
            u1 = (e - s) / max(1e-9, np.hypot(*(e - s)))
            u2 = (wr - e) / max(1e-9, np.hypot(*(wr - e)))
            n1 = np.array([-u1[1], u1[0]])
            n2 = np.array([-u2[1], u2[0]])
            fore = float(np.hypot(*(wr - e)))
            rec = {"shoulder": s.round(2).tolist(),
                   "elbow": e.round(2).tolist(),
                   "wrist": wr.round(2).tolist(),
                   "M1": m1.round(2).tolist(), "M2": m2.round(2).tolist(),
                   "M1_base": b1.round(2).tolist(),
                   "M2_base": b2.round(2).tolist(),
                   "M1_snap": round(sn1, 2), "M2_snap": round(sn2, 2),
                   "M1_shift": round(sh1, 2), "M2_shift": round(sh2, 2),
                   "forearm": round(fore, 2), "pieces": []}
            total = 0.0
            for tag, m, n in (("M1", b1, n1), ("M2", b2, n2)):
                for sign in (1.0, -1.0):
                    dvec = n * sign
                    body_ok = bool(body[int(round(m[1])), int(round(m[0]))])
                    db = (exit_dist(body, m, dvec, max(h, w))
                          if body_ok else 0.0)
                    used, clamped = db, False
                    lip_ok = True
                    if db <= 0 or db > fore:
                        msk = side_mask if side_mask[int(round(m[1])),
                                                     int(round(m[0]))] else lip_any
                        da = exit_dist(msk, m, dvec, max(h, w))
                        if 0 < da < db:
                            used, clamped = da, True
                        else:
                            lip_ok = False
                    rec["pieces"].append(
                        {"at": tag, "sign": int(sign), "body": round(db, 2),
                         "used": round(used, 2), "clamped": bool(clamped),
                         "lip_ok": bool(lip_ok),
                         "end": (m + used * dvec).round(2).tolist()})
                    total += used
            kept = float(np.hypot(*(m1 - e)) + np.hypot(*(m2 - e)))
            ratio = kept / total if total else None
            score = (min(100.0, max(0.0, (2.2 - ratio) * 100.0))
                     if ratio is not None else None)
            rec.update({"four_sum": round(total, 2),
                        "kept": round(kept, 2),
                        "ratio": round(ratio, 4) if ratio is not None else None,
                        "arm_score": round(score, 2) if score is not None else None,
                        "usable": selected in (side, "both")})
            sides[side] = rec
        usable = [s for s in ("left", "right") if sides[s]["usable"]]
        if len(usable) == 1:
            src = usable[0]
            dst = "right" if src == "left" else "left"
            sides[dst]["effective_from"] = src
            for key in ("four_sum", "kept", "ratio", "arm_score"):
                sides[dst][f"effective_{key}"] = sides[src][key]
        # effective total ratio (one arm is copied to the other)
        eff = []
        for s in ("left", "right"):
            if sides[s]["usable"]:
                eff.append(sides[s])
            elif len(usable) == 1:
                eff.append(sides[usable[0]])
        ratio = (sum(x["kept"] for x in eff) /
                 sum(x["four_sum"] for x in eff)) if eff else None
        score = (min(100.0, max(0.0, (2.2 - ratio) * 100.0))
                 if ratio is not None else None)
        for s in ("left", "right"):
            r = sides[s]
            print(f"{name:5s} {selected:>5s} {s:>5s} {r['four_sum']:7.1f} "
                  f"{r['kept']:7.1f} "
                  f"{(r['ratio'] if r['ratio'] is not None else float('nan')):6.2f} "
                  f"{(r['arm_score'] if r['arm_score'] is not None else float('nan')):6.1f}")
        vis = bgr.copy()
        # LIP-20 arm skin: usable green, other grey
        tint = vis.copy()
        for s, lv in (("left", 14), ("right", 15)):
            m = lip[lv]
            if s in usable:
                tint[m] = (40, 200, 40)
            else:
                tint[m] = (110, 110, 110)
        vis = cv2.addWeighted(tint, 0.38, vis, 0.62, 0)
        body_contours, _ = cv2.findContours(
            body.astype(np.uint8),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_NONE,
        )
        cv2.drawContours(
            vis, body_contours, -1, (0, 255, 0), 2, cv2.LINE_AA
        )
        for s in ("left", "right"):
            r = sides[s]
            s0 = tuple(np.round(r["shoulder"]).astype(int))
            e0 = tuple(np.round(r["elbow"]).astype(int))
            w0 = tuple(np.round(r["wrist"]).astype(int))
            on = r["usable"]
            col = (0, 220, 0) if on else (110, 110, 110)
            cv2.line(vis, s0, e0, col, 4 if on else 2, cv2.LINE_AA)
            cv2.line(vis, e0, w0, col, 4 if on else 2, cv2.LINE_AA)
            for p, c in ((s0, (255, 0, 0)), (e0, (0, 200, 255)),
                         (w0, (0, 0, 255))):
                cv2.circle(vis, p, 5, c, -1, cv2.LINE_AA)
            if on:
                for q in r["pieces"]:
                    end = tuple(np.round(q["end"]).astype(int))
                    base = tuple(np.round(
                        r["M1_base"] if q["at"] == "M1"
                        else r["M2_base"]).astype(int))
                    pcol = (0, 165, 255) if q["clamped"] else (0, 255, 255)
                    cv2.line(vis, base, end, pcol, 2, cv2.LINE_AA)
                    cv2.circle(vis, end, 4, (255, 255, 255), -1,
                               cv2.LINE_AA)
                m1 = tuple(np.round(r["M1_base"]).astype(int))
                m2 = tuple(np.round(r["M2_base"]).astype(int))
                cv2.circle(vis, m1, 4, (255, 0, 255), -1, cv2.LINE_AA)
                cv2.circle(vis, m2, 4, (255, 0, 255), -1, cv2.LINE_AA)
        txt = (f"{name}  use {selected}  total ratio "
               f"{ratio:.3f}" if ratio is not None else
               f"{name}  use {selected}  no measurement")
        cv2.putText(vis, txt, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(vis, txt, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (0, 255, 255), 1, cv2.LINE_AA)
        cv2.imencode(".png", vis)[1].tofile(
            str(args.out / f"armmeasure_{name}.png"))
        (args.out / f"armmeasure_{name}.json").write_text(json.dumps(
            {"image": name, "selected": selected,
             "reason": sel.get("reason"), "sides": sides,
             "effective_ratio": round(ratio, 4) if ratio is not None else None,
             "effective_score": round(score, 2) if score is not None else None,
             "note": "LIP-20 arm skin; body-outline perpendiculars; a piece "
                     "longer than the forearm is re-cut at the arm-skin edge; "
                     "one usable arm is copied to the other"}, indent=1),
            encoding="utf-8")
    print(f"\nwrote {args.out}/armmeasure_<n>.png/.json")


if __name__ == "__main__":
    main()
