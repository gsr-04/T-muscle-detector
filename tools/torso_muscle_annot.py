"""Person cut-out + SCHP arm/torso skin + DexiNed torso muscle lines.

Uses the body mask already in out2 (BiRefNet) to cut the person out, runs the
three SCHP schemes on it and merges them into arm skin / torso skin, then marks
the DexiNed edges that fall inside the torso.

  arm skin   = at least two of  LIP(14,15) | ATR(14,15) | PASCAL(3,4)
  torso skin = PASCAL(2)        - the only scheme with a bare-torso class
  muscle     = DexiNed edges inside the eroded torso

    python tools/torso_muscle_annot.py --reuse            # all data/*.png
    python tools/torso_muscle_annot.py --reuse --names 016
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))
from annotate_edges import dexined_edges               # noqa: E402
from torso_seg import clean_mask, load_bgr             # noqa: E402

SCHEMES = {                                            # name -> (dir, size)
    "lip": ("schp-lip-20", 473),
    "atr": ("schp-atr-18", 512),
    "pascal": ("schp-pascal-7", 512),
}
ARMS = {"lip": (14, 15), "atr": (14, 15), "pascal": (3, 4)}
TORSO = {"pascal": 2}
_SESS: dict[str, object] = {}


def session(key: str, size: int, model_dir: Path):
    import onnxruntime as ort

    if key not in _SESS:
        so = ort.SessionOptions()
        so.log_severity_level = 3
        path = (model_dir / SCHEMES[key][0] / "onnx" /
                f"{SCHEMES[key][0]}-int8-static.onnx")
        _SESS[key] = ort.InferenceSession(str(path), so,
                                          providers=["CPUExecutionProvider"])
    return _SESS[key]


def schp(key: str, bgr: np.ndarray, model_dir: Path) -> np.ndarray:
    """Class map of the whole image (uint8), neareast-resized back."""
    import onnxruntime as ort  # noqa: F401

    h, w = bgr.shape[:2]
    size = SCHEMES[key][1]
    cfg = json.loads((model_dir / SCHEMES[key][0] /
                      "preprocessor_config.json").read_text(encoding="utf-8"))
    mean = np.array(cfg["image_mean"], np.float32)
    std = np.array(cfg["image_std"], np.float32)
    small = cv2.resize(bgr, (size, size), interpolation=cv2.INTER_AREA)
    x = cv2.cvtColor(small, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    x = ((x - mean) / std).transpose(2, 0, 1)[None]
    sess = session(key, size, model_dir)
    out = np.asarray(sess.run(None, {sess.get_inputs()[0].name: x})[0])
    cls = out[0].argmax(0).astype(np.uint8)
    return cv2.resize(cls, (w, h), interpolation=cv2.INTER_NEAREST)


def runs_of(row: np.ndarray, min_len: int = 6):
    """[(start, end)] of the True runs in one row."""
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


def rtm_pose(name: str):
    """The RTMPose landmarks of this picture, if they have been written."""
    path = ROOT / "out3" / f"pose_{name}.json"
    if not path.exists():
        return None
    doc = json.loads(path.read_text(encoding="utf-8"))
    if "RTMPose" not in str(doc.get("source", "")):
        return None
    lm = {q["name"]: q for q in doc.get("landmarks", [])}
    need = ("left_shoulder", "left_elbow", "left_wrist",
            "right_shoulder", "right_elbow", "right_wrist")
    return lm if all(k in lm for k in need) else None


def fill_forearm(arm: np.ndarray, body: np.ndarray, lm: dict,
                 ref_torso: np.ndarray | None, name: str) -> int:
    """Complete the elbow->wrist part of the arm mask from the body mask.

    SCHP sometimes reads a bare forearm as background/upper-clothes (020: rows
    696..818).  The pose gives the forearm axis, and the body mask gives its
    silhouette.  A row is only filled when the forearm is a *separate* body run
    from the trunk: if the arm touches the trunk the merged run is left alone.
    """
    h, w = arm.shape
    diag = float(np.hypot(w, h))
    added = 0
    for side in ("left", "right"):
        try:
            s = np.array([lm[f"{side}_shoulder"]["x"],
                          lm[f"{side}_shoulder"]["y"]], float)
            e = np.array([lm[f"{side}_elbow"]["x"],
                          lm[f"{side}_elbow"]["y"]], float)
            wr = np.array([lm[f"{side}_wrist"]["x"],
                           lm[f"{side}_wrist"]["y"]], float)
        except (KeyError, TypeError, ValueError):
            continue
        if not all(-0.05 * w <= p[0] <= 1.05 * w and
                   -0.05 * h <= p[1] <= 1.05 * h for p in (s, e, wr)):
            continue
        se, ew, sw = (float(np.hypot(*(q - p)))
                      for p, q in ((s, e), (e, wr), (s, wr)))
        if min(se, ew) < 0.03 * diag or sw < 0.6 * max(se, ew):
            continue                     # missing joint or folded arm
        if abs(wr[1] - e[1]) < 0.45 * ew:
            continue                     # only the hanging forearm is handled
        hip = [lm.get(f"{side}_hip")] if lm.get(f"{side}_hip") else []
        mx = float(np.mean([s[0], wr[0]] + [q["x"] for q in hip]))
        if abs(0.5 * (e[0] + wr[0]) - mx) < 0.06 * w:
            continue                     # axis on the midline: not an arm
        y0 = int(max(0, round(min(e[1], wr[1]))))
        y1 = int(min(h - 1, round(max(e[1], wr[1]))))
        side_tot = 0
        for y in range(y0, y1 + 1):
            t = ((y - e[1]) / (wr[1] - e[1])
                 if abs(wr[1] - e[1]) > 1e-6 else 0.0)
            xc = e[0] + t * (wr[0] - e[0])
            cand = [r for r in runs_of(body[y]) if r[0] - 1 <= xc <= r[1] + 1]
            if not cand:
                continue
            r0, r1 = cand[0]
            trunk = [r for r in runs_of(body[y]) if r[0] - 1 <= mx <= r[1] + 1]
            if trunk and (r0, r1) == trunk[0]:
                continue                 # arm and trunk touch: do not guess
            if (0.5 * (r0 + r1) - mx) * (xc - mx) <= 0:
                continue                 # wrong side of the body
            half = max(8.0, 0.55 * (r1 - r0 + 1))
            a = max(r0, int(round(xc - half)))
            b = min(r1, int(round(xc + half)))
            if b < a:
                continue
            add = np.zeros(w, bool)
            add[a:b + 1] = True
            add &= body[y]
            if ref_torso is not None:
                add &= ref_torso[y] == 0
            new = add & ~arm[y]
            if new.any():
                arm[y] |= add
                side_tot += int(new.sum())
        if side_tot:
            print(f"{name:5s} {side} forearm gap filled: {side_tot} px")
            added += side_tot
    return added


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--names", default="")
    ap.add_argument("--data", type=Path, default=ROOT / "data")
    ap.add_argument("--out2", type=Path, default=ROOT / "out2")
    ap.add_argument("--models", type=Path, default=ROOT / "models")
    ap.add_argument("--schemes", default="lip,atr,pascal")
    ap.add_argument("--need", type=int, default=1,
                    help="how many schemes must call a pixel arm (1 = any; "
                         "LIP and ATR both miss arms on some photos)")
    ap.add_argument("--reuse", action="store_true",
                    help="use out2/bodymask_<n>.png (skip BiRefNet)")
    args = ap.parse_args()
    keys = [k.strip() for k in args.schemes.split(",") if k.strip()]
    want = [n for n in args.names.split(",") if n]
    names = want or sorted(p.stem for p in args.data.glob("*.png"))
    args.out2.mkdir(parents=True, exist_ok=True)
    print(f"{'img':5s} {'body px':>9s} {'arm skin':>9s} {'torso skin':>11s} "
          f"{'votes':>6s} {'time':>6s}")
    for name in names:
        t0 = time.time()
        bgr = load_bgr(args.data / f"{name}.png")
        h, w = bgr.shape[:2]
        mpath = args.out2 / f"bodymask_{name}.png"
        if not (args.reuse and mpath.exists()):
            raise SystemExit(f"missing {mpath} (run annotate_edges.py first)")
        body = cv2.imdecode(np.fromfile(str(mpath), np.uint8),
                            cv2.IMREAD_GRAYSCALE) > 127
        body = clean_mask((body.astype(np.uint8)) * 255,
                          max_hole=int(max(64.0, 0.002 * body.sum()))) > 0
        cls = {k: schp(k, bgr, args.models) for k in keys}
        arm_votes = np.zeros((h, w), np.uint8)
        for k in keys:
            if k in ARMS:
                a = np.isin(cls[k], ARMS[k])
                arm_votes += a.astype(np.uint8)
        arm = (arm_votes >= args.need) & body
        # the arm class stops a little inside the real arm edge, so a thin strip
        # between the arm and the belly used to be left in the trunk (blue)
        arm_w = cv2.dilate(arm.astype(np.uint8), np.ones((7, 7), np.uint8)) > 0
        # the strip belongs to the arm, not to nobody: taking it off the trunk
        # and *leaving it to the arm* avoids the unmarked sliver between the
        # blue trunk and the green outline
        arm_w &= body
        arm = arm_w
        # the arm may not swallow the trunk: the part model already knows where
        # the trunk is (out/simple_<n>.json), so keep the arm out of it
        ref_path = ROOT / "out" / f"simple_{name}.json"
        ref_torso = None
        if ref_path.exists():
            ref_doc = json.loads(ref_path.read_text(encoding="utf-8"))
            ref_torso = np.zeros((h, w), np.uint8)
            for poly in ref_doc["classes"].get("torso", {}).get("polygons", []):
                cv2.fillPoly(ref_torso,
                             [np.round(np.array(poly, float)).astype(np.int32)],
                             1)
            arm = arm & ~(ref_torso > 0)
        # pose-guided completion of a forearm SCHP missed (the M2 perpendicular
        # of the arm ratio lives here); only runs when an RTMPose is available
        lm_rtm = rtm_pose(name)
        if lm_rtm is not None:
            fill_forearm(arm, body, lm_rtm, ref_torso, name)
        # the trunk is defined by subtraction, not by a class: the body minus
        # the arms, minus the trousers, minus everything above the chin
        pants = np.zeros((h, w), bool)
        if "atr" in cls:                     # trousers + belt
            pants |= np.isin(cls["atr"], (5, 6, 8))
        if "lip" in cls:                     # pants, jumpsuit, skirt
            pants |= np.isin(cls["lip"], (9, 10, 12))
        if "pascal" in cls:                  # legs (below the waist in practice)
            pants |= np.isin(cls["pascal"], (5, 6))
        pants = (pants & body) & ~arm
        # the lower boundary is the trouser waistband, column by column: a
        # single horizontal floor (the crotch) cut the lower belly away
        from scipy import ndimage as ndi
        # lower boundary = the bottom of the trunk class of the existing part
        # model (out/torso_<n>.png), which ends at the trouser waistband that
        # the rest of the project already measures; the class masks above only
        # add to it
        top = np.full(w, -1, np.int32)
        tpath = ROOT / "out" / f"torso_{name}.png"
        if tpath.exists():
            tr = cv2.imdecode(np.fromfile(str(tpath), np.uint8),
                              cv2.IMREAD_UNCHANGED)
            tr = (tr[:, :, 3] > 127) if tr.ndim == 3 else (tr > 127)
            for x in np.flatnonzero(tr.any(0)):
                top[x] = int(np.flatnonzero(tr[:, x]).max()) + 1
        known = top >= 0
        if known.sum() >= 0.3 * w:
            top = ndi.median_filter(np.where(known, top, 10 ** 6),
                                   size=max(5, w // 30), mode="nearest")
            top[~known] = int(np.median(top[known]))
        else:                                  # no part model: hip landmarks
            pj = ROOT / "out1" / f"pose_{name}.json"
            if pj.exists():
                lm = {q["name"]: q for q in json.loads(
                    pj.read_text(encoding="utf-8"))["landmarks"]}
                S = float(np.hypot(lm["left_shoulder"]["x"]
                                   - lm["right_shoulder"]["x"],
                                   lm["left_shoulder"]["y"]
                                   - lm["right_shoulder"]["y"]))
                hy = 0.5 * (lm["left_hip"]["y"] + lm["right_hip"]["y"])
                top[:] = int(min(h - 1, hy - 0.02 * S))
                print(f"{name:5s} (no trunk class: waistband from the hip "
                      f"landmarks, row {int(top[0])})")
        if (top >= 0).any():
            pants |= np.arange(h)[:, None] >= np.maximum(top, 0)[None, :]
        # second opinion on the waistband: the clothes class of the selfie
        # multiclass model.  Where the trunk class stops early (016: 364 while
        # the trousers only start at 382+) this keeps the lower belly
        try:
            sys.path.insert(0, str(ROOT / "tools"))
            from multiclass_probe import multiclass            # noqa: WPS433
            mcls = multiclass(bgr, ROOT / "models" /
                              "selfie_multiclass.tflite").argmax(0)
            dress = (mcls == 4) & body
            line2 = np.full(w, -1, np.int32)
            for x in np.flatnonzero(dress.any(0)):
                line2[x] = int(np.flatnonzero(dress[:, x]).min())
            both = (line2 >= 0) & (top >= 0)
            if both.sum() >= 0.05 * w:
                lower = np.maximum(line2, top)
                top = np.where(both, lower, np.maximum(top, line2))
                top = ndi.median_filter(top, size=max(5, w // 30),
                                      mode="nearest")
                pants |= np.arange(h)[:, None] >= np.maximum(top, 0)[None, :]
                pants &= body & ~arm
        except Exception as exc:                                # noqa: BLE001
            print(f"    (clothes class unavailable: {type(exc).__name__})")
        pants &= body & ~arm
        # the chin, from the pose landmarks: the mouth plus a sixth of the
        # shoulder width (LIP's face class fails on these photos)
        chin = 0
        pj = ROOT / "out1" / f"pose_{name}.json"
        if pj.exists():
            lm = {q["name"]: q for q in json.loads(
                pj.read_text(encoding="utf-8"))["landmarks"]}
            if {"left_shoulder", "right_shoulder"} <= set(lm):
                sy = np.array([lm["left_shoulder"]["y"], lm["right_shoulder"]["y"]])
                sx = np.array([lm["left_shoulder"]["x"], lm["right_shoulder"]["x"]])
                sx = np.array([lm["left_shoulder"]["x"], lm["right_shoulder"]["x"]])
                S = float(np.hypot(*(sx - sx[::-1])) ) if False else float(
                    np.hypot(sx[0] - sx[1], sy[0] - sy[1]))
                my = np.mean([lm[q]["y"] for q in ("mouth_left", "mouth_right")
                              if q in lm]) if any(
                    q in lm for q in ("mouth_left", "mouth_right")) else None
                if my is None and "nose" in lm:
                    my = lm["nose"]["y"] + 0.05 * S
                if my is not None:
                    chin = int(my + 0.06 * S)
                    # a pose that reads the chest as the face puts the chin far
                    # below the shoulders: never cut the trunk below the
                    # shoulder line (that removed the whole upper chest on 012)
                    chin = min(chin, int(min(lm["left_shoulder"]["y"],
                                             lm["right_shoulder"]["y"])
                                         - 0.10 * S))
                    # a head above the frame gives a negative mouth/nose row;
                    # above[:chin+1] would then become a negative slice and
                    # remove the whole torso (031: mouth y=-79)
                    chin = max(0, chin)
        above = np.zeros((h, w), bool)
        above[:chin + 1] = True
        torso = body & ~arm_w & ~above          # the chin cut is pose-based: safe
        cut = torso & ~pants                  # the trousers/legs cut
        if cut.sum() >= 0.05 * body.sum():
            torso = cut
        else:
            # the class masks can fail badly (LIP calls the whole chest a
            # jumpsuit on 016): fall back to the waistband row of the trunk
            # class alone, which is what "remove the legs" needs
            line = np.zeros((h, w), bool)
            if (top >= 0).any():
                line = np.arange(h)[:, None] < np.maximum(top, 0)[None, :]
            alt = torso & line
            if alt.sum() >= 0.05 * body.sum():
                torso = alt
                print(f"{name:5s} (trousers classes unusable: cut at the trunk "
                      f"waistband row)")
            else:
                print(f"{name:5s} (trousers cut would empty the trunk, kept "
                      f"the rest)")
        # close pinholes left by the class boundaries, keep it inside the body
        torso = cv2.morphologyEx(torso.astype(np.uint8), cv2.MORPH_CLOSE,
                                 np.ones((9, 9), np.uint8)) > 0
        torso = clean_mask((torso.astype(np.uint8)) * 255,
                           max_hole=int(max(256.0, 0.002 * body.sum()))) > 0
        torso &= body & ~arm_w & ~above
        # never claim the strip between an arm and the trunk: the gaps are read
        # off the *raw* silhouette, because the body mask has small holes filled
        # and that strip is exactly such a hole when the arm touches below
        raw_path = ROOT / "out" / f"fgmask_{name}.png"
        if raw_path.exists():
            raw = cv2.imdecode(np.fromfile(str(raw_path), np.uint8),
                               cv2.IMREAD_GRAYSCALE)
            raw = (raw > 0).astype(np.uint8) * 255
            sys.path.insert(0, str(ROOT / "src"))
            from torso_seg import outside_gaps                  # noqa: WPS433
            torso &= outside_gaps(raw) > 0
        if not arm.any() or not torso.any():
            print(f"{name:5s} nothing segmented, skipped")
            continue
        # the picture: the cut-out person on a light field
        vis = np.full((h, w, 3), 236, np.uint8)
        vis[body] = bgr[body]
        tint = vis.copy()
        tint[torso] = (0.55 * tint[torso] + 0.45 * np.array([60, 150, 240]))
        tint[arm] = (0.60 * tint[arm] + 0.40 * np.array([235, 130, 60]))
        vis = tint.astype(np.uint8)
        cnts, _ = cv2.findContours(body.astype(np.uint8), cv2.RETR_CCOMP,
                                   cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(vis, cnts, -1, (0, 200, 0), 2, cv2.LINE_AA)
        ok, buf = cv2.imencode(".png", vis)
        buf.tofile(str(args.out2 / f"seg_{name}.png"))
        ok, buf = cv2.imencode(".png", (arm * 255).astype(np.uint8))
        buf.tofile(str(args.out2 / f"armmask_{name}.png"))
        ok, buf = cv2.imencode(".png", (torso * 255).astype(np.uint8))
        buf.tofile(str(args.out2 / f"torsomask_{name}.png"))
        agree = float(arm_votes[body & (arm_votes > 0)].mean()) if body.any() else 0
        print(f"{name:5s} {int(body.sum()):9d} {int(arm.sum()):8d} "
              f"{int(torso.sum()):8d} {arm_votes[body].mean():6.2f} "
              f"{time.time() - t0:5.1f}s")
    print(f"\nwrote {args.out2}/seg_<n>.png  (light blue = torso skin, "
          f"orange = arm skin, green = BiRefNet outline)")


if __name__ == "__main__":
    main()
