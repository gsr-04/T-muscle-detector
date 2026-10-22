"""One new picture through the outfinal flow, with the results in outfinal/.

    1. 02_prepare_image.py imports the picture and builds only the
       prerequisites
       needed by the current chain
    2. annotate_edges.py     BiRefNet body mask + the green/red edge figures
       pose_rtmpose_person.py top-down RTMPose-m on the original picture
    3. torso_muscle_annot.py SCHP arm/trunk skin, pose-guided forearm gap fill
                              and the "trunk" split
    4. trunk_span.py          six-cell rows -> the middle span (reference only)
       trunkspan_min.py       the old narrowest row (reference only)
    5. reclass_split.py       arms/trunk apart, crop (its MediaPipe pose is a
                              fallback only)
       pose_rtmpose_person.py top-down RTMPose-m on the *original* picture
                              (person box from the out1 pose / body mask)
       waist_shoulder_green.py shoulder-midpoint green split belly line
       poseline_new.py        new inward shoulder line + green belly line
    6. 07_detect_usable_arms.py
                              usable-arm rules + LIP-20 arm measurement
    7. 09_analyze_abdomen.py
                              abdomen rectangle -> pooled annotation -> centres
                              -> curve objective -> open angles -> score
    8. score = clamp((240 - mean_open_angle) * 100 / 180, 0, 100)

    python pipeline/01_run_pipeline.py path/to/photo.jpg
    python pipeline/01_run_pipeline.py path/to/photo.png --name 017
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = Path(os.environ.get("TMUSCLE_PYTHON", sys.executable))


def run(script: Path, extra: list[str], quiet: bool = True) -> str:
    cmd = [str(PY), str(script)] + extra
    print("  $ " + " ".join(str(p) for p in cmd[1:]))
    r = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True)
    out = (r.stdout or "") + (r.stderr or "")
    if r.returncode != 0:
        tail = "\n".join([ln for ln in out.strip().splitlines() if ln][-8:])
        raise SystemExit(f"{script.name} failed ({r.returncode}):\n{tail}")
    if not quiet:
        print(out)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("image", help="path of the picture to test")
    ap.add_argument("--name", default="",
                    help="force the data/<name>.png number (default: next free)")
    ap.add_argument("--waist-source", choices=("skin", "sixcell", "greensplit"),
                    default="greensplit",
                    help="belly line from the abdominal skin or the six-cell span")
    args = ap.parse_args()
    img = Path(args.image)
    if not img.is_absolute():
        img = (ROOT / img).resolve()
    if not img.exists():
        raise SystemExit(f"no such picture: {img}")

    print("1/8 prepare input image and torso prerequisites")
    run(
        ROOT / "pipeline" / "02_prepare_image.py",
        [str(img)] + (["--name", args.name] if args.name else []),
        quiet=False,
    )
    reg = json.loads((ROOT / "out1" / "image_registry.json").read_text(
        encoding="utf-8"))
    name = None
    for key, val in reg.items():
        if Path(key).resolve() == img:
            name = val
    if name is None:
        name = args.name
    if not name:
        raise SystemExit("could not work out the image number")
    print(f"  -> data/{name}.png")

    print("2/8 annotate_edges.py  (BiRefNet body mask)")
    run(ROOT / "tools" / "annotate_edges.py", ["--names", name])
    print("    pose_rtmpose_person.py  (RTMPose before the arm mask)")
    try:
        run(ROOT / "tools" / "pose_rtmpose_person.py", ["--names", name])
    except SystemExit as exc:
        print(f"  (RTMPose unavailable: {exc}) - forearm fill skipped")
    print("3/8 torso_muscle_annot.py  (SCHP arm/trunk skin)")
    run(ROOT / "tools" / "torso_muscle_annot.py",
        ["--reuse", "--names", name])
    print("4/8 trunk_span.py + trunkspan_min.py  (belly's narrowest line)")
    run(ROOT / "tools" / "trunk_span.py", [])
    run(ROOT / "tools" / "trunkspan_min.py", [])
    print("5/8 reclass_split.py  (arms/trunk apart, crop)")
    run(ROOT / "tools" / "reclass_split.py", ["--names", name])
    print("    pose_rtmpose_person.py  (RTMPose-m on the original picture)")
    try:
        run(ROOT / "tools" / "pose_rtmpose_person.py", ["--names", name])
    except SystemExit as exc:
        # keep the MediaPipe pose reclass_split wrote when the model is absent
        print(f"  (RTMPose pose unavailable: {exc}) - using reclass_split pose")
    if args.waist_source == "greensplit":
        print("    waist_shoulder_green.py  (shoulder-midpoint green split)")
        run(ROOT / "tools" / "waist_shoulder_green.py",
            ["--names", name])
    # the new shoulder line and belly line have to exist before the score
    print("    poseline_new.py  (inward shoulder line + green belly)")
    run(ROOT / "tools" / "poseline_new.py", ["--names", name])
    print("6/8 available-arm detection (annotate usable arms only)")
    run(
        ROOT / "pipeline" / "07_detect_usable_arms.py",
        ["--names", name],
        quiet=False,
    )
    print("7/8 LIP-cut abdomen rectangle, centres and objective")
    run(
        ROOT / "pipeline" / "09_analyze_abdomen.py",
        ["--names", name],
        quiet=False,
    )
    run(
        ROOT / "pipeline" / "11_export_total_score.py",
        [
            "--name", name,
            "--label", name,
            "--out", str(ROOT / "outfinal"),
        ],
        quiet=True,
    )
    print("8/8 final score and output tables")
    scores = json.loads(
        (ROOT / "outfinal" / "score.json").read_text(encoding="utf-8")
    )
    row = next(
        (r for r in scores["rows"] if r["image"] == name), None
    )
    print(f"\n=== {name} ({img.name})")
    if row:
        print(
            f"  平均开放角 {float(row['mean_open_angle_deg']):.3f}°"
        )
        print(
            f"  腹肌分数 {float(row['abdomen_score']):.2f}"
        )
    total_path = ROOT / "outfinal" / f"薄肌分数_{name}.json"
    if total_path.exists():
        total_row = json.loads(total_path.read_text(encoding="utf-8"))
        print(f"  总分 {float(total_row['total']):.2f}")
    print(
        "\n图片与表格: outfinal/score.csv、score.json、score.png、"
        f"available_arm_{name}.png、abdomen_openangle_rule_{name}.png；"
        "逐中心结果在 outfinal/lipcut_roi_annotation/center_objective/"
    )


if __name__ == "__main__":
    main()
