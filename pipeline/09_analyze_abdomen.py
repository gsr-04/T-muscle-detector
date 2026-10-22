"""Current abdominal score chain from contour to the final open-angle score.

The abdominal score is:

    score = clamp((240 - mean_open_angle) * 100 / 180, 0, 100)

The mean is computed independently for every image. A maximum open angle is
removed only when it is strictly greater than twice the second maximum; this
rule is implemented by tools/export_lipcut_center_open_angles.py.

Examples:

    python pipeline/09_analyze_abdomen.py --score-only
    python pipeline/09_analyze_abdomen.py --names 001,002 --prepare
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
plt.rcParams["font.sans-serif"] = [
    "Microsoft YaHei",
    "SimHei",
    "DejaVu Sans",
]
plt.rcParams["axes.unicode_minus"] = False


ROOT = Path(__file__).resolve().parents[1]
PY = Path(os.environ.get("TMUSCLE_PYTHON", sys.executable))
DEFAULT_NAMES = (
    "001,002,003,004,006,010,011,013,015,029,035,037,038"
)
DEFAULT_AVERAGES = (
    ROOT / "outfinal" / "lipcut_roi_annotation"
    / "center_objective" / "center_open_angle_averages.csv"
)


def run(script: Path, extra: list[str], quiet=False):
    command = [str(PY), str(script)] + extra
    print("  $ " + " ".join(command[1:]), flush=True)
    result = subprocess.run(
        command,
        cwd=str(ROOT),
        capture_output=True,
        text=True,
    )
    output = (result.stdout or "") + (result.stderr or "")
    if not quiet:
        tail = [
            line for line in output.strip().splitlines() if line.strip()
        ][-12:]
        print("\n".join(tail), flush=True)
    if result.returncode != 0:
        raise SystemExit(
            f"{script.name} failed ({result.returncode})"
        )


def prepare_upstream(names: str):
    run(ROOT / "tools" / "annotate_edges.py", ["--names", names])
    run(
        ROOT / "tools" / "pose_rtmpose_person.py",
        ["--names", names],
    )
    run(
        ROOT / "tools" / "torso_muscle_annot.py",
        ["--reuse", "--names", names],
    )
    run(ROOT / "tools" / "trunk_span.py", [])
    run(ROOT / "tools" / "trunkspan_min.py", [])
    run(ROOT / "tools" / "reclass_split.py", ["--names", names])
    run(
        ROOT / "tools" / "pose_rtmpose_person.py",
        ["--names", names],
    )
    run(
        ROOT / "tools" / "waist_shoulder_green.py",
        ["--names", names],
    )
    run(
        ROOT / "tools" / "poseline_new.py",
        ["--names", names],
    )


def run_abdomen_chain(names: str):
    run(
        ROOT / "tools" / "run_lipcut_roi_annotation_batch.py",
        ["--names", names],
    )
    run(
        ROOT / "tools" / "export_lipcut_annotation_functions.py",
        ["--names", names, "--sigma", "2"],
    )
    run(
        ROOT / "tools" / "lipcut_rowcol_intersections.py",
        ["--names", names],
    )
    run(
        ROOT / "tools" / "run_lipcut_center_objective.py",
        [
            "--names", names,
            "--distance-weight", "1.3",
            "--min-candidates-for-removal", "30",
            "--max-initial-selected-for-removal", "3",
            "--near-centre-fraction", "0.3",
        ],
    )
    run(
        ROOT / "tools" / "render_lipcut_center_retained_original.py",
        ["--names", names],
    )
    out1 = ROOT / "out1"
    out1.mkdir(parents=True, exist_ok=True)
    run(
        ROOT / "tools" / "export_lipcut_center_open_angles.py",
        ["--names", names, "--out", str(out1)],
    )


def load_scores(path: Path, names_filter: set[str]):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        source_rows = list(csv.DictReader(handle))
    rows = []
    for row in source_rows:
        name = row["image"]
        if names_filter and name not in names_filter:
            continue
        mean_angle = float(row["mean_open_angle_deg"])
        raw = (240.0 - mean_angle) * 100.0 / 180.0
        score = float(np.clip(raw, 0.0, 100.0))
        rows.append(
            {
                "image": name,
                "mean_open_angle_deg": round(mean_angle, 6),
                "raw_score": round(raw, 6),
                "abdomen_score": round(score, 6),
                "capped": raw > 100.0,
                "floored": raw < 0.0,
                "selected_center_count": int(row["selected_center_count"]),
                "bottom_row_clipped_over_one_third": (
                    row.get(
                        "bottom_row_clipped_over_one_third", "false"
                    ).lower() == "true"
                ),
                "bottom_row_clip_fraction": float(
                    row.get("bottom_row_clip_fraction", 0.0) or 0.0
                ),
                "excluded_centres": row.get(
                    "excluded_centres", ""
                ),
                "max_removed": (
                    row["max_exceeds_twice_second"].lower() == "true"
                ),
                "removed_max_centre": row["removed_max_centre"],
            }
        )
    if not rows:
        raise SystemExit("no images left to score")
    return rows


def preserve_legacy():
    out = ROOT / "outfinal"
    for suffix in ("csv", "json", "png"):
        current = out / f"score.{suffix}"
        backup = out / f"score_legacy_covariance.{suffix}"
        if current.exists() and not backup.exists():
            shutil.copy2(current, backup)


def write_rule_figures(rows, source_path: Path):
    """Draw which centre points entered each image's open-angle mean."""
    angles_path = source_path.parent / "center_open_angles.csv"
    if not angles_path.exists():
        return
    with angles_path.open(encoding="utf-8-sig", newline="") as handle:
        angle_rows = list(csv.DictReader(handle))
    by_image = {}
    for row in angle_rows:
        by_image.setdefault(row["image"], []).append(row)

    outputs = []
    for score_row in rows:
        name = score_row["image"]
        centres = sorted(
            by_image.get(name, []),
            key=lambda row: int(row["centre_index"]),
        )
        if not centres:
            continue
        excluded = {
            int(value.strip().lstrip("C"))
            for value in score_row["excluded_centres"].split()
            if value.strip()
        }
        removed_max = (
            int(score_row["removed_max_centre"].lstrip("C"))
            if score_row["removed_max_centre"]
            else None
        )
        indices = [int(row["centre_index"]) for row in centres]
        angles = [float(row["open_angle_deg"]) for row in centres]
        colours = []
        for index in indices:
            if index in excluded:
                colours.append("#9e9e9e")
            elif index == removed_max:
                colours.append("#d17c00")
            else:
                colours.append("#4c72b0")
        figure, axis = plt.subplots(figsize=(6.6, 3.5))
        axis.bar(
            [f"C{index}" for index in indices],
            angles,
            color=colours,
        )
        for index, (label, value) in enumerate(zip(indices, angles)):
            if label in excluded:
                axis.text(
                    index,
                    value,
                    "底边越界",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                    color="#555555",
                )
            elif label == removed_max:
                axis.text(
                    index,
                    value,
                    "去最大值",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                    color="#8a4b00",
                )
        axis.axhline(
            float(score_row["mean_open_angle_deg"]),
            color="#c0392b",
            linestyle="--",
            linewidth=1.5,
            label=f"参与平均 {float(score_row['mean_open_angle_deg']):.3f}°",
        )
        axis.set_ylim(0, max(360.0, max(angles) * 1.08))
        axis.set_ylabel("开放角（度）")
        axis.set_title(
            f"{name} 腹肌分数 {float(score_row['abdomen_score']):.2f}"
        )
        axis.grid(axis="y", color="0.9")
        axis.set_axisbelow(True)
        axis.legend(fontsize=8)
        figure.tight_layout()
        target = ROOT / "outfinal" / f"abdomen_openangle_rule_{name}.png"
        figure.savefig(target, dpi=170)
        plt.close(figure)
        outputs.append(target)
    return outputs


def write_results(rows, source_path: Path):
    formula = "clamp((240 - mean_open_angle) * 100 / 180, 0, 100)"
    payload = {
        "score_kind": "abdomen_open_angle",
        "formula": formula,
        "source": str(source_path),
        "max_rule": (
            "per image: remove maximum only when maximum > "
            "2 * second maximum"
        ),
        "rows": rows,
    }
    for directory in (
        ROOT / "out1",
        ROOT / "outfinal",
        ROOT / "outfinal" / "lipcut_roi_annotation" / "center_objective",
    ):
        directory.mkdir(parents=True, exist_ok=True)
        with (directory / "abdomen_openangle_score.csv").open(
            "w", newline="", encoding="utf-8-sig"
        ) as handle:
            writer = csv.writer(handle)
            writer.writerow(
                [
                    "image",
                    "mean_open_angle_deg",
                    "raw_score",
                    "abdomen_score",
                    "capped",
                    "floored",
                    "selected_center_count",
                    "bottom_row_clipped_over_one_third",
                    "bottom_row_clip_fraction",
                    "excluded_centres",
                    "max_removed",
                    "removed_max_centre",
                ]
            )
            for row in rows:
                writer.writerow(
                    [
                        row["image"],
                        f"{row['mean_open_angle_deg']:.6f}",
                        f"{row['raw_score']:.6f}",
                        f"{row['abdomen_score']:.6f}",
                        str(row["capped"]).lower(),
                        str(row["floored"]).lower(),
                        row["selected_center_count"],
                        str(
                            row["bottom_row_clipped_over_one_third"]
                        ).lower(),
                        f"{row['bottom_row_clip_fraction']:.6f}",
                        row["excluded_centres"],
                        str(row["max_removed"]).lower(),
                        row["removed_max_centre"],
                    ]
                )
        (directory / "abdomen_openangle_score.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    write_rule_figures(rows, source_path)
    preserve_legacy()
    out = ROOT / "outfinal"
    with (out / "score.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["image", "mean_open_angle_deg", "abdomen_score"]
        )
        for row in rows:
            writer.writerow(
                [
                    row["image"],
                    f"{row['mean_open_angle_deg']:.6f}",
                    f"{row['abdomen_score']:.6f}",
                ]
            )
    (out / "score.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    labels = [row["image"] for row in rows]
    values = [row["abdomen_score"] for row in rows]
    figure, axis = plt.subplots(
        figsize=(9.2, max(4.2, 0.42 * len(rows) + 1.8))
    )
    axis.barh(labels, values, color="#4c72b0")
    for index, value in enumerate(values):
        axis.text(
            min(98.0, value + 1.0),
            index,
            f"{value:.1f}",
            va="center",
            fontsize=9,
        )
    axis.set_xlim(0, 105)
    axis.set_xlabel("腹肌分数")
    axis.set_title("腹肌分数 = (240 - 开放角平均值) × 100 / 180")
    axis.grid(axis="x", color="0.9")
    axis.set_axisbelow(True)
    axis.invert_yaxis()
    figure.tight_layout()
    for directory in (out, out / "lipcut_roi_annotation" / "center_objective"):
        figure.savefig(
            directory / "abdomen_openangle_score.png", dpi=160
        )
    figure.savefig(out / "score.png", dpi=160)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--names", default=DEFAULT_NAMES)
    parser.add_argument(
        "--prepare",
        action="store_true",
        help="run body contour, pose, LIP-20, waist and shoulder preparation",
    )
    parser.add_argument(
        "--score-only",
        action="store_true",
        help="only recalculate the score from existing open-angle averages",
    )
    parser.add_argument(
        "--averages",
        type=Path,
        default=None,
        help="open-angle averages CSV",
    )
    args = parser.parse_args()
    names = [value for value in args.names.split(",") if value]
    name_text = ",".join(names)

    if args.prepare:
        print("1/3 contour, pose, LIP-20, waist and shoulder preparation")
        prepare_upstream(name_text)
    if not args.score_only:
        print("2/3 abdomen rectangle, annotations, centres and objective")
        run_abdomen_chain(name_text)
    else:
        print("1/1 score from existing open-angle averages")

    averages = args.averages
    if averages is None:
        out1_averages = ROOT / "out1" / "center_open_angle_averages.csv"
        averages = (
            out1_averages if out1_averages.exists() else DEFAULT_AVERAGES
        )
    out1 = ROOT / "out1"
    out1.mkdir(parents=True, exist_ok=True)
    target_averages = out1 / "center_open_angle_averages.csv"
    if averages.resolve() != target_averages.resolve():
        shutil.copy2(averages, target_averages)
    for filename in (
        "center_open_angles.csv",
        "center_open_angles.md",
        "center_open_angle_averages.json",
    ):
        source_file = averages.parent / filename
        target_file = out1 / filename
        if source_file.exists() and (
            not target_file.exists()
            or source_file.stat().st_mtime > target_file.stat().st_mtime
        ):
            shutil.copy2(source_file, target_file)
    names_filter = set(names) if args.names != DEFAULT_NAMES else set()
    rows = load_scores(averages, names_filter)
    write_results(rows, averages)
    print(f"\n{'image':>5s} {'mean_angle':>12s} {'score':>8s}")
    for row in rows:
        print(
            f"{row['image']:>5s} "
            f"{row['mean_open_angle_deg']:12.4f} "
            f"{row['abdomen_score']:8.3f}"
        )
    print("wrote out1/ and outfinal/ abdomen_openangle_score.*")


if __name__ == "__main__":
    main()
