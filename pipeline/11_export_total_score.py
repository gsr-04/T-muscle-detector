"""Build the legacy-format total score table with the new abdominal score."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle


ROOT = Path(__file__).resolve().parents[1]
plt.rcParams["font.sans-serif"] = [
    "Microsoft YaHei",
    "SimHei",
    "DejaVu Sans",
]
plt.rcParams["axes.unicode_minus"] = False


def clipped(value):
    return float(np.clip(value, 0.0, 100.0))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True)
    parser.add_argument("--label", default="")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--abdomen-json", type=Path, default=None)
    args = parser.parse_args()
    name = args.name
    label = args.label or name
    args.out.mkdir(parents=True, exist_ok=True)

    abdomen_path = args.abdomen_json or (
        ROOT / "out1" / "abdomen_openangle_score.json"
    )
    abdomen_payload = json.loads(
        abdomen_path.read_text(encoding="utf-8")
    )
    abdomen_row = (
        next(
            row for row in abdomen_payload["rows"]
            if row["image"] == name
        )
        if "rows" in abdomen_payload
        else abdomen_payload
    )
    legacy_path = ROOT / "out3" / "score_reclass.json"
    legacy = (
        json.loads(legacy_path.read_text(encoding="utf-8"))
        if legacy_path.exists() else {}
    )
    legacy_row = next(
        (
            row for row in legacy.get("rows", [])
            if row["image"] == name
        ),
        {},
    )
    weights = [float(value) for value in legacy.get(
        "weights", [0.55, 0.30, 0.15]
    )]
    waist_shift, waist_span = 1.3, 0.8
    arm_shift, arm_span = [
        float(value) for value in legacy.get("arm", [2.2, 1.0])
    ]

    abdomen_score = clipped(float(abdomen_row["abdomen_score"]))
    waist_raw = float(
        legacy_row.get("shoulder_over_waist") or 0.0
    )
    shoulder_path = ROOT / "out3" / f"shoulderline_{name}.json"
    waist_path = ROOT / "out3" / f"waist_{name}.json"
    greensplit_path = ROOT / "out3" / f"greensplit_{name}.json"
    if shoulder_path.exists() and (
        waist_path.exists() or greensplit_path.exists()
    ):
        shoulder_doc = json.loads(
            shoulder_path.read_text(encoding="utf-8")
        )
        if waist_path.exists():
            waist_doc = json.loads(
                waist_path.read_text(encoding="utf-8")
            )
            waist_length = float(
                waist_doc.get("torso_width_at_waist") or 0.0
            )
        else:
            green_doc = json.loads(
                greensplit_path.read_text(encoding="utf-8")
            )
            waist_length = float(green_doc.get("width") or 0.0)
        shoulder_length = float(
            shoulder_doc.get("three_segments") or 0.0
        )
        if waist_length > 0 and shoulder_length > 0:
            waist_raw = shoulder_length / waist_length
    arm_raw = float(legacy_row.get("arm_ratio") or 0.0)
    arm_measure_path = ROOT / "out3" / f"armmeasure_{name}.json"
    if arm_measure_path.exists():
        arm_measure = json.loads(
            arm_measure_path.read_text(encoding="utf-8")
        )
        if arm_measure.get("effective_ratio") is not None:
            arm_raw = float(arm_measure["effective_ratio"])
    waist_score = clipped(
        (waist_raw - waist_shift) * 100.0 / waist_span
    )
    arm_score = clipped(
        (arm_shift - arm_raw) * 100.0 / arm_span
    )
    scores = [abdomen_score, waist_score, arm_score]
    weighted = [
        score * weight for score, weight in zip(scores, weights)
    ]
    total = sum(weighted)

    raw_values = [
        f"{float(abdomen_row['mean_open_angle_deg']):.4f}",
        f"{waist_raw:.4f}",
        f"{arm_raw:.4f}",
        "",
    ]
    score_values = [
        f"{abdomen_score:.2f}",
        f"{waist_score:.2f}",
        f"{arm_score:.2f}",
        "",
    ]
    weighted_values = [
        f"{weighted[0]:.2f}",
        f"{weighted[1]:.2f}",
        f"{weighted[2]:.2f}",
        f"{total:.2f}",
    ]
    table = [
        ["指标", "腹肌分数", "相对肩宽分", "手臂维度分", "总分"],
        ["原始值", *raw_values],
        ["得分(0-100)", *score_values],
        ["加权分", *weighted_values],
    ]
    for target in (
        args.out / f"{label}_total_score.csv",
        ROOT / "outfinal" / f"薄肌分数_{name}.csv",
    ):
        with target.open("w", newline="", encoding="utf-8-sig") as handle:
            csv.writer(handle).writerows(table)

    payload = {
        "image": name,
        "source_label": label,
        "weights": weights,
        "abdomen": {
            "raw": float(abdomen_row["mean_open_angle_deg"]),
            "measure": "open-angle mean",
            "score": abdomen_score,
        },
        "waist": {
            "raw": waist_raw,
            "measure": "shoulder / waist",
            "score": waist_score,
        },
        "arm": {
            "raw": arm_raw,
            "measure": "kept line / four perpendiculars",
            "score": arm_score,
        },
        "weighted": {
            "abdomen": weighted[0],
            "waist": weighted[1],
            "arm": weighted[2],
        },
        "total": total,
    }
    for target in (
        args.out / f"{label}_total_score.json",
        ROOT / "outfinal" / f"薄肌分数_{name}.json",
    ):
        target.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    figure, axis = plt.subplots(figsize=(7.2, 9.6))
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.axis("off")
    colours = ("#4c72b0", "#dd8452", "#55a868", "#8172b2")
    labels = ("腹肌分数", "相对肩宽分", "手臂维度分", "总分")
    values = [*scores, total]
    axis.text(
        0.5,
        0.76,
        f"薄肌分数  {total:.1f}",
        ha="center",
        va="center",
        fontsize=26,
        fontweight="bold",
    )
    ys = (0.62, 0.52, 0.42, 0.30)
    x0, x1 = 0.40, 0.78
    for label_text, value, colour, y in zip(
        labels, values, colours, ys
    ):
        is_total = label_text == "总分"
        text_colour = "red" if is_total else "black"
        axis.text(
            0.03,
            y,
            f"{label_text}：",
            ha="left",
            va="center",
            fontsize=23,
            color=text_colour,
            fontweight="bold" if is_total else "normal",
        )
        axis.add_patch(
            Rectangle(
                (x0, y - 0.035),
                x1 - x0,
                0.07,
                facecolor="white",
                edgecolor="0.35",
                linewidth=1.6,
            )
        )
        fill = (x1 - x0) * max(0.0, min(100.0, value)) / 100.0
        axis.add_patch(
            Rectangle(
                (x0, y - 0.035),
                fill,
                0.07,
                facecolor="red" if is_total else colour,
                edgecolor="none",
            )
        )
        axis.text(
            0.795,
            y,
            f"{value:.1f}",
            ha="left",
            va="center",
            fontsize=19,
            fontweight="bold",
            color=text_colour,
        )
    for target in (
        args.out / f"{label}_total_score.png",
        ROOT / "outfinal" / f"薄肌分数_{name}.png",
    ):
        figure.savefig(target, dpi=130)
    plt.close(figure)

    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
