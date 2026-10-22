"""Collect one open-angle abdomen test into a self-contained output folder."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
LIPCUT = ROOT / "outfinal" / "lipcut_roi_annotation"
CENTRE = LIPCUT / "center_objective"


def copy_file(source: Path, target: Path):
    if not source.exists():
        raise FileNotFoundError(source)
    shutil.copy2(source, target)


def filter_csv(source: Path, target: Path, image: str):
    with source.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames
        rows = [row for row in reader if row["image"] == image]
    with target.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True)
    parser.add_argument("--label", default="")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    name = args.name
    label = args.label or name
    args.out.mkdir(parents=True, exist_ok=True)

    rois = {
        row["image"]: row
        for row in json.loads(
            (LIPCUT / "roi_annotation.json").read_text(encoding="utf-8")
        )["records"]
    }
    roi = rois[name]
    left, top, right, bottom = roi["rectangle_px"]

    annotation = Image.open(
        LIPCUT / f"annotation_{name}.png"
    ).convert("RGB")
    annotation.crop((left, top, right, bottom)).save(
        args.out / f"{label}_abdominal_rectangle_annotation.png"
    )
    Image.open(
        LIPCUT / "rowcol_intersections"
        / f"body_rectangle_{name}.png"
    ).save(
        args.out / f"{label}_abdominal_rectangle.png"
    )
    copy_file(
        LIPCUT / f"roi_{name}.png",
        args.out / f"{label}_abdomen_rectangle_full.png",
    )
    copy_file(
        ROOT / "outfinal" / f"armmid_{name}.png",
        args.out / f"{label}_armmid.png",
    )
    copy_file(
        ROOT / "outfinal" / f"available_arm_{name}.png",
        args.out / f"{label}_available_arm.png",
    )
    arm_measure = ROOT / "out3" / f"armmeasure_{name}.png"
    if arm_measure.exists():
        copy_file(
            arm_measure,
            args.out / f"{label}_arm_measure.png",
        )
    copy_file(
        LIPCUT / "rowcol_intersections"
        / f"roi_intersections_{name}.png",
        args.out / f"{label}_centres.png",
    )
    copy_file(
        CENTRE / f"center_contact_{name}.png",
        args.out / f"{label}_six_centres_retained.png",
    )
    rule_figure = ROOT / "outfinal" / f"abdomen_openangle_rule_{name}.png"
    if rule_figure.exists():
        copy_file(
            rule_figure,
            args.out / f"{label}_openangle_rule.png",
        )
    for suffix in ("C01", "C02", "C03", "C04", "C05", "C06"):
        copy_file(
            CENTRE / "original_overlays"
            / f"original_{name}_{suffix}.png",
            args.out / f"{label}_{suffix}_original.png",
        )
        copy_file(
            CENTRE / f"center_{name}_{suffix}.png",
            args.out / f"{label}_{suffix}_rectangle.png",
        )

    filter_csv(
        CENTRE / "selected_points.csv",
        args.out / f"{label}_selected_points.csv",
        name,
    )
    filter_csv(
        CENTRE / "center_objective.csv",
        args.out / f"{label}_center_objective.csv",
        name,
    )
    filter_csv(
        CENTRE / "nearest_center_removed_points.csv",
        args.out / f"{label}_nearest_removed_points.csv",
        name,
    )

    score_payload = json.loads(
        (ROOT / "out1" / "abdomen_openangle_score.json").read_text(
            encoding="utf-8"
        )
    )
    score_row = next(
        row for row in score_payload["rows"] if row["image"] == name
    )
    with (args.out / f"{label}_score.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "image",
                "mean_open_angle_deg",
                "abdomen_score",
                "selected_center_count",
                "max_removed",
                "removed_max_centre",
            ]
        )
        writer.writerow(
            [
                name,
                score_row["mean_open_angle_deg"],
                score_row["abdomen_score"],
                score_row["selected_center_count"],
                str(score_row["max_removed"]).lower(),
                score_row["removed_max_centre"],
            ]
        )
    (args.out / f"{label}_score.json").write_text(
        json.dumps(
            {
                "image": name,
                "source_label": label,
                "formula": score_payload["formula"],
                **score_row,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    figure, axis = plt.subplots(figsize=(6.0, 2.5))
    axis.barh(
        [f"{label} ({name})"],
        [float(score_row["abdomen_score"])],
        color="#4c72b0",
    )
    axis.set_xlim(0, 105)
    axis.set_xlabel("abdomen score")
    axis.set_title(
        f"mean open angle={float(score_row['mean_open_angle_deg']):.3f} deg"
    )
    axis.grid(axis="x", alpha=0.25)
    figure.tight_layout()
    figure.savefig(args.out / f"{label}_score.png", dpi=170)
    plt.close(figure)

    (args.out / "README.md").write_text(
        "\n".join(
            [
                f"# {label} ({name})",
                "",
                f"- 平均开放角：{float(score_row['mean_open_angle_deg']):.6f}°",
                f"- 腹肌分数：{float(score_row['abdomen_score']):.6f}",
                f"- 计分公式：`{score_payload['formula']}`",
                "",
                "文件：armmid、腹部矩形标注、中心交点、六中心保留点、"
                "逐中心图、目标函数和分数表。",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
