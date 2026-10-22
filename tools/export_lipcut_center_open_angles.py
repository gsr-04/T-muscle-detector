"""Export final open angles and per-image means for LIP-cut centre points."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = (
    ROOT
    / "outfinal"
    / "lipcut_roi_annotation"
    / "center_objective"
    / "center_objective.csv"
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument(
        "--names",
        default="",
        help="comma-separated image names; empty uses every image in source",
    )
    args = parser.parse_args()
    out = args.out or args.source.parent
    out.mkdir(parents=True, exist_ok=True)

    with args.source.open(encoding="utf-8-sig", newline="") as handle:
        source_rows = list(csv.DictReader(handle))
    available_names = sorted({row["image"] for row in source_rows})
    requested = [value for value in args.names.split(",") if value]
    names = requested or available_names
    unknown = set(names) - set(available_names)
    if unknown:
        raise SystemExit(f"unknown image names: {sorted(unknown)}")

    by_image = {
        name: [
            row for row in source_rows if row["image"] == name
        ]
        for name in names
    }
    if any(len(rows) != 6 for rows in by_image.values()):
        raise SystemExit("every image must have exactly six centre points")

    selected_rows = [
        row for name in names for row in by_image[name]
    ]
    maximum_by_image = {}
    second_by_image = {}
    bottom_excluded_by_image = {}
    bottom_clip_by_image = {}
    used_by_image = {}
    removed_by_image = {}
    image_means = {}
    for name in names:
        bottom_row = max(int(row["row"]) for row in by_image[name])
        bottom_rows = [
            row for row in by_image[name]
            if int(row["row"]) == bottom_row
        ]
        bottom_clip = max(
            float(row.get("bottom_clip_fraction", 0.0) or 0.0)
            for row in bottom_rows
        )
        bottom_excluded = bottom_clip > (1.0 / 3.0)
        eligible_rows = (
            [
                row for row in by_image[name]
                if int(row["row"]) != bottom_row
            ]
            if bottom_excluded
            else list(by_image[name])
        )
        if not eligible_rows:
            raise SystemExit(f"{name}: no centres left after bottom clip")
        bottom_excluded_by_image[name] = bottom_excluded
        bottom_clip_by_image[name] = bottom_clip
        maximum_row = max(
            eligible_rows,
            key=lambda row: (
                float(row["open_angle"]),
                int(row["centre_index"]),
            ),
        )
        remaining_rows = [
            row for row in eligible_rows if row is not maximum_row
        ]
        second_row = max(
            remaining_rows,
            key=lambda row: (
                float(row["open_angle"]),
                int(row["centre_index"]),
            ),
        )
        maximum_by_image[name] = maximum_row
        second_by_image[name] = second_row
        remove = (
            float(maximum_row["open_angle"])
            > 2.0 * float(second_row["open_angle"])
        )
        removed_by_image[name] = remove
        used_by_image[name] = (
            remaining_rows if remove else eligible_rows
        )
        image_means[name] = statistics.fmean(
            float(row["open_angle"])
            for row in used_by_image[name]
        )

    with (out / "center_open_angles.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "image",
                "centre_index",
                "row",
                "column",
                "open_angle_deg",
                "objective",
                "minimum_retained_points",
                "selected_points",
            ]
        )
        for row in selected_rows:
            writer.writerow(
                [
                    row["image"],
                    row["centre_index"],
                    row["row"],
                    row["column"],
                    f"{float(row['open_angle']):.6f}",
                    row["objective"],
                    row["minimum_retained_points"],
                    row["selected_points"],
                ]
            )

    average_rows = []
    for name in names:
        used_angles = [
            float(row["open_angle"])
            for row in used_by_image[name]
        ]
        average_rows.append(
            {
                "image": name,
                "selected_center_count": len(used_angles),
                "max_exceeds_twice_second": removed_by_image[name],
                "bottom_row_clipped_over_one_third": (
                    bottom_excluded_by_image[name]
                ),
                "bottom_row_clip_fraction": bottom_clip_by_image[name],
                "excluded_centres": (
                    [
                        int(row["centre_index"])
                        for row in bottom_rows
                    ]
                    if bottom_excluded_by_image[name]
                    else []
                ),
                "removed_max_centre": (
                    int(maximum_by_image[name]["centre_index"])
                    if removed_by_image[name] else None
                ),
                "removed_max_open_angle_deg": float(
                    maximum_by_image[name]["open_angle"]
                ),
                "second_max_open_angle_deg": float(
                    second_by_image[name]["open_angle"]
                ),
                "mean_open_angle_deg": image_means[name],
                "min_remaining_open_angle_deg": min(used_angles),
                "max_remaining_open_angle_deg": max(used_angles),
            }
        )
    with (out / "center_open_angle_averages.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as handle:
        fieldnames = [
            "image",
            "selected_center_count",
            "max_exceeds_twice_second",
            "bottom_row_clipped_over_one_third",
            "bottom_row_clip_fraction",
            "excluded_centres",
            "removed_max_centre",
            "removed_max_open_angle_deg",
            "second_max_open_angle_deg",
            "mean_open_angle_deg",
            "min_remaining_open_angle_deg",
            "max_remaining_open_angle_deg",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in average_rows:
            writer.writerow(
                {
                    **row,
                    "max_exceeds_twice_second": str(
                        row["max_exceeds_twice_second"]
                    ).lower(),
                    "bottom_row_clipped_over_one_third": str(
                        row["bottom_row_clipped_over_one_third"]
                    ).lower(),
                    "bottom_row_clip_fraction": (
                        f"{row['bottom_row_clip_fraction']:.6f}"
                    ),
                    "excluded_centres": " ".join(
                        f"C{value}" for value in row["excluded_centres"]
                    ),
                    "removed_max_centre": (
                        f"C{row['removed_max_centre']}"
                        if row["removed_max_centre"] is not None
                        else ""
                    ),
                    **{
                        key: f"{float(row[key]):.6f}"
                        for key in (
                            "removed_max_open_angle_deg",
                            "second_max_open_angle_deg",
                            "mean_open_angle_deg",
                            "min_remaining_open_angle_deg",
                            "max_remaining_open_angle_deg",
                        )
                    },
                }
            )

    (out / "center_open_angle_averages.json").write_text(
        json.dumps(
            {
                "rule": (
                    "exclude the bottom centre row when its rectangle is "
                    "clipped by more than 1/3; then remove one maximum only "
                    "when maximum > 2 * second maximum"
                ),
                "images": average_rows,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    lines = [
        "# 每个中心点的开放角",
        "",
        "- 距离惩罚：1.3（当前结果）",
        "- 开放角单位为度，范围为 [0, 360].",
        "- 表中数值来自当前最终保留点集合。",
        "- 最下方中心行矩形被下边界裁掉超过 1/3 时，排除该行两个中心点。",
        "- 对保留下来的中心，仅当最大值严格大于第二大的两倍时剔除最大值。",
        "- 不计算跨图片的全局平均。",
        "",
        "| 图片 | C1 | C2 | C3 | C4 | C5 | C6 | 排除底行 | 是否去最大 | 参与中心数 | 平均 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | ---: | ---: |",
    ]
    for name in names:
        display_angles = [
            f"{float(row['open_angle']):.3f}"
            for row in by_image[name]
        ]
        lines.append(
            f"| {name} | "
            + " | ".join(display_angles)
            + f" | {'是' if bottom_excluded_by_image[name] else '否'} "
            + f" | {'是' if removed_by_image[name] else '否'} "
            + f"| {len(used_by_image[name])} "
            + f"| {image_means[name]:.3f} |"
        )
    (out / "center_open_angles.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print(
        f"exported {len(selected_rows)} open angles for {len(names)} images"
    )


if __name__ == "__main__":
    main()
