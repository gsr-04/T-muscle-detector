"""Row/column annotation-count functions for the LIP-cut pooled images."""

from __future__ import annotations

import argparse
import csv
import json
from fractions import Fraction
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.ndimage import gaussian_filter1d


ROOT = Path(__file__).resolve().parents[1]


def font():
    path = Path(r"C:\Windows\Fonts\msyh.ttc")
    return font_manager.FontProperties(fname=str(path)) if path.exists() else None


def count_text(value):
    return f"{float(value):.4f}".rstrip("0").rstrip(".")


def positive_runs(values):
    runs = []
    start = None
    for index, value in enumerate(list(values) + [0]):
        if value > 0 and start is None:
            start = index
        elif value == 0 and start is not None:
            runs.append((start, index - 1))
            start = None
    return runs


def merge_value_ones(counts):
    """Add each value 1 to the next right-side count greater than 1."""
    original = [int(value) for value in counts]
    merged = [float(value) for value in original]
    records = []

    for row, value in enumerate(original):
        if value != 1:
            continue

        target_row = next(
            (
                candidate
                for candidate in range(row + 1, len(original))
                if original[candidate] > 1
            ),
            None,
        )
        if target_row is None:
            records.append(
                {
                    "phase": "value1",
                    "row": row,
                    "value": "1",
                    "target": "none",
                    "target_start": None,
                    "target_end": None,
                    "increment": "0",
                    "skipped_isolated_rows": [],
                    "merged": False,
                    "reason": "no_right_value_greater_than_1",
                }
            )
            continue

        merged[target_row] += 1.0
        merged[row] = 0.0
        records.append(
            {
                "phase": "value1",
                "row": row,
                "value": "1",
                "target": "right",
                "target_start": target_row,
                "target_end": target_row,
                "increment": "1",
                "skipped_isolated_rows": [],
                "merged": True,
                "reason": "added_to_first_right_value_greater_than_1",
            }
        )

    return merged, records


def merge_isolated_rows(counts):
    """Merge isolated values into the first right-side continuous run."""
    runs = positive_runs(counts)
    isolated_rows = [start for start, end in runs if start == end]
    continuous_runs = [
        (start, end) for start, end in runs if end > start
    ]
    merged = [float(value) for value in counts]
    records = []

    for row in isolated_rows:
        value = merged[row]
        skipped_isolated = []
        target = next(
            (
                run
                for run in continuous_runs
                if run[0] > row
            ),
            None,
        )
        for start, end in runs:
            if start <= row:
                continue
            if end > start:
                break
            skipped_isolated.append(start)

        if target is None:
            records.append(
                {
                    "phase": "isolated",
                    "row": row,
                    "value": count_text(value),
                    "target": "none",
                    "target_start": None,
                    "target_end": None,
                    "increment": "0",
                    "skipped_isolated_rows": skipped_isolated,
                    "merged": False,
                    "reason": "no_right_continuous_run",
                }
            )
            continue

        increment = Fraction(int(round(value)), target[1] - target[0] + 1)
        increment_float = float(increment)
        for target_row in range(target[0], target[1] + 1):
            merged[target_row] += increment_float
        merged[row] = 0.0
        records.append(
            {
                "phase": "isolated",
                "row": row,
                "value": count_text(value),
                "target": "right",
                "target_start": target[0],
                "target_end": target[1],
                "increment": count_text(increment_float),
                "skipped_isolated_rows": skipped_isolated,
                "merged": True,
                "reason": "split_into_first_right_continuous_run",
            }
        )

    return merged, records


def gaussian_filter(values, sigma):
    return gaussian_filter1d(
        np.asarray(values, dtype=float), sigma=sigma, mode="reflect"
    )


def anchor_last_group(values, filtered, sigma, fraction):
    """Keep a sufficiently large final run represented by the filtered curve."""
    values = np.asarray(values, dtype=float)
    filtered = np.asarray(filtered, dtype=float).copy()
    runs = positive_runs(values)
    info = {
        "required_fraction": fraction,
        "last_group": None,
        "last_group_peak": None,
        "overall_peak": float(values.max()) if values.size else 0.0,
        "anchor_row": None,
        "anchor_value": None,
        "gaussian_before_anchor": None,
        "applied": False,
    }
    if not runs or values.max() <= 0:
        return filtered, info
    start, end = runs[-1]
    group = values[start:end + 1]
    group_peak = float(group.max())
    peak_row = start + int(np.argmax(group))
    info["last_group"] = [start, end]
    info["last_group_peak"] = group_peak
    info["anchor_row"] = peak_row
    info["anchor_value"] = group_peak
    info["gaussian_before_anchor"] = float(filtered[peak_row])
    if group_peak < fraction * info["overall_peak"]:
        return filtered, info
    delta = group_peak - float(filtered[peak_row])
    if delta <= 0:
        info["applied"] = True
        return filtered, info
    distance = np.arange(len(values), dtype=float) - peak_row
    correction = delta * np.exp(-0.5 * (distance / sigma) ** 2)
    filtered += correction
    filtered[peak_row] = group_peak
    info["applied"] = True
    return filtered, info


def draw_step(
    ax,
    x,
    counts,
    filtered,
    colour,
    fill,
    xlabel,
    title,
    sigma,
    source_label,
    removed=None,
    anchor=None,
):
    x = np.asarray(x, dtype=float)
    counts = np.asarray(counts, dtype=float)
    filtered = np.asarray(filtered, dtype=float)
    ax.bar(
        x,
        counts,
        width=0.92,
        color=fill,
        edgecolor="none",
        label=source_label,
        zorder=1,
    )
    dense_x = np.linspace(float(x.min()), float(x.max()), 600)
    dense_y = PchipInterpolator(x, filtered)(dense_x)
    ax.fill_between(
        dense_x, 0, dense_y, color=colour, alpha=0.10, zorder=2
    )
    ax.plot(
        dense_x,
        dense_y,
        color=colour,
        linewidth=2.0,
        label=f"高斯滤波 σ={sigma:g}",
        zorder=3,
    )
    if removed is not None and removed.any():
        ax.scatter(
            x[removed],
            np.ones(int(removed.sum())),
            marker="x",
            s=25,
            color="#777777",
            linewidths=1.0,
            label="去除的值1",
            zorder=5,
        )
    peak = int(np.argmax(filtered))
    ax.annotate(
        f"({int(x[peak])}, {filtered[peak]:.2f})",
        (x[peak], filtered[peak]),
        xytext=(6, 8),
        textcoords="offset points",
        fontsize=9,
        color=colour,
        fontproperties=font(),
    )
    if anchor is not None and anchor["applied"]:
        ax.scatter(
            anchor["anchor_row"],
            anchor["anchor_value"],
            marker="*",
            s=120,
            color="#d4aa00",
            edgecolor="#6b4c00",
            linewidth=0.8,
            zorder=6,
            label=(
                f"末组锚点 ({anchor['anchor_row']}, "
                f"{count_text(anchor['anchor_value'])})"
            ),
        )
    ax.set_xlabel(xlabel, fontproperties=font())
    ax.set_ylabel("标注点数量", fontproperties=font())
    ax.set_title(title, fontproperties=font(), fontsize=11, pad=7)
    ax.grid(True, axis="y", color="#e6e6e6", linewidth=0.7)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_ylim(0, max(1, float(filtered.max()) + 1))
    ax.legend(prop=font(), frameon=False, loc="upper right")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--names",
        default="001,002,003,004,006,010,011,013,015,029,035,037,038",
    )
    parser.add_argument("--sigma", type=float, default=2.0)
    parser.add_argument("--last-group-anchor-fraction", type=float, default=0.4)
    parser.add_argument(
        "--input",
        type=Path,
        default=ROOT / "outfinal" / "lipcut_roi_annotation",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "outfinal" / "lipcut_roi_annotation"
        / "annotation_functions",
    )
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    names = [name for name in args.names.split(",") if name]

    payload = {}
    long_rows = []
    summary_rows = []
    fig_all, axes_all = plt.subplots(
        len(names), 2, figsize=(12, 3.3 * len(names)),
        constrained_layout=True, squeeze=False
    )
    for row_index, name in enumerate(names):
        data = json.loads(
            (
                args.input / f"peakscolor50lipcut_{name}.json"
            ).read_text(encoding="utf-8")
        )
        width, height = (int(v) for v in data["size"])
        red = {
            (int(x), int(y)) for x, y in data["vertical_points"]
        }
        blue = {
            (int(x), int(y)) for x, y in data["horizontal_points"]
        }
        both = {
            (int(x), int(y)) for x, y in data["both_points"]
        }
        red -= both
        blue -= both
        row_counts = np.bincount(
            [point[1] for point in red], minlength=height
        ).astype(int)
        col_counts = np.bincount(
            [
                point[0] for point in blue
                if point[1] >= height // 2
            ],
            minlength=width,
        ).astype(int)
        col_removed = col_counts == 1
        col_clean = np.where(col_removed, 0, col_counts)
        row_after_ones, row_one_merge_records = merge_value_ones(
            row_counts
        )
        row_merged, row_isolated_merge_records = merge_isolated_rows(
            row_after_ones
        )
        row_filtered_unanchored = gaussian_filter(row_merged, args.sigma)
        row_filtered, last_group_anchor = anchor_last_group(
            row_merged,
            row_filtered_unanchored,
            args.sigma,
            args.last_group_anchor_fraction,
        )
        col_filtered = gaussian_filter(col_clean, args.sigma)
        row_x = np.arange(height)
        col_x = np.arange(width)
        payload[name] = {
            "image": name,
            "input_size": [width, height],
            "row_rule": "all columns; red points",
            "column_rule": (
                f"blue points in lower half y>={height // 2}"
            ),
            "gaussian_filter": {
                "dimension": 1,
                "sigma": args.sigma,
                "mode": "reflect",
                "last_group_anchor_fraction": (
                    args.last_group_anchor_fraction
                ),
            },
            "row_function": {
                "index": row_x.tolist(),
                "counts": row_counts.tolist(),
                "value1_merge_records": row_one_merge_records,
                "counts_after_value1_merge": row_after_ones,
                "isolated_merge_records": row_isolated_merge_records,
                "counts_after_isolated_merge": row_merged,
                "counts_after_gaussian_filter": (
                    row_filtered.round(6).tolist()
                ),
                "counts_gaussian_before_last_group_anchor": (
                    row_filtered_unanchored.round(6).tolist()
                ),
                "last_group_anchor": last_group_anchor,
                "peak": [
                    int(row_x[np.argmax(row_merged)]),
                    float(max(row_merged)),
                ],
                "gaussian_peak": [
                    int(row_x[np.argmax(row_filtered)]),
                    float(row_filtered.max()),
                ],
            },
            "column_function": {
                "index": col_x.tolist(),
                "counts": col_counts.tolist(),
                "singletons_removed": col_removed.tolist(),
                "counts_after_removing_value1": col_clean.tolist(),
                "counts_after_gaussian_filter": (
                    col_filtered.round(6).tolist()
                ),
                "peak": [
                    int(col_x[np.argmax(col_clean)]),
                    int(col_clean.max()),
                ],
                "gaussian_peak": [
                    int(col_x[np.argmax(col_filtered)]),
                    float(col_filtered.max()),
                ],
            },
        }
        for index, raw in zip(row_x, row_counts):
            one_record = next(
                (
                    record for record in row_one_merge_records
                    if record["row"] == int(index)
                ),
                None,
            )
            isolated_record = next(
                (
                    record for record in row_isolated_merge_records
                    if record["row"] == int(index)
                ),
                None,
            )
            merge_record = one_record or isolated_record
            long_rows.append(
                [
                    name,
                    "row_red",
                    int(index),
                    int(raw),
                    count_text(row_after_ones[int(index)]),
                    count_text(row_merged[int(index)]),
                    "" if merge_record is None else merge_record["target"],
                    (
                        ""
                        if not merge_record
                        or merge_record["target_start"] is None
                        else merge_record["target_start"]
                    ),
                    (
                        ""
                        if not merge_record
                        or merge_record["target_end"] is None
                        else merge_record["target_end"]
                    ),
                    round(float(row_filtered[int(index)]), 6),
                ]
            )
        for index, raw, clean in zip(col_x, col_counts, col_clean):
            long_rows.append(
                [
                    name,
                    "column_blue_lower_half",
                    int(index),
                    int(raw),
                    int(clean),
                    int(clean),
                    "",
                    "",
                    "",
                    round(float(col_filtered[int(index)]), 6),
                ]
            )
        summary_rows.append(
            [
                name,
                height,
                width,
                int(row_counts.sum()),
                count_text(sum(row_after_ones)),
                count_text(sum(row_merged)),
                int(col_clean.sum()),
                int(
                    sum(
                        record["merged"]
                        for record in row_one_merge_records
                    )
                ),
                int(
                    sum(
                        not record["merged"]
                        for record in row_one_merge_records
                    )
                ),
                int(col_removed.sum()),
                int(
                    sum(
                        record["merged"]
                        for record in row_isolated_merge_records
                    )
                ),
                int(
                    sum(
                        not record["merged"]
                        for record in row_isolated_merge_records
                    )
                ),
                payload[name]["row_function"]["peak"][0],
                count_text(payload[name]["row_function"]["peak"][1]),
                payload[name]["column_function"]["peak"][0],
                payload[name]["column_function"]["peak"][1],
                payload[name]["row_function"]["gaussian_peak"][0],
                count_text(
                    payload[name]["row_function"]["gaussian_peak"][1]
                ),
                payload[name]["column_function"]["gaussian_peak"][0],
                count_text(
                    payload[name]["column_function"]["gaussian_peak"][1]
                ),
            ]
        )

        fig, axes = plt.subplots(
            1, 2, figsize=(11.5, 3.7), constrained_layout=True
        )
        draw_step(
            axes[0],
            row_x,
            row_merged,
            row_filtered,
            "#c0392b",
            "#f2d4d0",
            "行号 x（w50）",
            f"{name}：值1与孤立值合并后的行函数（高斯滤波）",
            args.sigma,
            "合并后行函数",
            removed=None,
            anchor=last_group_anchor,
        )
        draw_step(
            axes[1],
            col_x,
            col_clean,
            col_filtered,
            "#1f5fa8",
            "#d4e1f2",
            "列号 y（w50）",
            f"{name}：去除值1后的列函数（高斯滤波）",
            args.sigma,
            "去除值1后列函数",
            removed=col_removed,
        )
        axes[0].set_xlim(-0.5, height - 0.5)
        axes[1].set_xlim(-0.5, width - 0.5)
        fig.savefig(args.out / f"annotation_function_{name}.png", dpi=180)
        plt.close(fig)

        draw_step(
            axes_all[row_index, 0],
            row_x,
            row_merged,
            row_filtered,
            "#c0392b",
            "#f2d4d0",
            "行号 x",
            f"{name}  值1与孤立值合并后的行函数（高斯滤波）",
            args.sigma,
            "合并后行函数",
            removed=None,
            anchor=last_group_anchor,
        )
        draw_step(
            axes_all[row_index, 1],
            col_x,
            col_clean,
            col_filtered,
            "#1f5fa8",
            "#d4e1f2",
            "列号 y",
            f"{name}  去除值1后的列函数（高斯滤波）",
            args.sigma,
            "去除值1后列函数",
            removed=col_removed,
        )
        axes_all[row_index, 0].set_xlim(-0.5, height - 0.5)
        axes_all[row_index, 1].set_xlim(-0.5, width - 0.5)
        print(
            f"{name}: row values1 merged="
            f"{sum(record['merged'] for record in row_one_merge_records)}, "
            f"isolated values merged="
            f"{sum(record['merged'] for record in row_isolated_merge_records)}, "
            f"column removed1={int(col_removed.sum())}"
        )

    fig_all.savefig(args.out / "all_annotation_functions.png", dpi=170)
    plt.close(fig_all)
    with (args.out / "annotation_functions_long.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "image",
                "direction",
                "index",
                "raw_count",
                "after_value1_merge",
                "after_isolated_merge",
                "merge_target",
                "merge_target_start",
                "merge_target_end",
                "after_gaussian_filter",
            ]
        )
        writer.writerows(long_rows)
    merge_rows = [
        [
            name,
            record["phase"],
            record["row"],
            record["value"],
            record["target"],
            "" if record["target_start"] is None else record["target_start"],
            "" if record["target_end"] is None else record["target_end"],
            record["increment"],
            ",".join(
                str(row) for row in record["skipped_isolated_rows"]
            ),
            str(record["merged"]).lower(),
            record["reason"],
        ]
        for name in names
        for record in (
            payload[name]["row_function"]["value1_merge_records"]
            + payload[name]["row_function"]["isolated_merge_records"]
        )
    ]
    with (args.out / "row_merge_details.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "image",
                "phase",
                "source_row",
                "source_value",
                "target_side",
                "target_start",
                "target_end",
                "increment_per_row",
                "skipped_isolated_rows",
                "merged",
                "reason",
            ]
        )
        writer.writerows(merge_rows)
    with (args.out / "annotation_functions_summary.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "image",
                "rows",
                "columns",
                "row_red_total_raw",
                "row_red_total_after_value1_merge",
                "row_red_total_after_isolated_merge",
                "column_blue_lower_half_total",
                "row_values_1_merged",
                "row_values_1_unresolved",
                "column_values_1_removed",
                "row_isolated_values_merged",
                "row_isolated_values_unresolved",
                "row_peak",
                "row_peak_count",
                "column_peak",
                "column_peak_count",
                "row_gaussian_peak",
                "row_gaussian_peak_count",
                "column_gaussian_peak",
                "column_gaussian_peak_count",
            ]
        )
        writer.writerows(summary_rows)
    (args.out / "annotation_functions.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
