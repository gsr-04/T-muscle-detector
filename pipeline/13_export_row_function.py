"""Export one image's final row function as CSV, JSON and a plot."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import PchipInterpolator


plt.rcParams["font.sans-serif"] = [
    "Microsoft YaHei",
    "SimHei",
    "DejaVu Sans",
]
plt.rcParams["axes.unicode_minus"] = False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True)
    parser.add_argument("--json", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    payload = json.loads(args.json.read_text(encoding="utf-8"))
    function = payload[args.name]["row_function"]
    args.out.mkdir(parents=True, exist_ok=True)

    index = np.asarray(function["index"], dtype=float)
    raw = np.asarray(function["counts"], dtype=float)
    after_ones = np.asarray(
        function["counts_after_value1_merge"], dtype=float
    )
    merged = np.asarray(
        function["counts_after_isolated_merge"], dtype=float
    )
    filtered = np.asarray(
        function["counts_after_gaussian_filter"], dtype=float
    )

    with (args.out / f"{args.name}_row_function.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "row",
                "raw_count",
                "after_value1_merge",
                "after_isolated_merge",
                "after_gaussian_filter",
            ]
        )
        for values in zip(index, raw, after_ones, merged, filtered):
            writer.writerow(
                [
                    int(values[0]),
                    int(values[1]),
                    f"{values[2]:.6f}",
                    f"{values[3]:.6f}",
                    f"{values[4]:.6f}",
                ]
            )
    (args.out / f"{args.name}_row_function.json").write_text(
        json.dumps(
            {
                "image": args.name,
                "index": index.astype(int).tolist(),
                "raw_count": raw.astype(int).tolist(),
                "after_value1_merge": after_ones.tolist(),
                "after_isolated_merge": merged.tolist(),
                "after_gaussian_filter": filtered.tolist(),
                "peak": function["gaussian_peak"],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    figure, axis = plt.subplots(figsize=(11.0, 4.2))
    axis.bar(
        index,
        raw,
        width=0.92,
        color="#eeeeee",
        edgecolor="#b0b0b0",
        linewidth=0.4,
        label="原始计数",
        zorder=1,
    )
    dense = np.linspace(index.min(), index.max(), 700)
    curve = PchipInterpolator(index, filtered)(dense)
    axis.plot(
        dense,
        curve,
        color="#c0392b",
        linewidth=2.2,
        label="高斯滤波行函数（σ=2）",
        zorder=4,
    )
    axis.scatter(
        index[merged > 0],
        merged[merged > 0],
        s=13,
        color="#7b241c",
        label="合并后离散值",
        zorder=5,
    )
    peak_row, peak_value = function["gaussian_peak"]
    axis.annotate(
        f"峰值 ({peak_row}, {peak_value:.3f})",
        (peak_row, peak_value),
        xytext=(8, 9),
        textcoords="offset points",
        color="#8b0000",
    )
    axis.set_xlim(-0.5, float(index.max()) + 0.5)
    axis.set_ylim(0, max(1.0, float(filtered.max()) + 1.2))
    axis.set_xlabel("行号 x")
    axis.set_ylabel("标注点数量")
    axis.set_title(f"{args.name} 行函数")
    axis.grid(axis="y", color="0.9")
    axis.set_axisbelow(True)
    axis.legend(frameon=False)
    figure.tight_layout()
    figure.savefig(args.out / f"{args.name}_row_function.png", dpi=180)
    plt.close(figure)
    print(f"wrote {args.out / f'{args.name}_row_function.png'}")


if __name__ == "__main__":
    main()
