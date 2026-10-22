"""Summarize several exported single-image test folders."""

from __future__ import annotations

import argparse
import csv
import json
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("folders", type=Path, nargs="+")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    rows = []
    for folder in args.folders:
        files = sorted(folder.glob("*_total_score.json"))
        if len(files) != 1:
            raise SystemExit(
                f"{folder}: expected one *_total_score.json"
            )
        payload = json.loads(files[0].read_text(encoding="utf-8"))
        abdomen_files = sorted(folder.glob("*_score.json"))
        abdomen = (
            json.loads(abdomen_files[0].read_text(encoding="utf-8"))
            if abdomen_files else {}
        )
        rows.append(
            {
                "image": payload["image"],
                "mean_open_angle_deg": payload["abdomen"]["raw"],
                "abdomen_score": payload["abdomen"]["score"],
                "waist_score": payload["waist"]["score"],
                "arm_score": payload["arm"]["score"],
                "weighted_abdomen": payload["weighted"]["abdomen"],
                "weighted_waist": payload["weighted"]["waist"],
                "weighted_arm": payload["weighted"]["arm"],
                "total": payload["total"],
                "selected_centers": abdomen.get(
                    "selected_center_count", ""
                ),
            }
        )

    with (args.out / "batch_score.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as handle:
        fieldnames = list(rows[0])
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    (args.out / "batch_score.json").write_text(
        json.dumps({"rows": rows}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    labels = [row["image"] for row in rows]
    metrics = (
        ("abdomen_score", "腹肌"),
        ("waist_score", "肩腰"),
        ("arm_score", "手臂"),
        ("total", "总分"),
    )
    x = np.arange(len(rows))
    width = 0.18
    figure, axis = plt.subplots(figsize=(9.0, 4.8))
    for index, (key, label) in enumerate(metrics):
        values = [float(row[key]) for row in rows]
        axis.bar(
            x + (index - 1.5) * width,
            values,
            width,
            label=label,
        )
        for xpos, value in zip(x, values):
            axis.text(
                xpos + (index - 1.5) * width,
                value + 1,
                f"{value:.1f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )
    axis.set_xticks(x)
    axis.set_xticklabels(labels)
    axis.set_ylim(0, 110)
    axis.set_ylabel("分数")
    axis.set_title("037–039 批量测试结果")
    axis.grid(axis="y", color="0.9")
    axis.set_axisbelow(True)
    axis.legend(ncol=4)
    figure.tight_layout()
    figure.savefig(args.out / "batch_score.png", dpi=180)
    plt.close(figure)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
