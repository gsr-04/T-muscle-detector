"""Mark selected row/column intersections inside the LIP-cut rectangles."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.signal import find_peaks


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "outfinal" / "lipcut_roi_annotation"


def load_font(size):
    path = Path(r"C:\Windows\Fonts\msyh.ttc")
    return (
        ImageFont.truetype(str(path), size)
        if path.exists()
        else ImageFont.load_default(size=size)
    )


def mpl_font(size=10):
    path = Path(r"C:\Windows\Fonts\msyh.ttc")
    return font_manager.FontProperties(fname=str(path), size=size)


def choose_rows(values):
    """Choose the first top-three maximum and its next three minima."""
    maxima = [int(value) for value in find_peaks(values)[0]]
    ranked_maxima = sorted(
        maxima,
        key=lambda index: (values[index], -index),
        reverse=True,
    )[:3]
    anchor = (
        min(ranked_maxima)
        if ranked_maxima
        else int(np.argmax(values))
    )
    minima = [int(value) for value in find_peaks(-values)[0]]
    following_minima = [
        row for row in minima if row > anchor
    ]
    selected = following_minima[:3]

    boundary_rows = []
    last = len(values) - 1
    if (
        len(selected) < 3
        and last not in selected
        and values[last] <= values[last - 1]
    ):
        selected.append(last)
        boundary_rows.append(last)

    selected = sorted(set(selected))
    return {
        "rows": selected,
        "all_minima": minima,
        "following_minima": following_minima,
        "boundary_rows": boundary_rows,
        "all_maxima": maxima,
        "top3_maxima": ranked_maxima,
        "anchor_maximum": anchor,
    }


def choose_columns(values):
    """Use the largest maximum that has a minimum on both sides.

    A high maximum close to an image edge may have no following minimum.  In
    that case, inspect the other internal maxima from high to low instead of
    immediately falling back to the 1/3 and 2/3 columns.
    """
    global_maximum = int(np.argmax(values))
    maxima = [int(value) for value in find_peaks(values)[0]]
    minima = [int(value) for value in find_peaks(-values)[0]]
    maximum = None
    left = []
    right = []
    for candidate in sorted(
        maxima,
        key=lambda index: (values[index], -index),
        reverse=True,
    ):
        candidate_left = [
            column for column in minima if column < candidate
        ]
        candidate_right = [
            column for column in minima if column > candidate
        ]
        if candidate_left and candidate_right:
            maximum = candidate
            left = candidate_left
            right = candidate_right
            break
    if maximum is not None:
        columns = [left[-1], right[0]]
        fallback = False
    else:
        maximum = global_maximum
        columns = [
            int(round(len(values) / 3.0)),
            int(round(2.0 * len(values) / 3.0)),
        ]
        fallback = True
    return {
        "columns": columns,
        "maximum": maximum,
        "maximum_value": (
            float(values[maximum]) if maximum is not None else None
        ),
        "global_maximum": global_maximum,
        "global_maximum_value": float(values[global_maximum]),
        "all_maxima": maxima,
        "all_minima": minima,
        "fallback": fallback,
    }


def plot_selection(
    name,
    row_values,
    column_values,
    row_info,
    column_info,
    output,
):
    figure, axes = plt.subplots(
        1, 2, figsize=(12.5, 4.1), constrained_layout=True
    )
    row_axis, column_axis = axes

    row_axis.plot(row_values, color="#c0392b", linewidth=1.8)
    row_axis.scatter(
        row_info["all_minima"],
        row_values[row_info["all_minima"]],
        s=24,
        color="#d98880",
        label="全部局部极小值",
        zorder=3,
    )
    row_axis.scatter(
        row_info["top3_maxima"],
        row_values[row_info["top3_maxima"]],
        s=58,
        marker="^",
        color="#e67e22",
        label="前三极大值",
        zorder=4,
    )
    row_axis.scatter(
        row_info["anchor_maximum"],
        row_values[row_info["anchor_maximum"]],
        s=120,
        marker="*",
        color="#d4aa00",
        edgecolor="#7a4b00",
        linewidth=0.8,
        label="最靠左的前三极大值",
        zorder=5,
    )
    row_axis.axvline(
        row_info["anchor_maximum"],
        color="#d4aa00",
        linestyle=":",
        linewidth=1.6,
    )
    row_axis.scatter(
        row_info["rows"],
        row_values[row_info["rows"]],
        s=72,
        marker="D",
        color="#8b0000",
        label="选中的三行",
        zorder=4,
    )
    for row in row_info["rows"]:
        row_axis.axvline(
            row, color="#8b0000", linestyle="--", alpha=0.45
        )
    row_axis.set_xlabel("行号 x", fontproperties=mpl_font())
    row_axis.set_ylabel("高斯滤波计数", fontproperties=mpl_font())
    row_axis.set_title(
        (
            f"{name}：前三极大值最左点 "
            f"x={row_info['anchor_maximum']} 后三个极小值"
        ),
        fontproperties=mpl_font(),
    )
    row_axis.grid(alpha=0.2)
    row_axis.legend(prop=mpl_font())

    column_axis.plot(column_values, color="#1f5fa8", linewidth=1.8)
    column_axis.scatter(
        column_info["maximum"],
        column_values[column_info["maximum"]],
        s=110,
        marker="*",
        color="#d4aa00",
        label="选定的极大值",
        zorder=5,
    )
    if column_info["global_maximum"] != column_info["maximum"]:
        column_axis.scatter(
            column_info["global_maximum"],
            column_values[column_info["global_maximum"]],
            s=80,
            marker="^",
            facecolor="white",
            edgecolor="#7a4b00",
            linewidth=1.5,
            label="全局最大值（无两侧极小值）",
            zorder=4,
        )
    column_axis.scatter(
        column_info["columns"],
        column_values[column_info["columns"]],
        s=70,
        color="#073b75",
        label="最大值两侧极小值",
        zorder=4,
    )
    for column in column_info["columns"]:
        column_axis.axvline(
            column, color="#073b75", linestyle="--", alpha=0.45
        )
    title = f"{name}：最大可用极大值两侧极小值"
    if column_info["fallback"]:
        title += "（1/3、2/3 回退）"
    column_axis.set_title(title, fontproperties=mpl_font())
    column_axis.set_xlabel("列号 y", fontproperties=mpl_font())
    column_axis.set_ylabel("高斯滤波计数", fontproperties=mpl_font())
    column_axis.grid(alpha=0.2)
    column_axis.legend(prop=mpl_font())

    figure.savefig(output, dpi=180)
    plt.close(figure)


def draw_intersections(
    name,
    source_path,
    rectangle_px,
    row_count,
    column_count,
    rows,
    columns,
    crop_output,
    overlay_output,
):
    source_image = Image.open(source_path).convert("RGB")
    left, top, right, bottom = rectangle_px
    if left < 0 or top < 0:
        raise ValueError(f"{name}: invalid rectangle origin")
    if right > source_image.width or bottom > source_image.height:
        raise ValueError(f"{name}: rectangle exceeds source image")
    image = source_image.crop((left, top, right, bottom))
    image.save(crop_output)
    draw = ImageDraw.Draw(image)
    width, height = image.size
    font = load_font(max(11, min(20, width // 28)))
    radius = max(5, min(12, min(width, height) // 55))

    row_y = {
        row: int(round((row + 0.5) * height / row_count - 0.5))
        for row in rows
    }
    column_x = {
        column: int(round(
            (column + 0.5) * width / column_count - 0.5
        ))
        for column in columns
    }

    for row, y in row_y.items():
        draw.line([(0, y), (width - 1, y)], fill=(215, 45, 45), width=2)
        draw.text(
            (4, max(1, y - font.size - 3)),
            f"行 {row}",
            font=font,
            fill=(135, 0, 0),
        )
    for column, x in column_x.items():
        draw.line([(x, 0), (x, height - 1)], fill=(30, 95, 190), width=2)
        draw.text(
            (min(width - 42, x + 3), 2),
            f"列 {column}",
            font=font,
            fill=(0, 55, 150),
        )

    intersections = []
    for row, y in row_y.items():
        for column, x in column_x.items():
            draw.ellipse(
                [
                    x - radius,
                    y - radius,
                    x + radius,
                    y + radius,
                ],
                outline=(255, 215, 0),
                width=3,
            )
            draw.ellipse(
                [x - 2, y - 2, x + 2, y + 2],
                fill=(20, 20, 20),
            )
            intersections.append(
                {
                    "row": row,
                    "column": column,
                    "x": x,
                    "y": y,
                }
            )

    draw.rectangle(
        [0, 0, width - 1, height - 1],
        outline=(255, 215, 0),
        width=3,
    )
    image.save(overlay_output)
    return intersections


def contact_sheet(records, output):
    cell_width, cell_height = 400, 560
    margin, gap, columns = 20, 14, 3
    rows = (len(records) + columns - 1) // columns
    sheet = Image.new(
        "RGB",
        (
            columns * cell_width + (columns + 1) * margin,
            rows * cell_height + (rows + 1) * margin,
        ),
        (245, 245, 245),
    )
    draw = ImageDraw.Draw(sheet)
    title_font = load_font(18)
    body_font = load_font(13)
    for index, record in enumerate(records):
        image = Image.open(record["overlay_png"]).convert("RGB")
        image.thumbnail(
            (cell_width - 18, cell_height - 75),
            Image.Resampling.LANCZOS,
        )
        column = index % columns
        row = index // columns
        x = margin + column * (cell_width + gap)
        y = margin + row * (cell_height + gap)
        draw.rectangle(
            [x, y, x + cell_width - 1, y + cell_height - 1],
            fill="white",
            outline=(205, 205, 205),
        )
        draw.text(
            (x + 9, y + 7),
            f"{record['image']}  交点 {record['intersection_count']}",
            fill=(20, 80, 100),
            font=title_font,
        )
        sheet.paste(
            image,
            (
                x + (cell_width - image.width) // 2,
                y + 36,
            ),
        )
        draw.text(
            (x + 9, y + cell_height - 31),
            (
                f"行 {record['rows']}  列 {record['columns']}"
                + ("  回退列" if record["column_fallback"] else "")
            ),
            fill=(55, 55, 55),
            font=body_font,
        )
    sheet.save(output)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--names",
        default="001,002,003,004,006,010,011,013,015,029,035,037,038",
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument(
        "--functions",
        type=Path,
        default=DEFAULT_INPUT
        / "annotation_functions"
        / "annotation_functions.json",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_INPUT / "rowcol_intersections",
    )
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    names = [name for name in args.names.split(",") if name]
    functions = json.loads(args.functions.read_text(encoding="utf-8"))
    roi_records = {
        row["image"]: row
        for row in json.loads(
            (args.input / "roi_annotation.json").read_text(
                encoding="utf-8"
            )
        )["records"]
    }

    records = []
    csv_rows = []
    for name in names:
        function = functions[name]
        row_values = np.asarray(
            function["row_function"]["counts_after_gaussian_filter"],
            dtype=float,
        )
        column_values = np.asarray(
            function["column_function"]["counts_after_gaussian_filter"],
            dtype=float,
        )
        row_info = choose_rows(row_values)
        column_info = choose_columns(column_values)
        rows = row_info["rows"]
        columns = column_info["columns"]

        crop_output = args.out / f"body_rectangle_{name}.png"
        overlay = args.out / f"roi_intersections_{name}.png"
        rectangle_px = [
            int(value)
            for value in roi_records[name]["rectangle_px"]
        ]
        source_path = ROOT / "data" / f"{name}.png"
        points = draw_intersections(
            name,
            source_path,
            rectangle_px,
            len(row_values),
            len(column_values),
            rows,
            columns,
            crop_output,
            overlay,
        )
        selection_plot = args.out / f"selection_{name}.png"
        plot_selection(
            name,
            row_values,
            column_values,
            row_info,
            column_info,
            selection_plot,
        )
        record = {
            "image": name,
            "source_png": str(source_path),
            "rectangle_px": rectangle_px,
            "roi_png": str(crop_output),
            "row_function_length": len(row_values),
            "column_function_length": len(column_values),
            "row_minima_all": row_info["all_minima"],
            "row_maxima_all": row_info["all_maxima"],
            "row_maxima_top3": row_info["top3_maxima"],
            "row_maximum_anchor": row_info["anchor_maximum"],
            "row_minima_after_anchor": row_info["following_minima"],
            "row_boundary_fallback": row_info["boundary_rows"],
            "rows": rows,
            "row_values": [
                round(float(row_values[row]), 6) for row in rows
            ],
            "column_minima_all": column_info["all_minima"],
            "column_maxima_all": column_info["all_maxima"],
            "columns": columns,
            "column_values": [
                round(float(column_values[column]), 6)
                for column in columns
            ],
            "column_maximum": column_info["maximum"],
            "column_maximum_value": round(
                column_info["maximum_value"], 6
            ),
            "column_global_maximum": column_info["global_maximum"],
            "column_global_maximum_value": round(
                column_info["global_maximum_value"], 6
            ),
            "column_fallback": column_info["fallback"],
            "intersections": points,
            "intersection_count": len(points),
            "overlay_png": str(overlay),
            "selection_plot_png": str(selection_plot),
        }
        records.append(record)
        for point in points:
            csv_rows.append(
                [
                    name,
                    point["row"],
                    point["column"],
                    point["x"],
                    point["y"],
                ]
            )
        print(
            f"{name}: rows={rows}, columns={columns}, "
            f"intersections={len(points)}, "
            f"column_fallback={column_info['fallback']}"
        )

    with (args.out / "intersections.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["image", "row", "column", "roi_x", "roi_y"]
        )
        writer.writerows(csv_rows)
    (args.out / "intersections.json").write_text(
        json.dumps({"records": records}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    lines = [
        "# 行函数与列函数极小值交点",
        "",
        "- 行：从左向右，在前三高的局部极大值中取最靠左的一个，再取它之后的三个局部极小值；不足三点时用右边界极小值补齐。",
        "- 列：从高到低检查内部极大值，取第一个左右两侧都有局部极小值的峰；全部候选都不满足时才使用 1/3、2/3 列。",
        "- 交点标注在对应的矩形裁剪图中。",
        "",
        "| 图片 | 行锚点 | 行 | 列 | 列最大值 | 回退列 | 交点数 |",
        "| --- | ---: | --- | --- | ---: | --- | ---: |",
    ]
    for record in records:
        lines.append(
            f"| {record['image']} | {record['row_maximum_anchor']} | "
            f"{record['rows']} | "
            f"{record['columns']} | {record['column_maximum']} | "
            f"{record['column_fallback']} | "
            f"{record['intersection_count']} |"
        )
    (args.out / "summary.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    contact_sheet(records, args.out / "intersections_contact.png")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
