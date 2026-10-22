"""Render retained centre points on the complete original images."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "outfinal" / "lipcut_roi_annotation"
RESULT_PATH = INPUT / "center_objective" / "result.json"
OUT = INPUT / "center_objective" / "original_overlays"
GROUP_COLOURS = [
    (255, 210, 0),
    (255, 125, 35),
    (160, 220, 35),
    (35, 210, 100),
    (30, 205, 205),
    (40, 150, 235),
    (100, 115, 250),
    (205, 75, 235),
    (255, 75, 135),
    (145, 210, 255),
    (255, 175, 180),
    (190, 160, 110),
]
CENTER_COLOURS = [
    (255, 225, 0),
    (60, 230, 255),
    (255, 110, 40),
    (80, 255, 110),
    (255, 80, 180),
    (155, 130, 255),
]


def load_font(size):
    path = Path(r"C:\Windows\Fonts\msyh.ttc")
    return (
        ImageFont.truetype(str(path), size)
        if path.exists()
        else ImageFont.load_default(size=size)
    )


def full_pixel(row, column, roi, pooled_h):
    left, top, right, bottom = roi["rectangle_px"]
    return (
        left + (column + 0.5) * (right - left) / 50.0,
        top + (row + 0.5) * (bottom - top) / float(pooled_h),
    )


def draw_label(draw, xy, text, font, colour):
    x, y = xy
    box = draw.textbbox((x, y), text, font=font, stroke_width=1)
    draw.rectangle(
        [box[0] - 3, box[1] - 2, box[2] + 3, box[3] + 2],
        fill=(0, 0, 0),
    )
    draw.text(
        (x, y),
        text,
        font=font,
        fill=colour,
        stroke_width=1,
        stroke_fill=(0, 0, 0),
    )


def render_center(
    record, selected_points, image_name, roi, pooled_h, output
):
    image = Image.open(ROOT / "data" / f"{image_name}.png").convert("RGB")
    draw = ImageDraw.Draw(image)
    left, top, right, bottom = roi["rectangle_px"]
    draw.rectangle(
        [left, top, right - 1, bottom - 1],
        outline=(255, 215, 0),
        width=3,
    )
    sx0, sy0, sx1, sy1 = record["rectangle"]
    x0 = left + int(round(sx0 * (right - left) / 50.0))
    x1 = left + int(round((sx1 + 1) * (right - left) / 50.0)) - 1
    y0 = top + int(round(sy0 * (bottom - top) / pooled_h))
    y1 = top + int(round((sy1 + 1) * (bottom - top) / pooled_h)) - 1
    draw.rectangle([x0, y0, x1, y1], outline=(30, 220, 255), width=3)

    radius = max(4, min(10, min(right - left, bottom - top) // 70))
    points_by_group = {}
    for point in selected_points:
        points_by_group.setdefault(
            int(point["group_id"]), []
        ).append(point)

    for group_id, points in sorted(points_by_group.items()):
        colour = GROUP_COLOURS[(group_id - 1) % len(GROUP_COLOURS)]
        ordered = sorted(
            points,
            key=lambda point: point.get(
                "angle",
                math.atan2(
                    int(point["row"]) - record["row"],
                    int(point["column"]) - record["column"],
                ),
            ),
        )
        pixels = [
            full_pixel(
                int(point["row"]), int(point["column"]), roi, pooled_h
            )
            for point in ordered
        ]
        if len(pixels) > 1:
            draw.line(pixels, fill=colour, width=3)
        for point, pixel in zip(ordered, pixels):
            x, y = (int(round(value)) for value in pixel)
            draw.ellipse(
                [x - radius, y - radius, x + radius, y + radius],
                fill=colour,
                outline=(0, 0, 0),
                width=2,
            )

    cx, cy = full_pixel(record["row"], record["column"], roi, pooled_h)
    cx, cy = int(round(cx)), int(round(cy))
    arm = radius + 6
    draw.line(
        [(cx - arm, cy), (cx + arm, cy)],
        fill=(255, 0, 255),
        width=4,
    )
    draw.line(
        [(cx, cy - arm), (cx, cy + arm)],
        fill=(255, 0, 255),
        width=4,
    )
    font = load_font(max(15, min(28, image.width // 40)))
    draw_label(
        draw,
        (left + 6, max(4, top - font.size - 8)),
        (
            f"{image_name} C{record['index']:02d} "
            f"F={record['objective']:.4f} "
            f"kept={record['selected_point_count']}"
        ),
        font,
        (255, 255, 0),
    )
    image.save(output)
    return output


def render_combined(image_name, records, roi, pooled_h, output):
    image = Image.open(ROOT / "data" / f"{image_name}.png").convert("RGB")
    draw = ImageDraw.Draw(image)
    left, top, right, bottom = roi["rectangle_px"]
    draw.rectangle(
        [left, top, right - 1, bottom - 1],
        outline=(255, 215, 0),
        width=3,
    )
    radius = max(4, min(8, min(right - left, bottom - top) // 90))
    font = load_font(max(15, min(28, image.width // 40)))
    memberships = {}
    for record in records:
        colour = CENTER_COLOURS[
            (record["index"] - 1) % len(CENTER_COLOURS)
        ]
        sx0, sy0, sx1, sy1 = record["rectangle"]
        x0 = left + int(round(sx0 * (right - left) / 50.0))
        x1 = left + int(round((sx1 + 1) * (right - left) / 50.0)) - 1
        y0 = top + int(round(sy0 * (bottom - top) / pooled_h))
        y1 = top + int(round((sy1 + 1) * (bottom - top) / pooled_h)) - 1
        draw.rectangle([x0, y0, x1, y1], outline=colour, width=2)
        cx, cy = full_pixel(record["row"], record["column"], roi, pooled_h)
        cx, cy = int(round(cx)), int(round(cy))
        draw.line(
            [(cx - 9, cy), (cx + 9, cy)], fill=colour, width=3
        )
        draw.line(
            [(cx, cy - 9), (cx, cy + 9)], fill=colour, width=3
        )
        for point in record["selected_points"]:
            cell = (point["row"], point["column"])
            memberships.setdefault(cell, set()).add(record["index"])

    for (row, column), center_ids in sorted(memberships.items()):
        x, y = full_pixel(row, column, roi, pooled_h)
        x, y = int(round(x)), int(round(y))
        ordered_ids = sorted(center_ids)
        outer = radius + len(ordered_ids)
        draw.ellipse(
            [x - outer, y - outer, x + outer, y + outer],
            fill=(0, 0, 0),
        )
        for offset, center_id in enumerate(ordered_ids):
            colour = CENTER_COLOURS[
                (center_id - 1) % len(CENTER_COLOURS)
            ]
            ring = outer - offset - 1
            if ring <= 1:
                break
            draw.ellipse(
                [x - ring, y - ring, x + ring, y + ring],
                outline=colour,
                width=2,
            )

    for index, record in enumerate(records):
        colour = CENTER_COLOURS[
            (record["index"] - 1) % len(CENTER_COLOURS)
        ]
        draw_label(
            draw,
            (7, 7 + index * (font.size + 7)),
            (
                f"C{record['index']} "
                f"F={record['objective']:.4f} "
                f"kept={record['selected_point_count']}"
            ),
            font,
            colour,
        )
    image.save(output)
    return output


def contact_sheet(paths, output):
    cell_width, cell_height = 360, 520
    margin, gap, columns = 18, 12, 3
    rows = math.ceil(len(paths) / columns)
    sheet = Image.new(
        "RGB",
        (
            columns * cell_width + (columns + 1) * margin,
            rows * cell_height + (rows + 1) * margin,
        ),
        (245, 245, 245),
    )
    draw = ImageDraw.Draw(sheet)
    font = load_font(15)
    for index, path in enumerate(paths):
        image = Image.open(path).convert("RGB")
        image.thumbnail(
            (cell_width - 12, cell_height - 45),
            Image.Resampling.LANCZOS,
        )
        column, row = index % columns, index // columns
        x = margin + column * (cell_width + gap)
        y = margin + row * (cell_height + gap)
        draw.rectangle(
            [x, y, x + cell_width - 1, y + cell_height - 1],
            fill="white",
            outline=(205, 205, 205),
        )
        sheet.paste(image, (x + (cell_width - image.width) // 2, y + 4))
        draw.text(
            (x + 7, y + cell_height - 32),
            path.stem.replace("original_centers_", ""),
            font=font,
            fill=(45, 45, 45),
        )
    sheet.save(output)
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--names",
        default="",
        help="comma-separated image names; empty uses every result image",
    )
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    result = json.loads(RESULT_PATH.read_text(encoding="utf-8"))
    with (INPUT / "center_objective" / "selected_points.csv").open(
        encoding="utf-8-sig", newline=""
    ) as handle:
        point_details = {}
        for row in csv.DictReader(handle):
            key = (row["image"], int(row["centre_index"]))
            point_details.setdefault(key, []).append(row)
    rois = {
        record["image"]: record
        for record in json.loads(
            (INPUT / "roi_annotation.json").read_text(encoding="utf-8")
        )["records"]
    }
    csv_rows = []
    combined_paths = []
    requested = [value for value in args.names.split(",") if value]
    image_records = [
        record for record in result["records"]
        if not requested or record["image"] in requested
    ]
    if not image_records:
        raise SystemExit("no matching centre-objective records")
    for image_record in image_records:
        name = image_record["image"]
        source = json.loads(
            (INPUT / f"peakscolor50lipcut_{name}.json").read_text(
                encoding="utf-8"
            )
        )
        pooled_h = int(source["size"][1])
        roi = rois[name]
        for record in image_record["centres"]:
            selected_points = point_details[
                (name, int(record["index"]))
            ]
            output = (
                OUT / f"original_{name}_C{record['index']:02d}.png"
            )
            render_center(
                record,
                selected_points,
                name,
                roi,
                pooled_h,
                output,
            )
            for point in selected_points:
                px, py = full_pixel(
                    int(point["row"]),
                    int(point["column"]),
                    roi,
                    pooled_h,
                )
                csv_rows.append(
                    [
                        name,
                        record["index"],
                        point["row"],
                        point["column"],
                        point["family"],
                        point["group_id"],
                        point["group_size"],
                        point["source_shape"],
                        point["segment_kind"],
                        round(px, 6),
                        round(py, 6),
                    ]
                )
        combined = OUT / f"original_centers_{name}.png"
        render_combined(
            name,
            image_record["centres"],
            roi,
            pooled_h,
            combined,
        )
        combined_paths.append(combined)

    with (OUT / "retained_points_fullres.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "image",
                "centre_index",
                "row",
                "column",
                "family",
                "group_id",
                "group_size",
                "source_shape",
                "segment_kind",
                "full_x",
                "full_y",
            ]
        )
        writer.writerows(csv_rows)
    contact_sheet(combined_paths, OUT / "original_centers_contact.png")
    print(
        f"rendered {len(csv_rows)} retained centre-point records "
        f"on {len(combined_paths)} original images"
    )


if __name__ == "__main__":
    main()
