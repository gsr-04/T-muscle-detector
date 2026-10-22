"""Run the curve-aware objective for the current LIP-cut centre intersections."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import armmid_curve_group_objective as curve  # noqa: E402


DEFAULT_INPUT = ROOT / "outfinal" / "lipcut_roi_annotation"
DEFAULT_NAMES = (
    "001,002,003,004,006,010,011,013,015,029,035,037,038"
)
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


def load_font(size):
    path = Path(r"C:\Windows\Fonts\msyh.ttc")
    return (
        ImageFont.truetype(str(path), size)
        if path.exists()
        else ImageFont.load_default(size=size)
    )


def load_candidates(path):
    data = json.loads(path.read_text(encoding="utf-8"))
    both = {tuple(map(int, point)) for point in data["both_points"]}
    red = {
        tuple(map(int, point)) for point in data["vertical_points"]
    } - both
    blue = {
        tuple(map(int, point)) for point in data["horizontal_points"]
    } - both
    points = [
        {"row": y, "column": x, "family": family}
        for family, cells in (
            ("red", red),
            ("blue", blue),
            ("white", both),
        )
        for x, y in sorted(cells)
    ]
    return data, points


def cardinal_neighbours(centre, centres):
    row = int(centre["row"])
    column = int(centre["column"])
    same_row = [
        other for other in centres
        if other["index"] != centre["index"]
        and int(other["row"]) == row
    ]
    same_column = [
        other for other in centres
        if other["index"] != centre["index"]
        and int(other["column"]) == column
    ]
    left = [
        other for other in same_row
        if int(other["column"]) < column
    ]
    right = [
        other for other in same_row
        if int(other["column"]) > column
    ]
    up = [
        other for other in same_column
        if int(other["row"]) < row
    ]
    down = [
        other for other in same_column
        if int(other["row"]) > row
    ]
    return {
        "left": (
            max(left, key=lambda other: int(other["column"]))
            if left else None
        ),
        "right": (
            min(right, key=lambda other: int(other["column"]))
            if right else None
        ),
        "up": (
            max(up, key=lambda other: int(other["row"]))
            if up else None
        ),
        "down": (
            min(down, key=lambda other: int(other["row"]))
            if down else None
        ),
    }


def choose_local_rectangle(centre, centres, row_count, column_count):
    row = int(centre["row"])
    column = int(centre["column"])
    neighbours = cardinal_neighbours(centre, centres)
    horizontal_limits = []
    vertical_limits = []
    if neighbours["left"] is not None:
        horizontal_limits.append(
            column - int(neighbours["left"]["column"]) - 1
        )
    if neighbours["right"] is not None:
        horizontal_limits.append(
            int(neighbours["right"]["column"]) - column - 1
        )
    if neighbours["up"] is not None:
        vertical_limits.append(row - int(neighbours["up"]["row"]) - 1)
    if neighbours["down"] is not None:
        vertical_limits.append(
            int(neighbours["down"]["row"]) - row - 1
        )
    half_width = (
        min(horizontal_limits)
        if horizontal_limits
        else min(column, column_count - 1 - column)
    )
    half_height = (
        min(vertical_limits)
        if vertical_limits
        else min(row, row_count - 1 - row)
    )
    if half_width <= 0 or half_height <= 0:
        raise RuntimeError(
            f"C{centre['index']}: invalid rectangle span"
        )
    desired = (
        column - half_width,
        row - half_height,
        column + half_width,
        row + half_height,
    )
    rectangle = (
        max(0, desired[0]),
        max(0, desired[1]),
        min(column_count - 1, desired[2]),
        min(row_count - 1, desired[3]),
    )
    contained = [
        other["index"] for other in centres
        if other["index"] != centre["index"]
        and rectangle[0] <= int(other["column"]) <= rectangle[2]
        and rectangle[1] <= int(other["row"]) <= rectangle[3]
    ]
    if contained:
        raise RuntimeError(
            f"C{centre['index']}: rectangle contains other centres "
            f"{contained}"
        )
    return rectangle, desired, neighbours, half_width, half_height


def evaluate_inventory_minimum(
    inventory, d_ref, distance_weight, minimum_points
):
    """Find the best visited whole-group selection meeting a point floor."""
    nearest = sorted(
        range(len(inventory)),
        key=lambda index: (
            inventory[index]["distance_sum"] / inventory[index]["size"]
        ),
    )
    seeds = nearest[:5]
    best = None

    def consider(result):
        nonlocal best
        if (
            result["redundant_group_ids"]
            or result["k"] < minimum_points
        ):
            return
        if best is None or result["objective"] < best["objective"]:
            best = result

    for seed in seeds:
        current = curve.optimize_007.evaluate_mask(
            1 << seed,
            inventory,
            d_ref,
            distance_weight=distance_weight,
        )
        consider(current)
        while True:
            step = None
            for index in range(len(inventory)):
                bit = 1 << index
                if current["mask"] & bit:
                    continue
                candidate = curve.optimize_007.evaluate_mask(
                    current["mask"] | bit,
                    inventory,
                    d_ref,
                    distance_weight=distance_weight,
                )
                consider(candidate)
                if candidate["redundant_group_ids"]:
                    continue
                if (
                    step is None
                    or candidate["objective"] < step["objective"]
                ):
                    step = candidate
            if step is None:
                break
            if current["k"] < minimum_points:
                current = step
                consider(current)
                continue
            if step["objective"] >= current["objective"]:
                break
            current = step
            consider(current)

    if best is None:
        # The greedy search can stop on a locally redundant combination even
        # when the candidate total is sufficient.  Fall back to exact
        # whole-group MILP, trying the first feasible cardinalities at or above
        # the required floor.
        group_span = max(
            (group["size"] for group in inventory), default=1
        )
        for required_k in range(
            minimum_points,
            minimum_points + group_span + 1,
        ):
            try:
                candidate, _, _ = (
                    curve.optimize_007.solve_group_selections_milp(
                        inventory,
                        d_ref,
                        fixed_k=required_k,
                        distance_weight=distance_weight,
                    )
                )
            except RuntimeError:
                continue
            if (
                candidate["k"] >= minimum_points
                and not candidate["redundant_group_ids"]
            ):
                return candidate
        raise RuntimeError(
            f"no group selection reaches {minimum_points} points"
        )
    return best


def draw_center_crop(
    name,
    centre,
    local_points,
    selected_cells,
    groups,
    rectangle,
    roi,
    pooled_h,
    objective,
    output,
):
    source_path = ROOT / "data" / f"{name}.png"
    source = Image.open(source_path).convert("RGB")
    left, top, right, bottom = roi["rectangle_px"]
    body = source.crop((left, top, right, bottom))
    body_width, body_height = body.size
    sx0, sy0, sx1, sy1 = rectangle
    crop_left = int(round(sx0 * body_width / 50.0))
    crop_right = int(round((sx1 + 1) * body_width / 50.0))
    crop_top = int(round(sy0 * body_height / pooled_h))
    crop_bottom = int(round((sy1 + 1) * body_height / pooled_h))
    crop_right = min(body_width, max(crop_left + 1, crop_right))
    crop_bottom = min(body_height, max(crop_top + 1, crop_bottom))
    image = body.crop((crop_left, crop_top, crop_right, crop_bottom))
    draw = ImageDraw.Draw(image)
    crop_width, crop_height = image.size
    scale_x = body_width / 50.0
    scale_y = body_height / float(pooled_h)
    radius = max(3, min(10, min(crop_width, crop_height) // 70))
    line_width = max(1, radius // 3)
    font = load_font(max(12, min(24, crop_width // 18)))

    def pixel(row, column):
        return (
            int(round((column + 0.5) * scale_x)) - crop_left,
            int(round((row + 0.5) * scale_y)) - crop_top,
        )

    for point in local_points:
        x, y = pixel(point["row"], point["column"])
        if (point["row"], point["column"]) in selected_cells:
            continue
        draw.ellipse(
            [x - 2, y - 2, x + 2, y + 2],
            fill=(135, 135, 135),
        )

    selected_group_ids = {
        group["group_id"]
        for group in groups
        if any(cell in selected_cells for cell in group["cells"])
    }
    point_lookup = {
        (point["row"], point["column"]): point for point in local_points
    }
    for group in groups:
        if group["group_id"] not in selected_group_ids:
            continue
        colour = GROUP_COLOURS[
            (group["group_id"] - 1) % len(GROUP_COLOURS)
        ]
        ordered = [
            cell for cell in group["cells"] if cell in selected_cells
        ]
        if len(ordered) > 1:
            draw.line(
                [pixel(*cell) for cell in ordered],
                fill=colour,
                width=line_width + 1,
            )
        for row, column in ordered:
            x, y = pixel(row, column)
            draw.ellipse(
                [x - radius, y - radius, x + radius, y + radius],
                fill=colour,
                outline=(20, 20, 20),
                width=1,
            )

    cx, cy = pixel(centre["row"], centre["column"])
    arm = radius + 4
    draw.line(
        [(cx - arm, cy), (cx + arm, cy)],
        fill=(255, 255, 0),
        width=3,
    )
    draw.line(
        [(cx, cy - arm), (cx, cy + arm)],
        fill=(255, 255, 0),
        width=3,
    )
    draw.ellipse(
        [cx - arm - 2, cy - arm - 2, cx + arm + 2, cy + arm + 2],
        outline=(255, 255, 0),
        width=2,
    )
    draw.text(
        (5, 4),
        (
            f"{name} C{centre['index']:02d}  "
            f"F={objective['objective']:.4f}"
        ),
        font=font,
        fill=(255, 255, 0),
        stroke_width=2,
        stroke_fill=(0, 0, 0),
    )
    image.save(output)


def draw_overview(name, records, roi, pooled_h, output):
    source = Image.open(ROOT / "data" / f"{name}.png").convert("RGB")
    left, top, right, bottom = roi["rectangle_px"]
    image = source.crop((left, top, right, bottom))
    draw = ImageDraw.Draw(image)
    font = load_font(max(12, min(24, image.width // 35)))
    for record in records:
        sx0, sy0, sx1, sy1 = record["rectangle"]
        p0 = (
            int(round(sx0 * image.width / 50.0)),
            int(round(sy0 * image.height / pooled_h)),
        )
        p1 = (
            int(round((sx1 + 1) * image.width / 50.0)) - 1,
            int(round((sy1 + 1) * image.height / pooled_h)) - 1,
        )
        colour = GROUP_COLOURS[
            (record["index"] - 1) % len(GROUP_COLOURS)
        ]
        draw.rectangle([p0, p1], outline=colour, width=2)
        draw.text(
            (p0[0] + 3, p0[1] + 2),
            f"C{record['index']}",
            font=font,
            fill=colour,
            stroke_width=2,
            stroke_fill=(0, 0, 0),
        )
    image.save(output)


def make_contact(records, output):
    cell_width, cell_height = 360, 380
    margin, gap, columns = 18, 12, 3
    rows = math.ceil(len(records) / columns)
    sheet = Image.new(
        "RGB",
        (
            columns * cell_width + (columns + 1) * margin,
            rows * cell_height + (rows + 1) * margin,
        ),
        (245, 245, 245),
    )
    draw = ImageDraw.Draw(sheet)
    font = load_font(14)
    for index, record in enumerate(records):
        image = Image.open(record["overlay_png"]).convert("RGB")
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
        sheet.paste(image, (x + (cell_width - image.width) // 2, y + 5))
        draw.text(
            (x + 7, y + cell_height - 34),
            (
                f"C{record['index']:02d} F={record['objective']:.4f} "
                f"points={record['selected_point_count']}"
            ),
            font=font,
            fill=(45, 45, 45),
        )
    sheet.save(output)
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--names", default=DEFAULT_NAMES)
    parser.add_argument("--distance-weight", type=float, default=1.3)
    parser.add_argument("--min-candidates-for-removal", type=int, default=30)
    parser.add_argument("--max-initial-selected-for-removal", type=int, default=3)
    parser.add_argument("--near-centre-fraction", type=float, default=0.3)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument(
        "--intersections",
        type=Path,
        default=DEFAULT_INPUT
        / "rowcol_intersections"
        / "intersections.json",
    )
    parser.add_argument(
        "--roi-json",
        type=Path,
        default=DEFAULT_INPUT / "roi_annotation.json",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_INPUT / "center_objective",
    )
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    names = [name for name in args.names.split(",") if name]
    intersection_doc = json.loads(
        args.intersections.read_text(encoding="utf-8")
    )
    intersections = {
        record["image"]: record["intersections"]
        for record in intersection_doc["records"]
    }
    rois = {
        record["image"]: record
        for record in json.loads(
            args.roi_json.read_text(encoding="utf-8")
        )["records"]
    }

    center_rows = []
    point_rows = []
    group_rows = []
    removed_rows = []
    nearest_removed_rows = []
    result_records = []
    contact_records = []
    for name in names:
        source, candidates = load_candidates(
            args.input / f"peakscolor50lipcut_{name}.json"
        )
        width, pooled_h = (int(value) for value in source["size"])
        centres = [
            {**centre, "index": index}
            for index, centre in enumerate(
                intersections[name], start=1
            )
        ]
        roi = rois[name]
        center_records = []
        print(f"{name}: {len(centres)} centres", flush=True)
        for centre in centres:
            rectangle, desired, neighbours, half_width, half_height = (
                choose_local_rectangle(
                    centre, centres, pooled_h, width
                )
            )
            desired_height = max(1, desired[3] - desired[1])
            bottom_clip_fraction = (
                max(0, desired[3] - rectangle[3]) / desired_height
            )
            rectangle_local = [
                point for point in candidates
                if rectangle[0] <= point["column"] <= rectangle[2]
                and rectangle[1] <= point["row"] <= rectangle[3]
            ]
            if not rectangle_local:
                raise RuntimeError(
                    f"{name} C{centre['index']}: no candidate points"
                )
            minimum_retained_points = math.ceil(
                0.5 * len(rectangle_local)
            )
            nearest_removed_point = min(
                rectangle_local,
                key=lambda point: (
                    math.hypot(
                        point["row"] - centre["row"],
                        point["column"] - centre["column"],
                    ),
                    point["row"],
                    point["column"],
                ),
            )
            nearest_removed_point = {
                **nearest_removed_point,
                "distance": round(
                    math.hypot(
                        nearest_removed_point["row"] - centre["row"],
                        nearest_removed_point["column"]
                        - centre["column"],
                    ),
                    9,
                ),
            }
            initial_local = [
                point for point in rectangle_local
                if (
                    point["row"],
                    point["column"],
                )
                != (
                    nearest_removed_point["row"],
                    nearest_removed_point["column"],
                )
            ]
            if not initial_local:
                raise RuntimeError(
                    f"{name} C{centre['index']}: nearest removal left "
                    "no candidate points"
                )
            (
                initial_components,
                initial_component_inventory,
                initial_groups,
            ) = (
                curve.preprocess_groups(
                    [
                        (point["row"], point["column"])
                        for point in initial_local
                    ]
                )
            )
            _, initial_inventory, initial_d_ref = curve.make_inventory(
                initial_local,
                initial_groups,
                float(centre["row"]),
                float(centre["column"]),
            )
            initial_best = evaluate_inventory_minimum(
                initial_inventory,
                initial_d_ref,
                args.distance_weight,
                minimum_retained_points,
            )
            initial_selected_group_ids = set(
                initial_best["group_ids"]
            )
            initial_selected_cells = {
                (
                    initial_local[index]["row"],
                    initial_local[index]["column"],
                )
                for group in initial_inventory
                if group["group_id"] in initial_selected_group_ids
                for index in group["points"]
            }
            initial_selected_points = [
                point for point in initial_local
                if (point["row"], point["column"])
                in initial_selected_cells
            ]
            initial_selected_distances = [
                math.hypot(
                    point["row"] - centre["row"],
                    point["column"] - centre["column"],
                )
                for point in initial_selected_points
            ]
            rectangle_short_side = min(
                rectangle[2] - rectangle[0],
                rectangle[3] - rectangle[1],
            )
            near_center_limit = (
                args.near_centre_fraction * rectangle_short_side
            )
            removal_applied = (
                len(initial_local)
                >= args.min_candidates_for_removal
                and len(initial_selected_points)
                <= args.max_initial_selected_for_removal
                and bool(initial_selected_distances)
                and max(initial_selected_distances) <= near_center_limit
            )
            removed_near_points = []
            if removal_applied:
                initial_cell_to_group = {
                    cell: group
                    for group in initial_groups
                    for cell in group["cells"]
                }
                for point in initial_selected_points:
                    group = initial_cell_to_group[
                        (point["row"], point["column"])
                    ]
                    removed_near_points.append(
                        {
                            **point,
                            "group_id": group["group_id"],
                            "group_size": len(group["cells"]),
                            "source_shape": group["source_shape"],
                            "segment_kind": group["segment_kind"],
                        }
                    )
                local = [
                    point for point in initial_local
                    if (point["row"], point["column"])
                    not in initial_selected_cells
                ]
                if not local:
                    raise RuntimeError(
                        f"{name} C{centre['index']}: removal left no points"
                    )
                components, component_inventory, groups = (
                    curve.preprocess_groups(
                        [
                            (point["row"], point["column"])
                            for point in local
                        ]
                    )
                )
                _, inventory, d_ref = curve.make_inventory(
                    local,
                    groups,
                    float(centre["row"]),
                    float(centre["column"]),
                )
                best = evaluate_inventory_minimum(
                    inventory,
                    d_ref,
                    args.distance_weight,
                    minimum_retained_points,
                )
            else:
                local = initial_local
                components = initial_components
                component_inventory = initial_component_inventory
                groups = initial_groups
                inventory = initial_inventory
                d_ref = initial_d_ref
                best = initial_best

            selected_group_ids = set(best["group_ids"])
            selected_cells = {
                (local[index]["row"], local[index]["column"])
                for group in inventory
                if group["group_id"] in selected_group_ids
                for index in group["points"]
            }
            selected_points = [
                point for point in local
                if (point["row"], point["column"]) in selected_cells
            ]
            overlay = args.out / (
                f"center_{name}_C{centre['index']:02d}.png"
            )
            draw_center_crop(
                name,
                centre,
                local,
                selected_cells,
                groups,
                rectangle,
                roi,
                pooled_h,
                best,
                overlay,
            )
            distances = sorted(
                (
                    float(
                        math.hypot(
                            centre["row"] - other["row"],
                            centre["column"] - other["column"],
                        )
                    ),
                    other["index"],
                )
                for other in centres
                if other["index"] != centre["index"]
            )
            d1, nearest_index = distances[0]
            d2 = distances[1][0] if len(distances) > 1 else None
            family_counts = {
                family: sum(
                    point["family"] == family
                    for point in selected_points
                )
                for family in ("red", "blue", "white")
            }
            record = {
                "index": centre["index"],
                "row": int(centre["row"]),
                "column": int(centre["column"]),
                "nearest_center_distance": round(d1, 6),
                "nearest_center_index": nearest_index,
                "second_nearest_distance": (
                    round(d2, 6) if d2 is not None else None
                ),
                "directional_neighbours": {
                    key: (
                        int(value["index"]) if value is not None else None
                    )
                    for key, value in neighbours.items()
                },
                "half_width_cells": half_width,
                "half_height_cells": half_height,
                "desired_rectangle": list(desired),
                "bottom_clip_fraction": round(
                    bottom_clip_fraction, 9
                ),
                "rectangle": list(rectangle),
                "width_cells": rectangle[2] - rectangle[0],
                "height_cells": rectangle[3] - rectangle[1],
                "edge_clipped": list(rectangle) != list(desired),
                "rectangle_candidate_points": len(rectangle_local),
                "minimum_retained_fraction": 0.5,
                "minimum_retained_points": minimum_retained_points,
                "nearest_center_pre_removed_point": (
                    nearest_removed_point
                ),
                "candidate_points_after_nearest_removal": len(
                    initial_local
                ),
                "initial_candidate_points": len(initial_local),
                "initial_objective": round(
                    float(initial_best["objective"]), 9
                ),
                "initial_selected_point_count": len(
                    initial_selected_points
                ),
                "near_center_removal_applied": removal_applied,
                "near_center_distance_limit": round(
                    near_center_limit, 6
                ),
                "removed_near_center_point_count": len(
                    removed_near_points
                ),
                "removed_near_center_points": removed_near_points,
                "candidate_points": len(local),
                "candidate_family_counts": {
                    family: sum(
                        point["family"] == family for point in local
                    )
                    for family in ("red", "blue", "white")
                },
                "component_count": len(components),
                "branched_component_count": sum(
                    item["shape"] == "branched"
                    for item in component_inventory
                ),
                "preprocessed_group_count": len(groups),
                "max_group_size": max(
                    len(group["cells"]) for group in groups
                ),
                "objective": round(float(best["objective"]), 9),
                "open_angle": round(float(best["open_angle"]), 6),
                "average_normalized_distance": round(
                    float(best["average_normalized_distance"]), 9
                ),
                "selected_point_count": len(selected_points),
                "minimum_retained_constraint_met": (
                    len(selected_points) >= minimum_retained_points
                ),
                "selected_group_count": len(selected_group_ids),
                "selected_group_ids": sorted(selected_group_ids),
                "selected_family_counts": family_counts,
                "selected_points": selected_points,
                "overlay_png": str(overlay),
            }
            nearest_removed_rows.append(
                [
                    name,
                    record["index"],
                    nearest_removed_point["row"],
                    nearest_removed_point["column"],
                    nearest_removed_point["family"],
                    nearest_removed_point["distance"],
                    record["rectangle_candidate_points"],
                    record["candidate_points_after_nearest_removal"],
                ]
            )
            center_records.append(record)
            contact_records.append(record)
            center_rows.append(
                [
                    name,
                    record["index"],
                    record["row"],
                    record["column"],
                    *record["rectangle"],
                    *record["desired_rectangle"],
                    str(record["edge_clipped"]).lower(),
                    f"{record['bottom_clip_fraction']:.9f}",
                    record["width_cells"],
                    record["height_cells"],
                    record["rectangle_candidate_points"],
                    record["minimum_retained_points"],
                    nearest_removed_point["row"],
                    nearest_removed_point["column"],
                    nearest_removed_point["family"],
                    nearest_removed_point["distance"],
                    record["candidate_points_after_nearest_removal"],
                    record["initial_candidate_points"],
                    record["initial_selected_point_count"],
                    record["initial_objective"],
                    str(record["near_center_removal_applied"]).lower(),
                    record["removed_near_center_point_count"],
                    record["candidate_points"],
                    record["preprocessed_group_count"],
                    record["objective"],
                    record["open_angle"],
                    record["average_normalized_distance"],
                    record["selected_point_count"],
                    record["selected_group_count"],
                    family_counts["red"],
                    family_counts["blue"],
                    family_counts["white"],
                ]
            )
            cell_to_group = {
                cell: group
                for group in groups
                for cell in group["cells"]
            }
            for point in selected_points:
                group = cell_to_group[
                    (point["row"], point["column"])
                ]
                point_rows.append(
                    [
                        name,
                        record["index"],
                        point["row"],
                        point["column"],
                        point["family"],
                        group["group_id"],
                        len(group["cells"]),
                        group["source_shape"],
                        group["segment_kind"],
                    ]
                )
            for point in removed_near_points:
                removed_rows.append(
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
                    ]
                )
            for group in groups:
                group_rows.append(
                    [
                        name,
                        record["index"],
                        group["group_id"],
                        group["component_id"],
                        len(group["cells"]),
                        group["source_shape"],
                        group["segment_kind"],
                        (
                            "true"
                            if group["group_id"] in selected_group_ids
                            else "false"
                        ),
                        " ".join(
                            f"{row},{column}"
                            for row, column in group["cells"]
                        ),
                    ]
                )
            print(
                f"  C{record['index']}: rect={record['width_cells']}x"
                f"{record['height_cells']} "
                f"local={record['rectangle_candidate_points']}->"
                f"{record['candidate_points_after_nearest_removal']}->"
                f"{len(local)} "
                f"groups={len(groups)} F={record['objective']:.6f} "
                f"kept={len(selected_points)} "
                f"removed_near={record['removed_near_center_point_count']}",
                flush=True,
            )

        overview = args.out / f"center_rectangles_{name}.png"
        draw_overview(name, center_records, roi, pooled_h, overview)
        contact = args.out / f"center_contact_{name}.png"
        make_contact(center_records, contact)
        result_records.append(
            {
                "image": name,
                "distance_weight": args.distance_weight,
                "centres": center_records,
                "center_rectangles_png": str(overview),
                "contact_png": str(contact),
            }
        )

    with (args.out / "center_objective.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "image",
                "centre_index",
                "row",
                "column",
                "rectangle_left",
                "rectangle_top",
                "rectangle_right",
                "rectangle_bottom",
                "desired_rectangle_left",
                "desired_rectangle_top",
                "desired_rectangle_right",
                "desired_rectangle_bottom",
                "edge_clipped",
                "bottom_clip_fraction",
                "width_cells",
                "height_cells",
                "rectangle_candidate_points",
                "minimum_retained_points",
                "nearest_removed_row",
                "nearest_removed_column",
                "nearest_removed_family",
                "nearest_removed_distance_cells",
                "candidate_points_after_nearest_removal",
                "initial_candidate_points",
                "initial_selected_points",
                "initial_objective",
                "near_center_removal_applied",
                "removed_near_center_points",
                "candidate_points",
                "groups",
                "objective",
                "open_angle",
                "average_normalized_distance",
                "selected_points",
                "selected_groups",
                "red",
                "blue",
                "white",
            ]
        )
        writer.writerows(center_rows)
    with (args.out / "selected_points.csv").open(
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
            ]
        )
        writer.writerows(point_rows)
    with (args.out / "removed_near_center_points.csv").open(
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
            ]
        )
        writer.writerows(removed_rows)
    with (args.out / "nearest_center_removed_points.csv").open(
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
                "distance_cells",
                "rectangle_candidate_points",
                "candidates_after_removal",
            ]
        )
        writer.writerows(nearest_removed_rows)
    with (args.out / "preprocessed_groups.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "image",
                "centre_index",
                "group_id",
                "component_id",
                "group_size",
                "source_shape",
                "segment_kind",
                "selected",
                "cells_row_col",
            ]
        )
        writer.writerows(group_rows)
    (args.out / "result.json").write_text(
        json.dumps(
            {
                "distance_weight": args.distance_weight,
                "objective_formula": (
                    "F = open_angle / 360 + distance_weight * "
                    "average_normalized_distance"
                ),
                "open_angle_endpoint_padding": (
                    "atan(0.707 / corresponding endpoint distance)"
                ),
                "near_center_removal_rule": {
                    "min_candidates": args.min_candidates_for_removal,
                    "max_initial_selected": (
                        args.max_initial_selected_for_removal
                    ),
                    "near_centre_fraction": args.near_centre_fraction,
                    "passes": 1,
                },
                "nearest_center_pre_removal": (
                    "remove one nearest annotation point before "
                    "curve grouping and objective calculation"
                ),
                "minimum_retained_rule": {
                    "fraction": 0.5,
                    "rounding": "ceil(rectangle_candidate_points * 0.5)",
                },
                "records": result_records,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    make_contact(contact_records, args.out / "all_center_contact.png")

    lines = [
        "# 当前中心点目标函数结果",
        "",
        f"- 距离惩罚：{args.distance_weight:g}",
        (
            "- 目标函数：F = open_angle / 360 + "
            f"{args.distance_weight:g} × 平均归一化距离。"
        ),
        "- 开放角端点扩展：atan(0.707 / 对应端点距离)。",
        "- 连续点集最多 5 点；有分支点集拆分为无分支路径和单点分支。",
        "- 矩形由上下左右最近中心点确定，不包含其他中心点；超出边界时直接裁到边界。",
        "- 计算前先剔除矩形内距离中心最近的一个红、蓝或白标注点。",
        "- 保留点数量不得少于矩形内原始标注点数量的 1/2，按向上取整。",
        (
            "- 近中心低保留过滤：候选点至少 "
            f"{args.min_candidates_for_removal}，首次仅保留不超过 "
            f"{args.max_initial_selected_for_removal} 个近中心点时，"
            "剔除这些点后重新计算一次。"
        ),
        "",
        "| 图片 | 中心 | C1 | C2 | C3 | C4 | C5 | C6 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for record in result_records:
        values = [
            centre["selected_point_count"]
            for centre in record["centres"]
        ]
        lines.append(
            f"| {record['image']} | {len(values)} | "
            + " | ".join(str(value) for value in values)
            + " |"
        )
    lines.extend(
        [
            "",
            "表中的数值为每个中心点最终保留的标注点数量。",
        ]
    )
    (args.out / "summary.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
