"""Curve-aware annotation grouping followed by the objective search.

All kept red, blue and white annotations participate.  Each 8-connected
component is first classified as an unbranched open curve, a closed curve or a
branched structure:

  * unbranched components are traversed in curve order and split into
    contiguous groups of at most five points;
  * for branched components, branch vertices become one-point degenerate
    curves, and the remaining graph is split into unbranched path components;
    each path is also split into groups of at most five points.

Every resulting group is indivisible inside the objective:

    F = open_angle / 360 + 2.0 * average_normalized_distance

The objective is evaluated independently for every retained row/column
intersection.  Since preprocessed groups are small and numerous, the exact
objective evaluator is combined with a deterministic multi-start group search.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import deque
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import optimize_007  # noqa: E402


def imread(path: Path):
    image = cv2.imdecode(
        np.fromfile(str(path), np.uint8), cv2.IMREAD_COLOR
    )
    if image is None:
        raise RuntimeError(f"cannot read {path}")
    return image


def imwrite(path: Path, image: np.ndarray):
    ok, buffer = cv2.imencode(".png", image)
    if not ok:
        raise RuntimeError(f"cannot encode {path}")
    buffer.tofile(str(path))


def load_font(size):
    path = Path(r"C:\Windows\Fonts\msyh.ttc")
    return (
        ImageFont.truetype(str(path), size)
        if path.exists()
        else ImageFont.load_default(size=size)
    )


def neighbours(cell, cells):
    row, col = cell
    out = []
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            other = (row + dr, col + dc)
            if other in cells:
                out.append(other)
    return sorted(out)


def connected_components(cells):
    remaining = set(cells)
    components = []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        queue = deque([start])
        component = []
        while queue:
            cell = queue.popleft()
            component.append(cell)
            for other in neighbours(cell, cells):
                if other in remaining:
                    remaining.remove(other)
                    queue.append(other)
        components.append(sorted(component))
    return components


def order_path(cells):
    cells = set(cells)
    if len(cells) <= 1:
        return list(cells)
    adjacency = {cell: neighbours(cell, cells) for cell in cells}
    endpoints = sorted(cell for cell in cells if len(adjacency[cell]) <= 1)
    start = endpoints[0] if endpoints else min(cells)
    ordered = [start]
    visited = {start}
    current = start
    previous = None
    while True:
        choices = [cell for cell in adjacency[current] if cell != previous]
        choices = [cell for cell in choices if cell not in visited] or [
            cell for cell in choices if cell == start and len(visited) < len(cells)
        ]
        if not choices:
            break
        nxt = sorted(choices)[0]
        if nxt == start:
            break
        ordered.append(nxt)
        visited.add(nxt)
        previous, current = current, nxt
    # A malformed/disconnected ordering must not silently drop a point.
    if len(ordered) != len(cells):
        remaining = sorted(cells - visited)
        ordered.extend(remaining)
    return ordered


def split_path(path, limit=5):
    return [path[index:index + limit] for index in range(0, len(path), limit)]


def preprocess_groups(cells):
    cell_set = set(cells)
    components = connected_components(cell_set)
    groups = []
    component_rows = []

    for component_id, component in enumerate(components, start=1):
        component_set = set(component)
        adjacency = {
            cell: neighbours(cell, component_set) for cell in component
        }
        degrees = {cell: len(adjacency[cell]) for cell in component}
        branches = sorted(
            cell for cell in component if degrees[cell] > 2
        )
        branch_set = set(branches)
        if branches:
            shape = "branched"
            rest = component_set - branch_set
            for branch in branches:
                groups.append(
                    {
                        "component_id": component_id,
                        "source_shape": shape,
                        "segment_kind": "branch_singleton",
                        "cells": [branch],
                    }
                )
            unseen = set(rest)
            while unseen:
                start = min(unseen)
                unseen.remove(start)
                queue = deque([start])
                part = []
                while queue:
                    cell = queue.popleft()
                    part.append(cell)
                    for other in neighbours(cell, rest):
                        if other in unseen:
                            unseen.remove(other)
                            queue.append(other)
                path = order_path(part)
                for chunk in split_path(path):
                    groups.append(
                        {
                            "component_id": component_id,
                            "source_shape": shape,
                            "segment_kind": "unbranched_path",
                            "cells": chunk,
                        }
                    )
        else:
            shape = (
                "closed_curve"
                if len(component) > 2
                and all(degrees[cell] == 2 for cell in component)
                else "open_curve"
            )
            path = order_path(component)
            for chunk in split_path(path):
                groups.append(
                    {
                        "component_id": component_id,
                        "source_shape": shape,
                        "segment_kind": shape,
                        "cells": chunk,
                    }
                )
        component_rows.append(
            {
                "component_id": component_id,
                "point_count": len(component),
                "shape": shape,
                "max_degree": max(degrees.values()),
                "branch_points": [list(cell) for cell in branches],
                "endpoint_count": sum(
                    degree <= 1 for degree in degrees.values()
                ),
            }
        )

    for group_id, group in enumerate(groups, start=1):
        group["group_id"] = group_id
    return components, component_rows, groups


def make_inventory(candidate_points, groups, centre_row, centre_col):
    point_index = {
        (point["row"], point["column"]): index
        for index, point in enumerate(candidate_points)
    }
    points = []
    for index, point in enumerate(candidate_points):
        dr = point["row"] - centre_row
        dc = point["column"] - centre_col
        points.append(
            {
                **point,
                "source_row": point["row"],
                "source_column": point["column"],
                "index": index,
                "row_relative": dr,
                "column_relative": dc,
                "row": dr,
                "col": dc,
                "x": dc,
                "y": dr,
                "distance": float(math.hypot(dr, dc)),
                "angle": float(
                    math.degrees(math.atan2(dr, dc)) % 360.0
                ),
            }
        )
    d_ref = max(point["distance"] for point in points)
    inventory = []
    for group in groups:
        indices = [
            point_index[cell]
            for cell in group["cells"]
            if cell in point_index
        ]
        if not indices:
            continue
        ordered_indices = sorted(
            indices, key=lambda index: points[index]["angle"]
        )
        angles = [points[index]["angle"] for index in ordered_indices]
        distance_sum = sum(points[index]["distance"] for index in indices)
        circular = np.asarray(angles + [angles[0] + 360.0])
        gaps = np.diff(circular)
        largest_gap = int(np.argmax(gaps))
        first = (largest_gap + 1) % len(angles)
        last = (first + len(angles) - 1) % len(angles)
        left_distance = max(
            points[ordered_indices[first]]["distance"], 1e-9
        )
        right_distance = max(
            points[ordered_indices[last]]["distance"], 1e-9
        )
        left_pad = math.degrees(math.atan(0.707 / left_distance))
        right_pad = math.degrees(math.atan(0.707 / right_distance))
        raw_contribution = 360.0 - float(gaps[largest_gap])
        if len(indices) == 1:
            entry = {
                "kind": "isolated",
                "angle_half_width": (left_pad + right_pad) / 2.0,
                "angle_min": angles[0],
                "angle_max": angles[0],
                "angle_contribution": left_pad + right_pad,
                "angle_wraps_zero": (
                    angles[0] - (left_pad + right_pad) / 2.0 < 0
                    or angles[0] + (left_pad + right_pad) / 2.0 > 360.0
                ),
            }
        else:
            start = (angles[first] - left_pad) % 360.0
            contribution = raw_contribution + left_pad + right_pad
            entry = {
                "kind": "continuous",
                "angle_half_width": None,
                "angle_min": start,
                "angle_max": (start + contribution) % 360.0,
                "angle_contribution": contribution,
                "angle_wraps_zero": start + contribution > 360.0 + 1e-10,
            }
        entry.update(
            {
                "group_id": group["group_id"],
                "points": indices,
                "size": len(indices),
                "distance_sum": distance_sum,
                "raw_angle_contribution": raw_contribution,
                "left_endpoint_pad": left_pad,
                "right_endpoint_pad": right_pad,
                "left_endpoint_distance": left_distance,
                "right_endpoint_distance": right_distance,
                "segment_kind": group["segment_kind"],
                "source_shape": group["source_shape"],
                "bounding_rows": [
                    min(cell[0] for cell in group["cells"]),
                    max(cell[0] for cell in group["cells"]),
                ],
                "bounding_cols": [
                    min(cell[1] for cell in group["cells"]),
                    max(cell[1] for cell in group["cells"]),
                ],
            }
        )
        inventory.append(entry)
    return points, inventory, d_ref


def evaluate_inventory(inventory, d_ref, weight=2.0):
    if not inventory:
        raise RuntimeError("no annotation groups")
    nearest = sorted(
        range(len(inventory)),
        key=lambda index: inventory[index]["distance_sum"] / inventory[index]["size"],
    )
    # With hundreds of five-point groups, a small nearest-group seed set keeps
    # the exhaustive per-addition evaluation tractable while avoiding a
    # degenerate zero-angle seed.  The evaluator remains exact.
    seeds = nearest[:5]
    best = None
    best_by_k = {}

    def consider(result):
        nonlocal best
        if result["redundant_group_ids"]:
            return
        k = result["k"]
        if k not in best_by_k or result["objective"] < best_by_k[k]["objective"]:
            best_by_k[k] = result
        if best is None or result["objective"] < best["objective"]:
            best = result

    for seed in seeds:
        current = optimize_007.evaluate_mask(
            1 << seed, inventory, d_ref, distance_weight=weight
        )
        consider(current)
        while True:
            step = None
            for index in range(len(inventory)):
                bit = 1 << index
                if current["mask"] & bit:
                    continue
                candidate = optimize_007.evaluate_mask(
                    current["mask"] | bit,
                    inventory,
                    d_ref,
                    distance_weight=weight,
                )
                consider(candidate)
                if candidate["redundant_group_ids"]:
                    continue
                if step is None or candidate["objective"] < step["objective"]:
                    step = candidate
            if step is None or step["objective"] >= current["objective"]:
                break
            current = step
            consider(current)
    return best, best_by_k


def draw_overlay(name, rois, h50, centres, selected_cells, all_points, output):
    image = imread(ROOT / "outfinal" / f"armmid_{name}.png")
    left, top, right, bottom = rois[name]["rectangle_px"]
    width, height = right - left, bottom - top
    cv2.rectangle(
        image, (left, top), (right - 1, bottom - 1),
        (0, 255, 255), 2, cv2.LINE_AA
    )
    colour_by_family = {
        "red": (0, 0, 255),
        "blue": (255, 0, 0),
        "white": (255, 255, 255),
    }
    for point in all_points:
        x = left + int(round(point["column"] * width / 50.0))
        y = top + int(round(point["row"] * height / float(h50)))
        cell = (point["row"], point["column"])
        if cell in selected_cells:
            cv2.circle(image, (x, y), 4, (0, 0, 0), -1, cv2.LINE_AA)
            cv2.circle(
                image, (x, y), 3, colour_by_family[point["family"]],
                -1, cv2.LINE_AA
            )
        else:
            cv2.circle(image, (x, y), 2, (130, 130, 130), -1, cv2.LINE_AA)
    for centre in centres:
        x = left + int(round(centre["column"] * width / 50.0))
        y = top + int(round(centre["row"] * height / float(h50)))
        cv2.drawMarker(
            image, (x, y), (0, 255, 255),
            cv2.MARKER_CROSS, 18, 2, cv2.LINE_AA
        )
        cv2.circle(image, (x, y), 8, (0, 255, 255), 2, cv2.LINE_AA)
    path = output / f"curve_objective_selected_{name}.png"
    imwrite(path, image)
    return path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--names", default="001,006,020,026,036")
    parser.add_argument("--distance-weight", type=float, default=2.0)
    parser.add_argument(
        "--input",
        type=Path,
        default=ROOT / "outfinal" / "roi_pool50_peakkernel",
    )
    parser.add_argument(
        "--intersections",
        type=Path,
        default=ROOT / "outfinal" / "roi_rowcol_intersections"
        / "rowcol_intersections.json",
    )
    parser.add_argument(
        "--roi-json",
        type=Path,
        default=ROOT / "outfinal" / "abdomen_roi" / "armmid_roi.json",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "outfinal" / "roi_curve_group_objective",
    )
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    names = [name for name in args.names.split(",") if name]
    intersections = {
        record["image"]: record
        for record in json.loads(
            args.intersections.read_text(encoding="utf-8")
        )["records"]
    }
    rois = {
        record["image"]: record
        for record in json.loads(args.roi_json.read_text(encoding="utf-8"))[
            "records"
        ]
    }

    result_records = []
    centre_rows = []
    point_rows = []
    for name in names:
        source = json.loads(
            (args.input / f"kernel5_lit_{name}.json").read_text(
                encoding="utf-8"
            )
        )
        candidates = [
            {
                "row": int(y),
                "column": int(x),
                "family": family,
            }
            for x, y, family, keep, darkness in source["points"]
            if keep
        ]
        components, component_inventory, groups = preprocess_groups(
            [(point["row"], point["column"]) for point in candidates]
        )
        cell_to_group = {
            cell: group["group_id"]
            for group in groups
            for cell in group["cells"]
        }
        centres = []
        selected_memberships = {}
        for centre_index, centre in enumerate(
            intersections[name]["intersections"], start=1
        ):
            print(
                f"  {name} centre {centre_index}/"
                f"{len(intersections[name]['intersections'])} "
                f"({centre['row']},{centre['column']})",
                flush=True,
            )
            points, inventory, d_ref = make_inventory(
                candidates,
                groups,
                float(centre["row"]),
                float(centre["column"]),
            )
            best, best_by_k = evaluate_inventory(
                inventory, d_ref, args.distance_weight
            )
            selected_group_ids = set(best["group_ids"])
            selected_point_indices = sorted(
                index
                for group in inventory
                if group["group_id"] in selected_group_ids
                for index in group["points"]
            )
            selected_cells = []
            for index in selected_point_indices:
                point = points[index]
                cell = (point["source_row"], point["source_column"])
                selected_cells.append(cell)
                selected_memberships.setdefault(cell, set()).add(centre_index)
            centre_payload = {
                "index": centre_index,
                "row": centre["row"],
                "column": centre["column"],
                "objective": round(float(best["objective"]), 9),
                "open_angle": round(float(best["open_angle"]), 6),
                "average_normalized_distance": round(
                    float(best["average_normalized_distance"]), 9
                ),
                "selected_points": int(best["k"]),
                "selected_groups": len(selected_group_ids),
                "selected_group_ids": sorted(selected_group_ids),
            }
            centres.append(centre_payload)
            centre_rows.append(
                [
                    name,
                    centre_index,
                    centre["row"],
                    centre["column"],
                    centre_payload["objective"],
                    centre_payload["open_angle"],
                    centre_payload["average_normalized_distance"],
                    centre_payload["selected_points"],
                    centre_payload["selected_groups"],
                ]
            )
        selected_points = []
        point_lookup = {
            (point["row"], point["column"]): point for point in candidates
        }
        for cell, members in sorted(selected_memberships.items()):
            point = point_lookup[cell]
            group = groups[cell_to_group[cell] - 1]
            selected_points.append(
                {
                    "row": cell[0],
                    "column": cell[1],
                    "family": point["family"],
                    "group_id": group["group_id"],
                    "group_size": len(group["cells"]),
                    "source_shape": group["source_shape"],
                    "segment_kind": group["segment_kind"],
                    "centre_memberships": sorted(members),
                }
            )
            point_rows.append(
                [
                    name,
                    cell[0],
                    cell[1],
                    point["family"],
                    group["group_id"],
                    len(group["cells"]),
                    group["source_shape"],
                    group["segment_kind"],
                    " ".join(map(str, sorted(members))),
                ]
            )
        pooled_h = int(source["size"][1])
        overlay = draw_overlay(
            name,
            rois,
            pooled_h,
            intersections[name]["intersections"],
            set(selected_memberships),
            candidates,
            args.out,
        )
        selected_family_counts = {
            family: sum(
                point["family"] == family for point in selected_points
            )
            for family in ("red", "blue", "white")
        }
        record = {
            "image": name,
            "distance_weight": args.distance_weight,
            "total_candidate_points": len(candidates),
            "candidate_family_counts": {
                family: sum(point["family"] == family for point in candidates)
                for family in ("red", "blue", "white")
            },
            "components": component_inventory,
            "component_count": len(components),
            "unbranched_component_count": sum(
                component["shape"] != "branched"
                for component in component_inventory
            ),
            "branched_component_count": sum(
                component["shape"] == "branched"
                for component in component_inventory
            ),
            "preprocessed_group_count": len(groups),
            "max_group_size": max(len(group["cells"]) for group in groups),
            "preprocessed_groups": groups,
            "centres": centres,
            "mean_objective": round(
                float(np.mean([centre["objective"] for centre in centres])), 9
            ),
            "selected_points": selected_points,
            "selected_point_count": len(selected_points),
            "selected_family_counts": selected_family_counts,
            "overlay_png": str(overlay),
        }
        result_records.append(record)
        print(
            f"{name}: components={record['component_count']} "
            f"(branch={record['branched_component_count']}), "
            f"groups={record['preprocessed_group_count']}, "
            f"centres={len(centres)}, meanF={record['mean_objective']:.6f}, "
            f"selected={record['selected_point_count']}"
        )

    with (args.out / "curve_groups.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "image",
                "component_id",
                "point_count",
                "shape",
                "max_degree",
                "branch_points",
            ]
        )
        for record in result_records:
            for component in record["components"]:
                writer.writerow(
                    [
                        record["image"],
                        component["component_id"],
                        component["point_count"],
                        component["shape"],
                        component["max_degree"],
                        " ".join(
                            f"{row},{col}"
                            for row, col in component["branch_points"]
                        ),
                    ]
                )
    with (args.out / "preprocessed_groups.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "image",
                "group_id",
                "component_id",
                "group_size",
                "source_shape",
                "segment_kind",
                "cells_row_col",
            ]
        )
        for record in result_records:
            for group in record["preprocessed_groups"]:
                writer.writerow(
                    [
                        record["image"],
                        group["group_id"],
                        group["component_id"],
                        len(group["cells"]),
                        group["source_shape"],
                        group["segment_kind"],
                        " ".join(
                            f"{row},{col}"
                            for row, col in group["cells"]
                        ),
                    ]
                )
    with (args.out / "objective_by_centre.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "image",
                "centre_index",
                "row",
                "column",
                "objective",
                "open_angle",
                "average_normalized_distance",
                "selected_points",
                "selected_groups",
            ]
        )
        writer.writerows(centre_rows)
    with (args.out / "selected_points.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "image",
                "row",
                "column",
                "family",
                "group_id",
                "group_size",
                "source_shape",
                "segment_kind",
                "centre_memberships",
            ]
        )
        writer.writerows(point_rows)
    (args.out / "result.json").write_text(
        json.dumps({"records": result_records}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    lines = [
        "# 曲线组预处理与目标函数",
        "",
        "- 红、蓝、白标注点全部参与。",
        "- 8 连通无分支曲线按顺序切成每组最多 5 点。",
        "- 有分支的集合在分支点处分段；分支点作为单点退化曲线。",
        "- 距离惩罚：2。",
        "",
        "| 图片 | 点数 | 连通集合 | 分支集合 | 预处理组 | 中心 | 平均 F | 选中点 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for record in result_records:
        lines.append(
            f"| {record['image']} | {record['total_candidate_points']} | "
            f"{record['component_count']} | "
            f"{record['branched_component_count']} | "
            f"{record['preprocessed_group_count']} | "
            f"{len(record['centres'])} | {record['mean_objective']:.6f} | "
            f"{record['selected_point_count']} |"
        )
    (args.out / "summary.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
