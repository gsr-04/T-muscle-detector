"""Select whole 8-connected annotation groups around the centre of 007.png.

All red and blue annotation marks are first detected. They are then split into
indivisible 8-connected components. A component may be selected as a whole or
omitted, but no point inside a component may be selected alone.

Objective:
    F = open_angle / 360
        + distance_weight * mean(distance / max_candidate_distance)

For each selected component:
    locate the two angular endpoints by removing the largest circular gap.
    The angular interval is then expanded at its endpoints:
        left  -= atan(0.707 / distance_of_left_endpoint)
        right += atan(0.707 / distance_of_right_endpoint)
    For a single point both endpoints are that same point.
    all selected angular intervals are merged; overlap counts only once
    open_angle = 360 - union interval length

With only 17 components, all 2^17 group selections are enumerated exactly.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_IMAGE = ROOT / "data" / "007.png"
DEFAULT_OUT = ROOT / "output" / "007_encirclement_groups"
PIXEL_DIAGONAL = math.sqrt(2.0)


def connected_centres(mask: np.ndarray) -> list[tuple[float, float]]:
    """Centres of the circular red/blue annotation marks."""
    count, _, stats, centroids = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8), connectivity=8
    )
    centres = []
    for i in range(1, count):
        _, _, width, height, area = stats[i]
        if area >= 100 and width >= 12 and height >= 12:
            centres.append((float(centroids[i, 0]), float(centroids[i, 1])))
    return centres


def white_centre(rgb: np.ndarray) -> tuple[float, float]:
    """Locate the compact white marker containing the centre label."""
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    mask = (r > 235) & (g > 235) & (b > 235)
    count, _, stats, centroids = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8), connectivity=8
    )
    candidates = []
    for i in range(1, count):
        x, y, width, height, area = stats[i]
        if 15 <= width <= 28 and 15 <= height <= 28 and 150 <= area <= 350:
            candidates.append(
                (
                    float(centroids[i, 0]),
                    float(centroids[i, 1]),
                    int(area),
                    x,
                    y,
                )
            )
    if not candidates:
        raise RuntimeError("未找到中心白色标记")
    cx, cy, _, _, _ = min(candidates, key=lambda item: abs(item[2] - 240))
    return cx, cy


def estimate_grid_step(
    centres: list[tuple[float, float]], centre: tuple[float, float]
) -> float:
    """Estimate lattice pitch from repeated coordinate separations."""
    del centre  # Kept in the signature for callers and future calibration.
    pitches = []
    coords = np.asarray(centres, dtype=float)
    for axis in (0, 1):
        values = np.array(sorted({round(float(v), 1) for v in coords[:, axis]}))
        diffs = np.diff(values)
        pitches.extend(diffs[(diffs > 10.0) & (diffs < 30.0)].tolist())
    if not pitches:
        raise RuntimeError("无法估计网格步长")
    return float(np.median(pitches))


def extract_candidates(image_path: Path):
    rgb = np.asarray(Image.open(image_path).convert("RGB"))
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    red = (r > 200) & (g < 70) & (b < 70)
    blue = (b > 200) & (r < 70) & (g < 70)

    raw = [(x, y, "red") for x, y in connected_centres(red)]
    raw += [(x, y, "blue") for x, y in connected_centres(blue)]
    cx, cy = white_centre(rgb)
    step = estimate_grid_step([(x, y) for x, y, _ in raw], (cx, cy))

    rows = []
    for x, y, colour in raw:
        col = (x - cx) / step
        row = (y - cy) / step
        distance = math.hypot(row, col)
        angle = math.degrees(math.atan2(row, col)) % 360.0
        rows.append(
            {
                "row": row,
                "col": col,
                "x": x,
                "y": y,
                "colour": colour,
                "distance": distance,
                "angle": angle,
            }
        )

    rows.sort(key=lambda p: (p["angle"], p["distance"]))
    for i, point in enumerate(rows):
        point["index"] = i
    d_ref = max(point["distance"] for point in rows)
    return rgb, (cx, cy), step, rows, d_ref


def are_adjacent(a: dict, b: dict) -> bool:
    """Eight-neighbour relation with tolerance for marker-centroid jitter."""
    dr = abs(a["row"] - b["row"])
    dc = abs(a["col"] - b["col"])
    return (
        (dr > 1e-6 or dc > 1e-6)
        and dr <= 1.35
        and dc <= 1.35
    )


def connected_groups(selected: list[int], points: list[dict]) -> list[list[int]]:
    """All 8-connected components of a candidate index set."""
    unseen = set(selected)
    groups = []
    while unseen:
        start = min(unseen, key=lambda i: points[i]["index"])
        stack = [start]
        unseen.remove(start)
        group = []
        while stack:
            i = stack.pop()
            group.append(i)
            for j in list(unseen):
                if are_adjacent(points[i], points[j]):
                    unseen.remove(j)
                    stack.append(j)
        groups.append(sorted(group, key=lambda i: points[i]["index"]))
    return groups


def describe_groups(points: list[dict]):
    groups = connected_groups(list(range(len(points))), points)
    inventory = []
    for group_id, group in enumerate(groups, 1):
        size = len(group)
        distance_sum = sum(points[i]["distance"] for i in group)
        ordered_group = sorted(group, key=lambda i: points[i]["angle"])
        angles = [points[i]["angle"] for i in ordered_group]
        circular = np.asarray(angles + [angles[0] + 360.0])
        gaps = np.diff(circular)
        largest_gap_index = int(np.argmax(gaps))
        first_index = (largest_gap_index + 1) % size
        last_index = (first_index + size - 1) % size
        first_point = points[ordered_group[first_index]]
        last_point = points[ordered_group[last_index]]
        left_pad = math.degrees(
            math.atan(
                0.707 / max(float(first_point["distance"]), 1e-12)
            )
        )
        right_pad = math.degrees(
            math.atan(
                0.707 / max(float(last_point["distance"]), 1e-12)
            )
        )
        raw_contribution = 360.0 - float(gaps[largest_gap_index])
        padded_start = (angles[first_index] - left_pad) % 360.0
        contribution = raw_contribution + left_pad + right_pad
        if size == 1:
            kind = "isolated"
            start_angle = angles[0]
            end_angle = angles[0]
            half_width = (left_pad + right_pad) / 2.0
            wraps = (
                angles[0] - half_width < 0
                or angles[0] + half_width > 360.0
            )
        else:
            kind = "continuous"
            start_angle = padded_start
            end_angle = (padded_start + contribution) % 360.0
            half_width = None
            wraps = padded_start + contribution > 360.0 + 1e-10
        inventory.append(
            {
                "group_id": group_id,
                "kind": kind,
                "points": group,
                "size": size,
                "distance_sum": distance_sum,
                "angle_contribution": contribution,
                "raw_angle_contribution": raw_contribution,
                "left_endpoint_pad": left_pad,
                "right_endpoint_pad": right_pad,
                "left_endpoint_distance": float(first_point["distance"]),
                "right_endpoint_distance": float(last_point["distance"]),
                "angle_half_width": half_width,
                "angle_min": start_angle,
                "angle_max": end_angle,
                "angle_wraps_zero": wraps,
                "bounding_rows": [
                    min(points[i]["row"] for i in group),
                    max(points[i]["row"] for i in group),
                ],
                "bounding_cols": [
                    min(points[i]["col"] for i in group),
                    max(points[i]["col"] for i in group),
                ],
            }
        )
    return groups, inventory


def merge_intervals(intervals):
    merged = []
    for start, end in sorted(intervals):
        if not merged or start > merged[-1][1] + 1e-10:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)
    return merged


def format_angle_interval(group: dict) -> str:
    if group["kind"] == "isolated":
        start = (
            group["angle_min"] - group["angle_half_width"]
        ) % 360.0
        end = (
            group["angle_max"] + group["angle_half_width"]
        ) % 360.0
    else:
        start = group["angle_min"]
        end = group["angle_max"]

    if group["angle_wraps_zero"]:
        return f"[{start:.6f},360] U [0,{end:.6f}]"
    return f"[{start:.6f},{end:.6f}]"


def group_interval_segments(group: dict):
    if group["kind"] == "continuous":
        start = group["angle_min"]
        end = (
            start + group["angle_contribution"]
        ) % 360.0
        if group["angle_wraps_zero"]:
            return [(start, 360.0), (0.0, end)]
        return [(start, end)]

    start = (
        group["angle_min"] - group["angle_half_width"]
    ) % 360.0
    end = (
        group["angle_max"] + group["angle_half_width"]
    ) % 360.0
    if start <= end:
        return [(start, end)]
    return [(0.0, end), (start, 360.0)]


def evaluate_mask(
    mask: int,
    inventory: list[dict],
    d_ref: float,
    distance_weight: float = 1.0,
):
    chosen = [
        group for i, group in enumerate(inventory) if mask & (1 << i)
    ]
    k = sum(group["size"] for group in chosen)
    distance_sum = sum(group["distance_sum"] for group in chosen)
    group_intervals = {}
    for group in chosen:
        group_intervals[group["group_id"]] = group_interval_segments(group)

    intervals = [
        interval
        for group_intervals_for_id in group_intervals.values()
        for interval in group_intervals_for_id
    ]
    merged_intervals = merge_intervals(intervals)
    angle_union_length = sum(end - start for start, end in merged_intervals)
    open_angle = 360.0 - angle_union_length
    average_distance = distance_sum / (k * d_ref)
    objective = (
        open_angle / 360.0 + distance_weight * average_distance
    )

    redundant_group_ids = []
    for group in chosen:
        own_intervals = group_intervals[group["group_id"]]
        other_intervals = sorted(
            interval
            for other in chosen
            if other["group_id"] != group["group_id"]
            for interval in group_intervals[other["group_id"]]
        )
        other_merged = merge_intervals(other_intervals)
        fully_covered = True
        for start, end in own_intervals:
            cursor = start
            for other_start, other_end in other_merged:
                if other_end < cursor - 1e-10:
                    continue
                if other_start > cursor + 1e-10:
                    break
                cursor = max(cursor, other_end)
                if cursor >= end - 1e-10:
                    break
            if cursor < end - 1e-10:
                fully_covered = False
                break
        if fully_covered:
            redundant_group_ids.append(group["group_id"])

    return {
        "mask": mask,
        "group_ids": [group["group_id"] for group in chosen],
        "k": k,
        "objective": objective,
        "open_angle": open_angle,
        "average_normalized_distance": average_distance,
        "angle_union_length": angle_union_length,
        "isolated_nominal_width_sum": sum(
            group["angle_contribution"]
            for group in chosen
            if group["kind"] == "isolated"
        ),
        "merged_angle_intervals": merged_intervals,
        "redundant_group_ids": redundant_group_ids,
        "isolated_group_count": sum(
            group["kind"] == "isolated" for group in chosen
        ),
        "continuous_group_count": sum(
            group["kind"] == "continuous" for group in chosen
        ),
    }


def enumerate_group_selections(
    inventory: list[dict],
    d_ref: float,
    fixed_k=None,
    excluded_group_ids=None,
    distance_weight: float = 1.0,
):
    best_overall = None
    best_by_k = {}
    excluded_group_ids = set(excluded_group_ids or [])
    excluded_mask = sum(
        1 << (group["group_id"] - 1)
        for group in inventory
        if group["group_id"] in excluded_group_ids
    )
    for mask in range(1, 1 << len(inventory)):
        if mask & excluded_mask:
            continue
        result = evaluate_mask(
            mask, inventory, d_ref, distance_weight=distance_weight
        )
        if result["redundant_group_ids"]:
            continue
        if fixed_k is not None and result["k"] != fixed_k:
            continue
        if fixed_k is not None:
            if best_overall is None or result["objective"] < best_overall["objective"]:
                best_overall = result
            continue
        k = result["k"]
        if k not in best_by_k or result["objective"] < best_by_k[k]["objective"]:
            best_by_k[k] = result
        if best_overall is None or result["objective"] < best_overall["objective"]:
            best_overall = result
    if best_overall is None:
        raise RuntimeError(f"不存在总点数恰为 {fixed_k} 的整组选择方案")
    if fixed_k is not None:
        best_by_k[fixed_k] = best_overall
    return best_overall, best_by_k


def solve_group_selections_milp(
    inventory: list[dict],
    d_ref: float,
    fixed_k=None,
    excluded_group_ids=None,
    distance_weight: float = 1.0,
):
    """Exact whole-group selection for larger component counts.

    The angular axis is split at every interval endpoint. Coverage variables
    represent the union on each elementary segment. Additional binary
    variables enforce that every selected group owns at least one segment not
    covered by any other selected group.
    """
    excluded_group_ids = set(excluded_group_ids or [])
    groups = [
        group for group in inventory
        if group["group_id"] not in excluded_group_ids
    ]
    m = len(groups)
    group_intervals = [
        group_interval_segments(group) for group in groups
    ]

    breakpoints = {0.0, 360.0}
    for intervals in group_intervals:
        for start, end in intervals:
            breakpoints.add(start)
            breakpoints.add(end)
    breakpoints = sorted(breakpoints)

    segments = []
    for start, end in zip(breakpoints, breakpoints[1:]):
        if end <= start + 1e-10:
            continue
        covering = [
            g for g, intervals in enumerate(group_intervals)
            if any(s <= start + 1e-10 and e >= end - 1e-10
                   for s, e in intervals)
        ]
        if covering:
            segments.append((start, end, covering))

    y_start = m
    u_start = y_start + len(segments)
    unique_vars = {}
    variable_count = u_start
    for g, intervals in enumerate(group_intervals):
        for t, (_, _, covering) in enumerate(segments):
            if g in covering:
                unique_vars[g, t] = variable_count
                variable_count += 1

    rows = []
    lower = []
    upper = []

    def add_constraint(coefficients, lb, ub):
        rows.append(coefficients)
        lower.append(lb)
        upper.append(ub)

    for t, (_, _, covering) in enumerate(segments):
        y = y_start + t
        for g in covering:
            add_constraint({y: 1.0, g: -1.0}, 0.0, np.inf)

    for g in range(m):
        own_u = []
        for t, (_, _, covering) in enumerate(segments):
            if g not in covering:
                continue
            u = unique_vars[g, t]
            own_u.append(u)
            add_constraint({u: 1.0, g: -1.0}, -np.inf, 0.0)
            for other in covering:
                if other != g:
                    add_constraint({u: 1.0, other: 1.0}, -np.inf, 1.0)
        add_constraint(
            {u: 1.0 for u in own_u} | {g: -1.0},
            0.0,
            np.inf,
        )

    feasible_sums = {0}
    for group in groups:
        feasible_sums |= {
            value + group["size"] for value in list(feasible_sums)
        }
    feasible_sums.discard(0)
    k_values = [fixed_k] if fixed_k is not None else sorted(feasible_sums)
    if any(k <= 0 or k not in feasible_sums for k in k_values):
        raise RuntimeError(f"不存在总点数恰为 {fixed_k} 的整组选择方案")

    row_indices = []
    col_indices = []
    values = []
    row_lower = []
    row_upper = []
    for row_index, (coefficients, lb, ub) in enumerate(
        zip(rows, lower, upper)
    ):
        for column, value in coefficients.items():
            row_indices.append(row_index)
            col_indices.append(column)
            values.append(value)
        row_lower.append(lb)
        row_upper.append(ub)

    base_matrix = coo_matrix(
        (values, (row_indices, col_indices)),
        shape=(len(rows), variable_count),
    ).tocsr()
    integrality = np.zeros(variable_count, dtype=np.uint8)
    integrality[:m] = 1
    integrality[u_start:] = 1

    best_overall = None
    best_by_k = {}
    for k in k_values:
        objective = np.zeros(variable_count)
        cardinality_coefficients = np.zeros(variable_count)
        for g, group in enumerate(groups):
            objective[g] = (
                distance_weight * group["distance_sum"] / (k * d_ref)
            )
            cardinality_coefficients[g] = group["size"]
        for t, (start, end, _) in enumerate(segments):
            objective[y_start + t] = -(end - start) / 360.0

        base = base_matrix.tocoo()
        matrix = coo_matrix(
            (
                np.r_[base.data, cardinality_coefficients],
                (
                    np.r_[
                        base.row,
                        np.full(variable_count, len(rows), dtype=int),
                    ],
                    np.r_[
                        base.col,
                        np.arange(variable_count),
                    ],
                ),
            ),
            shape=(len(rows) + 1, variable_count),
        ).tocsr()
        constraint = LinearConstraint(
            matrix,
            np.r_[row_lower, k],
            np.r_[row_upper, k],
        )
        result = milp(
            objective,
            integrality=integrality,
            bounds=Bounds(0.0, 1.0),
            constraints=constraint,
            options={"time_limit": 30.0, "mip_rel_gap": 0.0},
        )
        if not result.success:
            if getattr(result, "status", None) == 2:
                # A total-point count can exist as a group-size sum yet have
                # no redundancy-free selection.
                continue
            raise RuntimeError(
                f"MILP 求解失败 k={k}: {result.message}"
            )
        selected_groups = np.flatnonzero(result.x[:m] > 0.5)
        mask = sum(
            1 << (groups[g]["group_id"] - 1) for g in selected_groups
        )
        evaluated = evaluate_mask(
            mask, inventory, d_ref, distance_weight=distance_weight
        )
        if evaluated["redundant_group_ids"]:
            raise RuntimeError("MILP 返回了包含冗余组的解")
        if fixed_k is not None or (
            best_overall is None
            or evaluated["objective"] < best_overall["objective"] - 1e-10
        ):
            best_overall = evaluated
        best_by_k[k] = evaluated

    if best_overall is None:
        raise RuntimeError("没有可行的无冗余整组选择方案")

    return best_overall, best_by_k, {
        "segments": len(segments),
        "variables": variable_count,
        "constraints": len(rows) + 1,
        "k_values_solved": len(k_values),
    }


def solve_radial_prefixes(
    inventory: list[dict],
    points: list[dict],
    d_ref: float,
    fixed_k=None,
    excluded_group_ids=None,
    patience: int = 8,
    distance_weight: float = 1.0,
):
    """Expand whole groups from nearest to farthest.

    Each prefix is solved exactly. The search stops after `patience` newly
    added radial layers fail to improve the best objective, so distant groups
    are not explored when they provide no benefit.
    """
    excluded_group_ids = set(excluded_group_ids or [])
    ranked = sorted(
        (
            group for group in inventory
            if group["group_id"] not in excluded_group_ids
        ),
        key=lambda group: (
            min(points[i]["distance"] for i in group["points"]),
            group["group_id"],
        ),
    )

    best_overall = None
    best_by_k = {}
    misses = 0
    prefixes_solved = 0
    frontier_group_id = None
    frontier_distance = None

    for count in range(1, len(ranked) + 1):
        prefix_ids = {group["group_id"] for group in ranked[:count]}
        outside_ids = {
            group["group_id"] for group in ranked[count:]
        }
        result, per_k, _ = solve_group_selections_milp(
            inventory,
            d_ref,
            fixed_k=fixed_k,
            excluded_group_ids=excluded_group_ids | outside_ids,
            distance_weight=distance_weight,
        )
        prefixes_solved += 1

        improved = (
            best_overall is None
            or result["objective"] < best_overall["objective"] - 1e-10
        )
        if improved:
            best_overall = result
            misses = 0
            frontier_group_id = ranked[count - 1]["group_id"]
            frontier_distance = min(
                points[i]["distance"]
                for i in ranked[count - 1]["points"]
            )
        else:
            misses += 1

        for k, candidate in per_k.items():
            if (
                k not in best_by_k
                or candidate["objective"]
                < best_by_k[k]["objective"] - 1e-10
            ):
                best_by_k[k] = candidate

        if fixed_k is None and misses >= patience:
            break

    return best_overall, best_by_k, {
        "groups_sorted_by_nearest_distance": len(ranked),
        "prefixes_solved": prefixes_solved,
        "stopping_patience": patience,
        "best_frontier_group_id": frontier_group_id,
        "best_frontier_distance_cells": frontier_distance,
    }


def draw_result(rgb, centre, points, inventory, groups, best, output_path):
    out = Image.fromarray(rgb.copy()).convert("RGB")
    draw = ImageDraw.Draw(out)
    font = ImageFont.load_default()
    cx, cy = centre
    selected_groups = set(best["group_ids"])
    palette = [
        (255, 230, 0),
        (255, 150, 0),
        (160, 255, 0),
        (0, 255, 80),
        (0, 255, 220),
        (0, 180, 255),
        (120, 120, 255),
        (230, 80, 255),
        (255, 80, 120),
    ]

    for group, info in zip(groups, inventory):
        chosen = info["group_id"] in selected_groups
        colour = (
            palette[(info["group_id"] - 1) % len(palette)]
            if chosen
            else (100, 100, 100)
        )
        ordered = sorted(group, key=lambda i: points[i]["angle"])
        for i in ordered:
            x, y = points[i]["x"], points[i]["y"]
            radius = 12 if chosen else 10
            draw.ellipse(
                (x - radius, y - radius, x + radius, y + radius),
                outline=colour,
                width=3 if chosen else 1,
            )
        if chosen and len(ordered) > 1:
            polyline = [(points[i]["x"], points[i]["y"]) for i in ordered]
            draw.line(polyline, fill=colour, width=2)
        first = ordered[0]
        draw.text(
            (points[first]["x"] + 12, points[first]["y"] - 7),
            f"G{info['group_id']}" + ("" if chosen else "×"),
            fill=colour,
            font=font,
        )

    draw.line((cx - 10, cy, cx + 10, cy), fill=(255, 255, 255), width=2)
    draw.line((cx, cy - 10, cx, cy + 10), fill=(255, 255, 255), width=2)
    out.save(output_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=Path, default=DEFAULT_IMAGE)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        "--k",
        type=int,
        default=None,
        help="可选：要求整组选择后的总点数恰为 k；默认不限制",
    )
    parser.add_argument(
        "--exclude-group",
        type=int,
        action="append",
        default=[],
        help="强制排除的组号，可重复指定",
    )
    parser.add_argument(
        "--strategy",
        choices=("auto", "enumerate", "milp", "radial"),
        default="auto",
        help="auto: 小组枚举、大组径向扩展",
    )
    parser.add_argument(
        "--radial-patience",
        type=int,
        default=8,
        help="径向扩展连续多少个新增层无改进后停止",
    )
    parser.add_argument(
        "--distance-weight",
        type=float,
        default=1.0,
        help="归一化平均距离的惩罚系数（默认 1.0）",
    )
    args = parser.parse_args()
    if args.distance_weight < 0:
        raise ValueError("--distance-weight 不能为负数")

    rgb, centre, step, points, d_ref = extract_candidates(args.image)
    groups, inventory = describe_groups(points)
    available_group_ids = {group["group_id"] for group in inventory}
    unknown = set(args.exclude_group) - available_group_ids
    if unknown:
        raise ValueError(f"不存在的组号: {sorted(unknown)}")
    brute_combinations = 1 << len(inventory)
    if args.strategy == "auto":
        strategy = "enumerate" if len(inventory) <= 20 else "radial"
    else:
        strategy = args.strategy

    if strategy == "enumerate":
        best, best_by_k = enumerate_group_selections(
            inventory,
            d_ref,
            args.k,
            excluded_group_ids=args.exclude_group,
            distance_weight=args.distance_weight,
        )
        method = "exact enumeration of all whole-component subsets"
        solver_stats = {}
    elif strategy == "milp":
        best, best_by_k, solver_stats = solve_group_selections_milp(
            inventory,
            d_ref,
            args.k,
            excluded_group_ids=args.exclude_group,
            distance_weight=args.distance_weight,
        )
        method = (
            "exact mixed-integer optimization over angular elementary "
            "segments with unique-coverage constraints"
        )
    else:
        best, best_by_k, solver_stats = solve_radial_prefixes(
            inventory,
            points,
            d_ref,
            fixed_k=args.k,
            excluded_group_ids=args.exclude_group,
            patience=args.radial_patience,
            distance_weight=args.distance_weight,
        )
        method = (
            "radial outward expansion; each prefix solved by exact MILP; "
            "stop after patience layers without improvement"
        )
        solver_stats["candidate_combination_count"] = brute_combinations
    selected_indices = sorted(
        i
        for group in inventory
        if group["group_id"] in best["group_ids"]
        for i in group["points"]
    )

    args.out.mkdir(parents=True, exist_ok=True)
    with open(
        args.out / "group_inventory.csv", "w", newline="", encoding="utf-8-sig"
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "group_id",
                "kind",
                "size",
                "point_indices",
                "row_range",
                "col_range",
                "angle_min_deg",
                "angle_max_deg",
                "contribution_deg",
                "angle_interval_deg",
                "angle_half_width_deg",
                "distance_sum_cells",
            ]
        )
        for group in inventory:
            writer.writerow(
                [
                    group["group_id"],
                    group["kind"],
                    group["size"],
                    " ".join(map(str, group["points"])),
                    f"{group['bounding_rows'][0]}:{group['bounding_rows'][1]}",
                    f"{group['bounding_cols'][0]}:{group['bounding_cols'][1]}",
                    round(group["angle_min"], 6),
                    round(group["angle_max"], 6),
                    round(group["angle_contribution"], 6),
                    format_angle_interval(group),
                    (
                        ""
                        if group["angle_half_width"] is None
                        else round(group["angle_half_width"], 6)
                    ),
                    round(group["distance_sum"], 6),
                ]
            )

    with open(
        args.out / "selected_groups.csv", "w", newline="", encoding="utf-8-sig"
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "group_id",
                "kind",
                "size",
                "point_indices",
                "angle_contribution_deg",
                "angle_interval_deg",
                "angle_half_width_deg",
                "distance_sum_cells",
            ]
        )
        for group in inventory:
            if group["group_id"] in best["group_ids"]:
                writer.writerow(
                    [
                        group["group_id"],
                        group["kind"],
                        group["size"],
                        " ".join(map(str, group["points"])),
                        round(group["angle_contribution"], 6),
                        format_angle_interval(group),
                        (
                            ""
                            if group["angle_half_width"] is None
                            else round(group["angle_half_width"], 6)
                        ),
                        round(group["distance_sum"], 6),
                    ]
                )

    with open(
        args.out / "selected_points.csv", "w", newline="", encoding="utf-8-sig"
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "index",
                "group_id",
                "row_from_centre",
                "col_from_centre",
                "pixel_x",
                "pixel_y",
                "colour",
                "angle_deg",
                "distance_cells",
                "normalized_distance",
            ]
        )
        group_lookup = {
            i: group["group_id"]
            for group in inventory
            for i in group["points"]
        }
        for i in selected_indices:
            point = points[i]
            writer.writerow(
                [
                    i,
                    group_lookup[i],
                    point["row"],
                    point["col"],
                    round(point["x"], 3),
                    round(point["y"], 3),
                    point["colour"],
                    round(point["angle"], 6),
                    round(point["distance"], 6),
                    round(point["distance"] / d_ref, 8),
                ]
            )

    with open(
        args.out / "results_by_k.csv", "w", newline="", encoding="utf-8-sig"
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "total_points",
                "objective",
                "open_angle_deg",
                "average_normalized_distance",
                "group_ids",
            ]
        )
        for k in sorted(best_by_k):
            result = best_by_k[k]
            writer.writerow(
                [
                    k,
                    result["objective"],
                    result["open_angle"],
                    result["average_normalized_distance"],
                    " ".join(map(str, result["group_ids"])),
                ]
            )

    payload = {
        "image": str(args.image),
        "centre_pixel": {"x": centre[0], "y": centre[1]},
        "grid_step_px": step,
        "candidate_count": len(points),
        "candidate_group_count": len(inventory),
        "group_sizes": sorted(
            (group["size"] for group in inventory), reverse=True
        ),
        "d_ref_cells": d_ref,
        "fixed_k": args.k,
        "distance_weight": args.distance_weight,
        "excluded_group_ids": sorted(set(args.exclude_group)),
        "best": best,
        "selected_indices": selected_indices,
        "groups": inventory,
        "best_by_k": [best_by_k[k] for k in sorted(best_by_k)],
        "method": method,
        "solver_stats": solver_stats,
    }
    with open(args.out / "result.json", "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)

    draw_result(
        rgb,
        centre,
        points,
        inventory,
        groups,
        best,
        args.out / "selected_groups_overlay.png",
    )

    print(
        f"candidates={len(points)} groups={len(inventory)} "
        f"sizes={sorted((g['size'] for g in inventory), reverse=True)}"
    )
    print(
        f"best k={best['k']} F={best['objective']:.9f} "
        f"open={best['open_angle']:.6f} "
        f"avg_d={best['average_normalized_distance']:.9f}"
    )
    print("groups", " ".join(map(str, best["group_ids"])))
    print("wrote", args.out)


if __name__ == "__main__":
    main()
