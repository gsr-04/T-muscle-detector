"""Validate current LIP-cut centre objectives and retained annotation points."""

from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [
    str(ROOT / "tools"),
    str(ROOT / "src"),
]
import armmid_curve_group_objective as curve  # noqa: E402
import optimize_007  # noqa: E402
import run_lipcut_center_objective as objective_runner  # noqa: E402


INPUT = ROOT / "outfinal" / "lipcut_roi_annotation"
OUT = INPUT / "center_objective"
ORIGINAL_OUT = OUT / "original_overlays"
NAMES = [
    "001", "002", "003", "004", "006", "010", "011",
    "013", "015", "029", "035", "037", "038",
]


def load_candidates(name):
    data = json.loads(
        (INPUT / f"peakscolor50lipcut_{name}.json").read_text(
            encoding="utf-8"
        )
    )
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


result = json.loads((OUT / "result.json").read_text(encoding="utf-8"))
DISTANCE_WEIGHT = float(result["distance_weight"])
assert DISTANCE_WEIGHT == 1.3
assert len(result["records"]) == len(NAMES)
intersections = {
    record["image"]: record["intersections"]
    for record in json.loads(
        (
            INPUT / "rowcol_intersections" / "intersections.json"
        ).read_text(encoding="utf-8")
    )["records"]
}
with (OUT / "center_objective.csv").open(
    encoding="utf-8-sig", newline=""
) as handle:
    center_csv = list(csv.DictReader(handle))
with (OUT / "selected_points.csv").open(
    encoding="utf-8-sig", newline=""
) as handle:
    point_csv = list(csv.DictReader(handle))
with (ORIGINAL_OUT / "retained_points_fullres.csv").open(
    encoding="utf-8-sig", newline=""
) as handle:
    fullres_csv = list(csv.DictReader(handle))
with (OUT / "removed_near_center_points.csv").open(
    encoding="utf-8-sig", newline=""
) as handle:
    removed_csv = list(csv.DictReader(handle))
with (OUT / "nearest_center_removed_points.csv").open(
    encoding="utf-8-sig", newline=""
) as handle:
    nearest_csv = list(csv.DictReader(handle))
rois = {
    row["image"]: row
    for row in json.loads(
        (INPUT / "roi_annotation.json").read_text(encoding="utf-8")
    )["records"]
}

assert len(center_csv) == 13 * 6
assert int(
    sum(int(row["selected_points"]) for row in center_csv)
) == len(point_csv)
assert len(fullres_csv) == len(point_csv)
assert len(removed_csv) == sum(
    int(row["removed_near_center_points"]) for row in center_csv
)
assert len(nearest_csv) == len(center_csv)
assert {
    (row["image"], int(row["centre_index"])) for row in nearest_csv
} == {
    (row["image"], int(row["centre_index"])) for row in center_csv
}
point_keys = {
    (
        row["image"],
        int(row["centre_index"]),
        int(row["row"]),
        int(row["column"]),
        row["family"],
    )
    for row in point_csv
}
fullres_keys = {
    (
        row["image"],
        int(row["centre_index"]),
        int(row["row"]),
        int(row["column"]),
        row["family"],
    )
    for row in fullres_csv
}
assert fullres_keys == point_keys
for row in fullres_csv:
    roi = rois[row["image"]]
    left, top, right, bottom = roi["rectangle_px"]
    assert left <= float(row["full_x"]) < right
    assert top <= float(row["full_y"]) < bottom

for image_record in result["records"]:
    name = image_record["image"]
    assert name in NAMES
    source, candidates = load_candidates(name)
    pooled_h = int(source["size"][1])
    centres = [
        {**centre, "index": index}
        for index, centre in enumerate(intersections[name], start=1)
    ]
    records = image_record["centres"]
    assert len(records) == len(centres) == 6

    for centre, record in zip(centres, records):
        assert record["index"] == centre["index"]
        assert (record["row"], record["column"]) == (
            centre["row"],
            centre["column"],
        )
        left, top, right, bottom = record["rectangle"]
        assert left <= centre["column"] <= right
        assert top <= centre["row"] <= bottom
        for other in centres:
            if other["index"] == centre["index"]:
                continue
            assert not (
                left <= other["column"] <= right
                and top <= other["row"] <= bottom
            )

        rectangle_local = [
            point for point in candidates
            if left <= point["column"] <= right
            and top <= point["row"] <= bottom
        ]
        assert len(rectangle_local) == record["rectangle_candidate_points"]
        minimum_retained_points = math.ceil(
            0.5 * len(rectangle_local)
        )
        assert (
            record["minimum_retained_points"]
            == minimum_retained_points
        )
        nearest_point = min(
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
        assert record["nearest_center_pre_removed_point"][
            "row"
        ] == nearest_point["row"]
        assert record["nearest_center_pre_removed_point"][
            "column"
        ] == nearest_point["column"]
        assert record["nearest_center_pre_removed_point"][
            "family"
        ] == nearest_point["family"]
        initial_local = [
            point for point in rectangle_local
            if (point["row"], point["column"])
            != (nearest_point["row"], nearest_point["column"])
        ]
        assert len(initial_local) == record["initial_candidate_points"]
        assert len(initial_local) == (
            record["candidate_points_after_nearest_removal"]
        )
        initial_groups = curve.preprocess_groups(
            [
                (point["row"], point["column"])
                for point in initial_local
            ]
        )[2]
        _, initial_inventory, initial_d_ref = curve.make_inventory(
            initial_local,
            initial_groups,
            float(centre["row"]),
            float(centre["column"]),
        )
        initial_best = objective_runner.evaluate_inventory_minimum(
            initial_inventory,
            initial_d_ref,
            DISTANCE_WEIGHT,
            minimum_retained_points,
        )
        initial_selected_cells = {
            (
                initial_local[index]["row"],
                initial_local[index]["column"],
            )
            for group in initial_inventory
            if group["group_id"] in set(initial_best["group_ids"])
            for index in group["points"]
        }
        removed_cells = {
            (point["row"], point["column"])
            for point in record["removed_near_center_points"]
        }
        removal_applied = record["near_center_removal_applied"]
        assert removal_applied == (
            len(initial_local) >= 30
            and len(initial_selected_cells) <= 3
            and bool(initial_selected_cells)
            and max(
                math.hypot(
                    row_value - centre["row"],
                    column_value - centre["column"],
                )
                for row_value, column_value in initial_selected_cells
            )
            <= 0.3 * min(right - left, bottom - top)
        )
        assert removed_cells == (
            initial_selected_cells if removal_applied else set()
        )
        assert abs(
            initial_best["objective"] - record["initial_objective"]
        ) < 1e-8
        assert len(initial_selected_cells) == (
            record["initial_selected_point_count"]
        )
        local = [
            point for point in initial_local
            if (point["row"], point["column"])
            not in removed_cells
        ]
        assert len(local) == record["candidate_points"]
        _, _, groups = curve.preprocess_groups(
            [(point["row"], point["column"]) for point in local]
        )
        _, inventory, d_ref = curve.make_inventory(
            local,
            groups,
            float(centre["row"]),
            float(centre["column"]),
        )
        selected_group_ids = set(record["selected_group_ids"])
        assert selected_group_ids == {
            group["group_id"] for group in inventory
            if group["group_id"] in selected_group_ids
        }
        mask = sum(1 << (group_id - 1) for group_id in selected_group_ids)
        evaluated = optimize_007.evaluate_mask(
            mask,
            inventory,
            d_ref,
            distance_weight=DISTANCE_WEIGHT,
        )
        assert not evaluated["redundant_group_ids"]
        assert abs(evaluated["objective"] - record["objective"]) < 1e-8
        assert abs(evaluated["open_angle"] - record["open_angle"]) < 1e-6
        assert abs(
            evaluated["average_normalized_distance"]
            - record["average_normalized_distance"]
        ) < 1e-8
        assert abs(
            record["objective"]
            - (
                record["open_angle"] / 360.0
                + DISTANCE_WEIGHT
                * record["average_normalized_distance"]
            )
        ) < 1e-8

        expected_cells = {
            (point["row"], point["column"])
            for group in inventory
            if group["group_id"] in selected_group_ids
            for point_index in group["points"]
            for point in [local[point_index]]
        }
        actual_cells = {
            (point["row"], point["column"])
            for point in record["selected_points"]
        }
        assert actual_cells == expected_cells
        assert len(actual_cells) == record["selected_point_count"]
        assert len(actual_cells) >= minimum_retained_points
        assert record["minimum_retained_constraint_met"] is True
        assert all(
            top <= row <= bottom and left <= column <= right
            for row, column in actual_cells
        )
        overlay = Path(record["overlay_png"])
        assert overlay.is_file() and overlay.stat().st_size > 0
        with Image.open(overlay) as image:
            assert image.width > 0 and image.height > 0

    assert (OUT / f"center_contact_{name}.png").stat().st_size > 0
    assert (OUT / f"center_rectangles_{name}.png").stat().st_size > 0
    assert (
        ORIGINAL_OUT / f"original_centers_{name}.png"
    ).stat().st_size > 0
    for centre in records:
        assert (
            ORIGINAL_OUT
            / f"original_{name}_C{centre['index']:02d}.png"
        ).stat().st_size > 0

assert (OUT / "all_center_contact.png").stat().st_size > 0
assert (OUT / "preprocessed_groups.csv").stat().st_size > 0
assert (ORIGINAL_OUT / "original_centers_contact.png").stat().st_size > 0
print(
    "validated 13 images, 78 centers, "
    f"{len(point_csv)} retained centre-point records and original overlays"
)
