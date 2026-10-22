"""Validate LIP-cut row/column intersection selections and overlays."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.signal import find_peaks


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outfinal" / "lipcut_roi_annotation" / "rowcol_intersections"
DATA = json.loads(
    (OUT / "intersections.json").read_text(encoding="utf-8")
)
RECORDS = DATA["records"]
FUNCTIONS = json.loads(
    (
        ROOT
        / "outfinal"
        / "lipcut_roi_annotation"
        / "annotation_functions"
        / "annotation_functions.json"
    ).read_text(encoding="utf-8")
)
ROI_RECORDS = {
    row["image"]: row
    for row in json.loads(
        (
            ROOT
            / "outfinal"
            / "lipcut_roi_annotation"
            / "roi_annotation.json"
        ).read_text(encoding="utf-8")
    )["records"]
}


assert len(RECORDS) == 13
for record in RECORDS:
    assert len(record["rows"]) == 3
    assert record["rows"] == sorted(set(record["rows"]))
    assert len(record["columns"]) == 2
    assert record["columns"] == sorted(set(record["columns"]))
    assert record["intersection_count"] == 6

    source = FUNCTIONS[record["image"]]
    row_values = np.asarray(
        source["row_function"]["counts_after_gaussian_filter"],
        dtype=float,
    )
    column_values = np.asarray(
        source["column_function"]["counts_after_gaussian_filter"],
        dtype=float,
    )
    assert np.allclose(
        record["row_values"],
        row_values[record["rows"]],
        atol=1e-6,
    )
    assert np.allclose(
        record["column_values"],
        column_values[record["columns"]],
        atol=1e-6,
    )
    maxima = [int(value) for value in find_peaks(row_values)[0]]
    top3 = sorted(
        maxima,
        key=lambda index: (row_values[index], -index),
        reverse=True,
    )[:3]
    anchor = min(top3) if top3 else int(np.argmax(row_values))
    minima = [int(value) for value in find_peaks(-row_values)[0]]
    following = [row for row in minima if row > anchor]
    expected_rows = following[:3]
    if (
        len(expected_rows) < 3
        and len(row_values) - 1 not in expected_rows
        and row_values[-1] <= row_values[-2]
    ):
        expected_rows.append(len(row_values) - 1)
    assert record["row_maxima_top3"] == top3
    assert record["row_maximum_anchor"] == anchor
    assert record["row_minima_after_anchor"] == following
    assert record["rows"] == sorted(expected_rows)

    maximum = int(np.argmax(column_values))
    assert record["column_maximum"] == maximum
    if record["column_fallback"]:
        assert record["columns"] == [
            round(len(column_values) / 3),
            round(2 * len(column_values) / 3),
        ]
    else:
        minima = [int(value) for value in find_peaks(-column_values)[0]]
        left = [column for column in minima if column < maximum]
        right = [column for column in minima if column > maximum]
        assert record["columns"] == [left[-1], right[0]]

    for row in record["rows"]:
        assert 0 <= row < record["row_function_length"]
    for column in record["columns"]:
        assert 0 <= column < record["column_function_length"]

    overlay = Path(record["overlay_png"])
    selection = Path(record["selection_plot_png"])
    assert overlay.is_file() and overlay.stat().st_size > 0
    assert selection.is_file() and selection.stat().st_size > 0
    with Image.open(overlay) as image:
        width, height = image.size
    roi = ROI_RECORDS[record["image"]]
    left, top, right, bottom = record["rectangle_px"]
    assert record["rectangle_px"] == roi["rectangle_px"]
    assert (width, height) == (right - left, bottom - top)
    with Image.open(record["roi_png"]) as crop_image:
        assert crop_image.size == (width, height)
        crop = np.asarray(crop_image.convert("RGB"))
    with Image.open(record["source_png"]) as source_image:
        expected_crop = np.asarray(
            source_image.convert("RGB").crop(
                (left, top, right, bottom)
            )
        )
    assert np.array_equal(crop, expected_crop)
    for point in record["intersections"]:
        assert 0 <= point["x"] < width
        assert 0 <= point["y"] < height


with (OUT / "intersections.csv").open(
    encoding="utf-8-sig", newline=""
) as handle:
    rows = list(csv.DictReader(handle))
assert len(rows) == 13 * 6
assert (OUT / "intersections_contact.png").stat().st_size > 0
print("validated 13 rectangles, 39 rows, 26 columns, 78 intersections")
