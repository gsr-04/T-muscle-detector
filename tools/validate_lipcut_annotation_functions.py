"""Validate the exported LIP-cut row/column annotation functions."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter1d


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outfinal" / "lipcut_roi_annotation" / "annotation_functions"
NAMES = [
    "001", "002", "003", "004", "006", "010", "011",
    "013", "015", "029", "035", "037", "038",
]


def positive_runs(values):
    result = []
    start = None
    for index, value in enumerate(list(values) + [0]):
        if value > 0 and start is None:
            start = index
        elif value == 0 and start is not None:
            result.append((start, index - 1))
            start = None
    return result


def merge_value_ones(counts):
    original = [int(value) for value in counts]
    merged = [float(value) for value in original]
    for row, value in enumerate(original):
        if value != 1:
            continue
        target = next(
            (
                candidate
                for candidate in range(row + 1, len(original))
                if original[candidate] > 1
            ),
            None,
        )
        if target is not None:
            merged[target] += 1.0
            merged[row] = 0.0
    return merged


def merge_isolated(counts):
    runs = positive_runs(counts)
    isolated = [start for start, end in runs if start == end]
    continuous = [
        (start, end) for start, end in runs if end > start
    ]
    merged = [float(value) for value in counts]
    for row in isolated:
        target = next(
            (run for run in continuous if run[0] > row), None
        )
        if target is None:
            continue
        increment = merged[row] / (target[1] - target[0] + 1)
        for target_row in range(target[0], target[1] + 1):
            merged[target_row] += increment
        merged[row] = 0.0
    return merged


def anchor_last_group(values, filtered, sigma, fraction):
    runs = positive_runs(values)
    if not runs or max(values) <= 0:
        return filtered
    start, end = runs[-1]
    group = values[start:end + 1]
    group_peak = max(group)
    peak_row = start + group.index(group_peak)
    if group_peak < fraction * max(values):
        return filtered
    delta = group_peak - filtered[peak_row]
    if delta <= 0:
        return filtered
    distance = np.arange(len(values), dtype=float) - peak_row
    result = filtered + delta * np.exp(
        -0.5 * (distance / sigma) ** 2
    )
    result[peak_row] = group_peak
    return result


def main():
    payload = json.loads(
        (OUT / "annotation_functions.json").read_text(encoding="utf-8")
    )
    assert set(payload) == set(NAMES)

    long_rows = list(
        csv.DictReader(
            (OUT / "annotation_functions_long.csv").open(
                encoding="utf-8-sig", newline=""
            )
        )
    )
    assert len(long_rows) == 1572

    for name in NAMES:
        image_path = OUT / f"annotation_function_{name}.png"
        assert image_path.is_file() and image_path.stat().st_size > 0

        image_payload = payload[name]
        row_function = image_payload["row_function"]
        raw = row_function["counts"]
        after_ones = row_function["counts_after_value1_merge"]
        merged = row_function["counts_after_isolated_merge"]
        assert len(row_function["index"]) == len(raw) == len(
            after_ones
        ) == len(merged)
        expected_ones = merge_value_ones(raw)
        expected_merged = merge_isolated(expected_ones)
        assert np.allclose(after_ones, expected_ones, atol=1e-9)
        assert np.allclose(merged, expected_merged, atol=1e-9)
        assert abs(sum(raw) - sum(after_ones)) < 1e-8
        assert abs(sum(after_ones) - sum(merged)) < 1e-8
        one_records = row_function["value1_merge_records"]
        for record in one_records:
            if record["merged"]:
                assert record["target_start"] == record["target_end"]
                assert merged[record["row"]] == 0
            else:
                assert record["reason"] == (
                    "no_right_value_greater_than_1"
                )
                assert after_ones[record["row"]] == 1

        isolated_records = row_function["isolated_merge_records"]
        for record in isolated_records:
            if record["merged"]:
                assert record["target_start"] < record["target_end"]
                assert merged[record["row"]] == 0

        column_function = image_payload["column_function"]
        col_raw = column_function["counts"]
        col_clean = column_function["counts_after_removing_value1"]
        assert column_function["singletons_removed"] == [
            count == 1 for count in col_raw
        ]
        assert col_clean == [
            0 if count == 1 else count for count in col_raw
        ]
        assert 1 not in col_clean

        gaussian = image_payload["gaussian_filter"]
        assert gaussian == {
            "dimension": 1,
            "sigma": 2.0,
            "mode": "reflect",
            "last_group_anchor_fraction": 0.4,
        }
        for function, source, output_key in (
            (row_function, merged, "counts_after_gaussian_filter"),
            (
                column_function,
                col_clean,
                "counts_after_gaussian_filter",
            ),
        ):
            actual = np.asarray(function[output_key], dtype=float)
            expected = gaussian_filter1d(
                np.asarray(source, dtype=float),
                sigma=gaussian["sigma"],
                mode=gaussian["mode"],
            )
            if function is row_function:
                expected = anchor_last_group(
                    merged,
                    expected,
                    gaussian["sigma"],
                    gaussian["last_group_anchor_fraction"],
                )
            assert actual.shape == expected.shape
            assert np.isfinite(actual).all()
            assert np.allclose(actual, expected, atol=1e-6)
            peak_index, peak_value = function["gaussian_peak"]
            assert peak_index == int(np.argmax(actual))
            assert abs(peak_value - float(actual.max())) < 1e-6

    with (OUT / "annotation_functions_summary.csv").open(
        encoding="utf-8-sig", newline=""
    ) as handle:
        summary = list(csv.reader(handle))
    assert len(summary) == len(NAMES) + 1
    assert all(len(row) == 20 for row in summary)
    assert all(len(row) == 10 for row in long_rows)
    for row in long_rows:
        direction = row["direction"]
        function = payload[row["image"]][
            "row_function"
            if direction == "row_red"
            else "column_function"
        ]
        index = int(row["index"])
        expected_after_ones = (
            function["counts_after_value1_merge"][index]
            if direction == "row_red"
            else function["counts_after_removing_value1"][index]
        )
        assert abs(
            float(row["after_value1_merge"]) - expected_after_ones
        ) < 1e-6
        assert abs(
            float(row["after_gaussian_filter"])
            - function["counts_after_gaussian_filter"][index]
        ) < 1e-6

    merge_rows = list(
        csv.DictReader(
            (OUT / "row_merge_details.csv").open(
                encoding="utf-8-sig", newline=""
            )
        )
    )
    assert merge_rows
    assert {row["phase"] for row in merge_rows} == {
        "value1", "isolated"
    }
    assert (OUT / "all_annotation_functions.png").stat().st_size > 0
    print(
        "validated 13 images, 1572 function rows, value1-right merging, "
        "isolated-right merging, Gaussian filtering, last-group anchoring"
    )


if __name__ == "__main__":
    main()
