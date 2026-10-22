"""Validate the final open-angle abdominal score."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCORE = ROOT / "outfinal" / "score.json"
AVERAGES = ROOT / "out1" / "center_open_angle_averages.csv"
CENTRES = (
    ROOT / "outfinal" / "lipcut_roi_annotation"
    / "center_objective" / "center_objective.csv"
)


payload = json.loads(SCORE.read_text(encoding="utf-8"))
assert payload["score_kind"] == "abdomen_open_angle"
with AVERAGES.open(encoding="utf-8-sig", newline="") as handle:
    average_rows = {
        row["image"]: row for row in csv.DictReader(handle)
    }
source = {
    name: float(row["mean_open_angle_deg"])
    for name, row in average_rows.items()
}
with CENTRES.open(encoding="utf-8-sig", newline="") as handle:
    centre_rows = list(csv.DictReader(handle))
by_image = {}
for row in centre_rows:
    by_image.setdefault(row["image"], []).append(row)
rows = {row["image"]: row for row in payload["rows"]}
assert set(rows) == set(source)

bad = 0
for name, mean in source.items():
    expected = float(
        np.clip((240.0 - mean) * 100.0 / 180.0, 0.0, 100.0)
    )
    row = rows[name]
    bad += int(abs(float(row["mean_open_angle_deg"]) - mean) > 1e-5)
    bad += int(abs(float(row["abdomen_score"]) - expected) > 1e-5)
    centres = by_image[name]
    bottom_row = max(int(centre["row"]) for centre in centres)
    bottom = [
        centre for centre in centres
        if int(centre["row"]) == bottom_row
    ]
    bottom_clip = max(
        float(centre["bottom_clip_fraction"]) for centre in bottom
    )
    expected_excluded = bottom_clip > 1.0 / 3.0
    actual_excluded = (
        average_rows[name]["bottom_row_clipped_over_one_third"]
        .lower() == "true"
    )
    bad += int(expected_excluded != actual_excluded)
    eligible_count = 4 if expected_excluded else 6
    expected_count = eligible_count - int(
        average_rows[name]["max_exceeds_twice_second"].lower()
        == "true"
    )
    bad += int(
        int(average_rows[name]["selected_center_count"])
        != expected_count
    )

with (ROOT / "outfinal" / "score.csv").open(
    encoding="utf-8-sig", newline=""
) as handle:
    csv_rows = list(csv.DictReader(handle))
assert len(csv_rows) == len(rows)
for row in csv_rows:
    assert abs(
        float(row["abdomen_score"])
        - float(rows[row["image"]]["abdomen_score"])
    ) < 1e-5

assert bad == 0
print(f"validated {len(rows)} open-angle abdominal scores")
