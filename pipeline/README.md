# Ordered pipeline

The pipeline scores the abdomen, shoulders and arms from a photograph. It
outputs the three component scores and a weighted overall score; higher values
indicate a thinner-muscle appearance.

The supported entrypoint remains the repository-root CLI:

```bash
python tmuscle.py test-image path/to/photo.jpg
```

It calls `pipeline/00_test_image.py`.

## Execution order

```text
00_test_image.py
  -> 01_run_pipeline.py
  -> 02_prepare_image.py
       -> 03_extract_subject.py
       -> 04_annotate_parts.py
       -> 05_cutout_torso.py
       -> 06_cutout_crop.py
  -> 07_detect_usable_arms.py
  -> 09_analyze_abdomen.py
       -> 08_pool_peak_marks.py
  -> 10_export_abdomen_bundle.py
  -> 11_export_total_score.py
```

| Stage | File | Purpose |
|---|---|---|
| 00 | `00_test_image.py` | Supported single-image entrypoint and result bundle |
| 01 | `01_run_pipeline.py` | Complete detection and scoring driver |
| 02 | `02_prepare_image.py` | Import image and build prerequisites |
| 03 | `03_extract_subject.py` | Subject extraction and body mask |
| 04 | `04_annotate_parts.py` | Part, skin and torso annotations |
| 05 | `05_cutout_torso.py` | Torso cutout |
| 06 | `06_cutout_crop.py` | Tight person crop |
| 07 | `07_detect_usable_arms.py` | Usable-arm selection and measurement |
| 08 | `08_pool_peak_marks.py` | Pooled annotations, Gaussian filter and peaks |
| 09 | `09_analyze_abdomen.py` | Rectangle, centres, objectives and open angles |
| 10 | `10_export_abdomen_bundle.py` | Single-image result bundle |
| 11 | `11_export_total_score.py` | Weighted total-score table |
| 12 | `12_export_batch_summary.py` | Batch score summary |
| 13 | `13_export_row_function.py` | Row-function export |
| 14 | `14_check_abdomen_score.py` | Abdomen-score validation |

## Scoring rules

The abdominal rectangle uses the green-outline narrowest waist line for its
width and waist row. Annotation points inside the rectangle that fall on
LIP-20 left/right arm classes (`14/15`) are removed before statistics and
objective evaluation.

Centre rows come directly from the row function; there is no vertical midpoint
constraint. For columns, peaks are checked from highest to lowest and the first
peak with a local minimum on both sides is used. The `1/3, 2/3` fallback is used
only when no internal peak satisfies both sides.

```text
abdomen = clamp((240 - mean_open_angle) * 100 / 180, 0, 100)
shoulder_waist = clamp((shoulder / waist - 1.3) * 100 / 0.8, 0, 100)
total = 0.55 * abdomen + 0.30 * shoulder_waist + 0.15 * arm
```

The bottom centre row is excluded when its local rectangle is cut by the image
border by more than one third. The maximum open angle is removed only when it
is greater than twice the second maximum.

Usable-arm measurement lines are drawn only for selected arms. Lines follow
the green body contour first and are cut back to the LIP-20 arm edge when the
contour span exceeds the forearm reference.

Historical numbered stages and experimental probes have been removed. Run
`python tmuscle.py doctor` when a model or dependency is missing.
