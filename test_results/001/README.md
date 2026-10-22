# Test 001

Generated results from the current T-Muscle Detector pipeline for input `001`.
The original input photograph and intermediate working files are not included.

> Research output only. These measurements are not medical advice or a
> diagnosis. The images contain a person; review them before redistribution.

## Result

| Metric | Value |
|---|---:|
| Centre rows | 38, 47, 60 |
| Mean open angle | 68.845518 deg |
| Abdomen score | 95.085823 |
| Shoulder / waist score | 87.883065 |
| Arm score | 100.000000 |
| **Overall score** | **93.662122** |

The row function uses two local minima after the anchor (`47`, `60`) and fills
the third row backward with `38`. No point after the final maximum (`66`) is
used.

## Files

- `001_armmid.png`: usable-arm overlay and narrowest abdominal line.
- `001_abdominal_rectangle_annotation.png`: annotations inside the retained
  abdominal rectangle.
- `001_C01_original.png` through `001_C06_original.png`: retained groups for
  each centre on the original image.
- `001_centres.png`: six selected centre points.
- `001_total_score.png`, `.csv`, `.json`: component scores and overall score.
