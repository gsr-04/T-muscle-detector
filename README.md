# T-Muscle Detector

T-Muscle Detector scores the abdomen, shoulders and arms from a photograph and
reports the three component scores plus an overall score. Every score runs
from 0 to 100; a higher score indicates a thinner-muscle appearance.

> Research software only. Scores are image-derived measurements and are not
> medical advice, a diagnosis, or a substitute for professional assessment.

## Detection flow

1. Extract the body silhouette and locate shoulder, elbow and wrist landmarks
   with pose estimation.
2. Use LIP-20 to segment arms, torso, skin and lower-body clothing, then remove
   the trousers and their boundary band.
3. Derive the narrowest abdominal line from the shoulder split and green body
   outline; measure the shoulder line and choose usable arms.
4. Build the abdominal rectangle from the narrowest-line width, the lowest
   shoulder point and the highest LIP-20 lower-clothing point.
5. Max-pool the retained rectangle, Gaussian-filter the annotation field and
   detect row/column extrema. Remove red, blue and white annotations that fall
   on LIP-20 arm labels.
6. Select six centre points from the row and column functions, then run the
   discrete-point objective inside each centre's maximal non-overlapping local
   rectangle.
7. Calculate open angles for the retained centre groups, derive the abdomen
   score, shoulder/waist score and arm score, then combine them into the overall
   score.

## Score

For each image, six centre points are evaluated inside the abdominal rectangle.
The bottom row of centres is excluded when its local rectangle is clipped by the
bottom image edge by more than one third. The remaining open angles are used as
follows:

```text
abdomen_score = (240 - mean_open_angle) * 100 / 180
abdomen_score = clamp(abdomen_score, 0, 100)
```

The optional total score uses the legacy weighting:

```text
total = 0.55 * abdomen_score + 0.30 * shoulder_waist_score
      + 0.15 * arm_score
```

## Structure

```text
data/                  input images (not included)
models/                downloaded or mounted model assets (not included)
pipeline/              ordered stages 00–14 and the supported entrypoint
tools/                 reusable image-processing and model adapters
src/                   torso, silhouette, pose and segmentation helpers
tmuscle/               public command-line interface
scripts/               installation and model-health helpers
```

The public current-flow entrypoint is `pipeline/00_test_image.py`. Historical
numbered stages and experimental probes have been removed.

## Install

Python 3.10-3.11 is recommended.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
python -m pip install -U pip
python -m pip install -r requirements.txt
python -m pip install -e .
```

If the active interpreter is not the one used to launch subprocesses, set:

```bash
# Windows PowerShell
$env:TMUSCLE_PYTHON = (Get-Command python).Source
# Bash
export TMUSCLE_PYTHON="$(command -v python)"
```

## Models

Model binaries are intentionally excluded. Place the following assets under
`models/`:

```text
models/pose_landmarker_full.task
models/birefnet_lite.onnx
models/dexined_2024sep.onnx
models/rtmpose-m-distill/onnx/rtmpose-m-distill-256x256.onnx
models/schp-lip-20/onnx/schp-lip-20-int8-static.onnx
models/schp-atr-18/onnx/schp-atr-18-int8-static.onnx
```

See [models/README.md](models/README.md) and `scripts/doctor.py`. Model
licences are separate from this repository's code licence.

## Input data

The repository does not ship portrait photographs. Put images under `data/`
with numeric names such as `data/001.png`, or use the end-to-end command with
any path:

```bash
python tmuscle.py test-image path/to/photo.jpg
```

The input is converted to `data/<number>.png`. Existing assignments are stored
in `out1/image_registry.json`.

## Commands

```bash
# Full test on one input image
python tmuscle.py test-image path/to/photo.jpg

# Detect usable arms only
python tmuscle.py arms 001,002

# Abdomen scoring only, using existing open-angle results
python tmuscle.py abdomen --score-only

# Environment and model check
python tmuscle.py doctor
```

For direct use of the numbered pipeline:

```bash
python pipeline/00_test_image.py path/to/photo.jpg
```

## Outputs

The current flow writes to:

```text
outfinal/score.csv
outfinal/score.json
outfinal/score.png
outfinal/available_arm_<n>.png
outfinal/abdomen_openangle_rule_<n>.png
outfinal/lipcut_roi_annotation/
outfinal/<test-name>_test/
```

The **last step**, `pipeline/00_test_image.py`, collects armmid, the abdominal
rectangle annotation, six centre points, retained points, the abdominal score,
the usable-arm overlay, and the weighted total-score table for one image.

## Data and privacy

Input photographs are not committed. Generated `out*`, registries, temporary
files, model weights, and environment files are ignored by default. Review
images before publishing examples.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). Run:

```bash
python tmuscle.py doctor
python -m compileall -q pipeline tools src tmuscle
```

## Licence

Code in this repository is released under the MIT License. Third-party models,
datasets, and derived model files retain their original licences; see
[NOTICE.md](NOTICE.md).
