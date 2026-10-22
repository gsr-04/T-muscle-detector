# Model assets

Model binaries are not distributed with this repository.

Create this layout:

```text
models/
  pose_landmarker_full.task
  birefnet_lite.onnx
  dexined_2024sep.onnx
  rtmpose-m-distill/onnx/rtmpose-m-distill-256x256.onnx
  schp-lip-20/onnx/schp-lip-20-int8-static.onnx
  schp-atr-18/onnx/schp-atr-18-int8-static.onnx
```

The repository includes helper scripts under `tools/fetch_*.py` for several
upstream assets. Available tools and network conditions vary by release. Check
`python tmuscle.py doctor` after installation.

Do not commit model files. Verify upstream licences before redistribution.
