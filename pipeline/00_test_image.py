"""Final end-to-end test of one input image.

Runs the complete contour -> pose -> LIP-20 -> abdomen -> usable-arm chain and
collects all requested outputs in one test folder.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PY = Path(os.environ.get("TMUSCLE_PYTHON", sys.executable))


def run(command):
    print("$ " + " ".join(command[1:]), flush=True)
    result = subprocess.run(
        command, cwd=str(ROOT), capture_output=True, text=True
    )
    output = (result.stdout or "") + (result.stderr or "")
    print("\n".join(output.splitlines()[-20:]), flush=True)
    if result.returncode != 0:
        raise SystemExit(
            f"{Path(command[1]).name} failed ({result.returncode})"
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("image", help="input image path")
    parser.add_argument("--name", default="")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    image = Path(args.image)
    if not image.is_absolute():
        image = (ROOT / image).resolve()
    command = [
        str(PY),
        str(ROOT / "pipeline" / "01_run_pipeline.py"),
        str(image),
    ]
    if args.name:
        command += ["--name", args.name]
    run(command)

    registry = json.loads(
        (ROOT / "out1" / "image_registry.json").read_text(
            encoding="utf-8"
        )
    )
    name = args.name or registry.get(str(image))
    if not name:
        raise SystemExit("could not resolve the input image number")
    label = args.name or f"input_{name}"
    out = args.out or ROOT / "outfinal" / f"{label}_test"
    run(
        [
            str(PY),
            str(ROOT / "pipeline" / "10_export_abdomen_bundle.py"),
            "--name", name,
            "--label", label,
            "--out", str(out),
        ]
    )
    run(
        [
            str(PY),
            str(ROOT / "pipeline" / "11_export_total_score.py"),
            "--name", name,
            "--label", label,
            "--out", str(out),
        ]
    )
    print(f"\nfinal input-image test: {out}")


if __name__ == "__main__":
    main()
