"""Import one image and build only the prerequisites used by the current flow."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
DATA = ROOT / "data"
OUT = ROOT / "out"
REGISTRY = ROOT / "out1" / "image_registry.json"


def run(script: Path, args: list[str]):
    command = [sys.executable, str(script)] + args
    print("  $ " + " ".join(command[1:]), flush=True)
    result = subprocess.run(
        command, cwd=str(ROOT), capture_output=True, text=True
    )
    if result.returncode != 0:
        text = (result.stdout or "") + (result.stderr or "")
        raise SystemExit(
            f"{script.name} failed:\n"
            + "\n".join(text.splitlines()[-12:])
        )


def next_name() -> str:
    used = {
        path.stem for path in DATA.glob("*.png")
        if path.stem.isdigit()
    }
    number = 1
    while f"{number:03d}" in used:
        number += 1
    return f"{number:03d}"


def filed_name(image: Path, requested: str) -> str:
    registry = (
        json.loads(REGISTRY.read_text(encoding="utf-8"))
        if REGISTRY.exists() else {}
    )
    key = str(image)
    if requested:
        name = requested
    elif key in registry and (DATA / f"{registry[key]}.png").exists():
        name = registry[key]
    else:
        name = next_name()
    registry[key] = name
    REGISTRY.parent.mkdir(parents=True, exist_ok=True)
    REGISTRY.write_text(
        json.dumps(
            {key: registry[key] for key in sorted(registry)},
            ensure_ascii=False,
            indent=1,
        ),
        encoding="utf-8",
    )
    return name


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("image", type=Path)
    parser.add_argument("--name", default="")
    parser.add_argument("--keep-temp", action="store_true")
    args = parser.parse_args()

    image = args.image
    if not image.is_absolute():
        image = (ROOT / image).resolve()
    if not image.is_file():
        raise SystemExit(f"input image not found: {image}")
    DATA.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    name = filed_name(image, args.name)
    raw = cv2.imdecode(
        np.fromfile(str(image), dtype=np.uint8),
        cv2.IMREAD_UNCHANGED,
    )
    if raw is None:
        raise SystemExit(f"cannot decode image: {image}")
    if raw.ndim == 3 and raw.shape[2] == 4:
        raw = raw[:, :, :3]
    ok, buffer = cv2.imencode(".png", raw)
    if not ok:
        raise SystemExit("cannot encode input image")
    buffer.tofile(str(DATA / f"{name}.png"))

    temporary = Path(tempfile.mkdtemp(prefix=f"tmuscle_prepare_{name}_"))
    try:
        (temporary / "data").mkdir()
        (temporary / "out").mkdir()
        shutil.copy2(
            DATA / f"{name}.png",
            temporary / "data" / f"{name}.png",
        )
        for script in (
            "03_extract_subject.py",
            "04_annotate_parts.py",
            "05_cutout_torso.py",
            "06_cutout_crop.py",
        ):
            extra = []
            if script == "03_extract_subject.py":
                extra = [
                    "--data", str(temporary / "data"),
                    "--out", str(temporary / "out"),
                ]
            elif script == "04_annotate_parts.py":
                extra = [
                    "--data", str(temporary / "data"),
                    "--out", str(temporary / "out"),
                ]
            else:
                extra = ["--out", str(temporary / "out")]
            run(HERE / script, extra)

        for source in (temporary / "out").iterdir():
            if source.name in ("crops.json", "crop_torso_boxes.json"):
                continue
            shutil.copy2(source, OUT / source.name)
        for filename in ("crops.json", "crop_torso_boxes.json"):
            new = json.loads(
                (temporary / "out" / filename).read_text(
                    encoding="utf-8"
                )
            )
            if name not in new:
                raise RuntimeError(
                    f"{filename} has no entry for image {name}"
                )
            target = OUT / filename
            old = (
                json.loads(target.read_text(encoding="utf-8"))
                if target.exists() else {}
            )
            old[name] = new[name]
            target.write_text(
                json.dumps(
                    {key: old[key] for key in sorted(old)},
                    ensure_ascii=False,
                    indent=1,
                ),
                encoding="utf-8",
            )
    finally:
        if args.keep_temp:
            print(f"temporary files retained: {temporary}")
        else:
            shutil.rmtree(temporary, ignore_errors=True)

    print(f"prepared data/{name}.png")


if __name__ == "__main__":
    main()
