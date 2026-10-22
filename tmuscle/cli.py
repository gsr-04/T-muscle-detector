"""Public command-line interface."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def run(script: str, args: list[str]):
    command = [sys.executable, str(ROOT / script)] + args
    result = subprocess.run(command, cwd=str(ROOT))
    if result.returncode != 0:
        raise SystemExit(result.returncode)


def doctor(_args):
    run("scripts/doctor.py", [])


def test_image(args):
    extra = ["--name", args.name] if args.name else []
    if args.out:
        extra += ["--out", str(args.out)]
    run("pipeline/00_test_image.py", [str(args.image), *extra])


def arms(args):
    run("pipeline/07_detect_usable_arms.py", ["--names", args.names])


def abdomen(args):
    forwarded = ["--names", args.names]
    if args.prepare:
        forwarded.append("--prepare")
    if args.score_only:
        forwarded.append("--score-only")
    if args.averages:
        forwarded += ["--averages", str(args.averages)]
    run("pipeline/09_analyze_abdomen.py", forwarded)


def main():
    parser = argparse.ArgumentParser(
        prog="tmuscle",
        description="T-Muscle Detector command-line interface",
    )
    parser.add_argument("--version", action="version", version="0.1.0")
    commands = parser.add_subparsers(dest="command", required=True)

    test = commands.add_parser(
        "test-image",
        help="run the complete pipeline on one image",
    )
    test.add_argument("image", type=Path)
    test.add_argument("--name", default="")
    test.add_argument("--out", type=Path, default=None)
    test.set_defaults(func=test_image)

    arm_parser = commands.add_parser(
        "arms",
        help="detect and annotate usable arms only",
    )
    arm_parser.add_argument("--names", required=True)
    arm_parser.set_defaults(func=arms)

    abdomen_parser = commands.add_parser(
        "abdomen",
        help="run or rescore open-angle abdominal scoring",
    )
    abdomen_parser.add_argument(
        "--names",
        default="001,002,003,004,006,010,011,013,015,029,035,037,038",
    )
    abdomen_parser.add_argument("--prepare", action="store_true")
    abdomen_parser.add_argument("--score-only", action="store_true")
    abdomen_parser.add_argument("--averages", type=Path, default=None)
    abdomen_parser.set_defaults(func=abdomen)

    doctor_parser = commands.add_parser(
        "doctor",
        help="check Python dependencies and model assets",
    )
    doctor_parser.set_defaults(func=doctor)

    args = parser.parse_args()
    args.func(args)
