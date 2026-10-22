"""Reject private paths and large binary assets from a public release."""

from __future__ import annotations

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TEXT_SUFFIXES = {".py", ".md", ".json", ".toml", ".txt", ".example"}
PRIVATE = re.compile(
    r"(?:[A-Z]:\\+(?:Users|人体图片分割)\\+"
    r"|/home/[A-Za-z0-9_.-]+/|mydata[/\\])",
    re.IGNORECASE,
)
LARGE_BINARY = {".onnx", ".task", ".tflite", ".pth", ".pt", ".safetensors"}
SKIP_DIRS = {
    ".git", "__pycache__", ".venv", "venv", "env",
    ".idea", ".vscode", "node_modules",
    "data", "models", "out", "out1", "out2", "out3", "outfinal",
}


def main():
    problems = []
    for path in ROOT.rglob("*"):
        if not path.is_file() or any(
            part in SKIP_DIRS for part in path.relative_to(ROOT).parts[:-1]
        ):
            continue
        if path == Path(__file__).resolve():
            continue
        if path.suffix.lower() in LARGE_BINARY:
            problems.append(f"binary asset: {path.relative_to(ROOT)}")
        if path.suffix.lower() in TEXT_SUFFIXES:
            try:
                text = path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                continue
            for line_number, line in enumerate(text.splitlines(), 1):
                if PRIVATE.search(line):
                    problems.append(
                        f"private path: {path.relative_to(ROOT)}:"
                        f"{line_number}: {line.strip()}"
                    )
    if problems:
        print("\n".join(problems))
        raise SystemExit(1)
    print("public-release scan passed")


if __name__ == "__main__":
    main()
