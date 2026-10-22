"""Check the local Python dependency graph of the supported pipeline."""

from __future__ import annotations

import ast
from collections import deque
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ENTRIES = (
    ROOT / "tmuscle" / "cli.py",
    ROOT / "pipeline" / "00_test_image.py",
    ROOT / "pipeline" / "07_detect_usable_arms.py",
    ROOT / "pipeline" / "09_analyze_abdomen.py",
)
SCAN_DIRS = (ROOT / "pipeline", ROOT / "tools", ROOT / "src")
UTILITIES = {
    "pipeline/12_export_batch_summary.py",
    "pipeline/13_export_row_function.py",
    "pipeline/14_check_abdomen_score.py",
    "tools/fetch_birefnet.py",
    "tools/fetch_dexined.py",
    "tools/fetch_rtmpose.py",
    "tools/fetch_rtmpose2.py",
    "tools/validate_lipcut_annotation_functions.py",
    "tools/validate_lipcut_rowcol_intersections.py",
    "tools/validate_lipcut_center_objective.py",
}


def local_map():
    result = {}
    for directory in SCAN_DIRS:
        for path in directory.glob("*.py"):
            result.setdefault(path.stem, []).append(path.resolve())
    return result


def resolve_string(value, known, local):
    if not value.endswith(".py"):
        return None
    candidate = Path(value)
    options = (
        [candidate] if candidate.is_absolute()
        else [
            ROOT / candidate,
            ROOT / "final" / candidate,
            ROOT / "tools" / candidate,
            ROOT / "src" / candidate,
        ]
    )
    for option in options:
        if option.is_file():
            return option.resolve()
    filename = candidate.name
    matches = [
        path for paths in local.values() for path in paths
        if path.name == filename
    ]
    if len(matches) == 1:
        return matches[0]
    return None


def main():
    local = local_map()
    queue = deque(path.resolve() for path in ENTRIES)
    reached = set()
    missing = []
    while queue:
        path = queue.popleft()
        if path in reached or not path.exists():
            continue
        reached.add(path)
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            names = []
            if isinstance(node, ast.Import):
                names = [alias.name.split(".")[0] for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                names = [node.module.split(".")[0]]
            for name in names:
                for match in local.get(name, []):
                    queue.append(match)
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                target = resolve_string(node.value, local, local)
                if target is None and node.value.endswith(".py"):
                    relative = (
                        path.parent / node.value
                        if not Path(node.value).is_absolute()
                        else Path(node.value)
                    )
                    if not relative.exists():
                        missing.append(
                            f"{path.relative_to(ROOT)} -> {node.value}"
                        )
                elif target is not None:
                    queue.append(target)

    missing = sorted(set(missing))
    if missing:
        print("missing local references:")
        print("\n".join(f"- {item}" for item in missing))
        raise SystemExit(1)

    reachable_relative = {
        str(path.relative_to(ROOT)).replace("\\", "/")
        for path in reached
    }
    unused = []
    for directory in SCAN_DIRS:
        for path in directory.glob("*.py"):
            relative = str(path.relative_to(ROOT)).replace("\\", "/")
            if relative not in reachable_relative and relative not in UTILITIES:
                unused.append(relative)
    if unused:
        print("unreachable Python files:")
        print("\n".join(f"- {item}" for item in sorted(unused)))
        raise SystemExit(1)
    print(f"reachable local modules: {len(reached)}")
    print("no unreachable runtime Python files")


if __name__ == "__main__":
    main()
