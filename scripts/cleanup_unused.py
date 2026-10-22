"""Remove historical experiments that are not reachable from the public CLI."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PIPELINE = ROOT / "pipeline"
LEGACY = ROOT / "final"
TOOLS = ROOT / "tools"
SRC = ROOT / "src"

KEEP_PIPELINE = {
    "00_test_image.py",
    "01_run_pipeline.py",
    "02_prepare_image.py",
    "03_extract_subject.py",
    "04_annotate_parts.py",
    "05_cutout_torso.py",
    "06_cutout_crop.py",
    "07_detect_usable_arms.py",
    "08_pool_peak_marks.py",
    "09_analyze_abdomen.py",
    "10_export_abdomen_bundle.py",
    "11_export_total_score.py",
    "12_export_batch_summary.py",
    "13_export_row_function.py",
    "14_check_abdomen_score.py",
}

KEEP_TOOLS = {
    "annotate_edges.py",
    "arm_measure_lip20.py",
    "armmid_curve_group_objective.py",
    "export_lipcut_annotation_functions.py",
    "export_lipcut_center_open_angles.py",
    "fetch_birefnet.py",
    "fetch_dexined.py",
    "fetch_rtmpose.py",
    "fetch_rtmpose2.py",
    "lipcut_rowcol_intersections.py",
    "multiclass_probe.py",
    "pool_to_width.py",
    "pose_rtmpose_person.py",
    "poseline_new.py",
    "reclass_split.py",
    "render_lipcut_center_retained_original.py",
    "run_lipcut_center_objective.py",
    "run_lipcut_roi_annotation_batch.py",
    "torso_muscle_annot.py",
    "trunk_span.py",
    "trunkspan_min.py",
    "validate_lipcut_annotation_functions.py",
    "validate_lipcut_rowcol_intersections.py",
    "validate_lipcut_center_objective.py",
    "waist_shoulder_green.py",
}

KEEP_SRC = {
    "anatomy.py",
    "chest.py",
    "optimize_007.py",
    "pose_mp.py",
    "snap.py",
    "torso_seg.py",
}


def candidates(directory: Path, keep: set[str]):
    for path in directory.glob("*.py"):
        if path.name not in keep:
            yield path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--apply",
        action="store_true",
        help="delete the listed files instead of previewing",
    )
    args = parser.parse_args()
    doomed = sorted(
        list(candidates(PIPELINE, KEEP_PIPELINE))
        + list(candidates(TOOLS, KEEP_TOOLS))
        + list(candidates(SRC, KEEP_SRC))
        + (list(LEGACY.glob("*.py")) if LEGACY.exists() else [])
    )
    for path in doomed:
        resolved = path.resolve()
        allowed = (PIPELINE.resolve(), TOOLS.resolve(), SRC.resolve())
        if not any(base in resolved.parents for base in allowed) and not (
            LEGACY.exists() and LEGACY.resolve() in resolved.parents
        ):
            raise RuntimeError(f"unsafe cleanup target: {resolved}")
        print(("delete " if args.apply else "would delete ")
              + str(path.relative_to(ROOT)))
        if args.apply:
            path.unlink()

    for cache in (
        ROOT / "__pycache__",
        PIPELINE / "__pycache__",
        TOOLS / "__pycache__",
        SRC / "__pycache__",
        LEGACY / "__pycache__",
    ):
        if cache.exists():
            print(("delete " if args.apply else "would delete ")
                  + str(cache.relative_to(ROOT)))
            if args.apply:
                shutil.rmtree(cache)
    print(f"{len(doomed)} legacy Python files")


if __name__ == "__main__":
    main()
