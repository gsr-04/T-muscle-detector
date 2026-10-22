"""Detect usable arms and annotate only the selected arm(s)."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
PY = Path(os.environ.get("TMUSCLE_PYTHON", sys.executable))
OUT3 = ROOT / "out3"
OUTFINAL = ROOT / "outfinal"
REQUIRED = (
    "left_shoulder", "left_elbow", "left_wrist",
    "right_shoulder", "right_elbow", "right_wrist",
)


def imread(path):
    image = cv2.imdecode(np.fromfile(str(path), np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"cannot read {path}")
    return image


def imwrite(path, image):
    ok, buffer = cv2.imencode(".png", image)
    if not ok:
        raise RuntimeError(f"cannot encode {path}")
    buffer.tofile(str(path))


def elbow_angle(shoulder, elbow, wrist):
    upper = np.asarray(shoulder, float) - np.asarray(elbow, float)
    fore = np.asarray(wrist, float) - np.asarray(elbow, float)
    cosine = float(
        upper @ fore
        / max(1e-9, np.linalg.norm(upper) * np.linalg.norm(fore))
    )
    return float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))


def choose_arm(sides):
    left = sides["left"]
    right = sides["right"]
    if left["wrist_above_elbow"] != right["wrist_above_elbow"]:
        lower = (
            "left" if not left["wrist_above_elbow"] else "right"
        )
        other = "right" if lower == "left" else "left"
        if sides[lower]["elbow_below_shoulder"]:
            return lower, f"rule1: {lower} wrist below elbow"
        if sides[other]["elbow_below_shoulder"]:
            return other, f"rule1 fallback: use {other}"
        return "both", "rule1 fallback: neither elbow below shoulder"
    if left["wrist_above_elbow"] and right["wrist_above_elbow"]:
        angles = (left["angle"], right["angle"])
        if all(value >= 90 for value in angles):
            return "both", "rule2: both elbow angles >= 90"
        if all(value < 90 for value in angles):
            side = "left" if left["angle"] >= right["angle"] else "right"
            return side, "rule2: both < 90, use larger elbow angle"
        side = "left" if left["angle"] >= 90 else "right"
        return side, "rule2: only one elbow angle >= 90"
    return "both", "both wrists are below their elbows"


def load_selected_measurement(name):
    path = OUT3 / f"armmeasure_{name}.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def draw_selected_arm(name, selected, sides, measurement, image_size):
    image = imread(ROOT / "data" / f"{name}.png")
    body_path = ROOT / "out2" / f"bodymask_{name}.png"
    if body_path.exists():
        body = cv2.imdecode(
            np.fromfile(str(body_path), np.uint8),
            cv2.IMREAD_GRAYSCALE,
        ) > 127
        contours, _ = cv2.findContours(
            body.astype(np.uint8),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_NONE,
        )
        cv2.drawContours(
            image, contours, -1, (0, 255, 0), 2, cv2.LINE_AA
        )
    chosen = (
        ["left", "right"] if selected == "both" else [selected]
    )
    for side in chosen:
        record = sides[side]
        shoulder = tuple(
            int(round(value)) for value in record["shoulder"]
        )
        elbow = tuple(int(round(value)) for value in record["elbow"])
        wrist = tuple(int(round(value)) for value in record["wrist"])
        cv2.line(image, shoulder, elbow, (0, 220, 0), 5, cv2.LINE_AA)
        cv2.line(image, elbow, wrist, (0, 220, 0), 5, cv2.LINE_AA)
        for point, colour in (
            (shoulder, (255, 0, 0)),
            (elbow, (0, 200, 255)),
            (wrist, (0, 0, 255)),
        ):
            cv2.circle(image, point, 8, (0, 0, 0), -1, cv2.LINE_AA)
            cv2.circle(image, point, 5, colour, -1, cv2.LINE_AA)
        cv2.putText(
            image,
            f"{side} {record['angle']:.0f} deg",
            (wrist[0] + 8, max(20, wrist[1] - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 220, 0),
            2,
            cv2.LINE_AA,
        )
        measured = measurement.get("sides", {}).get(side, {})
        for piece in measured.get("pieces") or []:
            base_key = (
                "M1_base" if piece.get("at") == "M1" else "M2_base"
            )
            base = tuple(
                int(round(value))
                for value in measured.get(base_key, [])
            )
            end = tuple(
                int(round(value))
                for value in piece.get("end", [])
            )
            if len(base) != 2 or len(end) != 2:
                continue
            perpendicular_colour = (
                (0, 165, 255)
                if piece.get("clamped")
                else (0, 255, 255)
            )
            cv2.line(
                image,
                base,
                end,
                perpendicular_colour,
                2,
                cv2.LINE_AA,
            )
            cv2.circle(
                image, end, 4, (255, 255, 255), -1, cv2.LINE_AA
            )
    ratio = measurement.get("effective_ratio")
    score = measurement.get("effective_score")
    label = f"{name}: usable {selected}"
    if ratio is not None:
        label += f"  ratio {float(ratio):.3f}  score {float(score):.2f}"
    cv2.putText(
        image, label, (8, 28), cv2.FONT_HERSHEY_SIMPLEX,
        0.66, (0, 0, 0), 5, cv2.LINE_AA
    )
    cv2.putText(
        image, label, (8, 28), cv2.FONT_HERSHEY_SIMPLEX,
        0.66, (0, 255, 255), 2, cv2.LINE_AA
    )
    target = OUTFINAL / f"available_arm_{name}.png"
    imwrite(target, image)
    return target


def draw_selected_armmid(name):
    """Build armmid from the usable-arm overlay, adding torso references."""
    target = OUTFINAL / f"armmid_{name}.png"
    source = OUTFINAL / f"available_arm_{name}.png"
    image = imread(source)

    waist_path = OUT3 / f"waist_{name}.json"
    if waist_path.exists():
        waist = json.loads(waist_path.read_text(encoding="utf-8"))
        row = int(waist.get("waist_row") or 0)
        left = waist.get("green_left")
        right = waist.get("green_right")
        if row and left is not None and right is not None:
            left, right = int(left), int(right)
            cv2.line(
                image,
                (left, row),
                (right, row),
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.line(
                image,
                (0, row),
                (image.shape[1] - 1, row),
                (0, 0, 170),
                1,
                cv2.LINE_AA,
            )
            cv2.putText(
                image,
                f"belly {row}",
                (6, max(14, row - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )

    shoulder_path = OUT3 / f"shoulderline_{name}.json"
    if shoulder_path.exists():
        shoulder = json.loads(
            shoulder_path.read_text(encoding="utf-8")
        )
        path = np.array(
            [
                shoulder["left_boundary"],
                shoulder["left_shoulder"],
                shoulder["right_shoulder"],
                shoulder["right_boundary"],
            ],
            np.int32,
        )
        cv2.polylines(
            image,
            [path],
            False,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )
        for point in (
            shoulder["left_boundary"],
            shoulder["right_boundary"],
        ):
            cv2.circle(
                image,
                (int(point[0]), int(point[1])),
                4,
                (255, 0, 0),
                -1,
                cv2.LINE_AA,
            )
    imwrite(target, image)
    return target


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--names", required=True)
    parser.add_argument("--out", type=Path, default=OUT3)
    parser.add_argument("--skip-measure", action="store_true")
    args = parser.parse_args()
    names = [name for name in args.names.split(",") if name]
    args.out.mkdir(parents=True, exist_ok=True)
    OUTFINAL.mkdir(parents=True, exist_ok=True)

    records = []
    selectable = []
    for name in names:
        pose_path = args.out / f"pose_{name}.json"
        if not pose_path.exists():
            raise RuntimeError(f"missing pose file: {pose_path}")
        pose_doc = json.loads(pose_path.read_text(encoding="utf-8"))
        source_image = imread(ROOT / "data" / f"{name}.png")
        source_height, source_width = source_image.shape[:2]
        landmarks = {
            item["name"]: item for item in pose_doc["landmarks"]
        }
        missing = [key for key in REQUIRED if key not in landmarks]
        if missing:
            records.append(
                {
                    "image": name,
                    "selected": "none",
                    "reason": f"missing landmarks: {missing}",
                    "sides": {},
                }
            )
            continue
        sides = {}
        for side in ("left", "right"):
            shoulder = [
                float(landmarks[f"{side}_shoulder"]["x"]),
                float(landmarks[f"{side}_shoulder"]["y"]),
            ]
            elbow = [
                float(landmarks[f"{side}_elbow"]["x"]),
                float(landmarks[f"{side}_elbow"]["y"]),
            ]
            wrist = [
                float(landmarks[f"{side}_wrist"]["x"]),
                float(landmarks[f"{side}_wrist"]["y"]),
            ]
            sides[side] = {
                "shoulder": shoulder,
                "elbow": elbow,
                "wrist": wrist,
                "wrist_above_elbow": wrist[1] < elbow[1],
                "elbow_below_shoulder": elbow[1] > shoulder[1],
                "angle": elbow_angle(shoulder, elbow, wrist),
            }
        selected, reason = choose_arm(sides)
        record = {
            "image": name,
            "size": [
                source_width,
                source_height,
            ],
            "selected": selected,
            "usable_sides": (
                ["left", "right"] if selected == "both" else [selected]
            ),
            "reason": reason,
            "sides": sides,
        }
        (args.out / f"armselect_{name}.json").write_text(
            json.dumps(record, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        records.append(record)
        selectable.append(name)

    if selectable and not args.skip_measure:
        command = [
            str(PY),
            str(ROOT / "tools" / "arm_measure_lip20.py"),
            "--names", ",".join(selectable),
            "--out", str(args.out),
        ]
        print("  $ " + " ".join(command[2:]))
        result = subprocess.run(
            command, cwd=str(ROOT), capture_output=True, text=True
        )
        output = (result.stdout or "") + (result.stderr or "")
        if result.returncode != 0:
            raise SystemExit(
                f"arm_measure_lip20.py failed ({result.returncode}):\n"
                + "\n".join(output.splitlines()[-8:])
            )
        print("\n".join(output.splitlines()[-10:]))

    with (OUTFINAL / "available_arms.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "image",
                "selected",
                "usable_sides",
                "left_angle_deg",
                "right_angle_deg",
                "reason",
                "overlay_png",
            ]
        )
        for record in records:
            if record["selected"] == "none":
                continue
            measurement = load_selected_measurement(record["image"])
            overlay = draw_selected_arm(
                record["image"],
                record["selected"],
                record["sides"],
                measurement,
                record["size"],
            )
            record["effective_ratio"] = measurement.get("effective_ratio")
            record["effective_score"] = measurement.get("effective_score")
            record["overlay_png"] = str(overlay)
            record["armmid_png"] = str(
                draw_selected_armmid(record["image"])
            )
            (args.out / f"available_arm_{record['image']}.json").write_text(
                json.dumps(record, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            writer.writerow(
                [
                    record["image"],
                    record["selected"],
                    " ".join(record["usable_sides"]),
                    f"{record['sides']['left']['angle']:.3f}",
                    f"{record['sides']['right']['angle']:.3f}",
                    record["reason"],
                    str(overlay),
                ]
            )
            print(
                f"{record['image']}: usable={record['selected']} "
                f"ratio={record['effective_ratio']} "
                f"score={record['effective_score']}"
            )


if __name__ == "__main__":
    main()
