"""Waist-stripe ROI after LIP-20 trouser removal, then pooled peak annotation."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "tools"), str(ROOT / "src")]
from pool_to_width import pool_to_width  # noqa: E402
from torso_muscle_annot import schp  # noqa: E402
from torso_seg import load_bgr  # noqa: E402


DEFAULT_NAMES = (
    "001,002,003,004,006,010,011,013,015,029,035,037,038"
)


def imread(path, flag=cv2.IMREAD_UNCHANGED):
    image = cv2.imdecode(np.fromfile(str(path), np.uint8), flag)
    if image is None:
        raise RuntimeError(f"cannot read {path}")
    return image


def imwrite(path, image):
    ok, buffer = cv2.imencode(".png", image)
    if not ok:
        raise RuntimeError(f"cannot encode {path}")
    buffer.tofile(str(path))


def font(size):
    path = Path(r"C:\Windows\Fonts\msyh.ttc")
    return (
        ImageFont.truetype(str(path), size)
        if path.exists()
        else ImageFont.load_default(size=size)
    )


def detect_lower_garment(name, bgr, body, waist_row):
    classes = schp("lip", bgr, ROOT / "models")
    lip = np.isin(classes, (9, 10, 12)) & body
    lip[:waist_row + 1] = False
    lip_raw = lip.copy()
    source = "LIP-20 Pants+Jumpsuits+Skirt"
    # LIP can miss the lower garment entirely on unusual poses/occlusion.
    # Use ATR only when the LIP lower-garment evidence is very weak.
    if lip.sum() < 0.005 * body.sum():
        atr = schp("atr", bgr, ROOT / "models")
        fallback = np.isin(atr, (5, 6, 8)) & body
        fallback[:waist_row + 1] = False
        if fallback.sum() > max(2 * lip.sum(), 1000):
            lip = fallback
            source = "ATR fallback Pants+Skirt+Belt (LIP lower garment weak)"
    return lip_raw, lip, source, classes


def make_record(name):
    shoulder = json.loads(
        (ROOT / "out3" / f"shoulderline_{name}.json").read_text(
            encoding="utf-8"
        )
    )
    torso_image = imread(ROOT / "out3" / f"torso_{name}.png")
    torso = (
        torso_image[:, :, 3] > 127
        if torso_image.ndim == 3 and torso_image.shape[2] == 4
        else torso_image > 127
    )
    body = imread(
        ROOT / "out2" / f"bodymask_{name}.png", cv2.IMREAD_GRAYSCALE
    ) > 127
    bgr = load_bgr(ROOT / "data" / f"{name}.png")
    narrowest_path = ROOT / "out3" / f"greensplit_{name}.json"
    if narrowest_path.exists():
        narrowest = json.loads(
            narrowest_path.read_text(encoding="utf-8")
        )
        waist_row = int(narrowest["min_row"])
        left = int(narrowest["left_green"])
        right_exclusive = int(narrowest["right_green"]) + 1
        waist_source = "green-split narrowest abdomen line"
    else:
        waist = json.loads(
            (ROOT / "out3" / f"waist_{name}.json").read_text(
                encoding="utf-8"
            )
        )
        waist_row = int(waist["waist_row"])
        waist_x = np.flatnonzero(torso[waist_row])
        if not waist_x.size:
            raise RuntimeError(f"{name}: empty waist row")
        left = int(waist_x[0])
        right_exclusive = int(waist_x[-1]) + 1
        waist_source = "legacy waist mask fallback"
    pants_raw, pants, pants_source, classes = detect_lower_garment(
        name, bgr, body, waist_row
    )
    pants_edge = cv2.dilate(
        pants.astype(np.uint8), np.ones((5, 5), np.uint8), iterations=1
    ) > 0
    pants_edge[:waist_row + 1] = False
    retained = body & ~pants_edge

    pants_stripe = pants_raw[:, left:right_exclusive]
    pants_rows = np.flatnonzero(pants_stripe.any(axis=1))
    pants_rows = pants_rows[pants_rows > waist_row]
    if not pants_rows.size:
        raise RuntimeError(
            f"{name}: LIP-20 lower garment has no point below the waist"
        )
    pants_top_row = int(pants_rows.min())
    # The annotation region ends just before the first lower-garment pixel.
    bottom_exclusive = pants_top_row
    left_shoulder = [float(v) for v in shoulder["left_shoulder"]]
    right_shoulder = [float(v) for v in shoulder["right_shoulder"]]
    lowest = max((left_shoulder, right_shoulder), key=lambda p: p[1])
    top = int(round(lowest[1]))
    if bottom_exclusive <= top:
        raise RuntimeError(f"{name}: invalid top/bottom")
    arms = np.isin(classes, (14, 15)) & body
    record = {
        "image": name,
        "image_size": [int(torso.shape[1]), int(torso.shape[0])],
        "waist_row": waist_row,
        "waist_left": left,
        "waist_right_inclusive": right_exclusive - 1,
        "waist_width_px": right_exclusive - left,
        "waist_source": waist_source,
        "top": top,
        "bottom_exclusive": bottom_exclusive,
        "pants_top_row": pants_top_row,
        "pants_top_source": "LIP-20 Pants+Jumpsuits+Skirt",
        "rectangle_px": [
            left, top, right_exclusive, bottom_exclusive
        ],
        "rectangle_inclusive_px": [
            left, top, right_exclusive - 1, bottom_exclusive - 1
        ],
        "left_shoulder": left_shoulder,
        "right_shoulder": right_shoulder,
        "lowest_shoulder": lowest,
        "pants_pixels": int(pants.sum()),
        "pants_source": pants_source,
        "pants_edge_pixels": int((pants_edge & ~pants).sum()),
        "rule": (
            "the green-split narrowest abdomen line gives the rectangle "
            "width and waist row; LIP-20 lower clothing is removed with a "
            "2-pixel edge band; the rectangle bottom is the highest LIP-20 "
            "lower-clothing point below the waist; top is the lowest "
            "shoulder annotation"
        ),
        "lip20_arm_pixels_in_rectangle": int(
            arms[top:bottom_exclusive, left:right_exclusive].sum()
        ),
    }
    return record, retained, pants_edge, bgr, arms


def prepare_pooled(record, retained, pants_edge, bgr, output):
    left, top, right, bottom = record["rectangle_px"]
    crop = bgr[top:bottom, left:right].copy()
    retained_crop = retained[top:bottom, left:right]
    pants_crop = pants_edge[top:bottom, left:right]
    crop[pants_crop] = (255, 255, 255)
    bgra = cv2.cvtColor(crop, cv2.COLOR_BGR2BGRA)
    bgra[:, :, 3] = np.where(retained_crop, 255, 0).astype(np.uint8)
    pooled = pool_to_width(bgra, 50, "max")
    valid = pool_to_width(
        retained_crop[:, :, None].astype(np.uint8) * 255,
        50,
        "darkest",
    )[:, :, 0]
    pooled[:, :, 3] = valid
    imwrite(output / f"w50_{record['image']}.png", pooled)
    return pooled.shape[1], pooled.shape[0]


def draw_roi(record, bgr, output):
    image = bgr.copy()
    left, top, right, bottom = record["rectangle_px"]
    cv2.rectangle(
        image, (left, top), (right - 1, bottom - 1),
        (0, 255, 255), 3, cv2.LINE_AA
    )
    cv2.line(
        image, (left, record["waist_row"]),
        (right - 1, record["waist_row"]),
        (0, 0, 255), 2, cv2.LINE_AA
    )
    path = output / f"roi_{record['image']}.png"
    imwrite(path, image)
    return path


def draw_annotations(name, record, pooled_h, bgr, output):
    data = json.loads(
        (output / f"peakscolor50lipcut_{name}.json").read_text(
            encoding="utf-8"
        )
    )
    left, top, right, bottom = record["rectangle_px"]
    image = bgr.copy()
    cv2.rectangle(
        image, (left, top), (right - 1, bottom - 1),
        (0, 255, 255), 2, cv2.LINE_AA
    )

    def map_xy(x, y):
        px = left + (float(x) + 0.5) * (right - left) / 50.0
        py = top + (float(y) + 0.5) * (bottom - top) / pooled_h
        return int(round(px)), int(round(py))

    both = {tuple(point) for point in data["both_points"]}
    vertical = {tuple(point) for point in data["vertical_points"]}
    horizontal = {tuple(point) for point in data["horizontal_points"]}
    for point in sorted(vertical - both):
        x, y = map_xy(*point)
        cv2.circle(image, (x, y), 5, (0, 0, 0), -1, cv2.LINE_AA)
        cv2.circle(image, (x, y), 3, (0, 0, 255), -1, cv2.LINE_AA)
    for point in sorted(horizontal - both):
        x, y = map_xy(*point)
        cv2.circle(image, (x, y), 5, (0, 0, 0), -1, cv2.LINE_AA)
        cv2.circle(image, (x, y), 3, (255, 0, 0), -1, cv2.LINE_AA)
    for point in sorted(both):
        x, y = map_xy(*point)
        cv2.circle(image, (x, y), 7, (0, 0, 0), -1, cv2.LINE_AA)
        cv2.circle(image, (x, y), 4, (255, 255, 255), -1, cv2.LINE_AA)
    path = output / f"annotation_{name}.png"
    imwrite(path, image)
    return path


def filter_lip20_arm_points(name, record, pooled_h, arms, output):
    """Remove pooled peak points whose native centre lands on a LIP-20 arm."""
    path = output / f"peakscolor50lipcut_{name}.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    left, top, right, bottom = record["rectangle_px"]
    width, height = right - left, bottom - top

    def point_xy(point):
        x, y = point
        full_x = left + (float(x) + 0.5) * width / 50.0
        full_y = top + (float(y) + 0.5) * height / float(pooled_h)
        return int(round(full_x)), int(round(full_y))

    def on_arm(point):
        x, y = point_xy(point)
        return bool(
            arms[
                max(0, y - 1):min(arms.shape[0], y + 2),
                max(0, x - 1):min(arms.shape[1], x + 2),
            ].any()
        )

    vertical = {tuple(map(int, point)) for point in data["vertical_points"]}
    horizontal = {
        tuple(map(int, point)) for point in data["horizontal_points"]
    }
    both = {tuple(map(int, point)) for point in data["both_points"]}
    before = {
        "vertical": len(vertical),
        "horizontal": len(horizontal),
        "both": len(both),
    }
    before_vertical = set(vertical)
    before_horizontal = set(horizontal)
    before_both = set(both)
    hits = {}

    def keep(point):
        if point not in hits:
            hits[point] = on_arm(point)
        return not hits[point]

    vertical = {point for point in vertical if keep(point)}
    horizontal = {point for point in horizontal if keep(point)}
    both = {point for point in both if keep(point)}
    removed = {
        "vertical": before["vertical"] - len(vertical),
        "horizontal": before["horizontal"] - len(horizontal),
        "both": before["both"] - len(both),
    }
    data["raw_vertical_points"] = data["vertical_points"]
    data["raw_horizontal_points"] = data["horizontal_points"]
    data["raw_both_points"] = data["both_points"]
    data["vertical_points"] = [list(point) for point in sorted(vertical)]
    data["horizontal_points"] = [list(point) for point in sorted(horizontal)]
    data["both_points"] = [list(point) for point in sorted(both)]
    data["vertical_after"] = len(vertical)
    data["horizontal_after"] = len(horizontal)
    data["both_after"] = len(both)
    data["lip20_arm_filter"] = {
        "classes": [14, 15],
        "rule": (
            "remove annotation points whose native pooled-cell centre is "
            "within one pixel of LIP-20 left/right arm labels"
        ),
        "before": before,
        "removed": removed,
        "after": {
            "vertical": len(vertical),
            "horizontal": len(horizontal),
            "both": len(both),
        },
        "unique_annotation_points_removed": len(
            (before_vertical | before_horizontal | before_both)
            - (vertical | horizontal | both)
        ),
    }
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=1),
        encoding="utf-8",
    )
    return data


def contact_sheet(records, output):
    cols, cell_w, cell_h, margin, gap = 4, 390, 440, 20, 12
    rows = (len(records) + cols - 1) // cols
    sheet = Image.new(
        "RGB",
        (
            cols * cell_w + (cols + 1) * margin,
            rows * cell_h + (rows + 1) * margin,
        ),
        (245, 245, 245),
    )
    draw = ImageDraw.Draw(sheet)
    label_font = font(16)
    for index, record in enumerate(records):
        image = Image.open(record["annotation_png"]).convert("RGB")
        image.thumbnail((cell_w - 16, cell_h - 58), Image.Resampling.LANCZOS)
        col, row = index % cols, index // cols
        x = margin + col * (cell_w + gap)
        y = margin + row * (cell_h + gap)
        draw.rectangle(
            [x, y, x + cell_w - 1, y + cell_h - 1],
            fill="white",
            outline=(205, 205, 205),
        )
        draw.text(
            (x + 8, y + 7),
            f"{record['image']}  V/H/B "
            f"{record['vertical_after']}/{record['horizontal_after']}/"
            f"{record['both_after']}",
            fill=(20, 80, 100),
            font=label_font,
        )
        sheet.paste(
            image,
            (
                x + (cell_w - image.width) // 2,
                y + 34 + (cell_h - 58 - image.height) // 2,
            ),
        )
    path = output / "annotation_contact.png"
    sheet.save(path)
    return path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--names", default=DEFAULT_NAMES)
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "outfinal" / "lipcut_roi_annotation",
    )
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    names = [name for name in args.names.split(",") if name]
    records = []
    pooled_sizes = {}
    arm_masks = {}
    for name in names:
        record, retained, pants_edge, bgr, arms = make_record(name)
        arm_masks[name] = arms
        records.append(record)
        pooled_sizes[name] = prepare_pooled(
            record, retained, pants_edge, bgr, args.out
        )
        record["roi_png"] = str(draw_roi(record, bgr, args.out))
        print(
            f"{name}: ROI {record['rectangle_inclusive_px']}, "
            f"pants removed {record['pants_pixels']}"
        )

    subprocess.run(
        [
            sys.executable,
            str(ROOT / "pipeline" / "08_pool_peak_marks.py"),
            "--out",
            str(args.out),
            "--prefix",
            "w50",
            "--method",
            "gaussian",
            "--d",
            "2",
            "--paint",
            "vrh",
            "--tag",
            "peakscolor50lipcut",
            "--names",
            ",".join(names),
        ],
        check=True,
        cwd=str(ROOT),
    )
    for record in records:
        name = record["image"]
        payload = json.loads(
            (
                args.out / f"peakscolor50lipcut_{name}.json"
            ).read_text(encoding="utf-8")
        )
        payload = filter_lip20_arm_points(
            name,
            record,
            int(pooled_sizes[name][1]),
            arm_masks[name],
            args.out,
        )
        record["vertical_before_arm_removal"] = int(
            payload["lip20_arm_filter"]["before"]["vertical"]
        )
        record["horizontal_before_arm_removal"] = int(
            payload["lip20_arm_filter"]["before"]["horizontal"]
        )
        record["both_before_arm_removal"] = int(
            payload["lip20_arm_filter"]["before"]["both"]
        )
        record["arm_annotation_points_removed"] = int(
            payload["lip20_arm_filter"][
                "unique_annotation_points_removed"
            ]
        )
        record["vertical_after"] = int(payload["vertical_after"])
        record["horizontal_after"] = int(payload["horizontal_after"])
        record["both_after"] = int(payload["both_after"])
        record["w50_size"] = list(payload["size"])
        bgr = load_bgr(ROOT / "data" / f"{name}.png")
        annotation = draw_annotations(
            name,
            record,
            int(pooled_sizes[name][1]),
            bgr,
            args.out,
        )
        record["annotation_png"] = str(annotation)
        print(
            f"{name}: removed arm points "
            f"V/H/B={payload['lip20_arm_filter']['removed']['vertical']}/"
            f"{payload['lip20_arm_filter']['removed']['horizontal']}/"
            f"{payload['lip20_arm_filter']['removed']['both']}"
        )

    with (args.out / "roi_annotation.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "image",
                "waist_row",
                "left",
                "top",
                "right",
                "bottom_inclusive",
                "width_px",
                "height_px",
                "pants_pixels",
                "pants_source",
                "pants_top_row",
                "vertical",
                "horizontal",
                "both",
                "vertical_before_arm_removal",
                "horizontal_before_arm_removal",
                "both_before_arm_removal",
                "arm_annotation_points_removed",
                "lip20_arm_pixels_in_rectangle",
                "roi_png",
                "annotation_png",
            ]
        )
        for record in records:
            left, top, right, bottom = record["rectangle_px"]
            writer.writerow(
                [
                    record["image"],
                    record["waist_row"],
                    left,
                    top,
                    right - 1,
                    bottom - 1,
                    right - left,
                    bottom - top,
                    record["pants_pixels"],
                    record["pants_source"],
                    record["pants_top_row"],
                    record["vertical_after"],
                    record["horizontal_after"],
                    record["both_after"],
                    record["vertical_before_arm_removal"],
                    record["horizontal_before_arm_removal"],
                    record["both_before_arm_removal"],
                    record["arm_annotation_points_removed"],
                    record["lip20_arm_pixels_in_rectangle"],
                    record["roi_png"],
                    record["annotation_png"],
                ]
            )
    (args.out / "roi_annotation.json").write_text(
        json.dumps({"records": records}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    lines = [
        "# LIP-20 去裤后矩形池化标注",
        "",
        "裤区使用 LIP-20 的 Pants+Jumpsuits+Skirt，并去除 2 像素边缘带。",
        "矩形底边取腹部宽度带内 LIP-20 下身衣物的最高点。",
        "矩形内标注点按原图坐标检测 LIP-20 手臂类别 14/15；落入左右手臂掩码的标注点在统计前去除。",
        "",
        "| 图片 | 矩形范围 | 尺寸 | 竖点 | 横点 | 重合点 |",
        "| --- | --- | --- | ---: | ---: | ---: |",
    ]
    for record in records:
        left, top, right, bottom = record["rectangle_px"]
        lines.append(
            f"| {record['image']} | "
            f"[{left},{top},{right - 1},{bottom - 1}] | "
            f"{right-left}×{bottom-top} | {record['vertical_after']} | "
            f"{record['horizontal_after']} | {record['both_after']} |"
        )
    (args.out / "summary.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    contact = contact_sheet(records, args.out)
    print(f"wrote {args.out}")
    print(f"contact: {contact}")


if __name__ == "__main__":
    main()
