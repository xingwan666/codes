from io import BytesIO
import re
import traceback

import cv2
import easyocr
import numpy as np
from flask import Flask, jsonify, render_template, request
from PIL import Image, ImageOps

app = Flask(__name__)

TARGET_PRESSURE_COUNT = 14
STRICT_PRESSURE_RANGE = (200, 250)
RELAXED_PRESSURE_RANGE = (180, 260)
MAX_OCR_DIM = 1400
MIN_OCR_DIM = 800
ROI_LIMIT = 2
MIN_CANDIDATE_PROB = 0.30
EARLY_STOP_COUNT = 14
SLOT_TEMPLATE = [
    ("top_left", 0.44, 0.15),
    ("top_right", 0.56, 0.15),
    ("left_1_left", 0.27, 0.41),
    ("left_1_right", 0.38, 0.41),
    ("left_2_left", 0.27, 0.56),
    ("left_2_right", 0.38, 0.56),
    ("left_3_left", 0.27, 0.72),
    ("left_3_right", 0.38, 0.72),
    ("right_1_left", 0.62, 0.41),
    ("right_1_right", 0.73, 0.41),
    ("right_2_left", 0.62, 0.56),
    ("right_2_right", 0.73, 0.56),
    ("right_3_left", 0.62, 0.72),
    ("right_3_right", 0.73, 0.72),
]

# Load OCR once when the app starts.
reader = easyocr.Reader(["en"], gpu=False)


def load_uploaded_image(file_storage):
    file_bytes = file_storage.read()
    if not file_bytes:
        raise ValueError("empty upload")

    pil_img = Image.open(BytesIO(file_bytes))
    pil_img = ImageOps.exif_transpose(pil_img).convert("RGB")
    return pil_img


def resize_for_ocr(pil_img, max_dim=MAX_OCR_DIM, min_dim=MIN_OCR_DIM):
    width, height = pil_img.size
    longest_edge = max(width, height)

    scale = 1.0
    if longest_edge > max_dim:
        scale = max_dim / longest_edge
    elif longest_edge < min_dim:
        scale = min_dim / longest_edge

    if scale == 1.0:
        return pil_img

    new_size = (max(1, int(width * scale)), max(1, int(height * scale)))
    return pil_img.resize(new_size, Image.LANCZOS)


def pil_to_gray(pil_img):
    rgb = np.array(pil_img)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)


def add_unique_region(regions, pil_img, name, bounds):
    width, height = pil_img.size
    x1, y1, x2, y2 = bounds
    x1 = max(0, min(width - 1, int(x1)))
    y1 = max(0, min(height - 1, int(y1)))
    x2 = max(x1 + 1, min(width, int(x2)))
    y2 = max(y1 + 1, min(height, int(y2)))

    crop_w = x2 - x1
    crop_h = y2 - y1
    if crop_w < width * 0.25 or crop_h < height * 0.2:
        return

    region_box = (x1, y1, x2, y2)
    for existing_name, existing_img, existing_box in regions:
        ex1, ey1, ex2, ey2 = existing_box
        inter_x1 = max(x1, ex1)
        inter_y1 = max(y1, ey1)
        inter_x2 = min(x2, ex2)
        inter_y2 = min(y2, ey2)
        inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
        crop_area = crop_w * crop_h
        existing_area = (ex2 - ex1) * (ey2 - ey1)
        union_area = crop_area + existing_area - inter_area
        if union_area and inter_area / union_area > 0.75:
            return

    regions.append((name, pil_img.crop(region_box), region_box))


def detect_display_regions(pil_img):
    width, height = pil_img.size
    gray = pil_to_gray(pil_img)

    detection_scale = 1.0
    longest_edge = max(width, height)
    if longest_edge > 1200:
        detection_scale = 1200 / longest_edge
        detection_img = cv2.resize(
            gray,
            (max(1, int(width * detection_scale)), max(1, int(height * detection_scale))),
            interpolation=cv2.INTER_AREA,
        )
    else:
        detection_img = gray

    blurred = cv2.GaussianBlur(detection_img, (5, 5), 0)
    edges = cv2.Canny(blurred, 40, 120)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    total_area = float(detection_img.shape[0] * detection_img.shape[1])
    scored_boxes = []

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        box_area = w * h
        if box_area < total_area * 0.05 or box_area > total_area * 0.95:
            continue

        aspect_ratio = w / max(h, 1)
        if aspect_ratio < 0.9 or aspect_ratio > 5.5:
            continue

        contour_area = cv2.contourArea(contour)
        fill_ratio = contour_area / max(box_area, 1)
        if fill_ratio < 0.35:
            continue

        center_x = x + w / 2
        center_y = y + h / 2
        offset_x = abs(center_x - detection_img.shape[1] / 2) / max(detection_img.shape[1], 1)
        offset_y = abs(center_y - detection_img.shape[0] / 2) / max(detection_img.shape[0], 1)

        score = (
            box_area / total_area * 2.5
            + fill_ratio * 0.8
            - offset_x * 0.8
            - offset_y * 0.6
            - abs(aspect_ratio - 2.0) * 0.12
        )
        scored_boxes.append((score, x, y, w, h))

    scored_boxes.sort(reverse=True)
    regions = []

    for index, (_, x, y, w, h) in enumerate(scored_boxes[:ROI_LIMIT], start=1):
        pad_x = int(w * 0.03)
        pad_y = int(h * 0.04)
        scaled_bounds = (
            (x - pad_x) / detection_scale,
            (y - pad_y) / detection_scale,
            (x + w + pad_x) / detection_scale,
            (y + h + pad_y) / detection_scale,
        )
        add_unique_region(regions, pil_img, f"roi_{index}", scaled_bounds)

    # Central fallback often works better than the full frame when the operator
    # roughly centers the display.
    add_unique_region(
        regions,
        pil_img,
        "center_crop",
        (width * 0.12, height * 0.2, width * 0.88, height * 0.82),
    )
    regions.append(("full_frame", pil_img, (0, 0, width, height)))

    return regions


def build_region_variants(region_img):
    region_img = resize_for_ocr(region_img)
    gray = pil_to_gray(region_img)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8)).apply(gray)
    blurred = cv2.GaussianBlur(gray, (0, 0), 2.2)
    sharpened = cv2.addWeighted(gray, 1.6, blurred, -0.6, 0)
    _, otsu = cv2.threshold(clahe, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _, otsu_inv = cv2.threshold(clahe, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    return [
        ("gray", gray),
        ("clahe", clahe),
        ("otsu", otsu),
        ("otsu_inv", otsu_inv),
        ("sharpened", sharpened),
    ]


def read_region_text(img_np):
    return reader.readtext(
        img_np,
        detail=1,
        paragraph=False,
        decoder="greedy",
        allowlist="0123456789",
        text_threshold=0.6,
        low_text=0.45,
        min_size=12,
        mag_ratio=1.2,
        contrast_ths=0.2,
        adjust_contrast=0.5,
        add_margin=0.02,
        width_ths=0.35,
        batch_size=1,
    )


def detect_text_boxes(img_np):
    horizontal_list, free_list = reader.detect(
        img_np,
        min_size=12,
        text_threshold=0.6,
        low_text=0.45,
        width_ths=0.35,
        add_margin=0.02,
        bbox_min_score=0.2,
        bbox_min_size=3,
        optimal_num_chars=3,
    )
    return horizontal_list[0] if horizontal_list else []


def crop_box(img_np, box):
    x_min, x_max, y_min, y_max = [int(v) for v in box]
    height, width = img_np.shape[:2]
    x_min = max(0, min(width - 1, x_min))
    y_min = max(0, min(height - 1, y_min))
    x_max = max(x_min + 1, min(width, x_max))
    y_max = max(y_min + 1, min(height, y_max))
    return img_np[y_min:y_max, x_min:x_max]


def recognize_box_digits(box_img):
    if box_img.size == 0:
        return []

    scaled = box_img
    h, w = box_img.shape[:2]
    if max(h, w) < 90:
        scale = 90 / max(h, w)
        scaled = cv2.resize(
            box_img,
            (max(1, int(w * scale)), max(1, int(h * scale))),
            interpolation=cv2.INTER_CUBIC,
        )

    variants = [scaled]
    _, otsu = cv2.threshold(scaled, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    variants.append(otsu)

    results = []
    for variant in variants:
        box_results = reader.recognize(
            variant,
            horizontal_list=[[0, variant.shape[1], 0, variant.shape[0]]],
            free_list=[],
            decoder="greedy",
            batch_size=1,
            allowlist="0123456789",
            detail=1,
            paragraph=False,
            contrast_ths=0.05,
            adjust_contrast=0.7,
            filter_ths=0.002,
        )
        results.extend(box_results)
    return results


def detect_and_recognize_boxes(region_name, variant_name, img_np):
    box_candidates = []
    horizontal_boxes = detect_text_boxes(img_np)
    for box_index, box in enumerate(horizontal_boxes):
        crop = crop_box(img_np, box)
        if crop.size == 0:
            continue

        crop_h, crop_w = crop.shape[:2]
        aspect_ratio = crop_w / max(crop_h, 1)
        if crop_h < 10 or crop_w < 12 or aspect_ratio < 0.4 or aspect_ratio > 4.5:
            continue

        for _, text, prob in recognize_box_digits(crop):
            box_candidates.append(
                (
                    [[box[0], box[2]], [box[1], box[2]], [box[1], box[3]], [box[0], box[3]]],
                    text,
                    prob,
                )
            )

    if box_candidates:
        return box_candidates, f"{region_name}:{variant_name}:boxes"

    return read_region_text(img_np), f"{region_name}:{variant_name}:full"


def compute_candidate_score(val, prob, digit_count):
    score = prob
    if digit_count == 3:
        score += 0.12
    if STRICT_PRESSURE_RANGE[0] <= val <= STRICT_PRESSURE_RANGE[1]:
        score += 0.25
    elif RELAXED_PRESSURE_RANGE[0] <= val <= RELAXED_PRESSURE_RANGE[1]:
        score += 0.12
    return score


def merge_ocr_results(all_results):
    candidates = []
    for entry in all_results:
        source_name = entry["source"]
        results = entry["results"]
        img_h, img_w = entry["shape"]
        region_name = source_name.split(":", 1)[0]
        for bbox, text, prob in results:
            if not text or prob < MIN_CANDIDATE_PROB:
                continue

            digits = re.sub(r"[^0-9]", "", text)
            if len(digits) not in (2, 3):
                continue

            try:
                val = int(digits)
            except ValueError:
                continue

            if not (80 <= val <= 320):
                continue

            x_min = min(pt[0] for pt in bbox)
            x_max = max(pt[0] for pt in bbox)
            y_min = min(pt[1] for pt in bbox)
            y_max = max(pt[1] for pt in bbox)

            width = x_max - x_min
            height = y_max - y_min
            if width <= 0 or height <= 0:
                continue

            candidates.append(
                {
                    "val": val,
                    "text": digits,
                    "x": (x_min + x_max) / 2,
                    "y": (y_min + y_max) / 2,
                    "nx": ((x_min + x_max) / 2) / max(img_w, 1),
                    "ny": ((y_min + y_max) / 2) / max(img_h, 1),
                    "w": width,
                    "h": height,
                    "prob": prob,
                    "score": compute_candidate_score(val, prob, len(digits)),
                    "bbox": bbox,
                    "source": source_name,
                    "region": region_name,
                }
            )

    if not candidates:
        return []

    candidates.sort(key=lambda candidate: candidate["score"], reverse=True)

    merged = []
    for candidate in candidates:
        duplicate_found = False
        for kept in merged:
            center_dx = abs(candidate["x"] - kept["x"])
            center_dy = abs(candidate["y"] - kept["y"])
            width_limit = min(candidate["w"], kept["w"]) * 0.8
            height_limit = min(candidate["h"], kept["h"]) * 0.8
            same_position = center_dx <= width_limit and center_dy <= height_limit
            same_value = candidate["val"] == kept["val"]
            if same_position or (same_value and center_dy <= height_limit * 0.8):
                duplicate_found = True
                break
        if not duplicate_found:
            merged.append(candidate)

    return merged


def cluster_rows_by_y(candidates, expected_rows=4):
    if not candidates:
        return []

    y_values = np.array([candidate["y"] for candidate in candidates], dtype=float)
    row_count = min(expected_rows, len(candidates))
    if row_count <= 1:
        return [candidates]

    centers = np.linspace(y_values.min(), y_values.max(), row_count)
    assignments = np.zeros(len(candidates), dtype=int)

    for _ in range(10):
        assignments = np.argmin(np.abs(y_values[:, None] - centers[None, :]), axis=1)
        new_centers = np.array(
            [
                y_values[assignments == index].mean() if np.any(assignments == index) else centers[index]
                for index in range(row_count)
            ],
            dtype=float,
        )
        if np.allclose(new_centers, centers):
            break
        centers = new_centers

    rows = [[] for _ in range(row_count)]
    for candidate, row_index in zip(candidates, assignments):
        rows[row_index].append(candidate)

    return [row for row, center in sorted(zip(rows, centers), key=lambda item: item[1]) if row]


def sort_pressures_spatially(candidates):
    rows = cluster_rows_by_y(candidates, expected_rows=4)
    ordered = []
    for row in rows:
        row.sort(key=lambda candidate: candidate["x"])
        ordered.extend(row)
    return ordered


def assign_region_candidates_to_slots(candidates):
    assigned = []
    used_indices = set()

    for slot_name, slot_x, slot_y in SLOT_TEMPLATE:
        best_index = None
        best_rank = None

        for index, candidate in enumerate(candidates):
            if index in used_indices:
                continue

            dx = abs(candidate["nx"] - slot_x)
            dy = abs(candidate["ny"] - slot_y)
            if dx > 0.16 or dy > 0.14:
                continue

            rank = dy * 1.8 + dx * 1.2 - candidate["score"] * 0.08
            if best_rank is None or rank < best_rank:
                best_rank = rank
                best_index = index

        if best_index is not None:
            used_indices.add(best_index)
            assigned.append(candidates[best_index])

    return assigned


def assign_candidates_to_layout(candidates):
    if not candidates:
        return []

    grouped = {}
    for candidate in candidates:
        grouped.setdefault(candidate["region"], []).append(candidate)

    preferred_region_order = ["roi_1", "center_crop", "roi_2", "full_frame"]
    normalized_groups = {
        region_name: sorted(region_candidates, key=lambda candidate: candidate["score"], reverse=True)
        for region_name, region_candidates in grouped.items()
    }

    assigned = []
    used_ids = set()

    for slot_name, slot_x, slot_y in SLOT_TEMPLATE:
        best_candidate = None
        best_rank = None

        for region_priority, region_name in enumerate(preferred_region_order):
            for candidate in normalized_groups.get(region_name, []):
                candidate_id = id(candidate)
                if candidate_id in used_ids:
                    continue

                dx = abs(candidate["nx"] - slot_x)
                dy = abs(candidate["ny"] - slot_y)
                if dx > 0.17 or dy > 0.15:
                    continue

                rank = dy * 1.8 + dx * 1.2 + region_priority * 0.02 - candidate["score"] * 0.08
                if best_rank is None or rank < best_rank:
                    best_rank = rank
                    best_candidate = candidate

        if best_candidate is not None:
            used_ids.add(id(best_candidate))
            assigned.append(best_candidate)

    return assigned


def filter_valid_tire_pressures(candidates):
    if not candidates:
        return []

    strict = [candidate for candidate in candidates if STRICT_PRESSURE_RANGE[0] <= candidate["val"] <= STRICT_PRESSURE_RANGE[1]]
    relaxed = [candidate for candidate in candidates if RELAXED_PRESSURE_RANGE[0] <= candidate["val"] <= RELAXED_PRESSURE_RANGE[1]]

    selected = strict if len(strict) >= 8 else relaxed
    if not selected:
        return []

    if len(selected) > TARGET_PRESSURE_COUNT:
        selected = sorted(selected, key=lambda candidate: candidate["score"], reverse=True)[:TARGET_PRESSURE_COUNT]
    return sort_pressures_spatially(selected)


def should_stop_early(valid_candidates):
    return len(valid_candidates) >= EARLY_STOP_COUNT


def recognize_tire_pressures(pil_img):
    region_candidates = detect_display_regions(pil_img)
    all_ocr_results = []
    best_valid = []
    strategies_used = []

    variant_priority = {
        "roi_1": ("clahe", "gray", "otsu"),
        "roi_2": ("clahe",),
        "center_crop": ("clahe", "gray"),
        "full_frame": ("clahe",),
    }

    for region_name, region_img, _ in region_candidates:
        variants = build_region_variants(region_img)
        preferred_names = variant_priority.get(region_name, ("gray", "clahe"))
        ordered_variants = [
            variant for variant in variants if variant[0] in preferred_names
        ] + [
            variant for variant in variants if variant[0] not in preferred_names
        ]

        for variant_name, img_np in ordered_variants:
            if region_name == "full_frame" and variant_name != "clahe":
                continue
            if region_name != "roi_1" and variant_name not in preferred_names:
                continue
            if region_name == "roi_1" and variant_name not in preferred_names:
                continue

            results, source_name = detect_and_recognize_boxes(region_name, variant_name, img_np)
            all_ocr_results.append(
                {
                    "source": source_name,
                    "results": results,
                    "shape": img_np.shape[:2],
                }
            )
            strategies_used.append(source_name)

            merged = merge_ocr_results(all_ocr_results)
            valid = filter_valid_tire_pressures(merged)
            if len(valid) > len(best_valid):
                best_valid = valid

            if should_stop_early(best_valid):
                return best_valid, all_ocr_results, strategies_used

    return best_valid, all_ocr_results, strategies_used


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/ocr_fill", methods=["POST"])
def ocr_fill():
    try:
        if "image" not in request.files:
            return jsonify({"success": False, "msg": "未接收到文件"})

        img_file = request.files["image"]
        if img_file.filename == "":
            return jsonify({"success": False, "msg": "文件名为空"})

        pil_img = load_uploaded_image(img_file)
        valid_data, all_ocr_results, strategies_used = recognize_tire_pressures(pil_img)
        tire_pressures = [candidate["val"] for candidate in valid_data[:TARGET_PRESSURE_COUNT]]

        raw_texts = []
        for entry in all_ocr_results:
            source_name = entry["source"]
            results = entry["results"]
            for _, text, prob in results:
                if text and text.strip():
                    raw_texts.append(f"[{source_name}] {text} ({prob:.2f})")

        return jsonify(
            {
                "success": True,
                "tire_pressures": tire_pressures,
                "recognized_count": len(tire_pressures),
                "debug": {
                    "raw_recognitions": raw_texts[:50],
                    "total_raw": sum(len(entry["results"]) for entry in all_ocr_results),
                    "merged_unique": len(merge_ocr_results(all_ocr_results)),
                    "valid_tire_data": len(valid_data),
                    "strategies_used": strategies_used,
                },
            }
        )

    except Exception as exc:
        traceback.print_exc()
        return jsonify({"success": False, "msg": f"识别失败: {exc}"})


@app.route("/check_rules", methods=["POST"])
def check_rules():
    try:
        data = request.get_json() or {}
        pressures = data.get("pressures", [])

        if len(pressures) != TARGET_PRESSURE_COUNT:
            return jsonify({"success": False, "msg": "胎压数据数量错误，需要14个（2前轮+12主轮）"})

        front_tires = pressures[:2]
        main_tires = pressures[2:]

        low_pressure_tires = []
        for index, pressure in enumerate(pressures):
            if pressure < 218:
                if index < 2:
                    low_pressure_tires.append(f"前轮{index + 1}（{pressure} PSI）")
                else:
                    low_pressure_tires.append(f"主轮{index - 1}（{pressure} PSI）")

        front_result = process_front_tires(front_tires)
        main_result = process_main_tires(main_tires)

        result = {
            "success": True,
            "rule1": {
                "has_low": len(low_pressure_tires) > 0,
                "tires": low_pressure_tires,
                "suggestion": "需要重点检查气门芯、压力传感器、热熔塔是否渗漏"
                if low_pressure_tires
                else "所有轮胎压力均 >= 218 PSI，无渗漏风险",
            },
            "front_check": front_result,
            "main_check": main_result,
        }

        return jsonify(result)

    except Exception as exc:
        traceback.print_exc()
        return jsonify({"success": False, "msg": f"校验失败: {exc}"})


def process_front_tires(front_tires):
    if len(front_tires) != 2:
        return {"error": "前轮数据数量错误"}

    low_pressure = min(front_tires)
    high_pressure = max(front_tires)
    difference = high_pressure - low_pressure
    ratio = (difference / high_pressure) * 100 if high_pressure else 0

    if ratio < 5:
        suggestion = "无需进一步处理"
    elif ratio < 10:
        suggestion = f"将较低胎压的前轮充气至另一前轮胎压（{high_pressure} PSI）"
    elif ratio < 20:
        suggestion = f"更换较低胎压的前轮（{low_pressure} PSI）"
    else:
        suggestion = "更换两个前轮"

    return {
        "tires": front_tires,
        "low_pressure": low_pressure,
        "high_pressure": high_pressure,
        "difference": difference,
        "ratio": round(ratio, 2),
        "suggestion": suggestion,
    }


def process_main_tires(main_tires):
    if len(main_tires) != 12:
        return {"error": "主轮数据数量错误"}

    min_pressure = min(main_tires)
    min_index = main_tires.index(min_pressure)
    other_tires = [pressure for idx, pressure in enumerate(main_tires) if idx != min_index]
    avg_pressure = sum(other_tires) / len(other_tires) if other_tires else 0
    difference = avg_pressure - min_pressure
    ratio = (difference / avg_pressure) * 100 if avg_pressure else 0

    if ratio < 5:
        suggestion = "无需进一步处理"
    elif ratio < 10:
        suggestion = f"将最低胎压的主轮（{min_pressure} PSI）充气至平均胎压（{round(avg_pressure, 1)} PSI）"
    elif ratio < 20:
        suggestion = f"更换最低胎压的主轮（{min_pressure} PSI）"
    else:
        suggestion = f"更换最低胎压的主轮（{min_pressure} PSI）及同轴主轮（小车架中间轴除外）"

    return {
        "tires": main_tires,
        "min_pressure": min_pressure,
        "avg_other_pressure": round(avg_pressure, 1),
        "difference": round(difference, 1),
        "ratio": round(ratio, 2),
        "suggestion": suggestion,
    }


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
