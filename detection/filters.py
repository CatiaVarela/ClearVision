"""Filtres géométriques et fusion de détections."""

from __future__ import annotations

import config
from config import (
    FACADE_MAX_HEIGHT_RATIO,
    FACADE_MAX_Y_CENTER,
    MIN_BOX_AREA_RATIO,
    MIN_VEGETATION_AREA_RATIO,
    NMS_IOU_COCO_WORLD,
    NMS_IOU_THRESHOLD,
    OVERLAP_REJECT_RATIO,
)


def box_area(x1: int, y1: int, x2: int, y2: int) -> int:
    return max(0, x2 - x1) * max(0, y2 - y1)


def iou(box_a: tuple, box_b: tuple) -> float:
    ax1, ay1, ax2, ay2 = box_a[:4]
    bx1, by1, bx2, by2 = box_b[:4]
    intersection_x1 = max(ax1, bx1)
    intersection_y1 = max(ay1, by1)
    intersection_x2 = min(ax2, bx2)
    intersection_y2 = min(ay2, by2)
    if intersection_x2 <= intersection_x1 or intersection_y2 <= intersection_y1:
        return 0.0
    intersection_area = (intersection_x2 - intersection_x1) * (intersection_y2 - intersection_y1)
    union_area = box_area(ax1, ay1, ax2, ay2) + box_area(bx1, by1, bx2, by2) - intersection_area
    return intersection_area / union_area if union_area > 0 else 0.0


def nms_boxes(
    boxes: list[tuple],
    threshold: float = NMS_IOU_THRESHOLD,
) -> list[tuple]:
    """boxes: list of (x1, y1, x2, y2, label_key, score, ...)"""
    sorted_boxes = sorted(boxes, key=lambda detection: detection[5], reverse=True)
    kept_boxes = []
    for candidate in sorted_boxes:
        if all(iou(candidate, kept) < threshold for kept in kept_boxes):
            kept_boxes.append(candidate)
    return kept_boxes


def is_overlapping(
    vegetation_box: tuple,
    human_boxes: list[tuple],
    overlap_ratio: float = OVERLAP_REJECT_RATIO,
) -> bool:
    veg_x1, veg_y1, veg_x2, veg_y2 = vegetation_box
    vegetation_area = box_area(veg_x1, veg_y1, veg_x2, veg_y2)
    if vegetation_area <= 0:
        return False

    for human_x1, human_y1, human_x2, human_y2 in human_boxes:
        intersection_x1 = max(veg_x1, human_x1)
        intersection_y1 = max(veg_y1, human_y1)
        intersection_x2 = min(veg_x2, human_x2)
        intersection_y2 = min(veg_y2, human_y2)

        if intersection_x2 > intersection_x1 and intersection_y2 > intersection_y1:
            intersection_area = (intersection_x2 - intersection_x1) * (intersection_y2 - intersection_y1)
            if intersection_area / vegetation_area > overlap_ratio:
                return True
    return False


def filter_by_min_area(
    boxes: list[tuple],
    image_area: int,
    min_ratio: float = MIN_BOX_AREA_RATIO,
) -> list[tuple]:
    return [
        box
        for box in boxes
        if box_area(box[0], box[1], box[2], box[3]) / image_area >= min_ratio
    ]


def merge_coco_and_world(
    coco_boxes: list[tuple],
    world_boxes: list[tuple],
    threshold: float = NMS_IOU_COCO_WORLD,
) -> list[tuple]:
    """Fusionne COCO et World : en cas de chevauchement, garde le score le plus élevé."""
    all_boxes = sorted(coco_boxes + world_boxes, key=lambda detection: detection[5], reverse=True)
    merged = []
    for candidate in all_boxes:
        overlapping = False
        for index, kept in enumerate(merged):
            if iou(candidate, kept) >= threshold:
                overlapping = True
                if candidate[5] > kept[5]:
                    merged[index] = candidate
                break
        if not overlapping:
            merged.append(candidate)
    return merged


def visible_vegetation_boxes(
    boxes: list[tuple],
    human_coordinates: list[tuple],
    image_area: int,
) -> list[tuple]:
    visible = []
    for x1, y1, x2, y2, label_key, score in boxes:
        if box_area(x1, y1, x2, y2) / image_area < MIN_VEGETATION_AREA_RATIO:
            continue
        if is_overlapping((x1, y1, x2, y2), human_coordinates):
            continue
        visible.append((x1, y1, x2, y2, label_key, score))
    return visible


def region_of_interest(image, x1: int, y1: int, x2: int, y2: int):
    image_height, image_width = image.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(image_width, x2), min(image_height, y2)
    return image[y1:y2, x1:x2]


def looks_like_vertical_pattern(region_of_interest_image) -> bool:
    """Volets / fenêtres : arêtes verticales dominantes."""
    import cv2
    import numpy as np

    if (
        region_of_interest_image.size == 0
        or region_of_interest_image.shape[0] < 12
        or region_of_interest_image.shape[1] < 12
    ):
        return False
    gray = cv2.cvtColor(region_of_interest_image, cv2.COLOR_BGR2GRAY)
    gradient_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gradient_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    mean_gradient_x = float(np.abs(gradient_x).mean())
    mean_gradient_y = float(np.abs(gradient_y).mean())
    if mean_gradient_y < 1e-6:
        return False
    width_over_height = region_of_interest_image.shape[1] / region_of_interest_image.shape[0]
    return mean_gradient_x > mean_gradient_y * 1.45 and width_over_height > 0.35


def is_facade_false_positive(x1: int, y1: int, x2: int, y2: int, image) -> bool:
    """Rejette les boîtes compactes en étage (volets verts, fenêtres)."""
    image_height, image_width = image.shape[:2]
    box_width, box_height = x2 - x1, y2 - y1
    if box_width <= 0 or box_height <= 0:
        return True

    center_y_ratio = (y1 + y2) / (2.0 * image_height)
    if center_y_ratio > FACADE_MAX_Y_CENTER or box_height / image_height > FACADE_MAX_HEIGHT_RATIO:
        return False

    aspect_ratio = box_width / box_height
    if not (0.3 <= aspect_ratio <= 3.5):
        return False

    roi = region_of_interest(image, x1, y1, x2, y2)
    return looks_like_vertical_pattern(roi)


def filter_vegetation_boxes(boxes: list[tuple], image) -> list[tuple]:
    return [
        box
        for box in boxes
        if not is_facade_false_positive(box[0], box[1], box[2], box[3], image)
    ]


def label_to_french(label_key: str) -> str:
    return config.LABEL_FR.get(label_key, label_key.replace("_", " "))
