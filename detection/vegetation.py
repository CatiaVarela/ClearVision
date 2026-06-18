"""Détection de végétation (YOLO-World + secours HSV)."""

from __future__ import annotations

import cv2
import numpy as np

from config import (
    CONF_VEGETATION,
    HSV_FOLIAGE_RANGES,
    MIN_GROUND_FOLIAGE_RATIO,
    MIN_VEGETATION_AREA_RATIO,
    MIN_WORLD_BOXES_BEFORE_HSV,
    NUM_TREE_COLUMNS,
    TREE_BAND_BOTTOM,
    TREE_BAND_TOP,
    TURF_EXCLUDE_X,
    TURF_EXCLUDE_Y,
    VEGETATION_CLASSES,
)
from detection.filters import (
    box_area,
    filter_vegetation_boxes,
    is_facade_false_positive,
    is_overlapping,
)


def ground_foliage_ratio(image, vertical_start_ratio: float = 0.4) -> float:
    """Part de pixels « feuillage » dans la partie basse de l'image."""
    image_height, image_width = image.shape[:2]
    foliage_mask = build_foliage_mask(image)
    lower_region = foliage_mask[int(image_height * vertical_start_ratio) :, :]
    return lower_region.sum() / (255.0 * image_height * image_width)


def scene_has_ground_vegetation(image) -> bool:
    return ground_foliage_ratio(image) >= MIN_GROUND_FOLIAGE_RATIO


def build_foliage_mask(image) -> np.ndarray:
    """Masque du feuillage en arrière-plan, avec plages HSV saisonnières."""
    image_height, image_width = image.shape[:2]
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    combined_mask = np.zeros((image_height, image_width), np.uint8)
    for lower_bound, upper_bound in HSV_FOLIAGE_RANGES:
        range_mask = cv2.inRange(hsv_image, lower_bound, upper_bound)
        combined_mask = cv2.bitwise_or(combined_mask, range_mask)

    vertical_band = np.zeros((image_height, image_width), np.uint8)
    vertical_band[int(image_height * TREE_BAND_TOP) : int(image_height * TREE_BAND_BOTTOM), :] = 255
    combined_mask = cv2.bitwise_and(combined_mask, vertical_band)

    turf_zone = np.zeros((image_height, image_width), np.uint8)
    turf_zone[
        int(image_height * TURF_EXCLUDE_Y) :,
        int(image_width * TURF_EXCLUDE_X) :,
    ] = 255
    bright_turf = cv2.inRange(hsv_image, (35, 70, 90), (80, 255, 255))
    combined_mask = cv2.bitwise_and(
        combined_mask,
        cv2.bitwise_not(cv2.bitwise_and(bright_turf, turf_zone)),
    )

    close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (55, 12))
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, close_kernel)
    open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, open_kernel, 1)
    return combined_mask


def detect_vegetation_world(world_model, image) -> list[tuple]:
    world_model.set_classes(VEGETATION_CLASSES)
    results = world_model.predict(image, conf=CONF_VEGETATION, imgsz=1280, verbose=False)
    found_boxes = []

    for result in results:
        if result.boxes is None:
            continue
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            class_id = int(box.cls[0])
            label_key = world_model.names[class_id]
            confidence = float(box.conf[0])
            found_boxes.append((x1, y1, x2, y2, label_key, confidence))

    return filter_vegetation_boxes(found_boxes, image)


def detect_tree_line(image) -> list[tuple]:
    """
    Découpe la haie d'arbres en bandes verticales (vue de parc au sol).
    Plus stable que le watershed pour ce type de scène.
    """
    image_height, image_width = image.shape[:2]
    image_area = image_height * image_width
    foliage_mask = build_foliage_mask(image)

    band_top = int(image_height * TREE_BAND_TOP)
    band_bottom = int(image_height * TREE_BAND_BOTTOM)
    band_mask = foliage_mask[band_top:band_bottom, :]
    minimum_pixels = 0.008 * image_area

    found_boxes = []
    for column_index in range(NUM_TREE_COLUMNS):
        column_start = int(column_index * image_width / NUM_TREE_COLUMNS)
        column_end = int((column_index + 1) * image_width / NUM_TREE_COLUMNS)
        column_patch = band_mask[:, column_start:column_end]

        if column_patch.sum() < minimum_pixels:
            continue

        row_indices, column_indices = np.where(column_patch > 0)
        box_x1 = column_start + int(column_indices.min())
        box_x2 = column_start + int(column_indices.max())
        box_y1 = band_top + int(row_indices.min())
        box_y2 = band_top + int(row_indices.max())

        if box_area(box_x1, box_y1, box_x2, box_y2) / image_area < MIN_VEGETATION_AREA_RATIO:
            continue
        if is_facade_false_positive(box_x1, box_y1, box_x2, box_y2, image):
            continue

        found_boxes.append((box_x1, box_y1, box_x2, box_y2, "tree_line", 0.6))

    return found_boxes


def collect_vegetation_boxes(
    source_image,
    world_model,
    *,
    fast: bool = False,
) -> tuple[list[tuple], str, bool]:
    """Retourne (boîtes, mode_description, hsv_utilisé)."""
    if fast:
        if scene_has_ground_vegetation(source_image):
            vegetation_boxes = detect_tree_line(source_image)
            mode = "rapide (YOLO + haie d'arbres)"
            return vegetation_boxes, mode, True
        return [], "rapide (YOLO seul)", False

    if world_model is None:
        raise ValueError("world_model requis si fast=False")

    vegetation_boxes = detect_vegetation_world(world_model, source_image)
    used_hsv_fallback = False

    if (
        scene_has_ground_vegetation(source_image)
        and len(vegetation_boxes) < MIN_WORLD_BOXES_BEFORE_HSV
    ):
        vegetation_boxes.extend(detect_tree_line(source_image))
        used_hsv_fallback = True

    mode = (
        "YOLO-World + haie d'arbres (secours)"
        if used_hsv_fallback
        else "YOLO-World"
    )
    return vegetation_boxes, mode, used_hsv_fallback
