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
    PERSON_GROUP_CENTER_DISTANCE_RATIO,
    PERSON_GROUP_MAX_GAP_RATIO,
    PERSON_GROUP_MIN_COUNT,
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


def select_nearby_obstacles(
    detections: list[dict],
    *,
    max_distance_m: float,
    max_count: int | None = None,
) -> list[dict]:
    """
    Garde uniquement les obstacles proches triés par distance.
    Exclut végétation et objets au-delà de max_distance_m.
    """
    from clearvision_voice import is_path_obstacle

    nearby_obstacles = []
    for detection in detections:
        label_key = detection["label_key"]
        source = detection.get("source", "coco_or_world")
        if not is_path_obstacle(label_key, source):
            continue

        distance_m = detection.get("distance_m")
        if distance_m is None or distance_m > max_distance_m:
            continue

        nearby_obstacles.append(detection)

    nearby_obstacles.sort(key=lambda item: item["distance_m"])
    if max_count is not None:
        nearby_obstacles = nearby_obstacles[:max_count]
    return nearby_obstacles


def _horizontal_gap_between_boxes(bbox_a: tuple, bbox_b: tuple) -> int:
    ax1, _ay1, ax2, _ay2 = bbox_a
    bx1, _by1, bx2, _by2 = bbox_b
    if ax2 < bx1:
        return bx1 - ax2
    if bx2 < ax1:
        return ax1 - bx2
    return 0


def _boxes_can_form_person_group(
    bbox_a: tuple,
    bbox_b: tuple,
    image_width: int,
    image_height: int,
) -> bool:
    ax1, ay1, ax2, ay2 = bbox_a
    bx1, by1, bx2, by2 = bbox_b

    horizontal_gap = _horizontal_gap_between_boxes(bbox_a, bbox_b)
    max_horizontal_gap = image_width * PERSON_GROUP_MAX_GAP_RATIO

    vertical_overlap = max(0, min(ay2, by2) - max(ay1, by1))
    minimum_height = max(1, min(ay2 - ay1, by2 - by1))

    if horizontal_gap <= max_horizontal_gap and vertical_overlap >= minimum_height * 0.25:
        return True

    center_a_x = (ax1 + ax2) / 2.0
    center_a_y = (ay1 + ay2) / 2.0
    center_b_x = (bx1 + bx2) / 2.0
    center_b_y = (by1 + by2) / 2.0
    center_distance = ((center_a_x - center_b_x) ** 2 + (center_a_y - center_b_y) ** 2) ** 0.5
    return center_distance <= image_width * PERSON_GROUP_CENTER_DISTANCE_RATIO


def _merge_person_group(members: list[dict]) -> dict:
    merged_x1 = min(member["bbox"][0] for member in members)
    merged_y1 = min(member["bbox"][1] for member in members)
    merged_x2 = max(member["bbox"][2] for member in members)
    merged_y2 = max(member["bbox"][3] for member in members)
    person_count = len(members)
    closest_distance = min(member["distance_m"] for member in members if member.get("distance_m") is not None)
    best_score = max(member.get("score") or 0.0 for member in members)

    if person_count >= PERSON_GROUP_MIN_COUNT:
        label_fr = f"Groupe de personnes ({person_count})"
        label_key = "person_group"
    else:
        label_fr = members[0].get("label_fr", "personne")
        label_key = "person"

    return {
        "label_key": label_key,
        "label_fr": label_fr,
        "bbox": (merged_x1, merged_y1, merged_x2, merged_y2),
        "score": best_score,
        "source": members[0].get("source", "coco_or_world"),
        "distance_m": closest_distance,
        "person_count": person_count,
    }


def group_nearby_persons(
    detections: list[dict],
    image_width: int,
    image_height: int,
) -> list[dict]:
    """
    Regroupe les personnes proches en une seule détection « Groupe de personnes ».
    Les autres obstacles (voitures, etc.) restent inchangés.
    """
    persons = [detection for detection in detections if detection.get("label_key") == "person"]
    other_obstacles = [detection for detection in detections if detection.get("label_key") != "person"]

    if len(persons) < PERSON_GROUP_MIN_COUNT:
        return detections

    parent = list(range(len(persons)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(index_a: int, index_b: int) -> None:
        root_a = find(index_a)
        root_b = find(index_b)
        if root_a != root_b:
            parent[root_b] = root_a

    for index_a in range(len(persons)):
        for index_b in range(index_a + 1, len(persons)):
            if _boxes_can_form_person_group(
                persons[index_a]["bbox"],
                persons[index_b]["bbox"],
                image_width,
                image_height,
            ):
                union(index_a, index_b)

    clusters: dict[int, list[dict]] = {}
    for index, person in enumerate(persons):
        cluster_root = find(index)
        clusters.setdefault(cluster_root, []).append(person)

    grouped_persons = [_merge_person_group(members) for members in clusters.values()]
    grouped_results = grouped_persons + other_obstacles
    grouped_results.sort(key=lambda item: item.get("distance_m") or 999.0)
    return grouped_results


def label_to_french(label_key: str) -> str:
    from object_learning.learned_labels import learned_label_fr

    learned = learned_label_fr(label_key)
    if learned:
        return learned
    return config.LABEL_FR.get(label_key, label_key.replace("_", " "))
