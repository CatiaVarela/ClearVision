"""Tests unitaires des filtres de détection."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from detection.filters import filter_by_min_area, iou, merge_coco_and_world, nms_boxes


def test_iou_identical_boxes():
    box = (10, 10, 50, 50, "dog", 0.9)
    assert iou(box, box) == 1.0


def test_iou_no_overlap():
    box_a = (0, 0, 10, 10, "dog", 0.9)
    box_b = (20, 20, 30, 30, "cat", 0.8)
    assert iou(box_a, box_b) == 0.0


def test_nms_removes_duplicate():
    boxes = [
        (10, 10, 50, 50, "dog", 0.9),
        (12, 12, 48, 48, "dog", 0.7),
        (200, 200, 250, 250, "cat", 0.8),
    ]
    kept = nms_boxes(boxes, threshold=0.3)
    assert len(kept) == 2


def test_merge_coco_and_world_keeps_higher_score():
    coco = [(10, 10, 50, 50, "dog", 0.6)]
    world = [(12, 12, 48, 48, "dog", 0.9)]
    merged = merge_coco_and_world(coco, world, threshold=0.3)
    assert len(merged) == 1
    assert merged[0][5] == 0.9


def test_filter_by_min_area():
    boxes = [
        (0, 0, 2, 2, "tiny", 0.9),
        (0, 0, 100, 100, "big", 0.8),
    ]
    filtered = filter_by_min_area(boxes, image_area=640 * 480, min_ratio=0.001)
    assert len(filtered) == 1
    assert filtered[0][4] == "big"


def test_select_nearby_obstacles():
    from detection.filters import select_nearby_obstacles

    detections = [
        {"label_key": "person", "label_fr": "personne", "bbox": (0, 0, 10, 10), "distance_m": 2.0, "source": "coco_or_world"},
        {"label_key": "car", "label_fr": "voiture", "bbox": (0, 0, 10, 10), "distance_m": 6.0, "source": "coco_or_world"},
        {"label_key": "tree_line", "label_fr": "arbre", "bbox": (0, 0, 10, 10), "distance_m": 1.0, "source": "vegetation"},
        {"label_key": "dog", "label_fr": "chien", "bbox": (0, 0, 10, 10), "distance_m": 3.5, "source": "coco_or_world"},
    ]
    nearby = select_nearby_obstacles(detections, max_distance_m=4.0, max_count=2)
    assert len(nearby) == 2
    assert nearby[0]["label_key"] == "person"
    assert nearby[1]["label_key"] == "dog"


def test_group_nearby_persons():
    from detection.filters import group_nearby_persons

    detections = [
        {"label_key": "person", "label_fr": "personne", "bbox": (100, 100, 150, 300), "distance_m": 3.0, "score": 0.9, "source": "coco_or_world"},
        {"label_key": "person", "label_fr": "personne", "bbox": (160, 105, 210, 305), "distance_m": 2.5, "score": 0.8, "source": "coco_or_world"},
        {"label_key": "person", "label_fr": "personne", "bbox": (500, 100, 550, 300), "distance_m": 4.0, "score": 0.7, "source": "coco_or_world"},
        {"label_key": "car", "label_fr": "voiture", "bbox": (600, 200, 700, 350), "distance_m": 2.0, "score": 0.95, "source": "coco_or_world"},
    ]
    grouped = group_nearby_persons(detections, image_width=1280, image_height=720)

    group_labels = [item["label_key"] for item in grouped]
    assert "person_group" in group_labels
    assert group_labels.count("person") == 1

    person_group = next(item for item in grouped if item["label_key"] == "person_group")
    assert person_group["person_count"] == 2
    assert person_group["distance_m"] == 2.5
    assert "Groupe de personnes (2)" in person_group["label_fr"]
