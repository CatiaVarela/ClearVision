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
