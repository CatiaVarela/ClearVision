"""Tests pipeline (nécessite les modèles YOLO téléchargés)."""

import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from config import YOLO_COCO_PATH


@pytest.mark.skipif(not YOLO_COCO_PATH.exists(), reason="Modèle YOLO non téléchargé")
def test_detect_objects_in_image_fast_mode():
    from detection import detect_objects_in_image

    synthetic_image = np.zeros((480, 640, 3), dtype=np.uint8)
    synthetic_image[100:300, 200:400] = (128, 128, 128)

    annotated_image, detection_info = detect_objects_in_image(synthetic_image, fast=True)

    assert annotated_image.shape == synthetic_image.shape
    assert "detections" in detection_info
    assert "drawn_count" in detection_info


@pytest.mark.skipif(not YOLO_COCO_PATH.exists(), reason="Modèle YOLO non téléchargé")
def test_detections_to_json():
    from detection import detections_to_json

    detections = [
        {
            "label_key": "person",
            "label_fr": "personne",
            "bbox": (10, 20, 100, 200),
            "score": 0.95,
            "source": "coco",
        }
    ]
    serialized = detections_to_json(detections)
    assert serialized[0]["bbox"]["x1"] == 10
    assert serialized[0]["label_fr"] == "personne"
