"""Module de détection ClearVision."""

from detection.drawing import draw_detections
from detection.models import load_coco_model, load_world_model, reset_model_cache
from detection.pipeline import (
    detect_objects_in_image,
    detections_to_json,
    process_frame,
)

__all__ = [
    "detect_objects_in_image",
    "detections_to_json",
    "draw_detections",
    "load_coco_model",
    "load_world_model",
    "process_frame",
    "reset_model_cache",
]
