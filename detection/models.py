"""Chargement et cache des modèles YOLO."""

from __future__ import annotations

from ultralytics import YOLO, YOLOWorld

from config import USE_CUSTOM_MODEL, YOLO_COCO_PATH, YOLO_CUSTOM_PATH, YOLO_WORLD_PATH


_cached_coco_model = None
_cached_world_model = None
_cached_custom_model = None


def load_coco_model():
    global _cached_coco_model
    if _cached_coco_model is None:
        _cached_coco_model = YOLO(str(YOLO_COCO_PATH))
    return _cached_coco_model


def load_custom_model():
    """Modèle entraîné sur objets appris (None si désactivé ou absent)."""
    global _cached_custom_model
    if not USE_CUSTOM_MODEL or not YOLO_CUSTOM_PATH.exists():
        return None
    if _cached_custom_model is None:
        _cached_custom_model = YOLO(str(YOLO_CUSTOM_PATH))
    return _cached_custom_model


def load_world_model():
    global _cached_world_model
    if _cached_world_model is None:
        _cached_world_model = YOLOWorld(str(YOLO_WORLD_PATH))
    return _cached_world_model


def reset_model_cache() -> None:
    """Utile pour les tests ou après changement de config."""
    global _cached_coco_model, _cached_world_model, _cached_custom_model
    _cached_coco_model = None
    _cached_world_model = None
    _cached_custom_model = None
