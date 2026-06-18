"""Détection YOLO-World par catégorie (animaux, plantes, objets)."""

from __future__ import annotations

from config import (
    CONF_WORLD_ANIMALS,
    CONF_WORLD_OBJECTS,
    CONF_WORLD_PLANTS,
    DEFAULT_WORLD_CATEGORIES,
    WORLD_CATEGORY_MAP,
    WORLD_CLASS_BATCH_SIZE,
)
from detection.filters import filter_vegetation_boxes


CONFIDENCE_BY_CATEGORY = {
    "animaux": CONF_WORLD_ANIMALS,
    "plantes": CONF_WORLD_PLANTS,
    "objets": CONF_WORLD_OBJECTS,
}


def _chunk_class_list(class_list: list[str], batch_size: int) -> list[list[str]]:
    return [
        class_list[start_index : start_index + batch_size]
        for start_index in range(0, len(class_list), batch_size)
    ]


def _predict_world_batch(world_model, image, class_batch: list[str], confidence_threshold: float) -> list[tuple]:
    world_model.set_classes(class_batch)
    results = world_model.predict(
        image,
        conf=confidence_threshold,
        imgsz=1280,
        verbose=False,
    )

    batch_detections = []
    for result in results:
        if result.boxes is None:
            continue
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            class_id = int(box.cls[0])
            label_key = world_model.names[class_id]
            confidence = float(box.conf[0])
            batch_detections.append((x1, y1, x2, y2, label_key, confidence))

    return batch_detections


def detect_with_world_categories(
    world_model,
    image,
    categories: tuple[str, ...] | list[str] | None = None,
) -> list[tuple]:
    """
    Lance YOLO-World pour chaque catégorie demandée.
    Les longues listes sont découpées en lots pour couvrir plus de classes.
    Retourne des tuples (x1, y1, x2, y2, label_key, score).
    """
    if categories is None:
        categories = DEFAULT_WORLD_CATEGORIES

    all_detections = []
    for category_name in categories:
        class_list = WORLD_CATEGORY_MAP.get(category_name)
        if not class_list:
            continue

        confidence_threshold = CONFIDENCE_BY_CATEGORY.get(category_name, CONF_WORLD_OBJECTS)
        class_batches = _chunk_class_list(class_list, WORLD_CLASS_BATCH_SIZE)

        for class_batch in class_batches:
            batch_detections = _predict_world_batch(
                world_model,
                image,
                class_batch,
                confidence_threshold,
            )
            all_detections.extend(batch_detections)

    if "plantes" in categories:
        return filter_vegetation_boxes(all_detections, image)

    return all_detections
