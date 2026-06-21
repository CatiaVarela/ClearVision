"""Pipeline principal de détection ClearVision."""

from __future__ import annotations

from config import CONF_COCO, CONF_PERSON, CUSTOM_MODEL_CONF, DEFAULT_WORLD_CATEGORIES
from detection.drawing import draw_detections
from detection.filters import (
    filter_by_min_area,
    group_nearby_persons,
    label_to_french,
    merge_coco_and_world,
    nms_boxes,
    select_nearby_obstacles,
    visible_vegetation_boxes,
)
from detection.models import load_custom_model
from detection.vegetation import collect_vegetation_boxes
from detection.world_detector import detect_with_world_categories


def detect_custom_objects(custom_model, source_image, confidence: float) -> list[tuple]:
    """Détections du modèle entraîné par l'utilisateur."""
    custom_boxes = []
    results = custom_model(source_image, conf=confidence, verbose=False)

    for result in results:
        if result.boxes is None:
            continue
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            class_id = int(box.cls[0])
            label_key = custom_model.names[class_id]
            confidence_score = float(box.conf[0])
            custom_boxes.append((x1, y1, x2, y2, label_key, confidence_score))

    return custom_boxes


def detection_to_dict(
    label_key: str,
    bbox: tuple,
    score: float,
    source: str,
    *,
    label_fr: str | None = None,
    distance_m: float | None = None,
) -> dict:
    return {
        "label_key": label_key,
        "label_fr": label_fr or label_to_french(label_key),
        "bbox": bbox,
        "score": score,
        "source": source,
        "distance_m": distance_m,
    }


def detect_coco_objects(yolo_model, source_image, predict_kwargs: dict) -> tuple[list[tuple], list[tuple]]:
    """Retourne (boîtes COCO, coordonnées des personnes)."""
    coco_boxes = []
    human_coordinates = []

    results_coco = yolo_model(source_image, **predict_kwargs)
    for result in results_coco:
        if result.boxes is None:
            continue
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            label_key = yolo_model.names[int(box.cls[0])]
            confidence = float(box.conf[0])
            bounding_box = (x1, y1, x2, y2)

            if label_key == "person":
                human_coordinates.append(bounding_box)

            coco_boxes.append((x1, y1, x2, y2, label_key, confidence))

    return coco_boxes, human_coordinates


def process_frame(
    source,
    yolo_model,
    world_model=None,
    *,
    fast: bool = False,
    estimate_distances: bool = False,
    world_categories: tuple[str, ...] | list[str] | None = None,
    skip_vegetation: bool = False,
    distance_focal_factor: float | None = None,
    obstacle_mode: bool = False,
    max_obstacle_distance_m: float | None = None,
    max_obstacles_display: int | None = None,
):
    """
    Analyse une frame BGR : COCO + YOLO-World + végétation HSV.
    obstacle_mode : obstacles proches uniquement, sans arbres/végétation, distances affichées.
    Retourne (image annotée, dict d'infos).
    """
    if obstacle_mode:
        skip_vegetation = True
        estimate_distances = True

    annotated_image = source.copy()
    image_height, image_width = annotated_image.shape[:2]
    image_area = image_height * image_width

    predict_kwargs = {"conf": CONF_PERSON if obstacle_mode else CONF_COCO, "verbose": False}
    if fast:
        predict_kwargs["imgsz"] = 960 if obstacle_mode else 640

    if estimate_distances:
        from clearvision_voice import estimate_distance_m, label_to_fr as voice_label_to_fr
    else:
        estimate_distance_m = None
        voice_label_to_fr = None

    if world_categories is None:
        world_categories = ("animaux",) if obstacle_mode else DEFAULT_WORLD_CATEGORIES

    coco_boxes, human_coordinates = detect_coco_objects(yolo_model, source, predict_kwargs)

    world_boxes = []
    if not fast and world_model is not None:
        world_boxes = detect_with_world_categories(world_model, source, world_categories)

    custom_model = load_custom_model()
    custom_boxes = []
    if custom_model is not None:
        custom_boxes = detect_custom_objects(custom_model, source, CUSTOM_MODEL_CONF)

    merged_boxes = merge_coco_and_world(coco_boxes, world_boxes)
    if custom_boxes:
        merged_boxes = merge_coco_and_world(merged_boxes, custom_boxes)
    merged_boxes = filter_by_min_area(merged_boxes, image_area)

    if skip_vegetation:
        vegetation_boxes = []
        vegetation_mode = "sans végétation (webcam/intérieur)"
        used_hsv = False
    else:
        vegetation_boxes, vegetation_mode, used_hsv = collect_vegetation_boxes(
            source,
            world_model,
            fast=fast,
        )
        vegetation_boxes = nms_boxes(vegetation_boxes)
    visible_vegetation = (
        [] if skip_vegetation else visible_vegetation_boxes(vegetation_boxes, human_coordinates, image_area)
    )

    all_detections: list[dict] = []

    for x1, y1, x2, y2, label_key, score in merged_boxes:
        bounding_box = (x1, y1, x2, y2)
        distance_m = None
        if estimate_distances:
            distance_m = estimate_distance_m(
                bounding_box,
                image_height,
                label_key,
                focal_length_factor=distance_focal_factor,
            )
            label_fr = voice_label_to_fr(label_key)
        else:
            label_fr = label_to_french(label_key)

        all_detections.append(
            detection_to_dict(
                label_key,
                bounding_box,
                score,
                "coco_or_world",
                label_fr=label_fr,
                distance_m=distance_m,
            )
        )

    existing_bboxes = [detection["bbox"] for detection in all_detections]

    for x1, y1, x2, y2, label_key, score in visible_vegetation:
        bounding_box = (x1, y1, x2, y2)
        if any(
            _boxes_overlap_significantly(bounding_box, existing_bbox)
            for existing_bbox in existing_bboxes
        ):
            continue

        distance_m = None
        if estimate_distances:
            distance_m = estimate_distance_m(
                bounding_box,
                image_height,
                label_key,
                focal_length_factor=distance_focal_factor,
            )

        all_detections.append(
            detection_to_dict(
                label_key,
                bounding_box,
                score,
                "vegetation",
                distance_m=distance_m,
            )
        )
        existing_bboxes.append(bounding_box)

    if obstacle_mode:
        from clearvision_voice import MAX_DISTANCE_M, MAX_OBSTACLES_DISPLAY

        distance_limit = max_obstacle_distance_m if max_obstacle_distance_m is not None else MAX_DISTANCE_M
        display_limit = max_obstacles_display if max_obstacles_display is not None else MAX_OBSTACLES_DISPLAY
        detections_to_draw = select_nearby_obstacles(
            all_detections,
            max_distance_m=distance_limit,
            max_count=None,
        )
        detections_to_draw = group_nearby_persons(
            detections_to_draw,
            image_width,
            image_height,
        )
        if display_limit is not None:
            detections_to_draw = detections_to_draw[:display_limit]
        vegetation_mode = "obstacles proches (sans végétation)"
    else:
        detections_to_draw = all_detections

    drawn_count = draw_detections(
        annotated_image,
        detections_to_draw,
        show_distance=estimate_distances,
    )

    return annotated_image, {
        "detections": detections_to_draw if obstacle_mode else all_detections,
        "all_detections": all_detections,
        "drawn_count": drawn_count,
        "veg_drawn": sum(1 for d in all_detections if d["source"] == "vegetation"),
        "mode": vegetation_mode,
        "used_hsv": used_hsv,
        "obstacle_mode": obstacle_mode,
    }


def _boxes_overlap_significantly(box_a: tuple, box_b: tuple, threshold: float = 0.5) -> bool:
    from detection.filters import iou

    return iou((*box_a, "", 0.0), (*box_b, "", 0.0)) >= threshold


def detect_objects_in_image(
    image,
    *,
    fast: bool = False,
    estimate_distances: bool = False,
    world_categories: tuple[str, ...] | list[str] | None = None,
):
    """Point d'entrée haut niveau : charge les modèles et analyse une image."""
    from detection.models import load_coco_model, load_world_model

    yolo_model = load_coco_model()
    world_model = None if fast else load_world_model()
    return process_frame(
        image,
        yolo_model,
        world_model,
        fast=fast,
        estimate_distances=estimate_distances,
        world_categories=world_categories,
    )


def detections_to_json(detections: list[dict]) -> list[dict]:
    """Sérialise les détections pour l'API JSON."""
    serialized = []
    for detection in detections:
        x1, y1, x2, y2 = detection["bbox"]
        serialized.append(
            {
                "label_key": detection["label_key"],
                "label_fr": detection["label_fr"],
                "bbox": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
                "score": detection.get("score"),
                "source": detection.get("source"),
                "distance_m": detection.get("distance_m"),
            }
        )
    return serialized
