"""Dessin unifié des détections (rectangles bleus)."""

from __future__ import annotations

import cv2

from config import DETECTION_BOX_COLOR, DETECTION_BOX_THICKNESS, DETECTION_TEXT_COLOR


def draw_detections(
    image,
    detections: list[dict],
    *,
    show_distance: bool = False,
) -> int:
    """
    Dessine toutes les détections en bleu sur l'image.
    Chaque détection : {label_fr, bbox, distance_m?}.
    Retourne le nombre de boîtes dessinées.
    """
    drawn_count = 0
    for detection in detections:
        x1, y1, x2, y2 = detection["bbox"]
        label_text = detection["label_fr"]

        if show_distance and detection.get("distance_m") is not None:
            label_text = f"{label_text} {detection['distance_m']:.0f}m"

        cv2.rectangle(
            image,
            (x1, y1),
            (x2, y2),
            DETECTION_BOX_COLOR,
            DETECTION_BOX_THICKNESS,
        )
        cv2.putText(
            image,
            label_text,
            (x1, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            DETECTION_TEXT_COLOR,
            1,
        )
        drawn_count += 1

    return drawn_count
