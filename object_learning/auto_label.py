"""Annotation automatique des images avec YOLO-World."""

from __future__ import annotations

from pathlib import Path

import cv2
import yaml


def _bbox_to_yolo_line(class_id: int, x1: int, y1: int, x2: int, y2: int, image_width: int, image_height: int) -> str:
    center_x = ((x1 + x2) / 2.0) / image_width
    center_y = ((y1 + y2) / 2.0) / image_height
    box_width = (x2 - x1) / image_width
    box_height = (y2 - y1) / image_height
    return f"{class_id} {center_x:.6f} {center_y:.6f} {box_width:.6f} {box_height:.6f}"


def auto_label_image_folder(
    image_paths: list[Path],
    labels_directory: Path,
    *,
    object_name: str,
    class_id: int,
    confidence_threshold: float = 0.20,
) -> list[tuple[Path, Path]]:
    """
    Annoter chaque image avec YOLO-World (prompt = nom de l'objet).
    Retourne les paires (image, label) réussies.
    """
    from ultralytics import YOLOWorld

    from config import YOLO_WORLD_PATH

    labels_directory.mkdir(parents=True, exist_ok=True)
    world_model = YOLOWorld(str(YOLO_WORLD_PATH))
    world_model.set_classes([object_name])

    labeled_pairs: list[tuple[Path, Path]] = []

    print(f"Annotation automatique ({len(image_paths)} image(s), classe {class_id})...")

    for image_path in image_paths:
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"  Ignoré (illisible) : {image_path.name}")
            continue

        image_height, image_width = image.shape[:2]
        results = world_model.predict(
            image,
            conf=confidence_threshold,
            imgsz=1280,
            verbose=False,
        )

        label_lines: list[str] = []
        for result in results:
            if result.boxes is None:
                continue
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                label_lines.append(
                    _bbox_to_yolo_line(class_id, x1, y1, x2, y2, image_width, image_height)
                )

        if not label_lines:
            print(f"  Aucune détection : {image_path.name} (image ignorée)")
            continue

        label_path = labels_directory / f"{image_path.stem}.txt"
        label_path.write_text("\n".join(label_lines) + "\n", encoding="utf-8")
        labeled_pairs.append((image_path, label_path))
        print(f"  OK : {image_path.name} → {len(label_lines)} boîte(s)")

    if not labeled_pairs:
        raise RuntimeError(
            "Aucune image n'a pu être annotée automatiquement. "
            "Essayez un nom en anglais (--search-en) ou baissez --conf."
        )

    return labeled_pairs


def save_learned_class_metadata(
    metadata_path: Path,
    class_slug: str,
    object_name: str,
    label_fr: str,
    class_id: int,
) -> None:
    metadata: dict = {}
    if metadata_path.exists():
        metadata = yaml.safe_load(metadata_path.read_text(encoding="utf-8")) or {}

    metadata[class_slug] = {
        "object_name": object_name,
        "label_fr": label_fr,
        "class_id": class_id,
    }
    metadata_path.write_text(
        yaml.safe_dump(metadata, allow_unicode=True, sort_keys=True),
        encoding="utf-8",
    )
