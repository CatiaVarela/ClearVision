"""Construction et mise à jour du dataset YOLO custom."""

from __future__ import annotations

import random
import shutil
from pathlib import Path

import yaml


def load_dataset_classes(dataset_yaml_path: Path) -> dict[int, str]:
    if not dataset_yaml_path.exists():
        return {}

    content = yaml.safe_load(dataset_yaml_path.read_text(encoding="utf-8")) or {}
    names = content.get("names", {})
    if isinstance(names, dict):
        return {int(class_id): str(class_name) for class_id, class_name in names.items()}
    return {index: name for index, name in enumerate(names)}


def _dataset_root_path(dataset_yaml_path: Path) -> Path:
    return dataset_yaml_path.parent.resolve()


def _save_dataset_yaml(dataset_yaml_path: Path, class_map: dict[int, str]) -> None:
    dataset_yaml_path.parent.mkdir(parents=True, exist_ok=True)
    dataset_root = _dataset_root_path(dataset_yaml_path)
    yaml_content = {
        "path": dataset_root.as_posix(),
        "train": "images/train",
        "val": "images/val",
        "names": {int(class_id): class_name for class_id, class_name in sorted(class_map.items())},
    }
    dataset_yaml_path.write_text(
        yaml.safe_dump(yaml_content, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )


def ensure_dataset_yaml(dataset_yaml_path: Path) -> None:
    """Corrige le chemin racine du dataset (Ultralytics exige un path absolu)."""
    class_map = load_dataset_classes(dataset_yaml_path)
    _save_dataset_yaml(dataset_yaml_path, class_map)


def register_object_class(dataset_yaml_path: Path, object_name: str) -> int:
    """Ajoute la classe si nécessaire. Retourne l'id de classe."""
    class_map = load_dataset_classes(dataset_yaml_path)

    for class_id, class_name in class_map.items():
        if class_name.lower() == object_name.lower():
            return class_id

    next_class_id = max(class_map.keys(), default=-1) + 1
    class_map[next_class_id] = object_name
    _save_dataset_yaml(dataset_yaml_path, class_map)
    return next_class_id


def add_object_to_dataset(
    labeled_pairs: list[tuple[Path, Path]],
    dataset_root: Path,
    *,
    validation_ratio: float = 0.2,
) -> tuple[int, int]:
    """
    Copie images + labels dans images/train|val et labels/train|val.
    Retourne (nombre train, nombre val).
    """
    random.shuffle(labeled_pairs)
    validation_count = max(1, int(len(labeled_pairs) * validation_ratio)) if len(labeled_pairs) >= 3 else 1
    validation_pairs = labeled_pairs[:validation_count]
    training_pairs = labeled_pairs[validation_count:]

    if not training_pairs:
        training_pairs = validation_pairs
        validation_pairs = []

    def _copy_pairs(pairs: list[tuple[Path, Path]], split_name: str) -> int:
        images_directory = dataset_root / "images" / split_name
        labels_directory = dataset_root / "labels" / split_name
        images_directory.mkdir(parents=True, exist_ok=True)
        labels_directory.mkdir(parents=True, exist_ok=True)

        copied_count = 0
        for image_path, label_path in pairs:
            destination_image = images_directory / image_path.name
            destination_label = labels_directory / label_path.name
            shutil.copy2(image_path, destination_image)
            shutil.copy2(label_path, destination_label)
            copied_count += 1
        return copied_count

    train_count = _copy_pairs(training_pairs, "train")
    val_count = _copy_pairs(validation_pairs, "val") if validation_pairs else 0
    return train_count, val_count
