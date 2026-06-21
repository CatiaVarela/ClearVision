"""Entraînement du modèle YOLO custom."""

from __future__ import annotations

from pathlib import Path

from ultralytics import YOLO

from config import MODEL_SIZE, PROJECT_ROOT
from object_learning.dataset_builder import ensure_dataset_yaml


def train_custom_model(
    *,
    dataset_yaml: Path,
    epochs: int = 40,
    run_name: str = "clearvision_custom",
) -> Path:
    dataset_yaml = Path(dataset_yaml).resolve()
    if not dataset_yaml.exists():
        raise FileNotFoundError(f"Dataset introuvable : {dataset_yaml}")

    ensure_dataset_yaml(dataset_yaml)

    dataset_root = dataset_yaml.parent
    for split_name in ("train", "val"):
        images_directory = dataset_root / "images" / split_name
        if not images_directory.exists() or not any(images_directory.iterdir()):
            raise FileNotFoundError(
                f"Aucune image dans {images_directory}. "
                "Relancez sans --skip-download ou ajoutez des images manuellement."
            )

    output_directory = PROJECT_ROOT / "datasets" / "custom" / "runs"
    base_model = f"yolo11{MODEL_SIZE}.pt"

    existing_weights = output_directory / run_name / "weights" / "best.pt"
    if existing_weights.exists():
        print(f"Fine-tuning depuis le modèle existant : {existing_weights.name}")
        model = YOLO(str(existing_weights))
    else:
        print(f"Entraînement depuis {base_model}...")
        model = YOLO(base_model)

    model.train(
        data=str(dataset_yaml),
        epochs=epochs,
        imgsz=640,
        project=str(output_directory),
        name=run_name,
        exist_ok=True,
        patience=15,
        batch=8,
    )

    best_weights = output_directory / run_name / "weights" / "best.pt"
    if not best_weights.exists():
        raise RuntimeError("Entraînement terminé mais best.pt introuvable.")

    return best_weights
