"""Entraînement YOLO custom sur dataset local."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ultralytics import YOLO

from config import MODEL_SIZE, PROJECT_ROOT


def main():
    dataset_yaml = PROJECT_ROOT / "datasets" / "custom" / "dataset.yaml"
    if not dataset_yaml.exists():
        raise FileNotFoundError(
            f"Dataset introuvable : {dataset_yaml}\n"
            "Créez le fichier et annotez vos images (voir datasets/README.md)."
        )

    base_model = f"yolo11{MODEL_SIZE}.pt"
    output_directory = PROJECT_ROOT / "datasets" / "custom" / "runs"

    model = YOLO(base_model)
    model.train(
        data=str(dataset_yaml),
        epochs=50,
        imgsz=640,
        project=str(output_directory),
        name="clearvision_custom",
        exist_ok=True,
    )

    print(f"Entraînement terminé. Modèle : {output_directory / 'clearvision_custom' / 'weights' / 'best.pt'}")


if __name__ == "__main__":
    main()
