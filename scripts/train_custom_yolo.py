"""Entraînement YOLO custom sur dataset local."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from config import PROJECT_ROOT
from object_learning.trainer import train_custom_model


def main():
    dataset_yaml = PROJECT_ROOT / "datasets" / "custom" / "dataset.yaml"
    best_weights = train_custom_model(dataset_yaml=dataset_yaml, epochs=50)
    print(f"Entraînement terminé. Modèle : {best_weights}")
    print("Activez USE_CUSTOM_MODEL = True dans config.py pour l'utiliser.")


if __name__ == "__main__":
    main()
