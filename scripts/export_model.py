"""Export du modèle custom en ONNX."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ultralytics import YOLO

from config import PROJECT_ROOT, YOLO_CUSTOM_PATH


def main():
    model_path = YOLO_CUSTOM_PATH
    if not model_path.exists():
        model_path = PROJECT_ROOT / "datasets" / "custom" / "runs" / "clearvision_custom" / "weights" / "best.pt"

    if not model_path.exists():
        raise FileNotFoundError(f"Modèle introuvable : {model_path}")

    model = YOLO(str(model_path))
    export_path = model.export(format="onnx")
    print(f"Modèle exporté : {export_path}")


if __name__ == "__main__":
    main()
