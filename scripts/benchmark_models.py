"""Compare les modèles YOLO sur le dossier images/."""

import sys
import time
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ultralytics import YOLO

from config import PROJECT_ROOT


def benchmark_model(model_filename: str, image_paths: list[Path]) -> dict:
    model = YOLO(model_filename)
    total_detections = 0
    total_seconds = 0.0

    for image_path in image_paths:
        source_image = cv2.imread(str(image_path))
        if source_image is None:
            continue

        start_time = time.perf_counter()
        results = model(source_image, conf=0.4, verbose=False)
        elapsed = time.perf_counter() - start_time
        total_seconds += elapsed

        for result in results:
            if result.boxes is not None:
                total_detections += len(result.boxes)

    image_count = len(image_paths)
    return {
        "model": model_filename,
        "images": image_count,
        "total_detections": total_detections,
        "avg_detections": total_detections / image_count if image_count else 0,
        "avg_seconds": total_seconds / image_count if image_count else 0,
        "fps": image_count / total_seconds if total_seconds > 0 else 0,
    }


def main():
    images_directory = PROJECT_ROOT / "images"
    image_paths = list(images_directory.glob("*.jpg")) + list(images_directory.glob("*.png"))

    if not image_paths:
        print(f"Aucune image dans {images_directory}. Ajoutez des photos de test.")
        return

    models_to_test = ["yolo11n.pt", "yolo11s.pt", "yolo11m.pt"]
    print(f"Benchmark sur {len(image_paths)} image(s)\n")

    for model_name in models_to_test:
        try:
            stats = benchmark_model(model_name, image_paths)
            print(
                f"{stats['model']}: "
                f"{stats['avg_detections']:.1f} dét./img, "
                f"{stats['avg_seconds']:.2f}s/img, "
                f"{stats['fps']:.1f} FPS"
            )
        except Exception as error:
            print(f"{model_name}: échec — {error}")


if __name__ == "__main__":
    main()
