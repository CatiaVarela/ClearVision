"""Point d'entrée principal — analyser une image et obtenir les détections."""

import argparse
import json
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from detection import detect_objects_in_image, detections_to_json

def build_output_path(image_path: Path) -> Path:
    return image_path.with_name(f"{image_path.stem}_detected{image_path.suffix}")

def main():
    parser = argparse.ArgumentParser(
        description="ClearVision — reconnaissance d'objets sur image (rectangles bleus)"
    )
    parser.add_argument("image", type=Path, help="Chemin de l'image à analyser")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Chemin de sortie (défaut : <nom>_detected.jpg)",
    )
    parser.add_argument("--fast", action="store_true", help="Mode rapide sans YOLO-World")
    parser.add_argument("--json", type=Path, default=None, help="Exporter les détections en JSON")
    parser.add_argument("--no-gui", action="store_true", help="Sans fenêtre OpenCV")
    arguments = parser.parse_args()

    source_image = cv2.imread(str(arguments.image))
    if source_image is None:
        raise FileNotFoundError(f"Image introuvable : {arguments.image.resolve()}")

    annotated_image, detection_info = detect_objects_in_image(source_image, fast=arguments.fast)

    output_path = arguments.output or build_output_path(arguments.image)
    cv2.imwrite(str(output_path), annotated_image)
    print(f"Image annotée : {output_path.resolve()}")
    print(f"Détections : {detection_info['drawn_count']}")

    for detection in detection_info["detections"]:
        print(f"  - {detection['label_fr']} ({detection['label_key']})")

    if arguments.json:
        json_payload = {
            "source": str(arguments.image),
            "detection_count": len(detection_info["detections"]),
            "detections": detections_to_json(detection_info["detections"]),
        }
        arguments.json.write_text(json.dumps(json_payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"JSON exporté : {arguments.json.resolve()}")

    if not arguments.no_gui:
        try:
            cv2.imshow("ClearVision", cv2.resize(annotated_image, (1000, 800)))
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        except cv2.error:
            pass

if __name__ == "__main__":
    main()
