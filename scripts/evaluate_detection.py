"""Évalue les détections sur un jeu d'images annotées."""

import json
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from detection import detect_objects_in_image


def main():
    fixtures_directory = ROOT / "tests" / "fixtures"
    if not fixtures_directory.exists():
        print(f"Dossier fixtures absent : {fixtures_directory}")
        return

    report_lines = ["# Rapport d'évaluation ClearVision\n"]

    for expected_file in fixtures_directory.glob("*_expected.json"):
        image_stem = expected_file.stem.replace("_expected", "")
        image_path = None
        for extension in (".jpg", ".png", ".webp"):
            candidate = fixtures_directory / f"{image_stem}{extension}"
            if candidate.exists():
                image_path = candidate
                break

        if image_path is None:
            continue

        expected_data = json.loads(expected_file.read_text(encoding="utf-8"))
        source_image = cv2.imread(str(image_path))
        _, detection_info = detect_objects_in_image(source_image, fast=True)

        detected_labels = {detection["label_key"] for detection in detection_info["detections"]}
        expected_labels = set(expected_data.get("labels_present", []))
        missing_labels = expected_labels - detected_labels

        report_lines.append(f"## {image_path.name}")
        report_lines.append(f"- Détections : {len(detection_info['detections'])}")
        report_lines.append(f"- Labels attendus manquants : {missing_labels or 'aucun'}\n")

    report_path = ROOT / "evaluation_report.md"
    report_path.write_text("\n".join(report_lines), encoding="utf-8")
    print(f"Rapport écrit : {report_path}")


if __name__ == "__main__":
    main()
