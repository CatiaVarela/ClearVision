"""
ClearVision — webcam avec détection, distance estimée et annonce vocale.

Annonce uniquement les objets estimés à au plus MAX_DISTANCE_M mètres (défaut : 4 m).
Calibration : ajuster FOCAL_LENGTH_FACTOR dans clearvision_voice.py si les distances
semblent trop proches ou trop lointaines.
"""

import importlib.util
import sys
import time
from pathlib import Path

import cv2
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from clearvision_voice import MAX_DISTANCE_M, VoiceAnnouncer

_YOLO_WORLD_SCRIPT = ROOT / "test-images" / "yolo-world-test.py"
YOLO_PATH = ROOT / "yolo11n.pt"
if not YOLO_PATH.exists():
    YOLO_PATH = Path(__file__).parent / "yolo11n.pt"


def _load_detection_module():
    spec = importlib.util.spec_from_file_location("yolo_world_test", _YOLO_WORLD_SCRIPT)
    if spec is None or spec.loader is None:
        raise ImportError(f"Impossible de charger {_YOLO_WORLD_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["yolo_world_test"] = module
    spec.loader.exec_module(module)
    return module


def main():
    yw = _load_detection_module()
    process_frame = yw.process_frame

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Webcam inaccessible (index 0).")

    yolo_model = YOLO(str(YOLO_PATH))
    announcer = VoiceAnnouncer(max_distance_m=MAX_DISTANCE_M)

    print("ClearVision — annonce vocale activée.")
    print(f"Portée : objets à au plus {MAX_DISTANCE_M:.0f} m. Touche q pour quitter.")

    while cap.isOpened():
        ok, frame = cap.read()
        if not ok:
            break

        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]

        annotated, info = process_frame(
            frame,
            yolo_model,
            world_model=None,
            fast=True,
            estimate_distances=True,
        )

        message = announcer.maybe_announce(info["detections"], w, h)
        if message:
            print(f"[voix] {message}")

        cv2.imshow("ClearVision — voix + distance", annotated)
        if cv2.waitKey(1) & 0xFF in (ord("q"), ord("Q")):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
