"""
ClearVision — webcam avec détection, distance estimée et annonce vocale.
"""

import sys
import time
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from clearvision_voice import MAX_DISTANCE_M, VoiceAnnouncer
from config import YOLO_COCO_PATH
from detection import load_coco_model, process_frame


def main():
    yolo_model = load_coco_model()
    from detection.models import load_world_model

    world_model = load_world_model()

    camera_capture = cv2.VideoCapture(0)
    if not camera_capture.isOpened():
        raise RuntimeError("Webcam inaccessible (index 0).")

    voice_announcer = VoiceAnnouncer(max_distance_m=MAX_DISTANCE_M)

    print("ClearVision — annonce vocale activée.")
    print(f"Modèle : {YOLO_COCO_PATH.name}")
    print(f"Portée : objets à au plus {MAX_DISTANCE_M:.0f} m. Touche q pour quitter.")

    while True:
        frame_read_ok, frame = camera_capture.read()
        if not frame_read_ok:
            break

        annotated_frame, detection_info = process_frame(
            frame,
            yolo_model,
            world_model,
            fast=True,
            estimate_distances=True,
        )

        frame_height, frame_width = frame.shape[:2]
        voice_announcer.maybe_announce(
            detection_info["detections"],
            frame_width,
            frame_height,
        )

        cv2.imshow("ClearVision", annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        time.sleep(0.01)

    camera_capture.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
