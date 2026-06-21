"""
ClearVision — webcam avec détection, distance estimée et annonce vocale.

Mode intérieur / webcam :
- pas de détection HSV « arbres » (évite les faux positifs sur plantes, murs, vêtements verts)
- calibration distance adaptée au gros plan (visage à ~50 cm du PC)
"""

import sys
import time
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from clearvision_voice import MAX_DISTANCE_M, WEBCAM_FOCAL_LENGTH_FACTOR, VoiceAnnouncer
from config import YOLO_COCO_PATH
from detection import load_coco_model, process_frame


def main():
    yolo_model = load_coco_model()

    camera_capture = cv2.VideoCapture(0)
    if not camera_capture.isOpened():
        raise RuntimeError("Webcam inaccessible (index 0).")

    voice_announcer = VoiceAnnouncer(max_distance_m=MAX_DISTANCE_M)

    print("ClearVision — annonce vocale activée.")
    print(f"Modèle : {YOLO_COCO_PATH.name}")
    print(f"Portée : objets à au plus {MAX_DISTANCE_M:.0f} m. Touche q pour quitter.")
    print("Mode webcam : sans détection d'arbres HSV (intérieur).")

    while True:
        frame_read_ok, frame = camera_capture.read()
        if not frame_read_ok:
            break

        frame = cv2.flip(frame, 1)

        annotated_frame, detection_info = process_frame(
            frame,
            yolo_model,
            world_model=None,
            fast=True,
            estimate_distances=True,
            skip_vegetation=True,
            obstacle_mode=True,
            distance_focal_factor=WEBCAM_FOCAL_LENGTH_FACTOR,
            max_obstacle_distance_m=MAX_DISTANCE_M,
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
