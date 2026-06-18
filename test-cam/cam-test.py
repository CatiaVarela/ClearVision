import sys
import time
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from config import DETECTION_BOX_COLOR, DETECTION_TEXT_COLOR
from detection import load_coco_model

model = load_coco_model()

camera_capture = cv2.VideoCapture(0)
window_name = "Détection YOLO - Toutes Classes"
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

start_time = time.time()
frame_count = 0

print("Détection en cours... Appuyez sur 'q' pour quitter.")

while time.time() - start_time < 60:
    frame_read_ok, frame = camera_capture.read()
    if not frame_read_ok:
        break
    frame = cv2.flip(frame, 1)
    results = model(frame, conf=0.25, verbose=False)

    for result in results:
        for box in result.boxes:
            class_id = int(box.cls[0])
            label = model.names[class_id]
            confidence = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            cv2.rectangle(frame, (x1, y1), (x2, y2), DETECTION_BOX_COLOR, 2)
            cv2.putText(
                frame,
                f"{label} {confidence:.2f}",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                DETECTION_TEXT_COLOR,
                2,
            )

    cv2.imshow(window_name, frame)
    frame_count += 1

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

camera_capture.release()
cv2.destroyAllWindows()

duration_seconds = time.time() - start_time
print("Session terminée.")
print(f"Total images : {frame_count} | Durée : {duration_seconds:.2f}s | FPS : {frame_count / duration_seconds:.2f}")
