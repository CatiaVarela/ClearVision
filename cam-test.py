import cv2
import time
from ultralytics import YOLO

# --- 1. CHARGEMENT DU MODÈLE ---
# yolo11n.pt connaît 80 catégories d'objets différentes
model = YOLO("yolo11n.pt")

# --- 2. CONFIGURATION VIDÉO ---
cap = cv2.VideoCapture(0)
window_name = "Détection YOLO - Toutes Classes"
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

start_time = time.time()
frames = 0

print("Détection en cours... Appuyez sur 'q' pour quitter.")

while time.time() < start_time + 60:
    success, frame = cap.read()
    if not success:
        break
    results = model(frame, conf=0.25, verbose=False)

    for result in results:
        for box in result.boxes:
            # Récupération des infos de l'objet
            class_id = int(box.cls[0])
            label = model.names[class_id]  # Le nom de l'objet (ex: 'person', 'dog', 'chair')
            confidence = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # --- DESSIN ---
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{label} {confidence:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # --- AFFICHAGE ---
    cv2.imshow(window_name, frame)
    frames += 1

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

duration = time.time() - start_time
print(f"Session terminée.")
print(f"Total images : {frames} | Durée : {duration:.2f}s | FPS : {frames/duration:.2f}")