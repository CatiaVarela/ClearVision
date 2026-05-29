import cv2
import time
import pytesseract
import pyttsx3
from ultralytics import YOLO

# --- 1. CONFIGURATION ---
# Chemin Tesseract (à vérifier selon ton installation)
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Chargement du modèle (YOLO11n est rapide pour la vidéo)
model = YOLO("yolo11n.pt")

# Initialisation de la voix
engine = pyttsx3.init()
engine.setProperty('rate', 165)

# --- 2. FONCTION DE PRIORITÉ (DISTANCE) ---
def get_distance_level(box_area, frame_area):
    """ Calcule si l'objet est proche selon la taille du carré """
    ratio = box_area / frame_area
    if ratio > 0.15: return "TRES PROCHE"
    if ratio > 0.05: return "PROCHE"
    return "LOINTAIN"

# --- 3. DÉMARRAGE CAMÉRA ---
cap = cv2.VideoCapture(0)
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
frame_area = frame_width * frame_height

last_speech_time = 0

print("Système ClearVision Activé. Appuyez sur 'q' pour quitter.")

while cap.isOpened():
    success, frame = cap.read()
    if not success: break

    # Miroir pour une navigation plus naturelle
    frame = cv2.flip(frame, 1)

    # Détection YOLO
    results = model(frame, conf=0.35, verbose=False)

    current_frame_objects = []

    for result in results:
        for box in result.boxes:
            # Infos de base
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls = int(box.cls[0])
            label = model.names[cls]
            conf = float(box.conf[0])

            # Calcul de la zone pour la distance
            area = (x2 - x1) * (y2 - y1)
            dist = get_distance_level(area, frame_area)

            # --- LOGIQUE D'ACTION ---
            # Couleur : Rouge si trES proche, Vert sinon
            color = (0, 0, 255) if dist == "TRES PROCHE" else (0, 255, 0)

            # Dessin
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"{label} - {dist}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # --- LOGIQUE VOCALE (Priorité) ---
            # On ne parle que toutes les 2 secondes pour ne pas saturer
            if time.time() - last_speech_time > 2.0:
                if dist != "LOINTAIN":
                    # Si c'est un panneau ou un objet avec du texte potentiel
                    if label in ["stop sign", "street sign", "potted plant"]:
                        # On tente l'OCR sur la zone (ROI)
                        roi = frame[y1:y2, x1:x2]
                        if roi.size > 0:
                            gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                            text = pytesseract.image_to_string(gray_roi, lang='fra', config='--psm 6')
                            if text.strip():
                                engine.say(f"Information lue : {text}")
                            else:
                                engine.say(f"{label} {dist}")
                    else:
                        engine.say(f"{label} {dist}")

                    engine.runAndWait()
                    last_speech_time = time.time()

    # Affichage
    cv2.imshow("ClearVision - Assistant de Rue", frame)

    if cv2.waitKey(1) & 0xFF == ord('q') or ord('Q'):
        break

cap.release()
cv2.destroyAllWindows()