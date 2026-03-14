import sys
import cv2
import pyttsx3
import time
import threading
import numpy as np
from pathlib import Path

# Initialisation voix
engine = pyttsx3.init()
engine.setProperty('rate', 130)  # lecture lente et compréhensible
engine.setProperty('volume', 1.0)

# Ajouter le chemin des modules
sys.path.append(str(Path(__file__).parent))

from modules.camera import CameraManager
from modules.detector import ObjectDetector
from modules.distance import DistanceEstimator
from modules.decision import DecisionEngine
from modules.audio import AudioManager
from modules.display import DisplayManager
from config import *

# --- Paramètres pour la distance ---
object_real_width = 20       # largeur réelle de l'objet en cm
known_distance = 100         # distance de calibration en cm
w_pixels_calibration = 150   # largeur de l'objet en pixels à known_distance
focal_length = (w_pixels_calibration * known_distance) / object_real_width

# Seuil pour déclencher alertes (en cm)
TRIGGER_DISTANCE_CM = 200  # 2 mètres

# Fonction pour lire la voix dans un thread
def speak_phrases(phrases):
    for phrase in phrases:
        engine.say(phrase)
    engine.runAndWait()

# Détection d'obstacles basique via l'image RGB
def detect_obstacles_rgb(frame, threshold=5000):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    if np.sum(edges) > threshold:
        # On retourne une indication générique si obstacle détecté
        return ["Obstacle devant détecté ! Ralentissez"]
    return []

def main():
    print("="*50)
    print("ClearVision - Prototype Python v0.6 (Seuil 2m)")
    print("="*50)

    try:
        # Initialisation des modules
        print("\n Initialisation caméra...")
        camera = CameraManager()
        print(" Chargement IA...")
        detector = ObjectDetector(MODEL_PATH, CONFIDENCE_THRESHOLD)
        print(" Initialisation distance...")
        distance_estimator = DistanceEstimator(FRAME_WIDTH, FRAME_HEIGHT)
        print(" Initialisation moteur décision...")
        decision_engine = DecisionEngine()
        print(" Initialisation audio...")
        audio = AudioManager()
        print(" Prêt ! Appuie sur 'q' pour quitter\n")
        last_announcement_time = 0
        while True:
            frame = camera.get_frame()
            if frame is None:
                print(" Impossible de lire la caméra")
                break

            # Détection objets connus
            objects = detector.detect(frame)
            objects_with_distance = distance_estimator.estimate(objects)

            # Alertes normales
            alerts_normal = decision_engine.analyze(objects_with_distance)

            # Détection obstacles RGB
            obstacles = detect_obstacles_rgb(frame)

            phrases = []
            current_time = time.time()

            # --- Filtrer objets et obstacles selon distance ---
            filtered_objects = []
            for obj in objects_with_distance:
                x, y, w, h = obj['bbox']
                distance_cm = (object_real_width * focal_length) / w
                if distance_cm <= TRIGGER_DISTANCE_CM:
                    frame_center = frame.shape[1] / 2
                    object_center = x + w / 2
                    if object_center < frame_center - 50:
                        direction = "gauche"
                    elif object_center > frame_center + 50:
                        direction = "droite"
                    else:
                        direction = "centre"
                    label = obj.get('class', 'un objet')
                    phrases.append(f"L'objet {label} est à {int(distance_cm)} cm et se trouve à {direction}")
                    filtered_objects.append(obj)

            # Obstacles génériques si distance < TRIGGER_DISTANCE_CM
            filtered_obstacles = obstacles if filtered_objects or obstacles else []
            if filtered_obstacles:
                phrases.extend(filtered_obstacles)

            # Si rien n'est proche, annoncer que la route est sûre
            if not filtered_objects and not filtered_obstacles and current_time - last_announcement_time >= 10:
                phrases.append("Votre chemin est sûr")
                last_announcement_time = current_time

            # Lancer la voix dans un thread
            if phrases:
                t = threading.Thread(target=speak_phrases, args=(phrases,))
                t.start()

            # Jouer bips pour alertes normales
            for alert in alerts_normal:
                audio.play_alert(alert)

            # Affichage
            DisplayManager.show(frame, filtered_objects, alerts_normal)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except Exception as e:
        print(f" Erreur : {e}")

    finally:
        print("\n Nettoyage...")
        if 'camera' in locals():
            camera.release()
        cv2.destroyAllWindows()
        print(" Arrêt du prototype")

if __name__ == "__main__":
    main()