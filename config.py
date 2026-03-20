# Configuration du prototype

# Paramètres caméra
CAMERA_ID = 0  # 0 = webcam intégrée, 1 = externe
FRAME_WIDTH = 640
FRAME_HEIGHT = 480

# Paramètres détection
CONFIDENCE_THRESHOLD = 0.5  # Seuil de confiance (50%)
MODEL_PATH = "models/yolov8n.pt"

# Seuils de distance (en mètres)
DANGER_ZONES = {
    "person": 2.0,
    "car": 5.0,
    "truck": 5.0,
    "bus": 5.0,
    "bicycle": 3.0,
    "motorcycle": 3.0,
    "dog": 2.0,
    "cat": 2.0,
    "chair": 1.5,
    "table": 1.5,
    "tree": 1.5,  # Branches en hauteur
    "pole": 1.5,  # Poteau
    "wall": 1.0,
    "door": 1.0
}

# Paramètres audio
SOUNDS_PATH = "sounds/"
ALERT_COOLDOWN = 2.0  # Secondes entre deux alertes