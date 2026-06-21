"""Configuration centralisée ClearVision."""

from pathlib import Path

from vocabulary import (
    LABEL_FR,
    VEGETATION_CLASSES,
    WORLD_ANIMAL_CLASSES,
    WORLD_OBJECT_CLASSES,
    WORLD_PLANT_CLASSES,
)

PROJECT_ROOT = Path(__file__).resolve().parent

# --- Modèles ---
MODEL_SIZE = "n"  # "n" | "s" | "m"
USE_CUSTOM_MODEL = False
YOLO_CUSTOM_PATH = PROJECT_ROOT / "datasets" / "custom" / "runs" / "clearvision_custom" / "weights" / "best.pt"
CUSTOM_MODEL_CONF = 0.35
LEARNED_CLASSES_PATH = PROJECT_ROOT / "datasets" / "custom" / "learned_classes.yaml"

WORLD_MODEL_SIZE = "s"  # yolov8{s,m,l}-world.pt

# Nombre max de prompts YOLO-World par passe (évite la surcharge)
WORLD_CLASS_BATCH_SIZE = 35


def yolo_coco_filename(size: str | None = None) -> str:
    model_size = size or MODEL_SIZE
    return f"yolo11{model_size}.pt"


def yolo_world_filename(size: str | None = None) -> str:
    model_size = size or WORLD_MODEL_SIZE
    return f"yolov8{model_size}-world.pt"


YOLO_COCO_PATH = PROJECT_ROOT / yolo_coco_filename()
YOLO_WORLD_PATH = PROJECT_ROOT / "test-images" / yolo_world_filename()

DEFAULT_IMAGE = PROJECT_ROOT / "images" / "coco-test.jpg"
DEFAULT_VIDEO = PROJECT_ROOT / "videos" / "38805-418875307.mp4"

# --- Seuils de confiance ---
CONF_COCO = 0.4
CONF_PERSON = 0.25  # seuil plus bas pour ne pas rater de passants
CONF_VEGETATION = 0.22
CONF_WORLD_ANIMALS = 0.22
CONF_WORLD_PLANTS = 0.20
CONF_WORLD_OBJECTS = 0.28

# --- Affichage (BGR OpenCV) ---
DETECTION_BOX_COLOR = (255, 0, 0)
DETECTION_TEXT_COLOR = (255, 0, 0)
DETECTION_BOX_THICKNESS = 2

# --- Filtres ---
MIN_VEGETATION_AREA_RATIO = 0.0015
MIN_BOX_AREA_RATIO = 0.0008
OVERLAP_REJECT_RATIO = 0.3
NMS_IOU_THRESHOLD = 0.45
NMS_IOU_COCO_WORLD = 0.50
MIN_WORLD_BOXES_BEFORE_HSV = 2
MIN_GROUND_FOLIAGE_RATIO = 0.008

# Regroupement de personnes proches
PERSON_GROUP_MIN_COUNT = 2
PERSON_GROUP_MAX_GAP_RATIO = 0.12  # écart horizontal max (ratio largeur image)
PERSON_GROUP_CENTER_DISTANCE_RATIO = 0.10  # proximité des centres

# --- Façades (faux positifs végétation) ---
FACADE_MAX_Y_CENTER = 0.52
FACADE_MAX_HEIGHT_RATIO = 0.22

# --- Haie d'arbres (HSV) ---
NUM_TREE_COLUMNS = 6
TREE_BAND_TOP = 0.06
TREE_BAND_BOTTOM = 0.55
TURF_EXCLUDE_X = 0.48
TURF_EXCLUDE_Y = 0.40

HSV_FOLIAGE_RANGES = [
    ((32, 45, 35), (88, 255, 210)),
    ((15, 30, 40), (35, 180, 180)),
]

WORLD_CATEGORY_MAP = {
    "animaux": WORLD_ANIMAL_CLASSES,
    "plantes": WORLD_PLANT_CLASSES,
    "objets": WORLD_OBJECT_CLASSES,
}

DEFAULT_WORLD_CATEGORIES = ("animaux", "plantes", "objets")

# Réexport pour compatibilité
__all__ = [
    "LABEL_FR",
    "VEGETATION_CLASSES",
    "WORLD_ANIMAL_CLASSES",
    "WORLD_PLANT_CLASSES",
    "WORLD_OBJECT_CLASSES",
]
