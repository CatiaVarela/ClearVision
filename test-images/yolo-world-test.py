import argparse
import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO, YOLOWorld

# --- CONFIG (défaut ; surchargeable avec --image) ---
DEFAULT_IMAGE = Path("../images/coco-test.jpg")
YOLO_COCO_PATH = "../yolo11n.pt"
YOLO_WORLD_PATH = "yolov8s-world.pt"

CONF_COCO = 0.4
CONF_VEGETATION = 0.22
MIN_VEGETATION_AREA_RATIO = 0.0015
OVERLAP_REJECT_RATIO = 0.3
MIN_WORLD_BOXES_BEFORE_HSV = 2
MIN_GROUND_FOLIAGE_RATIO = 0.008
FACADE_MAX_Y_CENTER = 0.52
FACADE_MAX_HEIGHT_RATIO = 0.22
NUM_TREE_COLUMNS = 6
TREE_BAND_TOP = 0.06
TREE_BAND_BOTTOM = 0.55
TURF_EXCLUDE_X = 0.48
TURF_EXCLUDE_Y = 0.40

VEGETATION_CLASSES = [
    "tree in park",
    "evergreen tree",
    "green bush",
    "hedge",
    "foliage",
]
LABEL_FR = {
    "tree in park": "Arbre",
    "evergreen tree": "Arbre",
    "green bush": "Buisson",
    "hedge": "Haie",
    "foliage": "Végétation",
    "hsv": "Végétation",
    "tree_line": "Arbre",
}


def is_overlapping(veg_box, human_boxes, ratio=OVERLAP_REJECT_RATIO):
    tx1, ty1, tx2, ty2 = veg_box
    area_veg = (tx2 - tx1) * (ty2 - ty1)
    if area_veg <= 0:
        return False

    for hx1, hy1, hx2, hy2 in human_boxes:
        ix1 = max(tx1, hx1)
        iy1 = max(ty1, hy1)
        ix2 = min(tx2, hx2)
        iy2 = min(ty2, hy2)

        if ix2 > ix1 and iy2 > iy1:
            area_intersection = (ix2 - ix1) * (iy2 - iy1)
            if area_intersection / area_veg > ratio:
                return True
    return False


def box_area(x1, y1, x2, y2):
    return max(0, x2 - x1) * max(0, y2 - y1)


def iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    union = box_area(*a) + box_area(*b) - inter
    return inter / union if union > 0 else 0.0


def nms_boxes(boxes, threshold=0.45):
    """boxes: list of (x1,y1,x2,y2,label_key,score)"""
    boxes = sorted(boxes, key=lambda b: b[5], reverse=True)
    kept = []
    for box in boxes:
        if all(iou(box[:4], other[:4]) < threshold for other in kept):
            kept.append(box)
    return kept


def _roi(image, x1, y1, x2, y2):
    h, w = image.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    return image[y1:y2, x1:x2]


def ground_foliage_ratio(image, y_start=0.4):
    """Part de pixels « feuillage » dans la partie basse (herbe, arbres au sol)."""
    h, w = image.shape[:2]
    mask = _foliage_mask(image)
    lower = mask[int(h * y_start) :, :]
    return lower.sum() / (255.0 * h * w)


def scene_has_ground_vegetation(image):
    return ground_foliage_ratio(image) >= MIN_GROUND_FOLIAGE_RATIO


def looks_like_vertical_pattern(roi):
    """Volets / fenêtres : arêtes verticales dominantes."""
    if roi.size == 0 or roi.shape[0] < 12 or roi.shape[1] < 12:
        return False
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    vx, vy = float(np.abs(gx).mean()), float(np.abs(gy).mean())
    if vy < 1e-6:
        return False
    return vx > vy * 1.45 and roi.shape[1] / roi.shape[0] > 0.35


def is_facade_false_positive(x1, y1, x2, y2, image):
    """Rejette les boîtes compactes en étage (volets verts, fenêtres)."""
    h, w = image.shape[:2]
    bw, bh = x2 - x1, y2 - y1
    if bw <= 0 or bh <= 0:
        return True

    cy = (y1 + y2) / (2.0 * h)
    if cy > FACADE_MAX_Y_CENTER or bh / h > FACADE_MAX_HEIGHT_RATIO:
        return False

    aspect = bw / bh
    if not (0.3 <= aspect <= 3.5):
        return False

    roi = _roi(image, x1, y1, x2, y2)
    return looks_like_vertical_pattern(roi)


def filter_vegetation_boxes(boxes, image):
    return [
        box
        for box in boxes
        if not is_facade_false_positive(box[0], box[1], box[2], box[3], image)
    ]


def detect_vegetation_world(world_model, image):
    world_model.set_classes(VEGETATION_CLASSES)
    results = world_model.predict(image, conf=CONF_VEGETATION, imgsz=1280, verbose=False)
    found = []

    for result in results:
        if result.boxes is None:
            continue
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            class_id = int(box.cls[0])
            label_key = world_model.names[class_id]
            found.append((x1, y1, x2, y2, label_key, float(box.conf[0])))

    return filter_vegetation_boxes(found, image)


def _foliage_mask(image):
    """Masque du feuillage en arrière-plan, sans pelouse synthétique bas-droite."""
    h, w = image.shape[:2]
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    mask = cv2.inRange(hsv, (32, 45, 35), (88, 255, 210))

    band = np.zeros((h, w), np.uint8)
    band[int(h * TREE_BAND_TOP) : int(h * TREE_BAND_BOTTOM), :] = 255
    mask = cv2.bitwise_and(mask, band)

    turf_zone = np.zeros((h, w), np.uint8)
    turf_zone[int(h * TURF_EXCLUDE_Y) :, int(w * TURF_EXCLUDE_X) :] = 255
    bright_turf = cv2.inRange(hsv, (35, 70, 90), (80, 255, 255))
    mask = cv2.bitwise_and(mask, cv2.bitwise_not(cv2.bitwise_and(bright_turf, turf_zone)))

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (55, 12))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(
        mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)), 1
    )
    return mask


def detect_tree_line(image):
    """
    Découpe la haie d'arbres en bandes verticales (vue de parc au sol).
    Plus stable que le watershed pour ce type de scène.
    """
    h, w = image.shape[:2]
    image_area = h * w
    mask = _foliage_mask(image)

    y0, y1 = int(h * TREE_BAND_TOP), int(h * TREE_BAND_BOTTOM)
    band_mask = mask[y0:y1, :]
    min_pixels = 0.008 * image_area

    found = []
    for col in range(NUM_TREE_COLUMNS):
        x_start = int(col * w / NUM_TREE_COLUMNS)
        x_end = int((col + 1) * w / NUM_TREE_COLUMNS)
        patch = band_mask[:, x_start:x_end]

        if patch.sum() < min_pixels:
            continue

        ys, xs = np.where(patch > 0)
        bx1 = x_start + int(xs.min())
        bx2 = x_start + int(xs.max())
        by1 = y0 + int(ys.min())
        by2 = y0 + int(ys.max())

        if box_area(bx1, by1, bx2, by2) / image_area < MIN_VEGETATION_AREA_RATIO:
            continue
        if is_facade_false_positive(bx1, by1, bx2, by2, image):
            continue

        found.append((bx1, by1, bx2, by2, "tree_line", 0.6))

    return found


def visible_vegetation_boxes(boxes, human_coords, image_area):
    visible = []
    for x1, y1, x2, y2, label_key, score in boxes:
        if box_area(x1, y1, x2, y2) / image_area < MIN_VEGETATION_AREA_RATIO:
            continue
        if is_overlapping((x1, y1, x2, y2), human_coords):
            continue
        visible.append((x1, y1, x2, y2, label_key, score))
    return visible


def draw_vegetation(image, boxes, human_coords, image_area, distances: dict | None = None):
    drawn = 0
    for x1, y1, x2, y2, label_key, _score in visible_vegetation_boxes(
        boxes, human_coords, image_area
    ):
        label_fr = LABEL_FR.get(label_key, "Végétation")
        tag = label_fr
        if distances and (x1, y1, x2, y2) in distances:
            tag = f"{label_fr} {distances[(x1, y1, x2, y2)]:.0f}m"

        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            image,
            tag,
            (x1, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 255, 0),
            1,
        )
        drawn += 1
    return drawn


def result_path_for(image_path: Path) -> Path:
    return image_path.with_name(f"{image_path.stem}_yoloworld{image_path.suffix}")


def process_frame(
    source,
    yolo_model,
    world_model=None,
    *,
    fast=False,
    estimate_distances: bool = False,
):
    """
    Analyse une frame BGR : COCO (bleu) + végétation YOLO-World / HSV (vert).
    fast=True : sans YOLO-World (beaucoup plus rapide, adapté à la vidéo).
    estimate_distances=True : ajoute distance_m sur chaque détection (pour la voix).
    Retourne (image annotée, dict d'infos).
    """
    image = source.copy()
    h, w = image.shape[:2]
    image_area = h * w
    human_coords = []
    detections = []
    predict_kw = {"conf": CONF_COCO, "verbose": False}
    if fast:
        predict_kw["imgsz"] = 640

    if estimate_distances:
        from clearvision_voice import estimate_distance_m, label_to_fr
    else:
        estimate_distance_m = None
        label_to_fr = None

    results_coco = yolo_model(source, **predict_kw)
    for result in results_coco:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            label = yolo_model.names[int(box.cls[0])]
            bbox = (x1, y1, x2, y2)

            if label == "person":
                human_coords.append(bbox)

            tag = label
            if estimate_distances:
                dist_m = estimate_distance_m(bbox, h, label)
                tag = f"{label} {dist_m:.0f}m"
                detections.append(
                    {
                        "label_key": label,
                        "label_fr": label_to_fr(label),
                        "bbox": bbox,
                        "distance_m": dist_m,
                    }
                )

            cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(
                image,
                tag,
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (255, 0, 0),
                1,
            )

    veg_boxes = []
    used_hsv = False

    if fast:
        if scene_has_ground_vegetation(source):
            veg_boxes = detect_tree_line(source)
            used_hsv = True
        mode = "rapide (YOLO + haie d'arbres)" if used_hsv else "rapide (YOLO seul)"
    else:
        if world_model is None:
            raise ValueError("world_model requis si fast=False")
        veg_boxes = detect_vegetation_world(world_model, source)
        if (
            scene_has_ground_vegetation(source)
            and len(veg_boxes) < MIN_WORLD_BOXES_BEFORE_HSV
        ):
            veg_boxes.extend(detect_tree_line(source))
            used_hsv = True
        mode = "YOLO-World + haie d'arbres (secours)" if used_hsv else "YOLO-World"

    veg_boxes = nms_boxes(veg_boxes)
    veg_visible = visible_vegetation_boxes(veg_boxes, human_coords, image_area)
    veg_distances = {}

    if estimate_distances:
        for x1, y1, x2, y2, label_key, _score in veg_visible:
            bbox = (x1, y1, x2, y2)
            dist_m = estimate_distance_m(bbox, h, label_key)
            veg_distances[bbox] = dist_m
            detections.append(
                {
                    "label_key": label_key,
                    "label_fr": LABEL_FR.get(label_key, "Végétation"),
                    "bbox": bbox,
                    "distance_m": dist_m,
                }
            )

    veg_drawn = draw_vegetation(
        image, veg_boxes, human_coords, image_area, veg_distances or None
    )

    return image, {
        "veg_drawn": veg_drawn,
        "mode": mode,
        "used_hsv": used_hsv,
        "detections": detections,
    }


def main(image_path: Path, show_gui: bool = True):
    source = cv2.imread(str(image_path))
    if source is None:
        raise FileNotFoundError(f"Image introuvable : {image_path.resolve()}")

    yolo_model = YOLO(YOLO_COCO_PATH)
    world_model = YOLOWorld(YOLO_WORLD_PATH)
    image, info = process_frame(source, yolo_model, world_model)

    out_path = result_path_for(image_path)
    cv2.imwrite(str(out_path), image)
    print(f"Image enregistrée : {out_path.resolve()}")
    print(
        f"Mode végétation : {info['mode']} — "
        f"{info['veg_drawn']} zone(s) verte(s) affichée(s)"
    )

    if not show_gui:
        return

    try:
        cv2.imshow("ClearVision - YOLO-World", cv2.resize(image, (1000, 800)))
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    except cv2.error:
        print("Affichage GUI indisponible (résultat déjà sauvegardé).")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Détection COCO + arbres/buissons")
    parser.add_argument(
        "--image",
        type=Path,
        default=DEFAULT_IMAGE,
        help="Chemin de l'image à analyser",
    )
    parser.add_argument(
        "--no-gui",
        action="store_true",
        help="Sauvegarder uniquement, sans fenêtre OpenCV",
    )
    args = parser.parse_args()
    main(image_path=args.image, show_gui=not args.no_gui)
