import cv2
from ultralytics import YOLO
from deepforest import main
from pathlib import Path

# --- FONCTION DE FILTRAGE ---
def is_overlapping(tree_box, human_boxes):
    tx1, ty1, tx2, ty2 = tree_box
    for (hx1, hy1, hx2, hy2) in human_boxes:
        # Calcul de l'intersection
        ix1 = max(tx1, hx1)
        iy1 = max(ty1, hy1)
        ix2 = min(tx2, hx2)
        iy2 = min(ty2, hy2)

        if ix2 > ix1 and iy2 > iy1:
            area_intersection = (ix2 - ix1) * (iy2 - iy1)
            area_tree = (tx2 - tx1) * (ty2 - ty1)
            # Si plus de 30% de l'arbre est sur l'humain, on rejette
            if area_intersection / area_tree > 0.3:
                return True
    return False

# 1. Chargement
yolo_model = YOLO("yolo11n.pt")
tree_model = main.deepforest()

image = cv2.imread(r"images\coco-test.jpg")
human_coords = []

# 2. Passage YOLO (On stocke les humains)
results_yolo = yolo_model(image, conf=0.4)
for result in results_yolo:
    for box in result.boxes:
        coords = map(int, box.xyxy[0])
        x1, y1, x2, y2 = coords
        human_coords.append((x1, y1, x2, y2)) # On mémorise la position
        cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)

# 3. Passage DeepForest avec filtrage
predictions = tree_model.predict_image(path=r"images\coco-test.jpg")

if predictions is not None:
    for index, row in predictions.iterrows():
        tx1, ty1, tx2, ty2 = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])

        # ON VÉRIFIE SI ÇA SUPERPOSE UN HUMAIN
        if not is_overlapping((tx1, ty1, tx2, ty2), human_coords):
            if row['score'] > 0.2: # Seuil
                cv2.rectangle(image, (tx1, ty1), (tx2, ty2), (0, 255, 0), 2)
                cv2.putText(image, "Arbre", (tx1, ty1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

result_path = Path("images") / "coco-test_resultat.jpg"
try:
    cv2.imshow("Resultat Propre", cv2.resize(image, (1000, 800)))
    cv2.waitKey(0)
    cv2.destroyAllWindows()
except cv2.error:
    cv2.imwrite(str(result_path), image)
    print(f"Affichage GUI indisponible. Image enregistrée dans : {result_path}")
