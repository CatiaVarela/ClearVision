"""Script CLI image — délègue au module detection."""

import argparse
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from config import DEFAULT_IMAGE
from detection import load_coco_model, load_world_model, process_frame


def result_path_for(image_path: Path) -> Path:
    return image_path.with_name(f"{image_path.stem}_yoloworld{image_path.suffix}")


def main(image_path: Path, show_gui: bool = True, fast: bool = False, categories: tuple[str, ...] = ()):
    source_image = cv2.imread(str(image_path))
    if source_image is None:
        raise FileNotFoundError(f"Image introuvable : {image_path.resolve()}")

    yolo_model = load_coco_model()
    world_model = None if fast else load_world_model()

    world_categories = categories if categories else None
    annotated_image, info = process_frame(
        source_image,
        yolo_model,
        world_model,
        fast=fast,
        world_categories=world_categories,
    )

    output_path = result_path_for(image_path)
    cv2.imwrite(str(output_path), annotated_image)
    print(f"Image enregistrée : {output_path.resolve()}")
    print(
        f"Mode végétation : {info['mode']} — "
        f"{info['drawn_count']} détection(s) affichée(s)"
    )

    if not show_gui:
        return

    try:
        cv2.imshow("ClearVision - Détection", cv2.resize(annotated_image, (1000, 800)))
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    except cv2.error:
        print("Affichage GUI indisponible (résultat déjà sauvegardé).")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Détection universelle sur image")
    parser.add_argument("--image", type=Path, default=DEFAULT_IMAGE, help="Image à analyser")
    parser.add_argument("--no-gui", action="store_true", help="Sans fenêtre OpenCV")
    parser.add_argument("--fast", action="store_true", help="Sans YOLO-World (plus rapide)")
    parser.add_argument(
        "--categories",
        type=str,
        default="animaux,plantes,objets",
        help="Catégories YOLO-World : animaux,plantes,objets",
    )
    arguments = parser.parse_args()
    category_list = tuple(c.strip() for c in arguments.categories.split(",") if c.strip())
    main(
        image_path=arguments.image,
        show_gui=not arguments.no_gui,
        fast=arguments.fast,
        categories=category_list,
    )
