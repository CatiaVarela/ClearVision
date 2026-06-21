"""
Apprendre un nouvel objet à ClearVision.

1. Cherche des images sur Internet (Bing, repli DuckDuckGo)
2. Les enregistre dans datasets/custom/sources/<nom_objet>/
3. Les annote automatiquement (YOLO-World)
4. Les ajoute au dataset d'entraînement
5. Entraîne (fine-tune) le modèle YOLO custom

Exemple :
    python scripts/learn_object.py "banc public" --count 10 --train
    python scripts/learn_object.py "fire hydrant" --search-en --count 10 --train
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from config import PROJECT_ROOT, YOLO_CUSTOM_PATH
from object_learning.auto_label import auto_label_image_folder, save_learned_class_metadata
from object_learning.dataset_builder import add_object_to_dataset, ensure_dataset_yaml, register_object_class
from object_learning.image_downloader import download_object_images, object_name_to_slug
from object_learning.trainer import train_custom_model


def main():
    parser = argparse.ArgumentParser(
        description="Télécharger des images, annoter et entraîner la détection d'un nouvel objet"
    )
    parser.add_argument("object_name", type=str, help='Nom de l\'objet (ex. "banc public", "poubelle")')
    parser.add_argument("--count", type=int, default=10, help="Nombre d'images à télécharger (défaut : 10)")
    parser.add_argument(
        "--search-en",
        action="store_true",
        help="Utiliser le nom tel quel pour la recherche (utile si déjà en anglais)",
    )
    parser.add_argument(
        "--label-fr",
        type=str,
        default=None,
        help="Libellé français affiché (défaut : nom de l'objet)",
    )
    parser.add_argument("--train", action="store_true", help="Lancer l'entraînement après collecte")
    parser.add_argument("--epochs", type=int, default=40, help="Époques d'entraînement (défaut : 40)")
    parser.add_argument(
        "--conf",
        type=float,
        default=0.20,
        help="Seuil confiance annotation auto (défaut : 0.20)",
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Réutiliser les images déjà dans sources/<objet>/",
    )
    arguments = parser.parse_args()

    object_name = arguments.object_name.strip()
    label_fr = arguments.label_fr or object_name
    class_slug = object_name_to_slug(object_name)
    search_query = object_name if arguments.search_en else f"{object_name} object"

    dataset_root = PROJECT_ROOT / "datasets" / "custom"
    sources_directory = dataset_root / "sources" / class_slug
    labels_scratch_directory = dataset_root / "labels_scratch" / class_slug
    dataset_yaml = dataset_root / "dataset.yaml"
    metadata_path = dataset_root / "learned_classes.yaml"

    print("=" * 60)
    print(f"ClearVision — apprentissage : {object_name}")
    print("=" * 60)

    class_id = register_object_class(dataset_yaml, object_name)
    print(f"Classe dataset : {class_id} → « {object_name} »")

    if arguments.skip_download:
        image_paths = sorted(
            list(sources_directory.glob("*.jpg"))
            + list(sources_directory.glob("*.jpeg"))
            + list(sources_directory.glob("*.png"))
            + list(sources_directory.glob("*.webp"))
        )
        if not image_paths:
            raise FileNotFoundError(f"Aucune image dans {sources_directory}")
        print(f"Réutilisation de {len(image_paths)} image(s) existante(s).")
    else:
        image_paths = download_object_images(
            object_name,
            sources_directory,
            image_count=arguments.count,
            search_query=search_query,
        )

    labeled_pairs = auto_label_image_folder(
        image_paths,
        labels_scratch_directory,
        object_name=object_name,
        class_id=class_id,
        confidence_threshold=arguments.conf,
    )

    train_count, val_count = add_object_to_dataset(labeled_pairs, dataset_root)
    print(f"Dataset mis à jour : {train_count} train, {val_count} val")

    save_learned_class_metadata(metadata_path, class_slug, object_name, label_fr, class_id)

    if arguments.train:
        ensure_dataset_yaml(dataset_yaml)
        print("\nEntraînement du modèle (peut prendre plusieurs minutes)...")
        best_weights = train_custom_model(
            dataset_yaml=dataset_yaml,
            epochs=arguments.epochs,
        )
        print(f"\nModèle entraîné : {best_weights}")
        print("\nPour activer le modèle custom, dans config.py :")
        print("  USE_CUSTOM_MODEL = True")
        print(f"  YOLO_CUSTOM_PATH = PROJECT_ROOT / \"datasets/custom/runs/clearvision_custom/weights/best.pt\"")
    else:
        print("\nImages collectées et annotées. Pour entraîner :")
        print("  python scripts/learn_object.py \"{0}\" --skip-download --train".format(object_name))

    print("\nTerminé.")


if __name__ == "__main__":
    main()
