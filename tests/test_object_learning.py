"""Tests du module d'apprentissage d'objets."""

from pathlib import Path

import yaml

from object_learning.dataset_builder import load_dataset_classes, register_object_class
from object_learning.image_downloader import object_name_to_slug
from object_learning.learned_labels import learned_label_fr, load_learned_classes


def test_object_name_to_slug():
    assert object_name_to_slug("Banc Public") == "banc_public"
    assert object_name_to_slug("  ") == "objet"


def test_register_object_class(tmp_path: Path):
    dataset_yaml = tmp_path / "dataset.yaml"
    class_id = register_object_class(dataset_yaml, "poubelle")
    assert class_id == 0

    class_map = load_dataset_classes(dataset_yaml)
    assert class_map[0] == "poubelle"

    yaml_content = yaml.safe_load(dataset_yaml.read_text(encoding="utf-8"))
    assert yaml_content["path"] == tmp_path.resolve().as_posix()
    assert yaml_content["train"] == "images/train"

    same_id = register_object_class(dataset_yaml, "poubelle")
    assert same_id == 0

    second_id = register_object_class(dataset_yaml, "banc public")
    assert second_id == 1


def test_learned_label_fr(tmp_path: Path, monkeypatch):
    metadata_path = tmp_path / "learned_classes.yaml"
    metadata_path.write_text(
        yaml.safe_dump(
            {
                "banc_public": {
                    "object_name": "banc public",
                    "label_fr": "banc",
                    "class_id": 0,
                }
            }
        ),
        encoding="utf-8",
    )

    import object_learning.learned_labels as learned_module

    monkeypatch.setattr(learned_module, "LEARNED_CLASSES_PATH", metadata_path)

    assert learned_label_fr("banc public") == "banc"
    assert learned_label_fr("banc_public") == "banc"
    assert load_learned_classes()["banc_public"]["label_fr"] == "banc"
