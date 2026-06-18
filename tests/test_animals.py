"""Tests traductions et labels animaux."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from clearvision_voice import label_to_fr
from detection.filters import label_to_french


def test_animal_labels_in_world_list():
    from vocabulary import WORLD_ANIMAL_CLASSES

    assert len(WORLD_ANIMAL_CLASSES) >= 80
    assert "dog" in WORLD_ANIMAL_CLASSES
    assert "squirrel" in WORLD_ANIMAL_CLASSES
    assert "penguin" in WORLD_ANIMAL_CLASSES
    assert "hamster" in WORLD_ANIMAL_CLASSES


def test_object_labels_in_world_list():
    from vocabulary import WORLD_OBJECT_CLASSES

    assert len(WORLD_OBJECT_CLASSES) >= 80
    assert "laptop" in WORLD_OBJECT_CLASSES
    assert "hammer" in WORLD_OBJECT_CLASSES


def test_plant_labels_in_world_list():
    from vocabulary import WORLD_PLANT_CLASSES

    assert len(WORLD_PLANT_CLASSES) >= 40
    assert "rose" in WORLD_PLANT_CLASSES
    assert "mushroom" in WORLD_PLANT_CLASSES


def test_french_translation_dog():
    assert label_to_french("dog") == "Chien"
    assert label_to_fr("dog") == "chien"


def test_french_translation_squirrel():
    assert label_to_french("squirrel") == "Écureuil"


def test_french_translation_penguin():
    assert label_to_french("penguin") == "Manchot"


def test_french_translation_hammer():
    assert label_to_french("hammer") == "Marteau"
