"""Tests végétation (masque HSV, sans modèle ML)."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from detection.vegetation import build_foliage_mask, scene_has_ground_vegetation


def test_foliage_mask_on_green_image(synthetic_green_image):
    mask = build_foliage_mask(synthetic_green_image)
    assert mask.sum() > 0


def test_scene_has_ground_vegetation_on_green(synthetic_green_image):
    assert bool(scene_has_ground_vegetation(synthetic_green_image))
