"""Fixtures pytest ClearVision."""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


@pytest.fixture
def synthetic_green_image():
    import numpy as np

    image = np.zeros((480, 640, 3), dtype=np.uint8)
    image[250:400, 100:500] = (40, 180, 40)
    return image
