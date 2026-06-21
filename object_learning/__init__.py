"""Apprentissage de nouveaux objets pour ClearVision."""

from object_learning.auto_label import auto_label_image_folder
from object_learning.dataset_builder import add_object_to_dataset, load_dataset_classes
from object_learning.image_downloader import download_object_images

__all__ = [
    "add_object_to_dataset",
    "auto_label_image_folder",
    "download_object_images",
    "load_dataset_classes",
]
