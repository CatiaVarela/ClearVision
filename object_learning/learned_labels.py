"""Libellés des objets appris par l'utilisateur."""

from __future__ import annotations

from pathlib import Path

import yaml

from config import LEARNED_CLASSES_PATH


def load_learned_classes() -> dict[str, dict]:
    if not LEARNED_CLASSES_PATH.exists():
        return {}

    content = yaml.safe_load(LEARNED_CLASSES_PATH.read_text(encoding="utf-8")) or {}
    return content if isinstance(content, dict) else {}


def learned_label_fr(label_key: str) -> str | None:
    learned = load_learned_classes()
    normalized_key = label_key.lower().replace(" ", "_")

    for slug, metadata in learned.items():
        object_name = str(metadata.get("object_name", "")).lower()
        if label_key.lower() == object_name or normalized_key == slug:
            return metadata.get("label_fr") or metadata.get("object_name")

    return None


def learned_object_names() -> set[str]:
    learned = load_learned_classes()
    names = set()
    for metadata in learned.values():
        if metadata.get("object_name"):
            names.add(str(metadata["object_name"]).lower())
    return names
