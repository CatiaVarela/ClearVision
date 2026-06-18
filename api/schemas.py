"""Schémas Pydantic pour l'API ClearVision."""

from pydantic import BaseModel


class BoundingBox(BaseModel):
    x1: int
    y1: int
    x2: int
    y2: int


class DetectionResult(BaseModel):
    label_key: str
    label_fr: str
    bbox: BoundingBox
    score: float | None = None
    source: str | None = None
    distance_m: float | None = None


class DetectionResponse(BaseModel):
    detection_count: int
    detections: list[DetectionResult]
    annotated_image_base64: str
