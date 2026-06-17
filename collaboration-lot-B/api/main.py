"""API FastAPI ClearVision — détection d'objets sur image."""

import base64
import sys
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, File, Query, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from api.schemas import BoundingBox, DetectionResponse, DetectionResult
from detection import detect_objects_in_image, detections_to_json

app = FastAPI(title="ClearVision API", version="1.0.0")

WEB_DIRECTORY = ROOT / "web"
if WEB_DIRECTORY.exists():
    app.mount("/static", StaticFiles(directory=str(WEB_DIRECTORY)), name="static")


@app.get("/")
async def serve_web_interface():
    index_path = WEB_DIRECTORY / "index.html"
    if index_path.exists():
        return FileResponse(index_path)
    return {"message": "ClearVision API — POST /detect avec une image"}


@app.post("/detect", response_model=DetectionResponse)
async def detect_image(
    file: UploadFile = File(...),
    fast: bool = Query(False, description="Mode rapide sans YOLO-World"),
):
    image_bytes = await file.read()
    image_array = np.frombuffer(image_bytes, dtype=np.uint8)
    source_image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

    if source_image is None:
        raise ValueError("Fichier image invalide ou format non supporté")

    annotated_image, detection_info = detect_objects_in_image(source_image, fast=fast)
    serialized_detections = detections_to_json(detection_info["detections"])

    encode_success, encoded_image = cv2.imencode(".jpg", annotated_image)
    if not encode_success:
        raise RuntimeError("Impossible d'encoder l'image annotée")

    image_base64 = base64.b64encode(encoded_image.tobytes()).decode("utf-8")

    detection_results = [
        DetectionResult(
            label_key=item["label_key"],
            label_fr=item["label_fr"],
            bbox=BoundingBox(**item["bbox"]),
            score=item.get("score"),
            source=item.get("source"),
            distance_m=item.get("distance_m"),
        )
        for item in serialized_detections
    ]

    return DetectionResponse(
        detection_count=len(detection_results),
        detections=detection_results,
        annotated_image_base64=image_base64,
    )
