from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .config import CLASS_NAMES
from .severity import estimate_severity
from .model import (
    class_code_for_id,
    generic_class_name_for_id,
    is_road_damage_model,
    resolve_inference_device,
)
from .video import VideoProcessingError, _validate_model_classes

ALLOWED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


@dataclass
class ImageResult:
    image_bytes: bytes
    detections: list[dict]
    width: int
    height: int


def process_image(image_bytes: bytes, filename: str, model, confidence: float = 0.25,
                  image_size: int = 640, device: str | int | None = None) -> ImageResult:
    """Detect supported road damage on one uploaded image and encode its annotation as PNG."""
    extension = Path(filename).suffix.lower()
    if extension not in ALLOWED_IMAGE_EXTENSIONS:
        raise VideoProcessingError("Only JPG, JPEG, and PNG image uploads are supported.")
    if not image_bytes:
        raise VideoProcessingError("The uploaded image is empty.")

    rdd_mode = is_road_damage_model(model)
    if rdd_mode:
        _validate_model_classes(model)

    if not 0.0 < confidence < 1.0:
        raise ValueError("Confidence must be between 0 and 1.")

    encoded = np.frombuffer(image_bytes, dtype=np.uint8)
    image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if image is None:
        raise VideoProcessingError("OpenCV could not decode this image. Check the file and try again.")
    if image_size <= 0:
        raise ValueError("Inference image size must be a positive integer.")
    resolved_device = resolve_inference_device(device)
    prediction_options = {"conf": confidence, "imgsz": image_size, "verbose": False}
    if resolved_device is not None:
        prediction_options["device"] = resolved_device
    height, width = image.shape[:2]
    if width <= 0 or height <= 0:
        raise VideoProcessingError("The image has invalid dimensions.")

    predictions = model.predict(image, **prediction_options)
    if not predictions:
        raise VideoProcessingError("The model returned no inference result for the image.")
    result = predictions[0]
    annotated = result.plot()
    rows: list[dict] = []
    if result.boxes is not None:
        for box in result.boxes:
            class_id = int(box.cls[0].item())
            score = float(box.conf[0].item())
            x1, y1, x2, y2 = (float(value) for value in box.xyxy[0].tolist())
            x1 = max(0.0, min(x1, width - 1.0))
            x2 = max(0.0, min(x2, width - 1.0))
            y1 = max(0.0, min(y1, height - 1.0))
            y2 = max(0.0, min(y2, height - 1.0))
            area_fraction = max(0.0, x2 - x1) * max(0.0, y2 - y1) / (width * height)

            if rdd_mode:
                code = class_code_for_id(result.names, class_id)
                label = CLASS_NAMES[code]
                severity, severity_score = estimate_severity(code, area_fraction)
            else:
                code = str(class_id)
                label = generic_class_name_for_id(result.names, class_id)
                # Simple area-based severity for generic models
                severity_score = 0 if area_fraction < 0.005 else (1 if area_fraction < 0.03 else 2)
                severity = "Minor" if severity_score == 0 else ("Moderate" if severity_score == 1 else "Severe")

            rows.append({
                "timestamp_seconds": None,
                "frame_number": None,
                "class_code": code,
                "class_name": label,
                "confidence": round(score, 5),
                "severity": severity,
                "severity_score": severity_score,
                "area_fraction": round(area_fraction, 6),
                "x1": round(x1, 2), "y1": round(y1, 2),
                "x2": round(x2, 2), "y2": round(y2, 2),
            })

    ok, output = cv2.imencode(".png", annotated)
    if not ok:
        raise VideoProcessingError("Could not encode the annotated image.")
    return ImageResult(image_bytes=output.tobytes(), detections=rows, width=width, height=height)
