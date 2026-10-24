from __future__ import annotations

from dataclasses import dataclass, field
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
from .vehicle_impact import (
    DamageAssessment,
    analyze_damaged_spot,
    draw_vehicle_warning_overlay,
    summarize_overall_assessment,
)

ALLOWED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


@dataclass
class ImageResult:
    image_bytes: bytes
    detections: list[dict]
    width: int
    height: int
    summary: dict = field(default_factory=dict)


def process_image(image_bytes: bytes, filename: str, model, confidence: float = 0.25,
                  image_size: int = 640, device: str | int | None = None) -> ImageResult:
    """Detect supported road damage on one uploaded image, evaluate vehicle impact, and encode annotation as PNG."""
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

    rows: list[dict] = []
    assessments: list[DamageAssessment] = []

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
                severity_score = 0 if area_fraction < 0.005 else (1 if area_fraction < 0.03 else 2)
                severity = "Minor" if severity_score == 0 else ("Moderate" if severity_score == 1 else "Severe")

            # Evaluate spot localization, vehicle impact severity, and generate warning
            assessment = analyze_damaged_spot(
                x1=x1,
                y1=y1,
                x2=x2,
                y2=y2,
                img_width=width,
                img_height=height,
                class_code=code,
                class_name=label,
                confidence=score,
            )
            assessments.append(assessment)

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
                "impact_level": assessment.impact.impact_level,
                "impact_score": assessment.impact.impact_score,
                "warning_level": assessment.warning.warning_level,
                "lane_zone": assessment.spot.lane_zone,
                "in_vehicle_path": assessment.spot.in_vehicle_path,
                "proximity_zone": assessment.spot.proximity_zone,
                "estimated_distance_m": round(assessment.spot.distance_m, 1),
                "recommended_action": assessment.warning.recommended_action,
                "vibration_shock": assessment.impact.vibration_shock,
                "suspension_risk": assessment.impact.suspension_risk,
                "suspension_risk_pct": assessment.impact.suspension_risk_pct,
                "tire_risk": assessment.impact.tire_risk,
                "tire_risk_pct": assessment.impact.tire_risk_pct,
                "steering_risk": assessment.impact.steering_risk,
                "chassis_risk": assessment.impact.chassis_risk,
                "two_wheeler_risk": assessment.impact.two_wheeler_risk,
                "two_wheeler_risk_pct": assessment.impact.two_wheeler_risk_pct,
            })

    # Render high-grade ADAS warning overlay with lane corridor, spot badges, and HUD banner
    annotated = draw_vehicle_warning_overlay(
        image=image,
        assessments=assessments,
        show_trajectory=True,
        show_hud_banner=True,
    )

    summary = summarize_overall_assessment(assessments)

    ok, output = cv2.imencode(".png", annotated)
    if not ok:
        raise VideoProcessingError("Could not encode the annotated image.")
    return ImageResult(
        image_bytes=output.tobytes(),
        detections=rows,
        width=width,
        height=height,
        summary=summary,
    )
