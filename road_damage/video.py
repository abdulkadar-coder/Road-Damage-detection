from __future__ import annotations

import csv
import io
import math
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import cv2

from .config import CLASS_NAMES
from .severity import estimate_severity
from .model import (
    class_code_for_id,
    generic_class_name_for_id,
    is_road_damage_model,
    resolve_inference_device,
    validate_model,
)

CSV_FIELDS = [
    "timestamp_seconds", "frame_number", "class_code", "class_name", "confidence",
    "severity", "severity_score", "area_fraction", "x1", "y1", "x2", "y2",
]
ALLOWED_EXTENSIONS = {".mp4", ".avi", ".mov"}


class VideoProcessingError(RuntimeError):
    """An upload could not be safely decoded, inferred on, or encoded."""


@dataclass
class VideoResult:
    video_bytes: bytes
    detections: list[dict]
    frame_count: int
    sampled_frames: int
    fps: float
    width: int
    height: int
    duration: float

    def csv_bytes(self) -> bytes:
        buffer = io.StringIO(newline="")
        writer = csv.DictWriter(buffer, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(self.detections)
        return buffer.getvalue().encode("utf-8")


def _validate_metadata(cap: cv2.VideoCapture) -> tuple[int, float, int, int]:
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if frame_count <= 0:
        raise VideoProcessingError("No frames were found. Try a different video file or re-encode it.")
    if not math.isfinite(fps) or fps <= 0:
        raise VideoProcessingError("The video has invalid frame-rate metadata.")
    if width <= 0 or height <= 0:
        raise VideoProcessingError("The video has invalid frame dimensions.")
    return frame_count, fps, width, height


def _make_writer(output_path: str, fps: float, width: int, height: int) -> cv2.VideoWriter:
    writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        writer.release()
        raise VideoProcessingError(
            "OpenCV could not initialize MP4 encoding. Install a full OpenCV build or re-run "
            "with a codec-enabled environment."
        )
    return writer


def _validate_model_classes(model) -> None:
    """Backward-compatible validation wrapper for existing inference callers."""
    validate_model(model)


def process_video(
    video_bytes: bytes,
    filename: str,
    model,
    confidence: float = 0.25,
    sample_every: int = 3,
    image_size: int = 640,
    progress_callback: Callable[[float, str], None] | None = None,
    device: str | int | None = None,
) -> VideoResult:
    """Infer on sampled frames; preserve all frames in the annotated output video."""
    extension = Path(filename).suffix.lower()
    if extension not in ALLOWED_EXTENSIONS:
        raise VideoProcessingError("Only MP4, AVI, and MOV video uploads are supported.")
    if not video_bytes:
        raise VideoProcessingError("The uploaded file is empty.")
    if not 0.0 < confidence < 1.0:
        raise ValueError("Confidence must be between 0 and 1.")
    if sample_every < 1:
        raise ValueError("Frame sampling interval must be at least 1.")

    rdd_mode = is_road_damage_model(model)
    if rdd_mode:
        _validate_model_classes(model)

    if image_size <= 0:
        raise ValueError("Inference image size must be a positive integer.")
    resolved_device = resolve_inference_device(device)
    prediction_options = {"conf": confidence, "imgsz": image_size, "verbose": False}
    if resolved_device is not None:
        prediction_options["device"] = resolved_device

    with tempfile.TemporaryDirectory(prefix="road_damage_") as temp_dir:
        input_path = os.path.join(temp_dir, f"input{extension}")
        output_path = os.path.join(temp_dir, "annotated.mp4")
        with open(input_path, "wb") as handle:
            handle.write(video_bytes)

        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            cap.release()
            raise VideoProcessingError("OpenCV could not read this video. Check that it is not corrupt.")
        writer = None
        rows: list[dict] = []
        sampled_frames = 0
        try:
            frame_count, fps, width, height = _validate_metadata(cap)
            writer = _make_writer(output_path, fps, width, height)
            frame_number = 0
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                if frame.shape[1] != width or frame.shape[0] != height:
                    raise VideoProcessingError("A decoded frame has inconsistent dimensions.")
                if frame_number % sample_every == 0:
                    predictions = model.predict(frame, **prediction_options)
                    if not predictions:
                        raise VideoProcessingError("The model returned no inference result for a frame.")
                    result = predictions[0]
                    annotated = result.plot()
                    boxes = result.boxes
                    if boxes is not None:
                        for box in boxes:
                            class_id = int(box.cls[0].item())
                            confidence_value = float(box.conf[0].item())
                            coords = box.xyxy[0].tolist()
                            x1, y1, x2, y2 = (float(value) for value in coords)
                            x1 = max(0.0, min(x1, width - 1.0))
                            x2 = max(0.0, min(x2, width - 1.0))
                            y1 = max(0.0, min(y1, height - 1.0))
                            y2 = max(0.0, min(y2, height - 1.0))
                            area_fraction = max(0.0, x2 - x1) * max(0.0, y2 - y1) / (width * height)

                            if rdd_mode:
                                code_upper = class_code_for_id(result.names, class_id)
                                display_name = CLASS_NAMES[code_upper]
                                severity, severity_score = estimate_severity(code_upper, area_fraction)
                            else:
                                code_upper = str(class_id)
                                display_name = generic_class_name_for_id(result.names, class_id)
                                severity_score = 0 if area_fraction < 0.005 else (1 if area_fraction < 0.03 else 2)
                                severity = "Minor" if severity_score == 0 else ("Moderate" if severity_score == 1 else "Severe")

                            rows.append({
                                "timestamp_seconds": round(frame_number / fps, 3),
                                "frame_number": frame_number,
                                "class_code": code_upper,
                                "class_name": display_name,
                                "confidence": round(confidence_value, 5),
                                "severity": severity,
                                "severity_score": severity_score,
                                "area_fraction": round(area_fraction, 6),
                                "x1": round(x1, 2), "y1": round(y1, 2),
                                "x2": round(x2, 2), "y2": round(y2, 2),
                            })
                    frame = annotated
                    sampled_frames += 1
                writer.write(frame)
                frame_number += 1
                if progress_callback and (frame_number % max(1, frame_count // 100) == 0 or frame_number == frame_count):
                    progress_callback(frame_number / frame_count, f"Processing frame {frame_number:,} of {frame_count:,}")
            if frame_number == 0:
                raise VideoProcessingError("No decodable frames were found in this video.")
            writer.release()
            writer = None
            if not os.path.isfile(output_path) or os.path.getsize(output_path) == 0:
                raise VideoProcessingError("The processed video could not be encoded.")
            verification = cv2.VideoCapture(output_path)
            if not verification.isOpened():
                verification.release()
                raise VideoProcessingError("OpenCV created an output file that it cannot decode as MP4.")
            readable, _ = verification.read()
            verification.release()
            if not readable:
                raise VideoProcessingError("The encoded MP4 contains no readable frames.")
            with open(output_path, "rb") as handle:
                output_bytes = handle.read()
            return VideoResult(
                video_bytes=output_bytes,
                detections=rows,
                frame_count=frame_number,
                sampled_frames=sampled_frames,
                fps=fps,
                width=width,
                height=height,
                duration=frame_number / fps,
            )
        except cv2.error as exc:
            raise VideoProcessingError(f"OpenCV failed while processing this video: {exc}") from exc
        finally:
            cap.release()
            if writer is not None:
                writer.release()
