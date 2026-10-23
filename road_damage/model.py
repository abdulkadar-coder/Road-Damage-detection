from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from .config import CLASS_ALIASES, CLASS_NAMES, DEFAULT_MODEL_PATH

# Sentinel indicating the loaded model uses generic COCO weights (not RDD2022).
_FALLBACK_MODEL_ATTR = "_is_generic_fallback"


class ModelLoadError(RuntimeError):
    """A road-damage checkpoint is missing, corrupted, or incompatible."""


def normalize_class_code(label: Any) -> str | None:
    text = str(label).strip()
    upper = text.upper()
    if upper in CLASS_NAMES:
        return upper
    return CLASS_ALIASES.get(text.lower())


def _class_names_by_id(names: Any) -> dict[int, str]:
    if isinstance(names, dict):
        try:
            items = sorted(((int(class_id), name) for class_id, name in names.items()), key=lambda pair: pair[0])
        except (TypeError, ValueError) as exc:
            raise ModelLoadError("Checkpoint class IDs must be integers 0 through 3.") from exc
    elif isinstance(names, (list, tuple)):
        items = list(enumerate(names))
    else:
        raise ModelLoadError("Checkpoint does not contain a readable detection class map.")

    if [class_id for class_id, _ in items] != list(range(len(items))):
        raise ModelLoadError("Checkpoint class IDs must be contiguous and start at 0.")

    mapped: dict[int, str] = {}
    for class_id, label in items:
        code = normalize_class_code(label)
        if code is None:
            raise ModelLoadError(
                f"Checkpoint class {class_id} ({label!r}) is not one of the RDD2022 classes "
                "(D00, D10, D20, D40)."
            )
        mapped[class_id] = code

    expected = set(CLASS_NAMES)
    if len(mapped) != len(expected) or set(mapped.values()) != expected:
        labels = ", ".join(CLASS_NAMES)
        raise ModelLoadError(
            f"Expected exactly the four RDD2022 detection classes ({labels}); "
            f"checkpoint exposes {len(mapped)} class(es)."
        )
    return mapped


def validate_model(model: Any) -> dict[int, str]:
    """Validate detector task and class metadata; return class ID -> RDD code."""
    task = getattr(model, "task", None)
    if task != "detect":
        raise ModelLoadError(
            f"Expected an Ultralytics object-detection checkpoint (task='detect'); got {task!r}."
        )
    return _class_names_by_id(getattr(model, "names", None))


def class_code_for_id(names: Any, class_id: int) -> str:
    mapping = _class_names_by_id(names)
    try:
        return mapping[int(class_id)]
    except (KeyError, TypeError, ValueError) as exc:
        raise ModelLoadError(f"Prediction refers to unknown class ID {class_id!r}.") from exc


def load_road_damage_model(weights_path: str | Path = DEFAULT_MODEL_PATH):
    """Load only the requested trained checkpoint; never fall back to generic weights."""
    path = Path(weights_path).expanduser()
    if not path.is_file():
        raise ModelLoadError(
            f"Trained road-damage weights were not found at {path}. "
            "Fine-tune YOLOv8n on RDD2022, then copy the generated best.pt to this path."
        )
    try:
        if path.stat().st_size == 0:
            raise ModelLoadError(f"Checkpoint is empty: {path}")
    except OSError as exc:
        raise ModelLoadError(f"Could not access checkpoint {path}: {exc}") from exc

    try:
        from ultralytics import YOLO
    except Exception as exc:
        raise ModelLoadError(
            "Could not import Ultralytics/PyTorch under Python "
            f"{sys.version.split()[0]}. Use a fresh environment with a supported Python/PyTorch "
            "combination and install this project's requirements there. "
            f"Import details: {exc}"
        ) from exc

    try:
        model = YOLO(str(path))
    except Exception as exc:
        raise ModelLoadError(
            f"Could not load checkpoint {path}. It may be corrupted or incompatible. "
            "Use the trained YOLOv8n RDD2022 detection checkpoint produced by scripts/train.py. "
            f"Loader details: {exc}"
        ) from exc

    validate_model(model)
    return model


def load_fallback_model():
    """Download the generic YOLOv8n COCO-pretrained model as a fallback.

    The returned model carries a sentinel attribute so callers can distinguish
    it from a trained RDD2022 checkpoint.
    """
    try:
        from ultralytics import YOLO
    except Exception as exc:
        raise ModelLoadError(
            "Could not import Ultralytics/PyTorch under Python "
            f"{sys.version.split()[0]}. Install this project's requirements. "
            f"Import details: {exc}"
        ) from exc

    try:
        model = YOLO("yolov8n.pt")  # auto-downloads from Ultralytics hub
    except Exception as exc:
        raise ModelLoadError(
            f"Could not download or load generic YOLOv8n weights: {exc}"
        ) from exc

    setattr(model, _FALLBACK_MODEL_ATTR, True)
    return model


def is_road_damage_model(model: Any) -> bool:
    """Return True if the model is a trained RDD2022 checkpoint (not the generic fallback)."""
    return not getattr(model, _FALLBACK_MODEL_ATTR, False)


def generic_class_name_for_id(names: Any, class_id: int) -> str:
    """Look up a human-readable class name from any YOLO model's name map."""
    if isinstance(names, dict):
        name = names.get(class_id, names.get(str(class_id)))
        if name is not None:
            return str(name)
    elif isinstance(names, (list, tuple)) and 0 <= class_id < len(names):
        return str(names[class_id])
    return f"class_{class_id}"


def resolve_inference_device(device: str | int | None) -> str | int | None:
    """Map Auto to Ultralytics' device default and reject unavailable CUDA devices clearly."""
    if device is None or str(device).strip().lower() in {"", "auto"}:
        return None
    normalized = str(device).strip().lower()
    if normalized == "cpu":
        return "cpu"
    if normalized.isdigit():
        try:
            import torch
        except Exception as exc:
            raise ModelLoadError("PyTorch could not be imported for CUDA device selection.") from exc
        if not torch.cuda.is_available():
            raise ModelLoadError(
                f"CUDA device {normalized} was selected, but CUDA is unavailable. Choose Auto or CPU."
            )
        index = int(normalized)
        if index >= torch.cuda.device_count():
            raise ModelLoadError(
                f"CUDA device {index} does not exist; this system reports {torch.cuda.device_count()} device(s)."
            )
        return normalized
    raise ModelLoadError("Inference device must be Auto, CPU, or a CUDA index such as 0.")

