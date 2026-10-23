from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from road_damage.config import CLASS_NAMES, DEFAULT_MODEL_PATH  # noqa: E402
from road_damage.model import ModelLoadError, load_road_damage_model, normalize_class_code  # noqa: E402

RUN_NAME = "yolov8n_rdd2022"


def _ordered_names(raw_names: Any) -> list[Any]:
    if isinstance(raw_names, dict):
        try:
            return [value for _, value in sorted(raw_names.items(), key=lambda item: int(item[0]))]
        except (TypeError, ValueError) as exc:
            raise ValueError("Dataset YAML class IDs must be integers.") from exc
    if isinstance(raw_names, (list, tuple)):
        return list(raw_names)
    raise ValueError("Dataset YAML must define class names as a list or numeric-keyed map.")


def validate_dataset_yaml(data_path: Path) -> dict[str, Any]:
    try:
        data = yaml.safe_load(data_path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ValueError(f"Could not read dataset YAML {data_path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError("Dataset YAML must contain a mapping.")
    names = _ordered_names(data.get("names"))
    codes = [normalize_class_code(label) for label in names]
    expected = list(CLASS_NAMES)
    if codes != expected:
        raise ValueError(
            "Dataset class IDs/names must be in this order: "
            + ", ".join(f"{class_id}={code}" for class_id, code in enumerate(expected))
        )

    root_value = data.get("path", data_path.parent)
    dataset_root = Path(root_value)
    if not dataset_root.is_absolute():
        dataset_root = data_path.parent / dataset_root
    for split in ("train", "val"):
        split_value = data.get(split)
        if not isinstance(split_value, str) or not split_value:
            raise ValueError(f"Dataset YAML is missing the {split!r} image path.")
        split_path = Path(split_value)
        if not split_path.is_absolute():
            split_path = dataset_root / split_path
        if not split_path.is_dir():
            raise ValueError(f"{split} image directory does not exist: {split_path}")
    return data


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fine-tune YOLOv8n on the four RDD2022 road-damage classes."
    )
    parser.add_argument("--data", type=Path, default=ROOT / "data/rdd2022_yolo/data.yaml")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="", help="For example 0 for CUDA, cpu, or blank for auto-detect.")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--project", type=Path, default=ROOT / "runs/road_damage")
    parser.add_argument("--base-weights", default="yolov8n.pt",
                        help="YOLOv8n initialization weights; these are fine-tuned before app inference.")
    parser.add_argument("--output-weights", type=Path, default=DEFAULT_MODEL_PATH,
                        help="Stable checkpoint path used by the Streamlit app.")
    args = parser.parse_args()

    data_path = args.data.expanduser().resolve()
    if not data_path.is_file():
        parser.error(f"Dataset YAML not found: {data_path}. Run scripts/prepare_dataset.py first.")
    try:
        validate_dataset_yaml(data_path)
    except ValueError as exc:
        parser.error(str(exc))
    if args.epochs < 1 or args.batch < 1 or args.imgsz < 32 or args.workers < 0:
        parser.error("epochs, batch, image size, and worker settings must be positive (workers may be zero).")

    try:
        from ultralytics import YOLO
    except Exception as exc:
        parser.error(f"Could not import Ultralytics/PyTorch in this environment: {exc}")

    project_dir = args.project.expanduser().resolve()
    run_dir = project_dir / RUN_NAME
    model = YOLO(args.base_weights)
    train_args = {
        "data": str(data_path),
        "epochs": args.epochs,
        "batch": args.batch,
        "imgsz": args.imgsz,
        "workers": args.workers,
        "project": str(project_dir),
        "name": RUN_NAME,
        "patience": 12,
        "seed": 42,
    }
    if args.device:
        train_args["device"] = args.device
    model.train(**train_args)

    trainer = getattr(model, "trainer", None)
    candidates: list[Path] = []
    trainer_best = getattr(trainer, "best", None)
    if trainer_best:
        candidates.append(Path(trainer_best))
    save_dir = getattr(trainer, "save_dir", None)
    if save_dir:
        candidates.append(Path(save_dir) / "weights" / "best.pt")
    best_path = next((path for path in candidates if path.is_file()), None)
    if best_path is None:
        raise SystemExit(
            "Training returned without a best.pt checkpoint. Check the Ultralytics run logs; "
            f"expected output under {run_dir / 'weights' / 'best.pt'}."
        )

    try:
        load_road_damage_model(best_path)
    except ModelLoadError as exc:
        raise SystemExit(f"Training produced a checkpoint that cannot be used by this app: {exc}") from exc

    output_path = args.output_weights.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(output_path.name + ".tmp")
    try:
        shutil.copy2(best_path, temporary_path)
        temporary_path.replace(output_path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()

    print(f"Training run's best checkpoint: {best_path}")
    print(f"Validated app checkpoint installed at: {output_path}")
    print("The app now loads this trained checkpoint; it never substitutes generic or random weights.")


if __name__ == "__main__":
    main()

