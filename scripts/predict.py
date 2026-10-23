from __future__ import annotations

import argparse
import csv
import io
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from road_damage.config import DEFAULT_MODEL_PATH  # noqa: E402
from road_damage.image import process_image  # noqa: E402
from road_damage.model import ModelLoadError, load_road_damage_model  # noqa: E402
from road_damage.video import CSV_FIELDS, VideoProcessingError  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Run real YOLOv8n RDD2022 inference on one road image.")
    parser.add_argument("--source", required=True, type=Path, help="JPG, JPEG, or PNG road image.")
    parser.add_argument("--weights", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--output", type=Path, default=ROOT / "runs/predict/annotated.png")
    parser.add_argument("--csv", type=Path, default=None, help="Optional detection CSV output path.")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="auto", help="auto, cpu, or CUDA index such as 0.")
    args = parser.parse_args()

    source = args.source.expanduser().resolve()
    weights = args.weights.expanduser().resolve()
    output = args.output.expanduser().resolve()
    csv_path = (args.csv.expanduser().resolve() if args.csv else
                output.with_name(output.stem + "_detections.csv"))
    if not source.is_file():
        parser.error(f"Road image does not exist: {source}")
    if not 0.0 < args.conf < 1.0:
        parser.error("--conf must be between 0 and 1.")
    if args.imgsz < 32:
        parser.error("--imgsz must be at least 32.")

    try:
        model = load_road_damage_model(weights)
        result = process_image(
            source.read_bytes(),
            source.name,
            model,
            confidence=args.conf,
            image_size=args.imgsz,
            device=args.device,
        )
    except (OSError, ModelLoadError, VideoProcessingError, ValueError) as exc:
        raise SystemExit(f"Inference failed: {exc}") from exc

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(result.image_bytes)
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=CSV_FIELDS)
    writer.writeheader()
    writer.writerows(result.detections)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path.write_text(buffer.getvalue(), encoding="utf-8-sig")

    print(f"Annotated image: {output}")
    print(f"Detection CSV: {csv_path}")
    print(f"Image dimensions: {result.width}x{result.height}")
    print(f"Real detections returned by the loaded checkpoint: {len(result.detections)}")
    for detection in result.detections:
        print(json.dumps(detection, ensure_ascii=False))


if __name__ == "__main__":
    main()

