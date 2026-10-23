from __future__ import annotations

import argparse
import hashlib
import random
import shutil
import sys
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

import cv2
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from road_damage.config import CLASS_ALIASES, CLASS_NAMES  # noqa: E402

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert RDD2022 Pascal VOC labels to Ultralytics YOLO format.")
    parser.add_argument("--source", required=True, type=Path, help="Root of extracted RDD2022 labeled training files.")
    parser.add_argument("--output", type=Path, default=ROOT / "data" / "rdd2022_yolo", help="New or empty output folder.")
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--copy-images", action="store_true", help="Copy images; default creates hardlinks where possible.")
    return parser.parse_args()


def _class_id(label: str) -> int | None:
    normalized = label.strip().lower()
    code = CLASS_ALIASES.get(normalized, label.strip().upper())
    order = list(CLASS_NAMES)
    return order.index(code) if code in order else None


def _find_image(xml_path: Path, images_by_name: dict[str, list[Path]]) -> Path | None:
    for extension in IMAGE_EXTENSIONS:
        sibling = xml_path.with_suffix(extension)
        if sibling.is_file():
            return sibling
    candidates = images_by_name.get(xml_path.stem.lower(), [])
    same_parent = [path for path in candidates if path.parent == xml_path.parent]
    if len(same_parent) == 1:
        return same_parent[0]
    return candidates[0] if len(candidates) == 1 else None


def _convert_annotation(xml_path: Path, image_path: Path) -> tuple[list[str], Counter[str]]:
    root = ET.parse(xml_path).getroot()
    size = root.find("size")
    width = int(size.findtext("width", "0")) if size is not None else 0
    height = int(size.findtext("height", "0")) if size is not None else 0
    if width <= 0 or height <= 0:
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Could not read image dimensions for {image_path}")
        height, width = image.shape[:2]
    lines: list[str] = []
    counts: Counter[str] = Counter()
    for obj in root.findall("object"):
        label = obj.findtext("name", "")
        class_id = _class_id(label)
        box = obj.find("bndbox")
        if class_id is None or box is None:
            continue
        try:
            xmin = float(box.findtext("xmin", "0"))
            ymin = float(box.findtext("ymin", "0"))
            xmax = float(box.findtext("xmax", "0"))
            ymax = float(box.findtext("ymax", "0"))
        except ValueError:
            continue
        xmin, xmax = sorted((min(max(xmin, 0), width), min(max(xmax, 0), width)))
        ymin, ymax = sorted((min(max(ymin, 0), height), min(max(ymax, 0), height)))
        if xmax <= xmin or ymax <= ymin:
            continue
        x_center = ((xmin + xmax) / 2) / width
        y_center = ((ymin + ymax) / 2) / height
        box_width = (xmax - xmin) / width
        box_height = (ymax - ymin) / height
        lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {box_width:.6f} {box_height:.6f}")
        counts[CLASS_NAMES[list(CLASS_NAMES)[class_id]]] += 1
    return lines, counts


def main() -> None:
    args = parse_args()
    if not args.source.is_dir():
        raise SystemExit(f"Source directory does not exist: {args.source}")
    if not 0.05 <= args.validation_fraction < 1.0:
        raise SystemExit("--validation-fraction must be at least 0.05 and less than 1.0")

    images_by_name: dict[str, list[Path]] = {}
    for image_path in args.source.rglob("*"):
        if image_path.is_file() and image_path.suffix.lower() in IMAGE_EXTENSIONS:
            images_by_name.setdefault(image_path.stem.lower(), []).append(image_path)
    records = []
    skipped = 0
    for xml_path in args.source.rglob("*.xml"):
        image_path = _find_image(xml_path, images_by_name)
        if image_path is None:
            skipped += 1
            continue
        try:
            lines, counts = _convert_annotation(xml_path, image_path)
        except (ET.ParseError, OSError, ValueError) as exc:
            print(f"Skipping {xml_path}: {exc}", file=sys.stderr)
            skipped += 1
            continue
        if not lines:
            skipped += 1
            continue
        records.append((image_path, lines, counts))
    if len(records) < 2:
        raise SystemExit("Fewer than two usable annotated images found; check --source and the extracted dataset layout.")

    random.Random(args.seed).shuffle(records)
    validation_count = max(1, round(len(records) * args.validation_fraction))
    validation_count = min(validation_count, len(records) - 1)
    validation = records[:validation_count]
    training = records[validation_count:]

    if args.output.exists() and any(args.output.iterdir()):
        raise SystemExit(f"Output directory is not empty: {args.output}. Choose a new --output folder to avoid overwriting data.")
    class_counts = {"train": Counter(), "val": Counter()}
    for split, items in (("train", training), ("val", validation)):
        image_dir = args.output / "images" / split
        label_dir = args.output / "labels" / split
        image_dir.mkdir(parents=True, exist_ok=True)
        label_dir.mkdir(parents=True, exist_ok=True)
        for image_path, lines, counts in items:
            # Include a path digest so identical basenames from different countries do not collide.
            digest = hashlib.sha1(str(image_path.resolve()).encode("utf-8")).hexdigest()[:10]
            output_stem = f"{image_path.stem}_{digest}"
            image_target = image_dir / f"{output_stem}{image_path.suffix.lower()}"
            if args.copy_images:
                shutil.copy2(image_path, image_target)
            else:
                try:
                    image_target.hardlink_to(image_path)
                except OSError:
                    shutil.copy2(image_path, image_target)
            (label_dir / f"{output_stem}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
            class_counts[split].update(counts)

    names = {index: name for index, name in enumerate(CLASS_NAMES.values())}
    data_yaml = {"path": args.output.resolve().as_posix(), "train": "images/train", "val": "images/val", "names": names}
    (args.output / "data.yaml").write_text(yaml.safe_dump(data_yaml, sort_keys=False, allow_unicode=True), encoding="utf-8")
    print(f"Prepared {len(training)} training and {len(validation)} validation images at {args.output}")
    print(f"Skipped {skipped} unlabeled, unmatched, or unusable XML annotations")
    for split in ("train", "val"):
        print(f"{split} instances by class: {dict(class_counts[split])}")
    print("Inspect several converted labels and images before starting a model run.")


if __name__ == "__main__":
    main()
