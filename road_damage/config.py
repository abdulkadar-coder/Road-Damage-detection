from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_PATH = ROOT_DIR / "models" / "road_damage_yolov8n.pt"
CLASS_NAMES = {
    "D00": "Longitudinal crack",
    "D10": "Transverse crack",
    "D20": "Alligator crack",
    "D40": "Pothole",
}
CLASS_ALIASES = {
    "D00": "D00", "longitudinal crack": "D00", "longitudinal_crack": "D00",
    "D10": "D10", "transverse crack": "D10", "transverse_crack": "D10",
    "D20": "D20", "alligator crack": "D20", "alligator_crack": "D20",
    "D40": "D40", "pothole": "D40",
}
