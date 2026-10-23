from __future__ import annotations


def estimate_severity(class_code: str, box_area_fraction: float) -> tuple[str, int]:
    """Return a transparent heuristic label and score, not an engineering rating.

    Potholes and alligator cracking have a one-point class prior. Normalized box
    area adds zero points below 0.5%, one point from 0.5% to 3%, and two points
    at 3% or larger. Scores 0, 1, and 2+ map to Minor, Moderate, and Severe.
    """
    area = min(max(float(box_area_fraction), 0.0), 1.0)
    class_prior = 1 if class_code in {"D20", "D40"} else 0
    area_points = 0 if area < 0.005 else (1 if area < 0.03 else 2)
    score = class_prior + area_points
    label = "Minor" if score == 0 else ("Moderate" if score == 1 else "Severe")
    return label, score
