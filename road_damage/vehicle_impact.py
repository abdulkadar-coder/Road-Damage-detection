from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np


@dataclass
class SpotLocation:
    center_x: float
    center_y: float
    box_width: float
    box_height: float
    area_fraction: float
    lane_zone: str
    in_vehicle_path: bool
    proximity_zone: str
    distance_m: float


@dataclass
class VehicleImpact:
    impact_level: str  # "CRITICAL", "HIGH", "MODERATE", "LOW"
    impact_score: int  # 0 to 100
    vibration_shock: str
    suspension_risk: str
    suspension_risk_pct: int
    tire_risk: str
    tire_risk_pct: int
    steering_risk: str
    steering_risk_pct: int
    chassis_risk: str
    chassis_risk_pct: int
    two_wheeler_risk: str
    two_wheeler_risk_pct: int
    sedan_risk: str
    suv_risk: str
    truck_risk: str


@dataclass
class WarningAlert:
    warning_level: str  # "CRITICAL", "HIGH", "CAUTION", "SAFE"
    warning_title: str
    warning_message: str
    recommended_action: str
    urgency: str  # "IMMEDIATE", "UPCOMING", "MONITOR"
    color_bgr: tuple[int, int, int]
    color_hex: str


@dataclass
class DamageAssessment:
    spot: SpotLocation
    impact: VehicleImpact
    warning: WarningAlert
    class_code: str
    class_name: str
    confidence: float
    x1: float
    y1: float
    x2: float
    y2: float

    def to_row_dict(self) -> dict[str, Any]:
        """Convert into a flat dictionary suitable for dataframe logs and exports."""
        return {
            "class_code": self.class_code,
            "class_name": self.class_name,
            "confidence": round(self.confidence, 5),
            "impact_level": self.impact.impact_level,
            "impact_score": self.impact.impact_score,
            "warning_level": self.warning.warning_level,
            "lane_zone": self.spot.lane_zone,
            "in_vehicle_path": self.spot.in_vehicle_path,
            "proximity_zone": self.spot.proximity_zone,
            "estimated_distance_m": round(self.spot.distance_m, 1),
            "recommended_action": self.warning.recommended_action,
            "vibration_shock": self.impact.vibration_shock,
            "suspension_risk": self.impact.suspension_risk,
            "suspension_risk_pct": self.impact.suspension_risk_pct,
            "tire_risk": self.impact.tire_risk,
            "tire_risk_pct": self.impact.tire_risk_pct,
            "steering_risk": self.impact.steering_risk,
            "chassis_risk": self.impact.chassis_risk,
            "two_wheeler_risk": self.impact.two_wheeler_risk,
            "two_wheeler_risk_pct": self.impact.two_wheeler_risk_pct,
            "area_fraction": round(self.spot.area_fraction, 6),
            "x1": round(self.x1, 2),
            "y1": round(self.y1, 2),
            "x2": round(self.x2, 2),
            "y2": round(self.y2, 2),
        }


def analyze_damaged_spot(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    img_width: int,
    img_height: int,
    class_code: str,
    class_name: str,
    confidence: float,
) -> DamageAssessment:
    """Analyze the spatial spot location, vehicle damage severity, and generate alerts."""
    width_safe = max(1, img_width)
    height_safe = max(1, img_height)

    # 1. Spot geometry and road position
    center_x = (x1 + x2) / 2.0
    center_y = (y1 + y2) / 2.0
    box_w = max(0.0, x2 - x1)
    box_h = max(0.0, y2 - y1)
    area_fraction = min(1.0, max(0.0, (box_w * box_h) / (width_safe * height_safe)))

    # Lateral lane localization (normalized x across road FOV)
    norm_x = center_x / width_safe
    if norm_x < 0.35:
        lane_zone = "Left Shoulder / Lane"
        in_vehicle_path = False
    elif norm_x <= 0.65:
        lane_zone = "Direct Vehicle Travel Path"
        in_vehicle_path = True
    else:
        lane_zone = "Right Shoulder / Lane"
        in_vehicle_path = False

    # Longitudinal proximity / distance estimation based on base of defect on road surface
    # In dashcam perspectives, road horizon is ~35%-50%, bottom (80%-100%) is right at the hood
    norm_y = max(0.0, min(1.0, y2 / height_safe))
    if norm_y >= 0.75:
        proximity_zone = "Immediate Hazard (< 5m)"
        # Distance roughly 1.0m to 5.0m
        distance_m = max(1.2, (1.0 - norm_y) * 20.0 + 1.0)
    elif norm_y >= 0.45:
        proximity_zone = "Approaching (5m - 15m)"
        # Distance roughly 5.0m to 15.0m
        distance_m = 5.0 + (0.75 - norm_y) / 0.30 * 10.0
    else:
        proximity_zone = "Distant Ahead (> 15m)"
        # Distance roughly 15.0m to 35.0m
        distance_m = 15.0 + (0.45 - norm_y) / 0.45 * 20.0
    distance_m = max(1.0, min(50.0, distance_m))

    spot = SpotLocation(
        center_x=center_x,
        center_y=center_y,
        box_width=box_w,
        box_height=box_h,
        area_fraction=area_fraction,
        lane_zone=lane_zone,
        in_vehicle_path=in_vehicle_path,
        proximity_zone=proximity_zone,
        distance_m=distance_m,
    )

    # 2. Vehicle Impact Assessment
    code_norm = str(class_code).upper()

    # Base damage characteristics by defect class
    if code_norm == "D40" or "pothole" in class_name.lower():
        # Potholes create severe vertical shock, rim denting, and suspension damage
        base_impact = 60
        susp_base = 80
        tire_base = 85
        steer_base = 70
        chassis_base = 65
        two_wheel_base = 95  # Lethal / extreme fall risk for motorcycles
    elif code_norm == "D20" or "alligator" in class_name.lower():
        # Alligator cracking causes continuous vibration, tire tread wear, traction loss
        base_impact = 40
        susp_base = 55
        tire_base = 70
        steer_base = 40
        chassis_base = 25
        two_wheel_base = 75
    elif code_norm == "D10" or "transverse" in class_name.lower():
        # Transverse crack creates sudden perpendicular jolt across both front wheels
        base_impact = 35
        susp_base = 60
        tire_base = 50
        steer_base = 45
        chassis_base = 20
        two_wheel_base = 60
    elif code_norm == "D00" or "longitudinal" in class_name.lower():
        # Longitudinal crack causes tramlining / rut wander; wheel trapping for bikes
        base_impact = 25
        susp_base = 35
        tire_base = 40
        steer_base = 65
        chassis_base = 15
        two_wheel_base = 85  # Motorcycle tires get trapped in the crack groove!
    else:
        # Fallback or generic detection
        base_impact = 20
        susp_base = 30
        tire_base = 30
        steer_base = 25
        chassis_base = 15
        two_wheel_base = 40

    # Area multiplier: larger area increases depth and footprint
    # 0.5% area gives +8 pts, 3% area gives +25 pts
    area_bonus = min(25.0, area_fraction * 800.0)

    # Path multiplier: in direct path = full impact; shoulder = lateral avoidance potential
    path_multiplier = 1.0 if in_vehicle_path else 0.45

    # Proximity urgency: immediate hazard has higher instantaneous threat
    proximity_bonus = 10.0 if norm_y >= 0.75 else (5.0 if norm_y >= 0.45 else 0.0)

    raw_impact = (base_impact + area_bonus + proximity_bonus) * (0.65 + 0.35 * path_multiplier)
    impact_score = int(max(5, min(100, round(raw_impact))))

    # Component risk percentages
    susp_pct = int(min(100, round((susp_base + area_bonus) * (0.6 + 0.4 * path_multiplier))))
    tire_pct = int(min(100, round((tire_base + area_bonus) * (0.6 + 0.4 * path_multiplier))))
    steer_pct = int(min(100, round((steer_base + area_bonus * 0.5) * (0.6 + 0.4 * path_multiplier))))
    chassis_pct = int(min(100, round((chassis_base + area_bonus * 0.8) * path_multiplier)))
    two_wheel_pct = int(min(100, round(two_wheel_base + area_bonus * 0.4)))

    # Categorical ratings
    if impact_score >= 70:
        impact_level = "CRITICAL"
        vibration_shock = "Severe Shock (High G-Force Spike)"
    elif impact_score >= 50:
        impact_level = "HIGH"
        vibration_shock = "Heavy Jolt (Suspension Compression)"
    elif impact_score >= 30:
        impact_level = "MODERATE"
        vibration_shock = "Moderate Vibration (Cabin Shudder)"
    else:
        impact_level = "LOW"
        vibration_shock = "Mild Surface Bump"

    def risk_descriptor(pct: int, label_high: str, label_med: str, label_low: str) -> str:
        if pct >= 75:
            return f"Critical ({label_high})"
        if pct >= 50:
            return f"Elevated ({label_med})"
        if pct >= 25:
            return f"Moderate ({label_low})"
        return "Low (Normal Wear)"

    suspension_risk = risk_descriptor(
        susp_pct,
        "Shock Damper Bottoming & Bushing Failure",
        "High Damper & Strut Fatigue",
        "Mild Shock Wear",
    )
    tire_risk = risk_descriptor(
        tire_pct,
        "Rim Denting & Sidewall Blowout Hazard",
        "Tread Gouging & Air Loss Risk",
        "Minor Surface Abrasion",
    )
    steering_risk = risk_descriptor(
        steer_pct,
        "Sudden Steering Jerk & Toe Misalignment",
        "Moderate Wheel Pull / Tramlining",
        "Slight Vibration Drift",
    )
    chassis_risk = risk_descriptor(
        chassis_pct,
        "Underbody Scrape & Oil Pan Impact Hazard",
        "Low-Clearance Bottoming Risk",
        "Negligible Underbody Threat",
    )
    two_wheeler_risk = (
        "EXTREME FALL DANGER: High wheel-trap / ejection risk!"
        if two_wheel_pct >= 75
        else (
            "HIGH INSTABILITY: Severe handlebar wobble & balance loss"
            if two_wheel_pct >= 50
            else "MODERATE: Ride with caution"
        )
    )

    # Vehicle type specific impacts
    sedan_risk = (
        "High Risk: Low clearance invites bumper scrapes and rim pinch flats."
        if (impact_score >= 50 or chassis_pct >= 50)
        else "Moderate: Suspension dampening engaged, proceed with caution."
    )
    suv_risk = (
        "Moderate-High: Adequate ground clearance, but high axle shock on dampers."
        if impact_score >= 60
        else "Low-Moderate: Elevated clearance mitigates underbody contact."
    )
    truck_risk = (
        "Elevated Heavy Axle Shock: Accelerates tire fatigue and road wear."
        if impact_score >= 50
        else "Low: Heavy suspension absorbs surface defect."
    )

    impact = VehicleImpact(
        impact_level=impact_level,
        impact_score=impact_score,
        vibration_shock=vibration_shock,
        suspension_risk=suspension_risk,
        suspension_risk_pct=susp_pct,
        tire_risk=tire_risk,
        tire_risk_pct=tire_pct,
        steering_risk=steering_risk,
        steering_risk_pct=steer_pct,
        chassis_risk=chassis_risk,
        chassis_risk_pct=chassis_pct,
        two_wheeler_risk=two_wheeler_risk,
        two_wheeler_risk_pct=two_wheel_pct,
        sedan_risk=sedan_risk,
        suv_risk=suv_risk,
        truck_risk=truck_risk,
    )

    # 3. Warning Alert & Driver Advisory Generation
    if in_vehicle_path and impact_score >= 65:
        warning_level = "CRITICAL"
        warning_title = f"🚨 CRITICAL HAZARD: {class_name.upper()} IN TRAVEL PATH"
        urgency = "IMMEDIATE"
        color_bgr = (40, 40, 240)  # Bright Red
        color_hex = "#EF4444"
        if norm_y >= 0.75:
            action = "IMMEDIATE ACTION: Brake firmly (< 20 km/h) & grip steering wheel firmly!"
            message = (
                f"Severe {class_name.lower()} directly under vehicle path ({distance_m:.1f}m). "
                "Immediate rim and suspension damage imminent."
            )
        else:
            action = "EVASIVE ACTION: Decelerate immediately and check mirrors for safe lane diversion."
            message = (
                f"Severe {class_name.lower()} approaching in center path ({distance_m:.1f}m). "
                "High vehicle damage risk if struck directly."
            )
    elif impact_score >= 45 or (in_vehicle_path and impact_score >= 35):
        warning_level = "HIGH"
        warning_title = f"⚠️ HIGH WARNING: {class_name.upper()} AHEAD"
        urgency = "UPCOMING"
        color_bgr = (25, 140, 255)  # Vibrant Amber/Orange
        color_hex = "#F97316"
        loc_str = "in direct driving path" if in_vehicle_path else f"on {lane_zone.lower()}"
        action = "CAUTION: Reduce speed to < 40 km/h and steer away from defect."
        message = (
            f"{class_name} detected {loc_str} ({distance_m:.1f}m). "
            f"Estimated vehicle impact {impact_score}%. Expect moderate suspension jolt."
        )
    elif impact_score >= 25:
        warning_level = "CAUTION"
        warning_title = f"⚡ ROAD DEFECT: {class_name.upper()} DETECTED"
        urgency = "MONITOR"
        color_bgr = (0, 215, 255)  # Yellow
        color_hex = "#EAB308"
        action = "ADVISORY: Maintain speed awareness; minor vibration expected."
        message = f"{class_name} noted {lane_zone.lower()} ({distance_m:.1f}m). Low vehicle risk."
    else:
        warning_level = "SAFE"
        warning_title = f"✓ MINOR DEFECT: {class_name.upper()}"
        urgency = "MONITOR"
        color_bgr = (88, 255, 180)  # Neon Green
        color_hex = "#22C55E"
        action = "NORMAL: Safe to proceed at road speed."
        message = f"Minor surface defect ({distance_m:.1f}m). Negligible vehicle wear."

    warning = WarningAlert(
        warning_level=warning_level,
        warning_title=warning_title,
        warning_message=message,
        recommended_action=action,
        urgency=urgency,
        color_bgr=color_bgr,
        color_hex=color_hex,
    )

    return DamageAssessment(
        spot=spot,
        impact=impact,
        warning=warning,
        class_code=class_code,
        class_name=class_name,
        confidence=confidence,
        x1=x1,
        y1=y1,
        x2=x2,
        y2=y2,
    )


def summarize_overall_assessment(assessments: list[DamageAssessment]) -> dict[str, Any]:
    """Summarize a collection of frame detections into an overall vehicle threat summary."""
    if not assessments:
        return {
            "has_hazard": False,
            "overall_warning_level": "CLEAR",
            "overall_impact_level": "NONE",
            "max_impact_score": 0,
            "avg_impact_score": 0,
            "critical_spots_count": 0,
            "high_spots_count": 0,
            "in_path_count": 0,
            "dominant_hazard": "None",
            "recommended_action": "Road surface is clear. Safe to proceed at regular speed.",
            "overall_color_hex": "#22C55E",
            "vulnerability": {
                "suspension": 0,
                "tires": 0,
                "steering": 0,
                "chassis": 0,
                "two_wheeler": 0,
            },
        }

    max_impact = max(a.impact.impact_score for a in assessments)
    avg_impact = int(round(sum(a.impact.impact_score for a in assessments) / len(assessments)))
    critical_count = sum(1 for a in assessments if a.warning.warning_level == "CRITICAL")
    high_count = sum(1 for a in assessments if a.warning.warning_level == "HIGH")
    in_path_count = sum(1 for a in assessments if a.spot.in_vehicle_path)

    # Determine highest urgency warning
    if critical_count > 0:
        overall_warning = "CRITICAL"
        overall_impact = "CRITICAL"
        color_hex = "#EF4444"
    elif high_count > 0:
        overall_warning = "HIGH"
        overall_impact = "HIGH"
        color_hex = "#F97316"
    elif any(a.warning.warning_level == "CAUTION" for a in assessments):
        overall_warning = "CAUTION"
        overall_impact = "MODERATE"
        color_hex = "#EAB308"
    else:
        overall_warning = "SAFE"
        overall_impact = "LOW"
        color_hex = "#22C55E"

    # Find dominant (worst) assessment
    worst = max(assessments, key=lambda a: a.impact.impact_score)

    max_susp = max(a.impact.suspension_risk_pct for a in assessments)
    max_tire = max(a.impact.tire_risk_pct for a in assessments)
    max_steer = max(a.impact.steering_risk_pct for a in assessments)
    max_chassis = max(a.impact.chassis_risk_pct for a in assessments)
    max_two_wheel = max(a.impact.two_wheeler_risk_pct for a in assessments)

    return {
        "has_hazard": True,
        "overall_warning_level": overall_warning,
        "overall_impact_level": overall_impact,
        "max_impact_score": max_impact,
        "avg_impact_score": avg_impact,
        "critical_spots_count": critical_count,
        "high_spots_count": high_count,
        "in_path_count": in_path_count,
        "dominant_hazard": f"{worst.class_name} ({worst.spot.proximity_zone})",
        "recommended_action": worst.warning.recommended_action,
        "overall_color_hex": color_hex,
        "worst_assessment": worst,
        "vulnerability": {
            "suspension": max_susp,
            "tires": max_tire,
            "steering": max_steer,
            "chassis": max_chassis,
            "two_wheeler": max_two_wheel,
        },
    }


def draw_vehicle_warning_overlay(
    image: np.ndarray,
    assessments: list[DamageAssessment],
    show_trajectory: bool = True,
    show_hud_banner: bool = True,
) -> np.ndarray:
    """Render an ADAS-grade tactical overlay showing vehicle trajectory, damage spots, and warnings."""
    out = image.copy()
    h, w = out.shape[:2]

    # 1. Draw subtle forward vehicle travel corridor (Trajectory guidelines)
    if show_trajectory:
        corridor = out.copy()
        # Horizon vanishing point estimate (center, 42% height)
        vanish_y = int(h * 0.44)
        vanish_x = int(w * 0.50)
        lane_w_top = int(w * 0.12)
        lane_w_bot = int(w * 0.40)

        # Left guide lane
        pt_bl = (int(vanish_x - lane_w_bot), h)
        pt_tl = (int(vanish_x - lane_w_top), vanish_y)
        # Right guide lane
        pt_br = (int(vanish_x + lane_w_bot), h)
        pt_tr = (int(vanish_x + lane_w_top), vanish_y)

        pts = np.array([pt_bl, pt_tl, pt_tr, pt_br], np.int32)

        # Highlight color based on whether an immediate threat is inside corridor
        corridor_color = (40, 200, 100)  # Default emerald green
        if any(a.spot.in_vehicle_path and a.impact.impact_score >= 60 for a in assessments):
            corridor_color = (40, 40, 230)  # Red warning
        elif any(a.spot.in_vehicle_path for a in assessments):
            corridor_color = (30, 160, 255)  # Amber warning

        cv2.fillPoly(corridor, [pts], corridor_color)
        cv2.addWeighted(corridor, 0.12, out, 0.88, 0, out)

        # Draw corridor dashed/dotted boundary edges
        cv2.line(out, pt_bl, pt_tl, corridor_color, 2, cv2.LINE_AA)
        cv2.line(out, pt_br, pt_tr, corridor_color, 2, cv2.LINE_AA)

    # 2. Draw styled Damage Spot bounding boxes and tactical telemetry badges
    for idx, item in enumerate(assessments, 1):
        x1, y1, x2, y2 = int(item.x1), int(item.y1), int(item.x2), int(item.y2)
        color = item.warning.color_bgr
        thickness = 2 if item.impact.impact_level in {"CRITICAL", "HIGH"} else 1

        # Corner bracket accent lines for high-tech HUD look
        box_w = max(4, x2 - x1)
        box_h = max(4, y2 - y1)
        corner_len = max(6, min(24, int(min(box_w, box_h) * 0.28)))

        # Main box
        cv2.rectangle(out, (x1, y1), (x2, y2), color, thickness, cv2.LINE_AA)

        # Corner brackets (top-left, top-right, bottom-left, bottom-right)
        bracket_color = (255, 255, 255) if item.spot.in_vehicle_path else color
        # TL
        cv2.line(out, (x1, y1), (x1 + corner_len, y1), bracket_color, thickness + 1)
        cv2.line(out, (x1, y1), (x1, y1 + corner_len), bracket_color, thickness + 1)
        # TR
        cv2.line(out, (x2, y1), (x2 - corner_len, y1), bracket_color, thickness + 1)
        cv2.line(out, (x2, y1), (x2, y1 + corner_len), bracket_color, thickness + 1)
        # BL
        cv2.line(out, (x1, y2), (x1 + corner_len, y2), bracket_color, thickness + 1)
        cv2.line(out, (x1, y2), (x1, y2 - corner_len), bracket_color, thickness + 1)
        # BR
        cv2.line(out, (x2, y2), (x2 - corner_len, y2), bracket_color, thickness + 1)
        cv2.line(out, (x2, y2), (x2, y2 - corner_len), bracket_color, thickness + 1)

        # Badge text: Spot number + Name + Impact + Distance
        path_tag = " [PATH!]" if item.spot.in_vehicle_path else ""
        badge_text = (
            f"#{idx} {item.class_name.upper()[:16]} | IMPACT {item.impact.impact_score}% "
            f"| {item.spot.distance_m:.1f}m{path_tag}"
        )

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.44
        badge_thick = 1
        (txt_w, txt_h), baseline = cv2.getTextSize(badge_text, font, font_scale, badge_thick)

        badge_y = max(y1 - 6, txt_h + 8)
        badge_x = max(2, min(x1, w - txt_w - 10))

        # Badge background pill
        sub_img = out.copy()
        cv2.rectangle(
            sub_img,
            (badge_x - 3, badge_y - txt_h - 4),
            (badge_x + txt_w + 5, badge_y + baseline + 2),
            (18, 20, 16),
            -1,
        )
        cv2.addWeighted(sub_img, 0.85, out, 0.15, 0, out)
        cv2.rectangle(
            out,
            (badge_x - 3, badge_y - txt_h - 4),
            (badge_x + txt_w + 5, badge_y + baseline + 2),
            color,
            1,
            cv2.LINE_AA,
        )

        # Text
        cv2.putText(
            out,
            badge_text,
            (badge_x, badge_y - 1),
            font,
            font_scale,
            (255, 255, 255),
            badge_thick,
            cv2.LINE_AA,
        )

    # 3. Top ADAS HUD Warning Banner
    if show_hud_banner:
        summary = summarize_overall_assessment(assessments)
        banner_h = int(min(64, max(44, h * 0.07)))
        banner_overlay = out.copy()

        # Banner color
        if summary["overall_warning_level"] == "CRITICAL":
            bg_color = (25, 25, 180)  # Crimson
            border_color = (60, 60, 255)
            icon_text = "CRITICAL ROAD HAZARD"
        elif summary["overall_warning_level"] == "HIGH":
            bg_color = (20, 95, 190)  # Amber
            border_color = (30, 150, 255)
            icon_text = "WARNING: ROAD DAMAGE DETECTED"
        elif summary["overall_warning_level"] == "CAUTION":
            bg_color = (25, 120, 150)  # Ocher / Yellow
            border_color = (0, 215, 255)
            icon_text = "CAUTION: SURFACE DEFECT"
        else:
            bg_color = (20, 60, 30)  # Deep emerald
            border_color = (80, 220, 120)
            icon_text = "ROAD SURFACE CLEAR"

        cv2.rectangle(banner_overlay, (0, 0), (w, banner_h), bg_color, -1)
        cv2.addWeighted(banner_overlay, 0.82, out, 0.18, 0, out)
        cv2.line(out, (0, banner_h), (w, banner_h), border_color, 2, cv2.LINE_AA)

        # Banner text
        title_font = cv2.FONT_HERSHEY_SIMPLEX
        action_text = summary["recommended_action"]
        if summary["has_hazard"]:
            banner_line1 = (
                f"{icon_text} | IMPACT INDEX: {summary['max_impact_score']}% "
                f"({summary['overall_impact_level']}) | {len(assessments)} SPOT(S)"
            )
        else:
            banner_line1 = f"{icon_text} | NO HAZARDS DETECTED"

        cv2.putText(
            out,
            banner_line1,
            (16, int(banner_h * 0.44)),
            title_font,
            0.52,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

        cv2.putText(
            out,
            f">> ACTION: {action_text}",
            (16, int(banner_h * 0.84)),
            title_font,
            0.42,
            (213, 255, 88),  # Acid lime accent
            1,
            cv2.LINE_AA,
        )

    return out
