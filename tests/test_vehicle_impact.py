from __future__ import annotations

import unittest
import numpy as np

from road_damage.vehicle_impact import (
    analyze_damaged_spot,
    summarize_overall_assessment,
    draw_vehicle_warning_overlay,
)


class VehicleImpactTests(unittest.TestCase):
    def test_pothole_in_direct_path_triggers_critical_alert(self):
        # A large pothole right in the driving path (center x=500, bottom y=900 out of 1000x1000)
        assessment = analyze_damaged_spot(
            x1=400.0,
            y1=750.0,
            x2=600.0,
            y2=920.0,
            img_width=1000,
            img_height=1000,
            class_code="D40",
            class_name="Pothole",
            confidence=0.88,
        )

        self.assertTrue(assessment.spot.in_vehicle_path)
        self.assertEqual(assessment.spot.lane_zone, "Direct Vehicle Travel Path")
        self.assertEqual(assessment.spot.proximity_zone, "Immediate Hazard (< 5m)")
        self.assertLess(assessment.spot.distance_m, 5.0)

        # Impact must be critical or high
        self.assertIn(assessment.impact.impact_level, {"CRITICAL", "HIGH"})
        self.assertGreaterEqual(assessment.impact.impact_score, 65)

        # Warning should be CRITICAL
        self.assertEqual(assessment.warning.warning_level, "CRITICAL")
        self.assertIn("BRAKE", assessment.warning.recommended_action.upper())

        # Two-wheeler risk should be extreme
        self.assertGreaterEqual(assessment.impact.two_wheeler_risk_pct, 75)
        self.assertIn("EXTREME", assessment.impact.two_wheeler_risk)

    def test_minor_crack_on_shoulder_generates_advisory(self):
        # Small transverse crack on left shoulder, distant ahead
        assessment = analyze_damaged_spot(
            x1=50.0,
            y1=200.0,
            x2=120.0,
            y2=250.0,
            img_width=1000,
            img_height=1000,
            class_code="D10",
            class_name="Transverse crack",
            confidence=0.65,
        )

        self.assertFalse(assessment.spot.in_vehicle_path)
        self.assertEqual(assessment.spot.lane_zone, "Left Shoulder / Lane")
        self.assertEqual(assessment.spot.proximity_zone, "Distant Ahead (> 15m)")
        self.assertIn(assessment.warning.warning_level, {"CAUTION", "SAFE"})

    def test_summary_aggregation(self):
        a1 = analyze_damaged_spot(
            x1=450, y1=800, x2=550, y2=900,
            img_width=1000, img_height=1000,
            class_code="D40", class_name="Pothole", confidence=0.9
        )
        a2 = analyze_damaged_spot(
            x1=100, y1=300, x2=150, y2=350,
            img_width=1000, img_height=1000,
            class_code="D00", class_name="Longitudinal crack", confidence=0.7
        )

        summary = summarize_overall_assessment([a1, a2])
        self.assertTrue(summary["has_hazard"])
        self.assertEqual(summary["overall_warning_level"], "CRITICAL")
        self.assertGreater(summary["max_impact_score"], 60)
        self.assertEqual(summary["in_path_count"], 1)
        self.assertIn("vulnerability", summary)
        self.assertGreater(summary["vulnerability"]["tires"], 50)

    def test_drawing_overlay(self):
        dummy_img = np.zeros((480, 640, 3), dtype=np.uint8)
        assessment = analyze_damaged_spot(
            x1=250, y1=300, x2=380, y2=420,
            img_width=640, img_height=480,
            class_code="D40", class_name="Pothole", confidence=0.85
        )
        overlay = draw_vehicle_warning_overlay(dummy_img, [assessment], show_trajectory=True, show_hud_banner=True)
        self.assertEqual(overlay.shape, dummy_img.shape)
        # Verify pixels were drawn
        self.assertGreater(overlay.sum(), 0)


if __name__ == "__main__":
    unittest.main()
