from __future__ import annotations

import unittest
import cv2
import numpy as np

from road_damage.model import load_fallback_model
from road_damage.image import process_image


class EndToEndInferenceTests(unittest.TestCase):
    def test_image_inference_with_fallback_model(self):
        # Create a synthetic road image with some shapes
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        # Add road surface grey
        img[200:, :] = 80
        # Add lane line
        cv2.line(img, (320, 200), (200, 480), (255, 255, 255), 4)
        cv2.line(img, (320, 200), (440, 480), (255, 255, 255), 4)

        ok, buf = cv2.imencode(".png", img)
        self.assertTrue(ok)
        img_bytes = buf.tobytes()

        model = load_fallback_model()
        result = process_image(img_bytes, "test_road.png", model, confidence=0.1, image_size=320)

        self.assertIsNotNone(result.image_bytes)
        self.assertGreater(len(result.image_bytes), 0)
        self.assertEqual(result.width, 640)
        self.assertEqual(result.height, 480)
        self.assertIn("overall_warning_level", result.summary)
        self.assertIn("vulnerability", result.summary)


if __name__ == "__main__":
    unittest.main()
