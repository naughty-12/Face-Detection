"""Tests for WIDER -> YOLO coordinate conversion.

This conversion decides where the model is told faces are, so an off-by-a-half-box
error here would quietly degrade training without failing anything.
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.convert import boxes_to_yolo  # noqa: E402


class TestBoxesToYolo(unittest.TestCase):
    def parse(self, line):
        parts = line.split()
        self.assertEqual(parts[0], "0", "class id for the single face class")
        return [float(v) for v in parts[1:]]

    def test_centre_and_size_are_normalised(self):
        cx, cy, w, h = self.parse(boxes_to_yolo([[10, 20, 30, 40]], 100, 200)[0])
        self.assertAlmostEqual(cx, 0.25)   # (10 + 30/2) / 100
        self.assertAlmostEqual(cy, 0.20)   # (20 + 40/2) / 200
        self.assertAlmostEqual(w, 0.30)
        self.assertAlmostEqual(h, 0.20)

    def test_full_image_box_is_exactly_one(self):
        cx, cy, w, h = self.parse(boxes_to_yolo([[0, 0, 100, 200]], 100, 200)[0])
        self.assertAlmostEqual(cx, 0.5)
        self.assertAlmostEqual(cy, 0.5)
        self.assertAlmostEqual(w, 1.0)
        self.assertAlmostEqual(h, 1.0)

    def test_single_pixel_box(self):
        cx, cy, w, h = self.parse(boxes_to_yolo([[50, 50, 1, 1]], 100, 100)[0])
        self.assertAlmostEqual(cx, 0.505)
        self.assertAlmostEqual(cy, 0.505)
        self.assertAlmostEqual(w, 0.01)
        self.assertAlmostEqual(h, 0.01)

    def test_empty_input_yields_no_lines(self):
        self.assertEqual(boxes_to_yolo([], 100, 100), [])

    def test_order_is_preserved(self):
        lines = boxes_to_yolo([[0, 0, 10, 10], [90, 90, 10, 10]], 100, 100)
        self.assertEqual(len(lines), 2)
        first = self.parse(lines[0])
        second = self.parse(lines[1])
        self.assertAlmostEqual(first[0], 0.05)
        self.assertAlmostEqual(second[0], 0.95)

    def test_values_are_serialisable_as_six_decimals(self):
        """The writer emits fixed 6-decimal text; make sure it round-trips closely."""
        line = boxes_to_yolo([[1, 2, 3, 4]], 640, 480)[0]
        _, cx, cy, w, h = line.split()
        for text in (cx, cy, w, h):
            self.assertLessEqual(len(text.split(".")[1]), 6)


if __name__ == "__main__":
    unittest.main(verbosity=2)
