"""Tests for the matching geometry used by the size-stratified error analysis.

``analyze_errors`` decides what counts as a hit by greedy IoU matching at 0.5, then bins
faces by equivalent side length. Both are easy to get subtly wrong and neither was
covered before, even though the published size-stratified recall numbers rest on them.
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.eval.analyze_errors import SIZE_BINS, _bin_of, _side, compute_iou  # noqa: E402


class TestComputeIou(unittest.TestCase):
    def test_identical_boxes(self):
        self.assertAlmostEqual(compute_iou([0, 0, 10, 10], [0, 0, 10, 10]), 1.0)

    def test_disjoint_boxes(self):
        self.assertEqual(compute_iou([0, 0, 10, 10], [20, 20, 30, 30]), 0.0)

    def test_touching_boxes_have_zero_overlap(self):
        self.assertEqual(compute_iou([0, 0, 10, 10], [10, 0, 20, 10]), 0.0)

    def test_half_overlap(self):
        # 10x10 each, overlap 5x10 = 50, union = 100 + 100 - 50 = 150
        self.assertAlmostEqual(compute_iou([0, 0, 10, 10], [5, 0, 15, 10]), 50 / 150)

    def test_contained_box(self):
        # inner 5x5 inside outer 10x10 -> 25 / 100
        self.assertAlmostEqual(compute_iou([0, 0, 10, 10], [0, 0, 5, 5]), 0.25)

    def test_zero_area_box_does_not_divide_by_zero(self):
        self.assertEqual(compute_iou([0, 0, 0, 0], [0, 0, 0, 0]), 0.0)

    def test_is_symmetric(self):
        a, b = [1, 2, 11, 12], [5, 6, 20, 25]
        self.assertAlmostEqual(compute_iou(a, b), compute_iou(b, a))


class TestSide(unittest.TestCase):
    def test_equivalent_side_is_geometric_mean_of_width_and_height(self):
        self.assertAlmostEqual(_side([0, 0, 30, 40]), (30 * 40) ** 0.5)

    def test_square(self):
        self.assertAlmostEqual(_side([0, 0, 32, 32]), 32.0)

    def test_degenerate_box_is_zero(self):
        self.assertEqual(_side([5, 5, 5, 5]), 0.0)


class TestBinOf(unittest.TestCase):
    def test_boundaries_are_lower_inclusive(self):
        # bins use lo <= side < hi
        self.assertEqual(_bin_of(0), 0)
        self.assertEqual(_bin_of(31.999), 0)
        self.assertEqual(_bin_of(32), 1)
        self.assertEqual(_bin_of(95.999), 1)
        self.assertEqual(_bin_of(96), 2)
        self.assertEqual(_bin_of(255.999), 2)
        self.assertEqual(_bin_of(256), 3)

    def test_very_large_falls_in_last_bin(self):
        self.assertEqual(_bin_of(100000.0), len(SIZE_BINS) - 1)

    def test_bins_are_contiguous_and_cover_infinity(self):
        self.assertEqual(SIZE_BINS[0][1], 0)
        for i in range(len(SIZE_BINS) - 1):
            self.assertEqual(SIZE_BINS[i][2], SIZE_BINS[i + 1][1])
        self.assertEqual(SIZE_BINS[-1][2], float("inf"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
