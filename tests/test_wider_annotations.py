"""Tests for the shared WIDER Face annotation parser.

This parser exists because four separate copies had drifted apart: only convert.py
handled the zero-face dummy line, so split.py raised

    ValueError: invalid literal for int() with base 10: '0--Parade/...jpg'

on the real annotation file. train_list.txt / val_list.txt were therefore never
generated, which silently disabled ONNX precision validation and the split evaluation.

The parsing rules are covered through ``parse_lines`` with in-memory input. The
file-backed paths use fixtures committed under ``tests/fixtures`` rather than
runtime-created temp files: a restricted file sandbox can deny writes to freshly created
directories, and a *skipped* test is not a safety net -- a generator/`with` bug in
``parse_samples`` (returning a generator whose file was already closed) slipped through
exactly that gap and broke ``src/data/split.py``.
"""
import os
import sys
import unittest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.data.wider_annotations import (  # noqa: E402
    parse_lines,
    parse_samples,
    parse_samples_with_image_root,
)

FIXTURES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures")


class TestParseLines(unittest.TestCase):
    """Pure parsing rules -- no files involved."""

    def test_single_entry_with_two_faces(self):
        samples = list(parse_lines([
            "a/1.jpg",
            "2",
            "10 20 30 40 0 0 0 0 0 0",
            "50 60 70 80 0 0 0 0 0 0",
        ]))
        self.assertEqual(samples, [("a/1.jpg", [[10, 20, 30, 40], [50, 60, 70, 80]])])

    def test_zero_face_dummy_line_is_consumed(self):
        """The exact regression that broke split.py."""
        samples = list(parse_lines([
            "a/1.jpg", "0", "0 0 0 0 0 0 0 0 0 0",
            "a/2.jpg", "1", "5 6 7 8 0 0 0 0 0 0",
        ]))
        self.assertEqual([s[0] for s in samples], ["a/1.jpg", "a/2.jpg"])
        self.assertEqual(samples[0][1], [])
        self.assertEqual(samples[1][1], [[5, 6, 7, 8]])

    def test_zero_face_without_dummy_line(self):
        samples = list(parse_lines([
            "a/1.jpg", "0",
            "a/2.jpg", "1", "5 6 7 8 0 0 0 0 0 0",
        ]))
        self.assertEqual([s[0] for s in samples], ["a/1.jpg", "a/2.jpg"])
        self.assertEqual(samples[0][1], [])

    def test_many_entries_stay_aligned(self):
        """A zero-face entry must not shift every following entry."""
        lines = []
        for i in range(6):
            if i % 2 == 0:
                lines += [f"a/{i}.jpg", "0", "0 0 0 0 0 0 0 0 0 0"]
            else:
                lines += [f"a/{i}.jpg", "1", f"{i} {i} 10 10 0 0 0 0 0 0"]
        samples = list(parse_lines(lines))
        self.assertEqual(len(samples), 6)
        self.assertEqual([s[0] for s in samples], [f"a/{i}.jpg" for i in range(6)])
        self.assertEqual(samples[3][1], [[3, 3, 10, 10]])

    def test_blank_lines_are_tolerated_everywhere(self):
        """Including between an image path and its count, and before a box line."""
        samples = list(parse_lines([
            "", "a/1.jpg", "", "2", "",
            "1 2 3 4 0 0 0 0 0 0", "", "5 6 7 8 0 0 0 0 0 0", "", "",
        ]))
        self.assertEqual(samples, [("a/1.jpg", [[1, 2, 3, 4], [5, 6, 7, 8]])])

    def test_crlf_and_surrounding_whitespace_are_stripped(self):
        samples = list(parse_lines(["  a/1.jpg  \r\n", " 1\r\n", " 1 2 3 4 0 0 0 0 0 0\r\n"]))
        self.assertEqual(samples, [("a/1.jpg", [[1, 2, 3, 4]])])

    def test_attribute_columns_are_ignored(self):
        """WIDER appends blur/expression/illumination/occlusion/pose after the box."""
        samples = list(parse_lines(["a/1.jpg", "1", "10 20 30 40 1 2 3 4 5"]))
        self.assertEqual(samples[0][1], [[10, 20, 30, 40]])

    def test_empty_input_yields_nothing(self):
        self.assertEqual(list(parse_lines([])), [])

    def test_truncated_final_entry_does_not_crash(self):
        """Real files can end mid-record; take what is there."""
        samples = list(parse_lines(["a/1.jpg", "2", "1 2 3 4 0 0 0 0 0 0"]))
        self.assertEqual(samples, [("a/1.jpg", [[1, 2, 3, 4]])])

    def test_malformed_count_raises_a_clear_error(self):
        with self.assertRaises(ValueError) as ctx:
            list(parse_lines(["a/1.jpg", "not-a-number"]))
        self.assertIn("expected a face count", str(ctx.exception))


class TestFileAccess(unittest.TestCase):
    """File-backed paths, against fixtures committed in the repository."""

    def test_parse_samples_reads_a_file(self):
        path = os.path.join(FIXTURES, "tiny_annotations.txt")
        samples = list(parse_samples(path))
        self.assertEqual(len(samples), 3)
        self.assertEqual(samples[0][0], "0--Parade/0_Parade_marchingband_1_849.jpg")
        self.assertEqual(samples[0][1], [[449, 330, 122, 149]])
        # the zero-face entry in the middle must not shift the entry after it
        self.assertEqual(samples[1][1], [])
        self.assertEqual(samples[2][0], "1--Handshaking/1_Handshaking_Handshaking_1_100.jpg")
        self.assertEqual(samples[2][1], [[10, 20, 30, 40], [50, 60, 70, 80]])

    def test_repeated_reads_give_the_same_result(self):
        """Directly guards the generator / already-closed-file regression."""
        path = os.path.join(FIXTURES, "tiny_annotations.txt")
        self.assertEqual(list(parse_samples(path)), list(parse_samples(path)))

    def test_missing_file_raises(self):
        with self.assertRaises(FileNotFoundError):
            list(parse_samples(os.path.join(FIXTURES, "does_not_exist.txt")))

    def test_images_that_do_not_exist_are_skipped(self):
        anno = os.path.join(FIXTURES, "images_annotations.txt")
        image_root = os.path.join(FIXTURES, "images")
        samples = list(parse_samples_with_image_root(anno, image_root))
        self.assertEqual(len(samples), 1)
        self.assertTrue(samples[0][0].endswith("present.jpg"))
        self.assertEqual(samples[0][1], [[1, 2, 3, 4]])


if __name__ == "__main__":
    unittest.main(verbosity=2)
