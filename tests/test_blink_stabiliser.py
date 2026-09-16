"""Blinking needs asymmetric smoothing **and** a hysteretic "snap shut".

Reports from the same person on 2026-09-16, in order:

1. with a symmetric EMA at ``--expression-alpha 0.65``: "**更抖**";
2. with the default 0.45: fast blinks never fully closed ("**大多数时候闭眼闭不上，有稍微一点点的睁开**");
3. after asymmetric smoothing: "**不抖但不实**" -- the jitter was gone, the closure still wasn't there.

A 150 s sampling run (values read back from VTube Studio, inverted through the active calibration)
showed why: their *natural* blink valleys land around raw 0.35-0.49, and the calibration's lower
bound (0.30) only pushes values below 0.30 all the way to closed. So most blinks kept a 5-25% gap.

The obvious fix -- raising the calibration's lower bound to ~0.5 -- is the wrong tool: the mapping
is linear, so shrinking the span from 0.69 to 0.47 would **amplify open-eye jitter by 1.5x**, i.e.
bring back complaint (1).

``BlinkStabiliser`` therefore snaps the output to exactly 0 once the smoothed value falls below
``snap_below``, and only lets go once the raw target rises above ``unsnap_above``. The hysteresis is
the important part: without it a value hovering at the threshold would flicker between closed and
half-open. Because the snap replaces the value instead of rescaling the range, the open-eye gain
(and therefore the jitter) is untouched.
"""
import os
import sys
import unittest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "apps", "vtube_bridge"))

# Direct import, no skipIf: a missing class must FAIL this file, not skip it silently.
from vtube_studio_bridge.vtube_studio_bridge import BlinkStabiliser  # noqa: E402


class TestBlinkStabiliser(unittest.TestCase):
    def make(self, **kwargs):
        return BlinkStabiliser(**kwargs)

    def test_the_first_value_is_adopted_without_smoothing(self):
        stabiliser = self.make()

        out = stabiliser.update({"EyeOpenLeft": 0.8})

        self.assertAlmostEqual(out["EyeOpenLeft"], 0.8, places=6)

    def test_a_genuine_blink_closes_fast(self):
        """State 1.0 -> target 0.0 with close_alpha=0.8 must land at 0.2 in one frame.

        The snap and the deep-closure rule are disabled here so this test isolates the
        asymmetric alpha itself.
        """
        stabiliser = self.make(close_alpha=0.8, open_alpha=0.45, snap_threshold=0.25,
                               snap_below=0.0, deep_below=0.0)
        stabiliser.update({"EyeOpenLeft": 1.0})

        out = stabiliser.update({"EyeOpenLeft": 0.0})

        self.assertAlmostEqual(out["EyeOpenLeft"], 0.2, places=6,
                               msg="a real blink must snap shut, not crawl")

    def test_small_noise_around_open_is_smoothed_gently(self):
        """A 0.10 wobble is not a blink: it must not be amplified by the fast alpha."""
        stabiliser = self.make(close_alpha=0.8, open_alpha=0.45, snap_threshold=0.25)
        stabiliser.update({"EyeOpenLeft": 1.0})

        out = stabiliser.update({"EyeOpenLeft": 0.9})

        self.assertAlmostEqual(out["EyeOpenLeft"], 0.955, places=6)

    def test_reopening_is_gentle(self):
        stabiliser = self.make(close_alpha=0.8, open_alpha=0.45, snap_threshold=0.25)
        stabiliser.update({"EyeOpenLeft": 0.2})

        out = stabiliser.update({"EyeOpenLeft": 1.0})

        self.assertAlmostEqual(out["EyeOpenLeft"], 0.56, places=6)

    def test_keys_it_does_not_manage_pass_through_untouched(self):
        stabiliser = self.make()

        out = stabiliser.update({"MouthOpen": 0.3, "EyeOpenLeft": 0.5})

        self.assertAlmostEqual(out["MouthOpen"], 0.3, places=6)
        self.assertAlmostEqual(out["EyeOpenLeft"], 0.5, places=6)

    def test_both_eyes_are_tracked_independently(self):
        stabiliser = self.make(close_alpha=0.8, open_alpha=0.45, snap_threshold=0.25,
                               snap_below=0.0, deep_below=0.0)
        stabiliser.update({"EyeOpenLeft": 1.0, "EyeOpenRight": 1.0})

        out = stabiliser.update({"EyeOpenLeft": 0.0, "EyeOpenRight": 1.0})

        self.assertAlmostEqual(out["EyeOpenLeft"], 0.2, places=6)
        self.assertAlmostEqual(out["EyeOpenRight"], 1.0, places=6)

    def test_reset_clears_state(self):
        stabiliser = self.make()
        stabiliser.update({"EyeOpenLeft": 0.0})
        stabiliser.reset()

        out = stabiliser.update({"EyeOpenLeft": 0.7})

        self.assertAlmostEqual(out["EyeOpenLeft"], 0.7, places=6)


class TestSquintVersusBlink(unittest.TestCase):
    """Depth alone cannot separate a blink from a squint on this face/camera.

    Measured valleys: natural blinks bottom at raw 0.35-0.49, and a deliberate squint sits around
    0.40-0.50 as well -- the two overlap, which is why a pure depth threshold swallowed squints
    ("眯眼很容易闭上"). Duration is what differs: a real blink is ~0.15 s (4-5 frames at 30 fps),
    a squint is sustained. So a shallow, *persistent* low value is released back to a partial
    closure, while a *deep* closure (raw < 0.38) stays closed for as long as it lasts.
    """

    def make(self, **kwargs):
        defaults = {"close_alpha": 0.8, "open_alpha": 0.45, "snap_threshold": 0.25,
                    "snap_below": 0.5, "unsnap_above": 0.7, "deep_below": 0.38,
                    "squint_frames": 10}
        defaults.update(kwargs)
        return BlinkStabiliser(**defaults)

    def test_a_fast_blink_still_reads_as_fully_closed(self):
        stabiliser = self.make()
        stabiliser.update({"EyeOpenLeft": 0.97})

        out = None
        for _ in range(3):                       # ~0.1 s: a real blink
            out = stabiliser.update({"EyeOpenLeft": 0.45})

        self.assertEqual(out["EyeOpenLeft"], 0.0, "a fast blink must still close all the way")

    def test_a_fast_blink_reopens_afterwards(self):
        stabiliser = self.make()
        stabiliser.update({"EyeOpenLeft": 0.97})
        for _ in range(3):
            stabiliser.update({"EyeOpenLeft": 0.45})

        out = None
        for _ in range(3):
            out = stabiliser.update({"EyeOpenLeft": 0.97})

        self.assertGreater(out["EyeOpenLeft"], 0.5, "the eye must be open again after the blink")

    def test_a_sustained_squint_is_not_swallowed(self):
        """The reported problem: squinting closed the eyes completely."""
        stabiliser = self.make()
        stabiliser.update({"EyeOpenLeft": 0.97})

        out = None
        for _ in range(25):                      # ~0.8 s of squinting
            out = stabiliser.update({"EyeOpenLeft": 0.45})

        self.assertGreater(out["EyeOpenLeft"], 0.2,
                           "a squint must stay a partial closure, not become a closed eye")

    def test_a_deep_closure_stays_closed_no_matter_how_long(self):
        stabiliser = self.make()

        stabiliser.update({"EyeOpenLeft": 0.97})
        out = None
        for _ in range(40):                      # 1.3 s with the eyes genuinely shut
            out = stabiliser.update({"EyeOpenLeft": 0.30})

        self.assertEqual(out["EyeOpenLeft"], 0.0,
                         "eyes deliberately shut must stay shut (this is not a squint)")

    def test_a_squint_released_early_still_closes_the_next_blink(self):
        """After a squint has been released, a new blink must close again."""
        stabiliser = self.make()
        stabiliser.update({"EyeOpenLeft": 0.97})
        for _ in range(25):
            stabiliser.update({"EyeOpenLeft": 0.45})     # squint, released
        stabiliser.update({"EyeOpenLeft": 0.97})

        out = None
        for _ in range(3):
            out = stabiliser.update({"EyeOpenLeft": 0.45})

        self.assertEqual(out["EyeOpenLeft"], 0.0)


class TestBlinkSnapShut(unittest.TestCase):
    """The measured bands (240 s sampling, snap disabled so the values invert back to raw):

        blink valleys / real closure : raw 0.300-0.365
        squint level                 : raw 0.410-0.550
        open                         : raw 0.63-0.97

    The thresholds therefore sit in the gap: a value that falls below 0.40 gets snapped shut (blinks
    do), while the squint band above it never is. These tests use the shipped defaults on purpose.
    """

    def make(self, **kwargs):
        defaults = {"close_alpha": 0.8, "open_alpha": 0.45, "snap_threshold": 0.25,
                    "snap_below": 0.40, "unsnap_above": 0.7, "deep_below": 0.33,
                    "squint_frames": 6}
        defaults.update(kwargs)
        return BlinkStabiliser(**defaults)

    def drive(self, stabiliser, value, frames):
        stabiliser.update({"EyeOpenLeft": 0.97})
        out = None
        for _ in range(frames):
            out = stabiliser.update({"EyeOpenLeft": value})
        return out["EyeOpenLeft"]

    def test_a_blink_valley_in_the_measured_band_reads_as_fully_closed(self):
        out = self.drive(self.make(), 0.36, 6)

        self.assertEqual(out, 0.0, "a blink valley at raw 0.36 must read as closed")

    def test_the_measured_squint_band_is_not_swallowed(self):
        """The reported problem: raw 0.41-0.55 is where squints live, and they must stay partial."""
        out = self.drive(self.make(), 0.45, 20)

        self.assertGreater(out, 0.15, "a squint must stay a partial closure, not a closed eye")

    def test_hysteresis_prevents_flicker_at_the_threshold(self):
        stabiliser = self.make()
        self.drive(stabiliser, 0.36, 5)                    # snapped shut

        out = stabiliser.update({"EyeOpenLeft": 0.60})     # above snap_below, below unsnap_above

        self.assertEqual(out["EyeOpenLeft"], 0.0,
                         "hovering between the thresholds must not flicker")

    def test_the_eye_reopens_smoothly_once_past_the_unsnap_threshold(self):
        stabiliser = self.make()
        self.drive(stabiliser, 0.36, 5)

        out = stabiliser.update({"EyeOpenLeft": 0.95})

        self.assertGreater(out["EyeOpenLeft"], 0.0, "the eye must reopen")
        self.assertLess(out["EyeOpenLeft"], 0.95, "and reopen smoothly, not jump")

    def test_an_open_eye_is_never_touched_by_the_snap(self):
        stabiliser = self.make()
        stabiliser.update({"EyeOpenLeft": 0.97})

        out = stabiliser.update({"EyeOpenLeft": 0.97})

        self.assertAlmostEqual(out["EyeOpenLeft"], 0.97, places=6)

    def test_a_deliberate_squint_above_the_threshold_stays_visible(self):
        """Raw 0.75 is a real half-closed look and must not be swallowed by the snap."""
        stabiliser = self.make()
        for _ in range(10):
            out = stabiliser.update({"EyeOpenLeft": 0.75})

        self.assertGreater(out["EyeOpenLeft"], 0.5)


if __name__ == "__main__":
    unittest.main(verbosity=2)
