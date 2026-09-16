"""Guard against the OpenCV distribution conflict this machine actually has.

Three PyPI distributions -- ``opencv-python``, ``opencv-python-headless`` and
``opencv-contrib-python`` -- all install into the *same* ``site-packages/cv2`` directory,
so only one of them can be the live build: whichever was installed last. pip installs all
three quite happily, because three unrelated packages ask for them:

    mediapipe      -> opencv-contrib-python
    albumentations -> opencv-python-headless
    requirements   -> opencv-python

The dangerous winner is ``opencv-python-headless``: it ships no GUI, so ``cv2.imshow``,
``cv2.waitKey``, ``cv2.destroyAllWindows``, ``cv2.getWindowProperty`` and
``cv2.WND_PROP_VISIBLE`` disappear. ``src/deploy/detect.py`` uses exactly those, so the
realtime preview would break the moment a headless build overwrites ``cv2/`` -- and the
traceback would point at ``cv2.imshow``, not at the install order that caused it.

Measured on this machine (2026-09-16): the live build is ``opencv-contrib-python 5.0.0.93``
(``cv2.__version__ == "5.0.0"``), while ``pip show opencv-python`` still reports 4.13.0.92
-- the metadata of three distributions describing one directory.

These tests deliberately **fail** rather than skip when the environment is broken: a guard
that reports "passed" by skipping proves nothing.
"""
import os
import sys
import unittest
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# The GUI symbols the project itself calls -- the list is derived from the code, not from
# what OpenCV happens to expose, so it stays honest if a module starts using more.
GUI_SYMBOLS_USED_BY_PROJECT = (
    "imshow",
    "waitKey",
    "destroyAllWindows",
    "getWindowProperty",
    "WND_PROP_VISIBLE",
)

FIX_HINT = (
    "当前生效的 cv2 缺少 GUI 符号：site-packages/cv2 几乎可以确定已被 "
    "opencv-python-headless 覆盖（三个分发包共用同一个 cv2/ 目录，最后安装者生效）。\n"
    "修复（必须在**你自己的**终端里跑 pip；受限 shell 里 pip 会退化成 user 安装，反而多装一份）：\n"
    "    python -m pip install --force-reinstall --no-deps opencv-contrib-python\n"
    "然后 python -m unittest discover -s tests -t . 必须回到 OK。"
)


def find_opencv_gui_problems(cv2_module):
    """Return the GUI symbol names the project calls that this cv2 build does not expose."""
    return [name for name in GUI_SYMBOLS_USED_BY_PROJECT if not hasattr(cv2_module, name)]


class TestOpenCVGuard(unittest.TestCase):
    def test_a_build_without_gui_reports_every_missing_symbol(self):
        headless_like = SimpleNamespace(__version__="4.13.0.92")

        missing = find_opencv_gui_problems(headless_like)

        self.assertEqual(list(missing), list(GUI_SYMBOLS_USED_BY_PROJECT),
                         "a headless build must report every GUI symbol the project needs")

    def test_a_partially_broken_build_reports_only_what_is_actually_missing(self):
        partly_gui = SimpleNamespace(
            __version__="4.13.0.92",
            imshow=lambda *a, **k: None,
            waitKey=lambda *a, **k: 0,
            destroyAllWindows=lambda: None,
            getWindowProperty=lambda *a, **k: 0,
        )

        self.assertEqual(find_opencv_gui_problems(partly_gui), ["WND_PROP_VISIBLE"])

    def test_the_live_cv2_build_has_the_gui_symbols_the_project_calls(self):
        import cv2

        missing = find_opencv_gui_problems(cv2)

        self.assertEqual(
            missing, [],
            f"{FIX_HINT}\nmissing={missing}\ncv2.__version__={cv2.__version__}\ncv2.__file__={cv2.__file__}",
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
