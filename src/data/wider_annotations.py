"""Single source of truth for parsing WIDER Face annotation files.

The format is a flat text stream, not a table:

    <image relative path>
    <number of faces>
    <x> <y> <w> <h> <blur> <expression> <illumination> <occlusion> <pose>   (repeated N times)

Two quirks make naive parsing fail:

1. **Zero-face images still emit a dummy all-zero box line.** With ``num_faces == 0``
   the official files still contain one line of ``0 0 0 0 0 0 0 0 0 0``. A parser that
   does not consume it loses alignment and then reads the *next image path* as a face
   count, which raises ``ValueError: invalid literal for int()``.
2. **Trailing blank lines and CRLF endings** appear in some distributions.

This module previously existed as four slightly different copies
(``convert.py``, ``qc.py``, ``split.py``, ``loader.py``); only ``convert.py`` had the
zero-face fix, so ``split.py`` crashed on the real annotation file. Keeping one
implementation removes that class of bug entirely.
"""
import os

# Tokens of the dummy all-zero line emitted for zero-face entries.
_DUMMY_TOKENS = 10


def parse_lines(lines):
    """Yield ``(image_name, boxes)`` from an iterable of raw annotation lines.

    Kept separate from file access so the parsing rules -- above all the zero-face dummy
    line -- can be exercised without touching the filesystem.
    """
    lines = [line.strip() for line in lines]

    i = 0
    total = len(lines)
    while i < total:
        image_name = lines[i]
        i += 1
        if not image_name:
            continue
        if i >= total:
            break

        # Blank lines are tolerated anywhere between records, including between an
        # image path and its face count -- the module docstring promises this.
        while i < total and not lines[i]:
            i += 1
        if i >= total:
            break

        try:
            num_faces = int(lines[i])
        except ValueError as exc:  # pragma: no cover - defensive
            raise ValueError(
                f"Malformed WIDER annotation near line {i + 1}: expected a face count, "
                f"got {lines[i]!r}"
            ) from exc
        i += 1

        boxes = []
        for _ in range(num_faces):
            while i < total and not lines[i]:
                i += 1
            if i >= total:
                break
            parts = lines[i].split()
            if len(parts) < 4:
                break
            boxes.append([int(v) for v in parts[:4]])
            i += 1

        # Quirk 1: consume the dummy all-zero line that follows a zero-face entry.
        if num_faces == 0 and i < total:
            parts = lines[i].split()
            if len(parts) == _DUMMY_TOKENS and all(p.isdigit() and int(p) == 0 for p in parts):
                i += 1

        yield image_name, boxes


def parse_samples(anno_file):
    """Yield ``(image_name, [[x, y, w, h], ...])`` for every entry in ``anno_file``.

    ``image_name`` is the raw relative path as written in the annotation file.
    No filesystem checks on the images are performed here -- callers add those as needed.
    """
    if not os.path.exists(anno_file):
        raise FileNotFoundError(f"WIDER annotation file not found: {anno_file}")
    # Read fully before delegating: parse_lines is a generator, so returning it from
    # inside the `with` would hand back a generator whose file is already closed.
    with open(anno_file, "r", encoding="utf-8", errors="replace") as f:
        lines = f.readlines()
    return parse_lines(lines)


def parse_samples_with_image_root(anno_file, image_root):
    """Yield ``(absolute_image_path, boxes)`` for entries whose image exists on disk."""
    for image_name, boxes in parse_samples(anno_file):
        img_path = os.path.normpath(os.path.join(image_root, image_name))
        if os.path.exists(img_path):
            yield img_path, boxes
