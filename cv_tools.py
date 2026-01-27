from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np

logger = logging.getLogger(__name__)


def _clamp_int(value: int, min_value: int, max_value: int) -> int:
    return max(min_value, min(value, max_value))


def _normalized_box_to_pixels(box: dict, width: int, height: int) -> tuple[int, int, int, int] | None:
    """Convert a normalized [x, y, w, h] box to pixel bounds."""
    try:
        x = float(box.get("x", 0.0))
        y = float(box.get("y", 0.0))
        w = float(box.get("w", 0.0))
        h = float(box.get("h", 0.0))
    except (TypeError, ValueError):
        return None

    if w <= 0 or h <= 0:
        return None

    x1 = int(x * width)
    y1 = int(y * height)
    x2 = int((x + w) * width)
    y2 = int((y + h) * height)

    x1 = _clamp_int(x1, 0, width - 1)
    y1 = _clamp_int(y1, 0, height - 1)
    x2 = _clamp_int(x2, 0, width)
    y2 = _clamp_int(y2, 0, height)

    if x2 <= x1 or y2 <= y1:
        return None

    return x1, y1, x2, y2


def _update_box_from_pixels(box: dict, x1: int, x2: int, width: int) -> None:
    """Write pixel x-bounds back into a normalized box while preserving the right edge."""
    x1 = _clamp_int(x1, 0, width - 1)
    x2 = _clamp_int(x2, 1, width)
    if x2 <= x1:
        return

    box["x"] = x1 / width
    box["w"] = (x2 - x1) / width


def _default_search_width_px(width: int) -> int:
    # ~5% of the page width, bounded to keep scanning fast and consistent.
    return _clamp_int(int(width * 0.05), 40, 140)


def fix_box_overlaps_with_vision(
    image_path: str | Path,
    boxes: Iterable[dict],
    *,
    search_width_px: int | None = None,
    padding_px: int = 6,
    min_width_px: int = 14,
    strip_half_height_px: int = 6,
    threshold: int = 200,
    scan_top_ratio: float = 0.3,
    scan_bottom_ratio: float = 0.7,
) -> list[dict]:
    """
    Use pixel scanning to push answer boxes rightward if they overlap nearby ink.

    This repo stores boxes as normalized {x, y, w, h} values. We scan a thin
    horizontal strip just to the left of each box, find the right-most ink pixel,
    and ensure the box starts after that pixel plus padding.
    """
    image_path = Path(image_path)
    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        logger.warning("CV_TOOLS: Could not load image: %s", image_path)
        return list(boxes)

    height, width = img.shape[:2]
    if width <= 2 or height <= 2:
        return list(boxes)

    if search_width_px is None or search_width_px <= 0:
        search_width_px = _default_search_width_px(width)

    padding_px = max(0, int(padding_px))
    min_width_px = max(1, int(min_width_px))
    strip_half_height_px = max(2, int(strip_half_height_px))
    scan_top_ratio = float(scan_top_ratio)
    scan_bottom_ratio = float(scan_bottom_ratio)
    if scan_bottom_ratio < scan_top_ratio:
        scan_top_ratio, scan_bottom_ratio = scan_bottom_ratio, scan_top_ratio
    scan_top_ratio = min(max(scan_top_ratio, 0.0), 1.0)
    scan_bottom_ratio = min(max(scan_bottom_ratio, 0.0), 1.0)

    # Work on a concrete list and mutate in place for simplicity.
    box_list = list(boxes)
    modified = 0

    for box in box_list:
        if not isinstance(box, dict):
            continue
        if box.get("type") not in {None, "answer"}:
            # Only adjust answer blanks; leave other box types untouched.
            continue

        bounds = _normalized_box_to_pixels(box, width, height)
        if bounds is None:
            continue
        x1, y1, x2, y2 = bounds

        if x2 - x1 < min_width_px:
            continue

        # Scan only the middle band of the target box to ignore underlines
        # (usually near the bottom) and symbols from nearby lines.
        box_h = y2 - y1
        band_top = y1 + int(box_h * scan_top_ratio)
        band_bottom = y1 + int(box_h * scan_bottom_ratio)
        band_top = _clamp_int(band_top, y1, max(y1, y2 - 1))
        band_bottom = _clamp_int(band_bottom, min(y2, y1 + 1), y2)

        if band_bottom - band_top < 4:
            mid_y = (y1 + y2) // 2
            y_top = _clamp_int(mid_y - strip_half_height_px, 0, height - 1)
            y_bottom = _clamp_int(mid_y + strip_half_height_px, 1, height)
        else:
            y_top = band_top
            y_bottom = band_bottom
        if y_bottom <= y_top:
            continue

        scan_start_x = max(0, x1 - search_width_px)
        scan_end_x = x1
        if scan_end_x - scan_start_x < 4:
            continue

        scan_strip = img[y_top:y_bottom, scan_start_x:scan_end_x]
        if scan_strip.size == 0:
            continue

        # Threshold for "ink": darker pixels become white (255) in the mask.
        _, binary = cv2.threshold(scan_strip, threshold, 255, cv2.THRESH_BINARY_INV)
        coords = cv2.findNonZero(binary)
        if coords is None:
            continue

        max_ink_x_local = int(np.max(coords[:, 0, 0]))
        global_ink_x = scan_start_x + max_ink_x_local
        # Snap the box to start just after the last detected ink pixel.
        # This allows both leftward and rightward adjustments, but keeps a
        # minimum width and never moves beyond the scan window.
        desired_x1 = global_ink_x + padding_px
        desired_x1 = max(scan_start_x, desired_x1)
        desired_x1 = min(desired_x1, x2 - min_width_px)

        if desired_x1 == x1:
            continue
        if (x2 - desired_x1) < min_width_px:
            continue

        _update_box_from_pixels(box, desired_x1, x2, width)
        modified += 1

    if modified:
        logger.info("CV_TOOLS: Adjusted %s boxes via pixel scan.", modified)

    return box_list
