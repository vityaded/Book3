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
    edge_scan_px: int | None = None,
) -> list[dict]:
    """
    Use pixel scanning to push answer boxes away from ink under their edges.

    This repo stores boxes as normalized {x, y, w, h} values. We scan a thin
    horizontal strip centered on the box height, then:
    - snap to the last letter just left of the box,
    - move right if ink appears under the left edge,
    - move left if ink appears under the right edge,
    - shrink if needed to reduce overlap.
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

        # Build one binary mask over a wider band around the box so we can
        # query left/outside and inside-edge regions cheaply.
        band_start_x = max(0, x1 - search_width_px)
        band_end_x = min(width, x2 + search_width_px)
        if band_end_x - band_start_x < 8:
            continue

        band_strip = img[y_top:y_bottom, band_start_x:band_end_x]
        if band_strip.size == 0:
            continue

        # Threshold for "ink": darker pixels become white (255) in the mask.
        _, band_binary = cv2.threshold(band_strip, threshold, 255, cv2.THRESH_BINARY_INV)

        band_width = band_end_x - band_start_x

        def _region_rel(start_x: int, end_x: int) -> tuple[int, int]:
            rel_start = _clamp_int(start_x - band_start_x, 0, band_width)
            rel_end = _clamp_int(end_x - band_start_x, 0, band_width)
            if rel_end < rel_start:
                rel_start, rel_end = rel_end, rel_start
            return rel_start, rel_end

        def _rightmost_ink(start_x: int, end_x: int) -> int | None:
            rel_start, rel_end = _region_rel(start_x, end_x)
            if rel_end - rel_start < 2:
                return None
            region = band_binary[:, rel_start:rel_end]
            coords = cv2.findNonZero(region)
            if coords is None:
                return None
            return band_start_x + rel_start + int(np.max(coords[:, 0, 0]))

        def _leftmost_ink(start_x: int, end_x: int) -> int | None:
            rel_start, rel_end = _region_rel(start_x, end_x)
            if rel_end - rel_start < 2:
                return None
            region = band_binary[:, rel_start:rel_end]
            coords = cv2.findNonZero(region)
            if coords is None:
                return None
            return band_start_x + rel_start + int(np.min(coords[:, 0, 0]))

        def _ink_count(start_x: int, end_x: int) -> int:
            rel_start, rel_end = _region_rel(start_x, end_x)
            if rel_end <= rel_start:
                return 0
            return int(np.count_nonzero(band_binary[:, rel_start:rel_end]))

        # How much of each edge to inspect for ink under the box itself.
        local_edge_scan_px = edge_scan_px
        if local_edge_scan_px is None or local_edge_scan_px <= 0:
            local_edge_scan_px = _clamp_int(int((x2 - x1) * 0.35), 20, min(120, x2 - x1))
        local_edge_scan_px = min(local_edge_scan_px, x2 - x1)
        if local_edge_scan_px < 6:
            continue

        # 1) Snap to the last ink pixel immediately to the left of the box.
        ext_left_start = max(0, x1 - search_width_px)
        ext_left_end = x1
        ext_last_ink = _rightmost_ink(ext_left_start, ext_left_end)
        desired_x1 = x1
        if ext_last_ink is not None:
            desired_x1 = ext_last_ink + padding_px

        # 2) If ink exists under the left edge of the box, push right.
        int_left_start = x1
        int_left_end = min(x1 + local_edge_scan_px, x2)
        left_edge_ink = _rightmost_ink(int_left_start, int_left_end)
        if left_edge_ink is not None:
            desired_x1 = max(desired_x1, left_edge_ink + padding_px)

        # 3) If ink exists under the right edge of the box, pull left.
        int_right_start = max(x2 - local_edge_scan_px, x1)
        int_right_end = x2
        right_edge_ink = _leftmost_ink(int_right_start, int_right_end)
        desired_x2 = x2
        if right_edge_ink is not None:
            desired_x2 = right_edge_ink - padding_px

        # Clamp desired edges to the band and image bounds.
        desired_x1 = _clamp_int(desired_x1, 0, width - 1)
        desired_x2 = _clamp_int(desired_x2, 1, width)

        # Allow shrinking below min_width_px when there is a clear gap between
        # ink on the left and right edges.
        gap_span = desired_x2 - desired_x1
        effective_min_width = min_width_px
        if 6 < gap_span < min_width_px:
            effective_min_width = gap_span

        def _candidate_from_edges(x1_c: int, x2_c: int) -> tuple[int, int]:
            x1_c = _clamp_int(x1_c, 0, width - 1)
            x2_c = _clamp_int(x2_c, 1, width)
            if x2_c - x1_c >= effective_min_width:
                return x1_c, x2_c
            # Enforce at least the effective minimum width.
            x2_c = min(width, x1_c + effective_min_width)
            if x2_c - x1_c < effective_min_width:
                x1_c = max(0, x2_c - effective_min_width)
            return x1_c, x2_c

        # Primary candidate: honor both edge constraints.
        cand_both = _candidate_from_edges(desired_x1, desired_x2)

        # Fallbacks when constraints conflict: move only one side.
        cand_right = _candidate_from_edges(desired_x1, desired_x1 + min_width_px)
        cand_left = _candidate_from_edges(desired_x2 - min_width_px, desired_x2)

        candidates = [cand_both, cand_right, cand_left]

        # Choose the candidate with the least ink under the band; tie-break by
        # smallest movement from the original box.
        best = None
        best_key = None
        for cand_x1, cand_x2 in candidates:
            if cand_x2 <= cand_x1:
                continue
            ink = _ink_count(cand_x1, cand_x2)
            movement = abs(cand_x1 - x1) + abs(cand_x2 - x2)
            key = (ink, movement)
            if best_key is None or key < best_key:
                best_key = key
                best = (cand_x1, cand_x2)

        if best is None:
            continue

        new_x1, new_x2 = best
        if new_x1 == x1 and new_x2 == x2:
            continue
        if new_x2 - new_x1 < 4:
            continue

        _update_box_from_pixels(box, new_x1, new_x2, width)
        modified += 1

    if modified:
        logger.info("CV_TOOLS: Adjusted %s boxes via pixel scan.", modified)

    return box_list
