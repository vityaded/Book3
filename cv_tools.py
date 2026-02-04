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
    x2 = _clamp_int(x2, 1, width)
    y2 = _clamp_int(y2, 1, height)

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


def _update_box_y_from_pixels(box: dict, y1: int, y2: int, height: int) -> None:
    """Write pixel y-bounds back into a normalized box."""
    y1 = _clamp_int(y1, 0, height - 1)
    y2 = _clamp_int(y2, 1, height)
    if y2 <= y1:
        return

    box["y"] = y1 / height
    box["h"] = (y2 - y1) / height


def _binarize_region(
    gray: np.ndarray,
    *,
    threshold: int,
    use_adaptive: bool = True,
) -> np.ndarray | None:
    if gray.size == 0:
        return None

    if use_adaptive and min(gray.shape[:2]) >= 8:
        block = max(15, min(gray.shape[:2]) // 2 * 2 + 1)
        block = _clamp_int(block, 15, 51)
        binary = cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            block,
            9,
        )
    else:
        _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY_INV)

    if np.count_nonzero(binary) == 0:
        _, binary_otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        if np.count_nonzero(binary_otsu) > 0:
            binary = binary_otsu

    return binary


def _middle_band(y1: int, y2: int, ratio: float) -> tuple[int, int]:
    box_h = y2 - y1
    band_h = max(3, int(box_h * ratio))
    mid = y1 + box_h // 2
    band_y1 = _clamp_int(mid - band_h // 2, y1, max(y1, y2 - 1))
    band_y2 = _clamp_int(band_y1 + band_h, band_y1 + 1, y2)
    return band_y1, band_y2


def _edge_avoid(
    binary_band: np.ndarray,
    *,
    band_x1: int,
    x1: int,
    x2: int,
    edge_scan_px: int,
    padding_px: int,
) -> tuple[int, int, bool, int | None, int | None]:
    width = binary_band.shape[1]

    def _region_rel(start_x: int, end_x: int) -> tuple[int, int]:
        rel_start = _clamp_int(start_x - band_x1, 0, width)
        rel_end = _clamp_int(end_x - band_x1, 0, width)
        if rel_end < rel_start:
            rel_start, rel_end = rel_end, rel_start
        return rel_start, rel_end

    def _rightmost_ink(start_x: int, end_x: int) -> int | None:
        rel_start, rel_end = _region_rel(start_x, end_x)
        if rel_end - rel_start < 2:
            return None
        region = binary_band[:, rel_start:rel_end]
        coords = cv2.findNonZero(region)
        if coords is None:
            return None
        return band_x1 + rel_start + int(np.max(coords[:, 0, 0]))

    def _leftmost_ink(start_x: int, end_x: int) -> int | None:
        rel_start, rel_end = _region_rel(start_x, end_x)
        if rel_end - rel_start < 2:
            return None
        region = binary_band[:, rel_start:rel_end]
        coords = cv2.findNonZero(region)
        if coords is None:
            return None
        return band_x1 + rel_start + int(np.min(coords[:, 0, 0]))

    left_edge_ink = _rightmost_ink(x1, min(x1 + edge_scan_px, x2))
    right_edge_ink = _leftmost_ink(max(x2 - edge_scan_px, x1), x2)

    overlap = left_edge_ink is not None or right_edge_ink is not None

    if not overlap:
        return x1, x2, False, left_edge_ink, right_edge_ink

    new_x1 = x1
    new_x2 = x2
    if left_edge_ink is not None:
        new_x1 = max(new_x1, left_edge_ink + padding_px)
    if right_edge_ink is not None:
        new_x2 = min(new_x2, right_edge_ink - padding_px)

    return new_x1, new_x2, True, left_edge_ink, right_edge_ink


def _collect_components(
    binary: np.ndarray,
    *,
    min_h: int,
    max_h: int,
    min_area: int,
    max_aspect: float,
) -> list[tuple[int, int, int, int]]:
    num, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    comps: list[tuple[int, int, int, int]] = []
    for idx in range(1, num):
        x, y, w, h, area = stats[idx]
        if h < min_h or h > max_h:
            continue
        if area < min_area:
            continue
        if w <= 1:
            continue
        aspect = w / max(h, 1)
        if aspect > max_aspect:
            continue
        comps.append((x, y, w, h))
    return comps


def _estimate_cap_height(
    img: np.ndarray,
    *,
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    search_width_px: int,
    threshold: int,
    scan_up_ratio: float,
    scan_down_ratio: float,
    row_ratio_min: float,
    row_ratio_scale: float,
    min_px: int,
    max_scale: float,
    max_shift_ratio: float,
) -> tuple[int, int] | None:
    height, width = img.shape[:2]
    box_h = y2 - y1
    box_w = x2 - x1
    if box_h <= 0 or box_w <= 0:
        return None

    scan_up = max(min_px, int(box_h * scan_up_ratio))
    scan_down = max(2, int(box_h * scan_down_ratio))
    window_y1 = _clamp_int(y1 - scan_up, 0, height - 1)
    window_y2 = _clamp_int(y2 + scan_down, 1, height)
    if window_y2 - window_y1 < min_px:
        return None

    region_w = min(search_width_px, max(18, int(box_w * 0.9)))
    if region_w < 6:
        return None

    regions = []
    left_x1 = max(0, x1 - region_w)
    left_x2 = x1
    if left_x2 - left_x1 >= 6:
        regions.append((left_x1, left_x2))
    right_x1 = x2
    right_x2 = min(width, x2 + region_w)
    if right_x2 - right_x1 >= 6:
        regions.append((right_x1, right_x2))

    if not regions:
        return None

    baseline_target = y2
    best_candidate = None

    for rx1, rx2 in regions:
        strip = img[window_y1:window_y2, rx1:rx2]
        binary = _binarize_region(strip, threshold=threshold, use_adaptive=True)
        if binary is None:
            continue

        # Light morphological cleanup to reduce pepper noise.
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)

        max_h = max(min_px, int(box_h * max_scale))
        min_area = max(6, int((rx2 - rx1) * 0.01))
        comps = _collect_components(
            binary,
            min_h=min_px,
            max_h=max_h,
            min_area=min_area,
            max_aspect=6.0,
        )

        if comps:
            bottoms = np.array([y + h for _, y, _, h in comps], dtype=np.float32)
            heights = np.array([h for _, _, _, h in comps], dtype=np.float32)
            global_bottoms = bottoms + window_y1

            distances = np.abs(global_bottoms - baseline_target)
            tol = max(8.0, box_h * 0.8)
            near_mask = distances <= tol
            if np.any(near_mask):
                selected_bottoms = global_bottoms[near_mask]
                selected_heights = heights[near_mask]
            else:
                idx_sorted = np.argsort(distances)[: min(12, len(distances))]
                selected_bottoms = global_bottoms[idx_sorted]
                selected_heights = heights[idx_sorted]

            baseline = float(np.median(selected_bottoms))
            height_median = float(np.median(selected_heights))
            if height_median < min_px:
                continue

            height_median = min(height_median, max_h)
            candidate = (baseline, height_median, len(selected_heights))
            if best_candidate is None or candidate[2] > best_candidate[2]:
                best_candidate = candidate

        # Fallback projection if components are sparse.
        if best_candidate is None:
            row_counts = np.count_nonzero(binary, axis=1).astype(np.float32)
            if row_counts.size == 0:
                continue
            if row_counts.max() <= 0:
                continue
            smoothed = np.convolve(row_counts, np.ones(5) / 5.0, mode="same")
            max_val = float(smoothed.max())
            if max_val <= 0:
                continue
            lower_half_start = int(smoothed.size * 0.45)
            baseline_idx = int(lower_half_start + np.argmax(smoothed[lower_half_start:]))
            threshold_ratio = max(row_ratio_min, max_val / max(rx2 - rx1, 1) * row_ratio_scale)
            threshold_value = threshold_ratio * max(rx2 - rx1, 1)

            top_idx = baseline_idx
            for idx in range(baseline_idx, -1, -1):
                if smoothed[idx] < threshold_value:
                    break
                top_idx = idx

            height_px = baseline_idx - top_idx + 1
            if height_px >= min_px:
                baseline = window_y1 + baseline_idx
                height_px = min(height_px, int(box_h * max_scale))
                best_candidate = (float(baseline), float(height_px), 1)

    if best_candidate is None:
        return None

    baseline, height_px, _ = best_candidate
    max_shift = max(6, int(box_h * max_shift_ratio))
    if abs(baseline - y2) > max_shift:
        return None

    new_y2 = int(round(baseline))
    new_y1 = int(round(baseline - height_px))
    new_y1 = _clamp_int(new_y1, 0, height - 1)
    new_y2 = _clamp_int(new_y2, new_y1 + 1, height)
    return new_y1, new_y2


def fix_box_overlaps_with_vision(
    image_path: str | Path,
    boxes: Iterable[dict],
    *,
    search_width_px: int | None = None,
    padding_px: int = 6,
    min_width_px: int = 14,
    min_width_ratio: float = 0.6,
    edge_scan_px: int | None = None,
    threshold: int = 200,
    middle_band_ratio: float = 0.1,
    cap_height_enabled: bool = True,
    cap_height_scan_up_ratio: float = 2.2,
    cap_height_scan_down_ratio: float = 0.6,
    cap_height_row_ratio_min: float = 0.03,
    cap_height_row_ratio_scale: float = 0.45,
    cap_height_min_px: int = 6,
    cap_height_max_scale: float = 2.2,
    cap_height_max_shift_ratio: float = 1.2,
    debug: bool = False,
    debug_pass_name: str | None = None,
) -> list[dict]:
    """
    Simplified CV alignment:
    - only avoid ink at box edges (using the middle 10% of the box height),
    - optionally set the box height to the detected cap height on the line.
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
        search_width_px = _clamp_int(int(width * 0.05), 40, 140)

    padding_px = max(0, int(padding_px))
    min_width_px = max(1, int(min_width_px))
    min_width_ratio = max(0.0, float(min_width_ratio))
    threshold = max(0, min(255, int(threshold)))
    middle_band_ratio = max(0.02, min(0.2, float(middle_band_ratio)))

    box_list = list(boxes)
    modified = 0

    for box in box_list:
        if not isinstance(box, dict):
            continue
        box_type = str(box.get("type") or "").strip().lower()
        if box_type and box_type not in {"answer", "underline", "box", "cloze_gap"}:
            continue

        bounds = _normalized_box_to_pixels(box, width, height)
        if bounds is None:
            continue
        x1, y1, x2, y2 = bounds
        orig_w = x2 - x1
        orig_h = y2 - y1
        if orig_w <= 0 or orig_h <= 0:
            continue

        steps_out = box.setdefault("cv_debug_steps", []) if debug else None
        pass_name = (debug_pass_name or "").strip()

        def _step_label(stage: str) -> str:
            return f"{pass_name}.{stage}" if pass_name else stage

        def _record(stage: str, x1_r: int, x2_r: int, y1_r: int, y2_r: int, note: str | None = None) -> None:
            if not debug or steps_out is None:
                return
            entry = {
                "step": _step_label(stage),
                "stage": stage,
                "pass": pass_name,
                "x": x1_r / width,
                "y": y1_r / height,
                "w": (x2_r - x1_r) / width,
                "h": (y2_r - y1_r) / height,
                "x1_px": int(x1_r),
                "x2_px": int(x2_r),
                "y1_px": int(y1_r),
                "y2_px": int(y2_r),
            }
            if note:
                entry["note"] = note
            steps_out.append(entry)

        _record("initial", x1, x2, y1, y2)

        # Edge avoidance using the middle band only.
        band_y1, band_y2 = _middle_band(y1, y2, middle_band_ratio)
        band_x1 = max(0, x1 - search_width_px)
        band_x2 = min(width, x2 + search_width_px)
        band_strip = img[band_y1:band_y2, band_x1:band_x2]

        if band_strip.size > 0:
            binary_band = _binarize_region(band_strip, threshold=threshold, use_adaptive=True)
        else:
            binary_band = None

        cur_x1, cur_x2 = x1, x2
        cur_y1, cur_y2 = y1, y2

        if binary_band is not None:
            local_edge_scan = edge_scan_px
            if local_edge_scan is None or local_edge_scan <= 0:
                local_edge_scan = _clamp_int(int(orig_w * 0.05), 5, min(80, orig_w))
            local_edge_scan = max(4, min(local_edge_scan, orig_w))

            new_x1, new_x2, moved, left_ink, right_ink = _edge_avoid(
                binary_band,
                band_x1=band_x1,
                x1=cur_x1,
                x2=cur_x2,
                edge_scan_px=local_edge_scan,
                padding_px=padding_px,
            )

            effective_min = max(min_width_px, 3)
            if min_width_ratio > 0 and orig_w >= max(min_width_px * 3, 36):
                effective_min = max(effective_min, int(orig_w * min_width_ratio))
            effective_min = min(effective_min, orig_w)

            if moved:
                new_x1 = _clamp_int(new_x1, 0, width - 1)
                new_x2 = _clamp_int(new_x2, 1, width)
                if new_x2 - new_x1 < effective_min:
                    new_x2 = min(width, new_x1 + effective_min)
                    if new_x2 - new_x1 < effective_min:
                        new_x1 = max(0, new_x2 - effective_min)

                cur_x1, cur_x2 = new_x1, new_x2
                _record(
                    "edge_avoid",
                    cur_x1,
                    cur_x2,
                    cur_y1,
                    cur_y2,
                    note=f"left_ink={left_ink} right_ink={right_ink}",
                )
            else:
                _record("edge_avoid", cur_x1, cur_x2, cur_y1, cur_y2, note="no_overlap")
        else:
            _record("edge_avoid", cur_x1, cur_x2, cur_y1, cur_y2, note="no_band")

        if cap_height_enabled:
            cap = _estimate_cap_height(
                img,
                x1=cur_x1,
                y1=cur_y1,
                x2=cur_x2,
                y2=cur_y2,
                search_width_px=search_width_px,
                threshold=threshold,
                scan_up_ratio=cap_height_scan_up_ratio,
                scan_down_ratio=cap_height_scan_down_ratio,
                row_ratio_min=cap_height_row_ratio_min,
                row_ratio_scale=cap_height_row_ratio_scale,
                min_px=cap_height_min_px,
                max_scale=cap_height_max_scale,
                max_shift_ratio=cap_height_max_shift_ratio,
            )
            if cap is not None:
                cur_y1, cur_y2 = cap
                _record("cap_height", cur_x1, cur_x2, cur_y1, cur_y2, note="aligned")
            else:
                _record("cap_height", cur_x1, cur_x2, cur_y1, cur_y2, note="no_match")
        else:
            _record("cap_height", cur_x1, cur_x2, cur_y1, cur_y2, note="disabled")

        if cur_x1 == x1 and cur_x2 == x2 and cur_y1 == y1 and cur_y2 == y2:
            _record("final", cur_x1, cur_x2, cur_y1, cur_y2, note="unchanged")
            continue

        _update_box_from_pixels(box, cur_x1, cur_x2, width)
        _update_box_y_from_pixels(box, cur_y1, cur_y2, height)
        _record("final", cur_x1, cur_x2, cur_y1, cur_y2)
        modified += 1

    if modified:
        logger.info("CV_TOOLS: Adjusted %s boxes via pixel scan.", modified)

    return box_list
