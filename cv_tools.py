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


def _update_box_y_from_pixels(box: dict, y1: int, y2: int, height: int) -> None:
    """Write pixel y-bounds back into a normalized box while preserving height."""
    y1 = _clamp_int(y1, 0, height - 1)
    y2 = _clamp_int(y2, 1, height)
    if y2 <= y1:
        return

    box["y"] = y1 / height
    box["h"] = (y2 - y1) / height


def _segments_from_row(row_mask: np.ndarray) -> list[tuple[int, int]]:
    """Return contiguous ink segments as (start, end) inclusive indices."""
    segments: list[tuple[int, int]] = []
    in_segment = False
    seg_start = 0
    for idx, value in enumerate(row_mask):
        if value and not in_segment:
            in_segment = True
            seg_start = idx
        elif not value and in_segment:
            segments.append((seg_start, idx - 1))
            in_segment = False
    if in_segment:
        segments.append((seg_start, len(row_mask) - 1))
    return segments


def _analyze_row_pattern(
    row_mask: np.ndarray,
    *,
    coverage_min: float,
    segments_min: int,
    gap_cv_max: float,
) -> dict | None:
    """Analyze a single row for underline-like patterns."""
    if row_mask.size == 0:
        return None
    ink_idx = np.flatnonzero(row_mask)
    if ink_idx.size == 0:
        return None

    width = int(row_mask.size)
    coverage_ratio = float(ink_idx.size / max(width, 1))
    segments = _segments_from_row(row_mask.astype(bool))
    segments_count = len(segments)
    span_start = int(ink_idx.min())
    span_end = int(ink_idx.max())
    center_x = int(np.median(ink_idx))

    gaps: list[int] = []
    for i in range(len(segments) - 1):
        gaps.append(max(0, segments[i + 1][0] - segments[i][1] - 1))

    gap_cv = float("inf")
    if gaps:
        mean_gap = float(np.mean(gaps))
        if mean_gap > 0:
            gap_cv = float(np.std(gaps) / mean_gap)
        else:
            gap_cv = 0.0

    dotted_like = segments_count >= segments_min and (gap_cv <= gap_cv_max or coverage_ratio >= coverage_min * 0.7)
    solid_like = coverage_ratio >= coverage_min
    real_line = bool(solid_like or dotted_like)

    # Score: prefer broader coverage and consistent dotted patterns.
    segment_term = min(segments_count, 8) / 8.0
    if gaps and np.isfinite(gap_cv):
        gap_term = max(0.0, 1.0 - min(gap_cv / max(gap_cv_max, 1e-6), 1.5))
    else:
        gap_term = 0.0
    score = coverage_ratio * 1.6 + segment_term * 0.25 + gap_term * 0.25
    if real_line:
        score += 0.35

    return {
        "coverage_ratio": coverage_ratio,
        "segments_count": segments_count,
        "gap_cv": gap_cv,
        "span_start": span_start,
        "span_end": span_end,
        "center_x": center_x,
        "real_line": real_line,
        "score": float(score),
    }


def _detect_underline_in_region(
    img: np.ndarray,
    *,
    x1: int,
    x2: int,
    y1: int,
    y2: int,
    threshold: int,
    coverage_min: float,
    segments_min: int,
    gap_cv_max: float,
    use_otsu: bool = True,
    blur_ksize: int = 3,
) -> dict | None:
    """Detect the best underline-like row within a region."""
    if x2 - x1 < 6 or y2 - y1 < 4:
        return None

    strip = img[y1:y2, x1:x2]
    if strip.size == 0:
        return None

    def analyze_binary(binary: np.ndarray) -> tuple[dict | None, dict | None]:
        best_line: dict | None = None
        best_fragment: dict | None = None
        for row_idx in range(binary.shape[0]):
            row = binary[row_idx, :] > 0
            analysis = _analyze_row_pattern(
                row,
                coverage_min=coverage_min,
                segments_min=segments_min,
                gap_cv_max=gap_cv_max,
            )
            if analysis is None:
                continue
            analysis["row_idx"] = int(row_idx)
            if analysis["real_line"]:
                if best_line is None or analysis["score"] > best_line["score"]:
                    best_line = analysis
            else:
                if best_fragment is None or analysis["score"] > best_fragment["score"]:
                    best_fragment = analysis
        return best_line, best_fragment

    _, binary = cv2.threshold(strip, threshold, 255, cv2.THRESH_BINARY_INV)
    best_line, best_fragment = analyze_binary(binary)

    if best_line is None and use_otsu:
        ksize = int(blur_ksize)
        if ksize < 3:
            ksize = 3
        if ksize % 2 == 0:
            ksize += 1
        blurred = cv2.GaussianBlur(strip, (ksize, ksize), 0)
        _, binary_otsu = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        best_line, best_fragment = analyze_binary(binary_otsu)

    if best_line is not None:
        row_idx = best_line["row_idx"]
        underline_y = y1 + row_idx
        span = (x1 + int(best_line["span_start"]), x1 + int(best_line["span_end"]))
        return {
            "real_line": True,
            "underline_y": int(underline_y),
            "span": span,
            "analysis": best_line,
        }

    if best_fragment is not None:
        row_idx = best_fragment["row_idx"]
        fragment_y = y1 + row_idx
        span = (x1 + int(best_fragment["span_start"]), x1 + int(best_fragment["span_end"]))
        center_x = x1 + int(best_fragment["center_x"])
        return {
            "real_line": False,
            "underline_y": int(fragment_y),
            "span": span,
            "center_x": int(center_x),
            "analysis": best_fragment,
        }

    return None


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
    underline_enabled: bool = True,
    underline_band_top_ratio: float = 0.62,
    underline_extra_bottom_px: int = 10,
    underline_margin_x_px: int | None = None,
    underline_row_ratio_min: float = 0.02,
    underline_min_row_pixels: int = 12,
    underline_line_coverage_min: float = 0.16,
    underline_segments_min: int = 3,
    underline_gap_cv_max: float = 1.4,
    underline_below_scan_px: int = 28,
    underline_below_margin_top_px: int = 2,
    underline_below_ratio: float = 0.5,
    underline_use_otsu: bool = True,
    underline_blur_ksize: int = 3,
    underline_threshold: int = 140,
    debug: bool = False,
    debug_pass_name: str | None = None,
) -> list[dict]:
    """
    Use pixel scanning to push answer boxes away from ink under their edges.

    This repo stores boxes as normalized {x, y, w, h} values. We scan a thin
    horizontal strip centered on the box height, then:
    - move left/right if letters are under the box edges,
    - if an underline/dotted line is behind the box, lift the box so its
      bottom aligns with that underline,
    - snap the left edge to the nearest letters on the left,
    - extend the right edge to the underline length when available.
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
    # Keep a very small hard minimum width to avoid collapsing the box.
    min_width_floor_px = 3
    scan_top_ratio = float(scan_top_ratio)
    scan_bottom_ratio = float(scan_bottom_ratio)
    if scan_bottom_ratio < scan_top_ratio:
        scan_top_ratio, scan_bottom_ratio = scan_bottom_ratio, scan_top_ratio
    scan_top_ratio = min(max(scan_top_ratio, 0.0), 1.0)
    scan_bottom_ratio = min(max(scan_bottom_ratio, 0.0), 1.0)
    underline_band_top_ratio = min(max(float(underline_band_top_ratio), 0.0), 1.0)
    underline_extra_bottom_px = max(0, int(underline_extra_bottom_px))
    underline_row_ratio_min = max(0.0, float(underline_row_ratio_min))
    underline_min_row_pixels = max(1, int(underline_min_row_pixels))
    underline_line_coverage_min = max(0.02, float(underline_line_coverage_min))
    underline_segments_min = max(2, int(underline_segments_min))
    underline_gap_cv_max = max(0.2, float(underline_gap_cv_max))
    underline_below_scan_px = max(0, int(underline_below_scan_px))
    underline_below_margin_top_px = max(0, int(underline_below_margin_top_px))
    underline_below_ratio = max(0.0, float(underline_below_ratio))
    underline_blur_ksize = max(1, int(underline_blur_ksize))
    underline_threshold = max(0, min(255, int(underline_threshold)))

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
        orig_box_h = y2 - y1
        orig_box_w = x2 - x1

        if x2 - x1 < min_width_floor_px:
            continue

        steps_out = box.setdefault("cv_debug_steps", []) if debug else None
        pass_name = (debug_pass_name or "").strip()

        def _step_label(stage: str) -> str:
            return f"{pass_name}.{stage}" if pass_name else stage

        def _norm(x1_n: int, x2_n: int, y1_n: int, y2_n: int) -> tuple[float, float, float, float]:
            return (
                x1_n / width,
                y1_n / height,
                (x2_n - x1_n) / width,
                (y2_n - y1_n) / height,
            )

        def _record(stage: str, x1_r: int, x2_r: int, y1_r: int, y2_r: int, note: str | None = None) -> None:
            if not debug or steps_out is None:
                return
            x_n, y_n, w_n, h_n = _norm(x1_r, x2_r, y1_r, y2_r)
            entry = {
                "step": _step_label(stage),
                "stage": stage,
                "pass": pass_name,
                "x": x_n,
                "y": y_n,
                "w": w_n,
                "h": h_n,
                "x1_px": int(x1_r),
                "x2_px": int(x2_r),
                "y1_px": int(y1_r),
                "y2_px": int(y2_r),
            }
            if note:
                entry["note"] = note
            steps_out.append(entry)
            logger.info(
                "CV_DEBUG page_box=%s step=%s x=%.4f y=%.4f w=%.4f h=%.4f note=%s",
                box.get("id", "?"),
                entry["step"],
                entry["x"],
                entry["y"],
                entry["w"],
                entry["h"],
                entry.get("note", ""),
            )

        _record("initial", x1, x2, y1, y2, note=f"w_px={orig_box_w} h_px={orig_box_h}")

        def _middle_band(y1_b: int, y2_b: int) -> tuple[int, int] | None:
            box_h_b = y2_b - y1_b
            band_top_b = y1_b + int(box_h_b * scan_top_ratio)
            band_bottom_b = y1_b + int(box_h_b * scan_bottom_ratio)
            band_top_b = _clamp_int(band_top_b, y1_b, max(y1_b, y2_b - 1))
            band_bottom_b = _clamp_int(band_bottom_b, min(y2_b, y1_b + 1), y2_b)

            if band_bottom_b - band_top_b < 4:
                mid_y_b = (y1_b + y2_b) // 2
                y_top_b = _clamp_int(mid_y_b - strip_half_height_px, 0, height - 1)
                y_bottom_b = _clamp_int(mid_y_b + strip_half_height_px, 1, height)
            else:
                y_top_b = band_top_b
                y_bottom_b = band_bottom_b
            if y_bottom_b <= y_top_b:
                return None
            return y_top_b, y_bottom_b

        band = _middle_band(y1, y2)
        if band is None:
            continue
        y_top, y_bottom = band

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
            local_edge_scan_px = _clamp_int(int((x2 - x1) * 0.05), 5, min(120, x2 - x1))
        local_edge_scan_px = min(local_edge_scan_px, x2 - x1)
        if local_edge_scan_px < 5:
            continue

        # 1) If ink exists under the left edge of the box, push right.
        int_left_start = x1
        int_left_end = min(x1 + local_edge_scan_px, x2)
        left_edge_ink = _rightmost_ink(int_left_start, int_left_end)
        desired_x1 = x1
        if left_edge_ink is not None:
            desired_x1 = max(desired_x1, left_edge_ink + padding_px)

        # 2) If ink exists under the right edge of the box, pull left.
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
        effective_min_width = min_width_floor_px
        if 0 < gap_span < effective_min_width:
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
        cand_right = _candidate_from_edges(desired_x1, desired_x1 + min_width_floor_px)
        cand_left = _candidate_from_edges(desired_x2 - min_width_floor_px, desired_x2)

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
        cur_x1, cur_x2 = new_x1, new_x2
        cur_y1, cur_y2 = y1, y2

        edge_note = (
            f"left_edge_ink={left_edge_ink} right_edge_ink={right_edge_ink} "
            f"cand_x1={cur_x1} cand_x2={cur_x2}"
        )
        _record("edge_avoid", cur_x1, cur_x2, cur_y1, cur_y2, note=edge_note)

        underline_span: tuple[int, int] | None = None
        underline_y: int | None = None
        underline_source: str | None = None

        # 4) Snap the left edge to the nearest letters on the left using the
        # current vertical placement (vertical alignment is disabled for now).
        snap_band = _middle_band(cur_y1, cur_y2)
        if snap_band is not None:
            snap_y_top, snap_y_bottom = snap_band
            snap_start_x = max(0, cur_x1 - search_width_px)
            snap_end_x = cur_x1
            if snap_end_x - snap_start_x >= 4:
                snap_strip = img[snap_y_top:snap_y_bottom, snap_start_x:snap_end_x]
                if snap_strip.size > 0:
                    _, snap_binary = cv2.threshold(snap_strip, threshold, 255, cv2.THRESH_BINARY_INV)
                    snap_coords = cv2.findNonZero(snap_binary)
                    if snap_coords is not None:
                        last_ink_local_x = int(np.max(snap_coords[:, 0, 0]))
                        last_ink_x = snap_start_x + last_ink_local_x
                        snap_x1 = last_ink_x + padding_px
                        # Keep enough width.
                        max_snap_x1 = max(0, cur_x2 - min_width_floor_px)
                        snap_x1 = min(snap_x1, max_snap_x1)
                        snap_x1 = _clamp_int(snap_x1, 0, width - 1)
                        if cur_x2 - snap_x1 >= min_width_floor_px:
                            cur_x1 = snap_x1
        _record("snap_left", cur_x1, cur_x2, cur_y1, cur_y2)

        # Underline extension is suspended for now.

        if underline_enabled:
            # After all horizontal alignment, look for underline/dotted line.
            margin_x = underline_margin_x_px
            if margin_x is None or margin_x <= 0:
                margin_x = max(12, min(search_width_px, 220))
            margin_x = max(0, int(margin_x))

            def _min_row_pixels(x1_r: int, x2_r: int) -> int:
                return max(underline_min_row_pixels, int((x2_r - x1_r) * underline_row_ratio_min))

            # Near-bottom scan first (relative to current box).
            near_x1 = max(0, cur_x1 - margin_x)
            near_x2 = min(width, cur_x2 + margin_x)
            near_y1 = cur_y1 + int(orig_box_h * underline_band_top_ratio)
            near_y2 = min(height, cur_y2 + underline_extra_bottom_px)
            near_y1 = _clamp_int(near_y1, cur_y1, max(cur_y1, near_y2 - 1))
            near_y2 = _clamp_int(near_y2, min(height, near_y1 + 1), height)

            near_result = _detect_underline_in_region(
                img,
                x1=near_x1,
                x2=near_x2,
                y1=near_y1,
                y2=near_y2,
                threshold=underline_threshold,
                coverage_min=underline_line_coverage_min,
                segments_min=underline_segments_min,
                gap_cv_max=underline_gap_cv_max,
                use_otsu=underline_use_otsu,
                blur_ksize=underline_blur_ksize,
            )

            if near_result and near_result.get("real_line"):
                # Enforce a minimum density to avoid letter fragments.
                analysis = near_result.get("analysis") or {}
                row_pixels_ok = analysis.get("coverage_ratio", 0.0) >= (
                    _min_row_pixels(near_x1, near_x2) / max(near_x2 - near_x1, 1)
                )
                if row_pixels_ok:
                    underline_y = int(near_result["underline_y"])
                    underline_span = near_result.get("span")
                    underline_source = "near"

            # Underline fragment adjustment is suspended for now.

            # If no underline yet, scan below the box for a real line.
            below_scan_px = underline_below_scan_px
            if underline_below_ratio > 0:
                below_scan_px = max(below_scan_px, int(orig_box_h * underline_below_ratio))

            if underline_y is None and below_scan_px > 0:
                below_x1 = max(0, cur_x1 - margin_x)
                below_x2 = min(width, cur_x2 + margin_x)
                below_y1 = min(height - 1, cur_y2 + underline_below_margin_top_px)
                below_y2 = min(height, below_y1 + below_scan_px)
                below_y1 = _clamp_int(below_y1, 0, max(0, below_y2 - 1))
                below_y2 = _clamp_int(below_y2, min(height, below_y1 + 1), height)

                below_result = _detect_underline_in_region(
                    img,
                    x1=below_x1,
                    x2=below_x2,
                    y1=below_y1,
                    y2=below_y2,
                    threshold=underline_threshold,
                    coverage_min=underline_line_coverage_min,
                    segments_min=underline_segments_min,
                    gap_cv_max=underline_gap_cv_max,
                    use_otsu=underline_use_otsu,
                    blur_ksize=underline_blur_ksize,
                )
                if below_result and below_result.get("real_line"):
                    analysis = below_result.get("analysis") or {}
                    row_pixels_ok = analysis.get("coverage_ratio", 0.0) >= (
                        _min_row_pixels(below_x1, below_x2) / max(below_x2 - below_x1, 1)
                    )
                    if row_pixels_ok:
                        underline_y = int(below_result["underline_y"])
                        underline_span = below_result.get("span")
                        underline_source = "below"

        # Last step: align the box bottom to the detected underline, if any.
        if underline_y is not None:
            new_y2 = _clamp_int(underline_y, 1, height)
            new_y1 = max(0, new_y2 - orig_box_h)
            cur_y1, cur_y2 = new_y1, new_y2
            _record(
                "underline_align",
                cur_x1,
                cur_x2,
                cur_y1,
                cur_y2,
                note=f"underline_y={underline_y} span={underline_span} source={underline_source}",
            )
        else:
            _record("underline_align", cur_x1, cur_x2, cur_y1, cur_y2, note="no_underline")

        cur_x1 = _clamp_int(cur_x1, 0, width - 1)
        cur_x2 = _clamp_int(cur_x2, 1, width)
        if cur_x2 <= cur_x1:
            continue
        if cur_x2 - cur_x1 < min_width_floor_px:
            continue

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
