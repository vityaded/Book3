from __future__ import annotations

import io
import json
import os
import re
import sys
from pathlib import Path

import cv2
import fitz  # PyMuPDF
import numpy as np
from PIL import Image, ImageDraw

BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE_DIR))

import app


EXAMPLES_DIR = BASE_DIR / "examples"
MANUAL_JSONS_DIR = BASE_DIR / "manual_jsons"
MANUAL_RENDER_DIR = BASE_DIR / "manual_process" / "renders"

VENOMOUS_GREEN = "#39FF14"
OVERLAY_LINE_WIDTH = 3
PDF_DPI = 200
TESSDATA_DIR_FALLBACK = "/usr/share/tesseract-ocr/5/tessdata"

# Ensure PyMuPDF can find tessdata even when system paths are not detected.
os.environ.setdefault("TESSDATA_PREFIX", TESSDATA_DIR_FALLBACK)


PROMPT = """
Task: Identify all user-fillable blanks (underlines, dotted/dashed lines, boxes, or empty writable spaces) in the document. Return their bounding boxes in strict JSON format.

For each fillable blank, identify two regions:
- Anchor: The bounding box of the specific word or symbol immediately to the left of the blank (e.g., "Name:", "1.", "a)").
- Target: The bounding box of the writable blank area (line/box/space for the answer).

Coordinate system:
- Use [ymin, xmin, ymax, xmax] relative to the image dimensions.
- Scale: 0-1000 (integer values).

Rules:
- The anchor_box and target_box must refer to the same blank.
- anchor_box must tightly bound only the anchor text/symbol (not the blank).
- target_box must tightly bound only the writable blank area (not surrounding text).
- The bottom edge (ymax) of the target_box should align with the writing baseline on that line.
- Use tight heights that match the cap height of surrounding text.
- Include filled: true when the blank already contains handwriting or typed text.
- If no blanks are found, return {"items": []}.

Workbook-specific requirements (very important):
- Do NOT return boxes for normal spaces at the end of a sentence line (right-margin whitespace).
- For sentence-level answer lines (e.g., "Correct these sentences ..."), return ONE target_box per numbered sentence covering the full answer line(s) below it. If there are multiple consecutive answer lines for the same number, merge them into a single target_box spanning all lines.
- When a blank line has no printed text immediately to its left, create anchor_box as a small letter-sized box at the left edge of that blank line (same line as the blank), with its bottom aligned to the writing baseline.
- If there are several consecutive unlabeled blank lines (one under another), return coordinates only for the topmost one.
""".strip()


def slugify(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "_", name).strip("_") or "document"


def render_page_to_png(doc: fitz.Document, page_index: int, out_path: Path) -> Image.Image:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    zoom = PDF_DPI / 72.0
    matrix = fitz.Matrix(zoom, zoom)
    page = doc.load_page(page_index)
    pix = page.get_pixmap(matrix=matrix, alpha=False)
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    img.save(out_path, format="PNG")
    return img


def call_gemini(image: Image.Image) -> dict:
    client = app.get_client()
    config = app.build_generation_config()
    response = app.call_gemini_with_retries(client, [image, PROMPT], config)
    text_candidates = app.extract_text_candidates(response)
    if not text_candidates:
        raise ValueError("Gemini returned no text parts.")

    last_error = None
    for text in text_candidates:
        if not text or not str(text).strip():
            continue
        try:
            return app.parse_json_response(str(text))
        except Exception as exc:
            last_error = exc

    if last_error:
        raise last_error
    raise ValueError("Gemini returned empty text response.")


def _normalize_box_coords(box: object) -> list[int] | None:
    if not isinstance(box, (list, tuple)) or len(box) != 4:
        return None
    try:
        y1, x1, y2, x2 = [float(v) for v in box]
    except (TypeError, ValueError):
        return None
    if x2 <= x1 or y2 <= y1:
        return None

    max_val = max(x1, y1, x2, y2)
    if max_val <= 1.5:
        y1 *= 1000.0
        x1 *= 1000.0
        y2 *= 1000.0
        x2 *= 1000.0
    # If max_val > 1000, we assume it's already pixels; we *can't* reliably
    # normalize without knowing the base image size. In that case, we skip.
    if max_val > 1000.0:
        return None

    y1_i = int(round(max(0.0, min(1000.0, y1))))
    x1_i = int(round(max(0.0, min(1000.0, x1))))
    y2_i = int(round(max(0.0, min(1000.0, y2))))
    x2_i = int(round(max(0.0, min(1000.0, x2))))
    if x2_i <= x1_i or y2_i <= y1_i:
        return None
    return [y1_i, x1_i, y2_i, x2_i]


def _to_px(box_1000: list[int], width: int, height: int) -> tuple[int, int, int, int]:
    y1, x1, y2, x2 = box_1000
    left = int(round(x1 / 1000.0 * width))
    right = int(round(x2 / 1000.0 * width))
    top = int(round(y1 / 1000.0 * height))
    bottom = int(round(y2 / 1000.0 * height))
    left = max(0, min(width - 1, left))
    right = max(left + 1, min(width, right))
    top = max(0, min(height - 1, top))
    bottom = max(top + 1, min(height, bottom))
    return left, top, right, bottom


def _to_1000(left: int, top: int, right: int, bottom: int, width: int, height: int) -> list[int]:
    y1 = int(round(top / max(height, 1) * 1000.0))
    x1 = int(round(left / max(width, 1) * 1000.0))
    y2 = int(round(bottom / max(height, 1) * 1000.0))
    x2 = int(round(right / max(width, 1) * 1000.0))
    y1 = max(0, min(1000, y1))
    x1 = max(0, min(1000, x1))
    y2 = max(0, min(1000, y2))
    x2 = max(0, min(1000, x2))
    if x2 <= x1 or y2 <= y1:
        x2 = min(1000, x1 + 1)
        y2 = min(1000, y1 + 1)
    return [y1, x1, y2, x2]


def _estimate_text_height_px(words: list[dict], height: int) -> int:
    """Estimate typical on-page text height in pixels from OCR/PDF words."""
    heights: list[int] = []
    for w in words:
        try:
            y0 = float(w.get("y0", 0.0))
            y1 = float(w.get("y1", 0.0))
        except (TypeError, ValueError):
            continue
        if y1 <= y0:
            continue
        h = int(round((y1 - y0) * height))
        if 6 <= h <= 80:
            heights.append(h)
    if not heights:
        return 22
    heights.sort()
    return int(heights[len(heights) // 2])


def _extract_words_for_page(pdf_path: Path, page_index: int, image: Image.Image) -> list[dict]:
    """Return OCR/PDF words as normalized boxes in [0..1] for this rendered page."""
    words: list[dict] = []
    try:
        doc = fitz.open(str(pdf_path))
        try:
            page = doc.load_page(page_index)
            raw = page.get_text("words") or []
            if raw:
                rect = page.rect
                width = rect.width or 1.0
                height = rect.height or 1.0
                for x0, y0, x1, y1, text, *_rest in raw:
                    if not text or not any(ch.isalnum() for ch in text):
                        continue
                    words.append(
                        {
                            "text": text,
                            "x0": max(0.0, min(1.0, (x0 - rect.x0) / width)),
                            "y0": max(0.0, min(1.0, (y0 - rect.y0) / height)),
                            "x1": max(0.0, min(1.0, (x1 - rect.x0) / width)),
                            "y1": max(0.0, min(1.0, (y1 - rect.y0) / height)),
                        }
                    )
                return words
        finally:
            doc.close()
    except Exception:
        words = []

    # Fallback: OCR the rendered image.
    doc = None
    try:
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        img_bytes = buffer.getvalue()
        doc = fitz.open()
        page = doc.new_page(width=image.width, height=image.height)
        page.insert_image(fitz.Rect(0, 0, image.width, image.height), stream=img_bytes)
        tessdata = os.getenv("TESSDATA_PREFIX") or TESSDATA_DIR_FALLBACK
        try:
            textpage = page.get_textpage_ocr(language="eng", tessdata=tessdata)
        except TypeError:
            textpage = page.get_textpage_ocr(tessdata=tessdata)
        for x0, y0, x1, y1, text, *_rest in textpage.extractWORDS():
            if not text or not any(ch.isalnum() for ch in text):
                continue
            words.append(
                {
                    "text": text,
                    "x0": max(0.0, min(1.0, x0 / image.width)),
                    "y0": max(0.0, min(1.0, y0 / image.height)),
                    "x1": max(0.0, min(1.0, x1 / image.width)),
                    "y1": max(0.0, min(1.0, y1 / image.height)),
                }
            )
        return words
    except Exception:
        return []
    finally:
        if doc is not None:
            doc.close()


def _find_anchor_box(words: list[dict], target_box: list[int]) -> list[int] | None:
    """Choose the OCR/PDF word immediately to the left of the target_box."""
    if not words:
        return None
    ty1, tx1, ty2, _tx2 = target_box
    tx1_n = tx1 / 1000.0
    ty1_n = ty1 / 1000.0
    ty2_n = ty2 / 1000.0

    best = None
    best_dx = None
    for w in words:
        try:
            x0 = float(w.get("x0", 0.0))
            y0 = float(w.get("y0", 0.0))
            x1 = float(w.get("x1", 0.0))
            y1 = float(w.get("y1", 0.0))
        except (TypeError, ValueError):
            continue
        if x1 <= x0 or y1 <= y0:
            continue

        # Require meaningful vertical overlap with the blank's band.
        overlap = min(y1, ty2_n) - max(y0, ty1_n)
        if overlap <= 0:
            continue
        if overlap / max(y1 - y0, 1e-6) < 0.35:
            continue

        # Anchor must be on the left side of the blank.
        if x1 > tx1_n:
            continue

        dx = tx1_n - x1
        if best_dx is None or dx < best_dx:
            best_dx = dx
            best = (x0, y0, x1, y1)

    if best is None:
        return None
    x0, y0, x1, y1 = best
    return [
        int(round(y0 * 1000.0)),
        int(round(x0 * 1000.0)),
        int(round(y1 * 1000.0)),
        int(round(x1 * 1000.0)),
    ]


def _find_anchor_box_and_label(
    words: list[dict],
    target_box: list[int],
    *,
    width: int,
    height: int,
    text_height_px: int,
) -> tuple[list[int] | None, str]:
    """Return (anchor_box_1000, label). Label is the anchor text when available."""
    if not words:
        return None, ""

    # Pass 1: strict overlap with the blank's band (same line).
    ty1, tx1, ty2, _tx2 = target_box
    tx1_n = tx1 / 1000.0
    ty1_n = ty1 / 1000.0
    ty2_n = ty2 / 1000.0

    best = None
    best_dx = None
    for w in words:
        try:
            x0 = float(w.get("x0", 0.0))
            y0 = float(w.get("y0", 0.0))
            x1 = float(w.get("x1", 0.0))
            y1 = float(w.get("y1", 0.0))
        except (TypeError, ValueError):
            continue
        if x1 <= x0 or y1 <= y0:
            continue
        if x1 > tx1_n:
            continue
        overlap = min(y1, ty2_n) - max(y0, ty1_n)
        if overlap <= 0:
            continue
        if overlap / max(y1 - y0, 1e-6) < 0.35:
            continue
        dx = tx1_n - x1
        if best_dx is None or dx < best_dx:
            best_dx = dx
            best = (x0, y0, x1, y1, str(w.get("text", "") or ""))

    if best is not None:
        x0, y0, x1, y1, text = best
        text_stripped = text.strip()
        anchor_box = [
            int(round(y0 * 1000.0)),
            int(round(x0 * 1000.0)),
            int(round(y1 * 1000.0)),
            int(round(x1 * 1000.0)),
        ]
        return anchor_box, text_stripped

    # Pass 2: allow the anchor to be on the previous line (common for answer lines
    # under numbered sentences).
    target_left, _target_top, _target_right, target_bottom = _to_px(target_box, width, height)
    baseline_y = target_bottom
    dy_limit = max(18, int(text_height_px * 3.0))

    best2 = None
    best_score = None
    for w in words:
        try:
            x0 = float(w.get("x0", 0.0))
            y0 = float(w.get("y0", 0.0))
            x1 = float(w.get("x1", 0.0))
            y1 = float(w.get("y1", 0.0))
        except (TypeError, ValueError):
            continue
        if x1 <= x0 or y1 <= y0:
            continue
        w_left = int(round(x0 * width))
        w_right = int(round(x1 * width))
        w_bottom = int(round(y1 * height))
        if w_right > target_left:
            continue
        dx_px = target_left - w_right
        dy_px = abs(w_bottom - baseline_y)
        if dy_px > dy_limit:
            continue
        score = -1.7 * float(dy_px) - 0.5 * float(dx_px)
        text = str(w.get("text", "") or "").strip()
        # Strongly prefer numeric question indices.
        if text and len(text) <= 3 and any(ch.isdigit() for ch in text):
            if re.fullmatch(r"\d+\.?", text):
                score += 80.0
            elif len(text) <= 2:
                score += 12.0
        if best_score is None or score > best_score:
            best_score = score
            best2 = (x0, y0, x1, y1, text)

    if best2 is None:
        return None, ""
    x0, y0, x1, y1, text = best2
    anchor_box = [
        int(round(y0 * 1000.0)),
        int(round(x0 * 1000.0)),
        int(round(y1 * 1000.0)),
        int(round(x1 * 1000.0)),
    ]
    return anchor_box, text.strip()


def _detect_fill_line(gray: np.ndarray, roi: tuple[int, int, int, int], expected_center_x: int, expected_y: int) -> tuple[int, int, int] | None:
    """
    Detect a horizontal underline/dotted line inside roi.
    Returns (y_baseline, x_left, x_right) in absolute pixel coords.
    """
    x1, y1, x2, y2 = roi
    if x2 - x1 < 10 or y2 - y1 < 6:
        return None

    sub = gray[y1:y2, x1:x2]
    if sub.size == 0:
        return None

    blur = cv2.GaussianBlur(sub, (3, 3), 0)
    _, binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Connect dotted lines but avoid merging whole text lines.
    close_w = max(3, min(13, int((x2 - x1) * 0.04)))
    close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (close_w, 1))
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, close_kernel, iterations=1)

    # Keep horizontal structures. Kernel is based on ROI width to keep short blanks.
    open_w = max(10, min(120, int((x2 - x1) * 0.12)))
    open_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (open_w, 1))
    hlines = cv2.morphologyEx(closed, cv2.MORPH_OPEN, open_kernel, iterations=1)

    contours, _hier = cv2.findContours(hlines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    best = None
    best_score = None
    for c in contours:
        cx, cy, cw, ch = cv2.boundingRect(c)
        if cw < 8:
            continue
        if ch > 12:
            continue
        abs_x = x1 + cx
        abs_y = y1 + cy
        baseline_y = abs_y + ch - 1
        center_x = abs_x + cw // 2
        # Prefer long lines close to expected y, and close in x.
        score = float(cw) - 1.8 * abs(baseline_y - expected_y) - 0.6 * abs(center_x - expected_center_x)
        if best_score is None or score > best_score:
            best_score = score
            best = (baseline_y, abs_x, abs_x + cw)

    return best


def _is_thin_line(
    gray: np.ndarray,
    *,
    x1: int,
    x2: int,
    y: int,
    min_row_pixels: int,
    threshold: int = 170,
    max_rows_with_ink: int = 2,
    center_ratio: float = 0.6,
) -> bool:
    """Heuristic: underlines/dots are usually 1-2px tall."""
    height, width = gray.shape[:2]
    if x2 - x1 < 4:
        return False
    y0 = max(0, y - 2)
    y1 = min(height, y + 3)
    if y1 <= y0:
        return False
    strip = gray[y0:y1, x1:x2]
    if strip.size == 0:
        return False
    _, binary = cv2.threshold(strip, threshold, 255, cv2.THRESH_BINARY_INV)
    row_counts = np.count_nonzero(binary, axis=1)
    total_ink = int(row_counts.sum())
    if total_ink == 0:
        return False
    center_idx = y - y0
    if center_idx < 0 or center_idx >= row_counts.size:
        return False
    rows_with_ink = sum(count >= min_row_pixels for count in row_counts)
    if rows_with_ink > max_rows_with_ink:
        return False
    center_count = int(row_counts[center_idx])
    if (center_count / total_ink) < center_ratio:
        return False
    return True


def _find_line_for_target(
    image: Image.Image,
    target_box: list[int],
    *,
    text_height_px: int,
    anchor_box: list[int] | None,
) -> tuple[int, int, int] | None:
    width, height = image.size
    gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)

    left, top, right, bottom = _to_px(target_box, width, height)
    expected_center_x = (left + right) // 2
    expected_y = bottom

    margin_x = max(120, min(width // 2, int(width * 0.38)))
    margin_y = max(12, min(90, int(text_height_px * 3.0)))
    sx1 = max(0, left - margin_x)
    sx2 = min(width, right + margin_x)
    sy1 = max(0, bottom - margin_y)
    sy2 = min(height, bottom + margin_y)

    detected = _detect_fill_line(gray, (sx1, sy1, sx2, sy2), expected_center_x, expected_y)
    if detected is None:
        return None
    baseline_y, line_left, line_right = detected

    # Validate thinness to avoid snapping to text rows.
    line_w = max(1, line_right - line_left)
    min_row_pixels = max(2, int(line_w * 0.05))
    if not _is_thin_line(
        gray,
        x1=line_left,
        x2=line_right,
        y=baseline_y,
        min_row_pixels=min_row_pixels,
        threshold=200,
        max_rows_with_ink=3,
        center_ratio=0.45,
    ):
        return None
    if line_w < max(8, int(text_height_px * 0.8)):
        return None
    return baseline_y, line_left, line_right


def refine_target_box_cv(
    image: Image.Image,
    target_box: list[int],
    *,
    text_height_px: int,
    anchor_box: list[int] | None,
) -> tuple[list[int], bool]:
    width, height = image.size

    left, top, right, bottom = _to_px(target_box, width, height)
    box_w = right - left
    box_h = bottom - top

    # Small square-ish targets are likely checkboxes; keep them.
    if box_w <= max(14, int(text_height_px * 1.5)) and box_h <= max(14, int(text_height_px * 1.5)):
        if 0.55 <= (box_w / max(box_h, 1)) <= 1.8:
            return target_box, False

    detected = _find_line_for_target(
        image,
        target_box,
        text_height_px=text_height_px,
        anchor_box=anchor_box,
    )
    if detected is None:
        return target_box, False

    baseline_y, line_left, line_right = detected
    pad_x = max(2, int(text_height_px * 0.15))
    new_left = max(0, line_left - pad_x)
    new_right = min(width, line_right + pad_x)

    # Don't let the target overlap the anchor.
    if anchor_box is not None:
        a_left, a_top, a_right, a_bottom = _to_px(anchor_box, width, height)
        new_left = max(new_left, a_right + 2)

    new_bottom = int(baseline_y)
    new_top = max(0, new_bottom - max(10, int(text_height_px * 1.15)))

    if new_right - new_left < 4 or new_bottom - new_top < 4:
        return target_box, False

    return _to_1000(new_left, new_top, new_right, new_bottom, width, height), True


def _looks_filled(image: Image.Image, target_box: list[int]) -> bool:
    """Heuristic: detect handwriting/typed content above the underline/dots."""
    width, height = image.size
    left, top, right, bottom = _to_px(target_box, width, height)
    if right - left < 6 or bottom - top < 6:
        return False
    crop = image.crop((left, top, right, bottom)).convert("RGB")
    arr = np.array(crop)
    if arr.size == 0:
        return False
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    box_h = int(gray.shape[0])
    box_w = int(gray.shape[1])

    # Checkbox-like squares: consider full interior so ticks/crosses count as filled.
    if box_w <= 80 and box_h <= 80 and 0.6 <= (box_w / max(box_h, 1)) <= 1.6:
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        dark = int(np.count_nonzero(binary))
        area = int(binary.size)
        if area <= 0:
            return False
        return (dark / area) >= 0.05

    # Underline blanks: ignore the bottom band where underline/dots live.
    ignore_rows = max(1, int(box_h * 0.22))
    if box_h - ignore_rows < 2:
        return False
    content = gray[: box_h - ignore_rows, :]
    _, binary = cv2.threshold(content, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    dark = int(np.count_nonzero(binary))
    area = int(binary.size)
    if area <= 0:
        return False
    return (dark / area) >= 0.015


def ensure_anchor_box(item: dict) -> None:
    if item.get("anchor_box"):
        return
    target_box = item.get("target_box")
    if not target_box:
        return
    y1, x1, y2, x2 = target_box
    # Small "letter-sized" anchor at left edge of blank line, baseline-aligned.
    h = max(10, int((y2 - y1) * 0.9))
    w = max(8, int(h * 0.6))
    # Prefer placing the anchor just to the LEFT of the target so it does not
    # overlap the writable area (important for checkboxes at the margin).
    ax2 = int(x1)
    ax1 = max(0, int(x1 - w))
    if ax2 <= ax1:
        ax1 = int(x1)
        ax2 = min(1000, int(x1 + w))
    item["anchor_box"] = [max(0, y2 - h), ax1, y2, ax2]


def _rect_iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
        return 0.0
    inter = float((inter_x2 - inter_x1) * (inter_y2 - inter_y1))
    area_a = float((ax2 - ax1) * (ay2 - ay1))
    area_b = float((bx2 - bx1) * (by2 - by1))
    denom = area_a + area_b - inter
    if denom <= 0:
        return 0.0
    return inter / denom


def _dedup_rects(rects: list[tuple[int, int, int, int]], *, iou_thresh: float = 0.85) -> list[tuple[int, int, int, int]]:
    if not rects:
        return rects
    rects_sorted = sorted(rects, key=lambda r: (r[2] - r[0]) * (r[3] - r[1]), reverse=True)
    kept: list[tuple[int, int, int, int]] = []
    for r in rects_sorted:
        if any(_rect_iou(r, k) >= iou_thresh for k in kept):
            continue
        kept.append(r)
    kept.sort(key=lambda r: (r[1], r[0]))
    return kept


def _words_overlapping_span(
    words: list[dict],
    *,
    span_left_px: int,
    span_right_px: int,
    y_bottom_px: int,
    width: int,
    height: int,
    max_dy_px: int,
) -> bool:
    """True when printed OCR words sit directly above and overlap the line span."""
    if not words:
        return False
    for w in words:
        try:
            x0 = float(w.get("x0", 0.0))
            y0 = float(w.get("y0", 0.0))
            x1 = float(w.get("x1", 0.0))
            y1 = float(w.get("y1", 0.0))
        except (TypeError, ValueError):
            continue
        wx0 = int(round(x0 * width))
        wx1 = int(round(x1 * width))
        wy1 = int(round(y1 * height))
        if wx1 <= span_left_px or wx0 >= span_right_px:
            continue
        if abs(wy1 - y_bottom_px) > max_dy_px:
            continue
        if wy1 <= y_bottom_px:
            return True
    return False


def detect_targets_cv(image: Image.Image, *, words: list[dict], text_height_px: int) -> list[list[int]]:
    """
    Detect fillable blanks from pixels:
    - underlines / dotted lines (horizontal segments)
    - small square boxes (checkboxes)

    Returns target boxes in 0..1000 coords.
    """
    width, height = image.size
    gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    _, binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    close_w = max(3, min(15, int(text_height_px * 0.6)))
    close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (close_w, 1))
    connected = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, close_kernel, iterations=1)

    line_rects: list[tuple[int, int, int, int]] = []
    kernel_ws = {
        max(10, min(260, int(width * 0.18))),
        max(10, min(220, int(width * 0.12))),
        max(10, min(180, int(width * 0.08))),
        max(10, min(120, int(width * 0.05))),
        max(10, min(90, int(text_height_px * 2.8))),
        max(8, min(70, int(text_height_px * 2.0))),
        max(8, min(50, int(text_height_px * 1.6))),
    }
    max_line_h = max(4, int(text_height_px * 0.4))
    min_line_w = max(8, int(text_height_px * 0.8))

    for kw in sorted(kernel_ws, reverse=True):
        open_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (int(kw), 1))
        hlines = cv2.morphologyEx(connected, cv2.MORPH_OPEN, open_kernel, iterations=1)
        contours, _hier = cv2.findContours(hlines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in contours:
            x, y, w, h = cv2.boundingRect(c)
            if w < min_line_w:
                continue
            if h > max_line_h:
                continue
            if w / max(h, 1) < 3.0:
                continue
            baseline_y = y + h - 1
            if _words_overlapping_span(
                words,
                span_left_px=x,
                span_right_px=x + w,
                y_bottom_px=baseline_y,
                width=width,
                height=height,
                max_dy_px=max(10, int(text_height_px * 0.7)),
            ):
                continue
            line_rects.append((x, y, x + w, y + h))

    line_rects = _dedup_rects(line_rects, iou_thresh=0.88)

    targets: list[list[int]] = []
    for x1, y1, x2, y2 in line_rects:
        baseline_y = y2 - 1
        bottom = int(baseline_y + 1)
        top = max(0, bottom - max(10, int(text_height_px * 1.2)))
        targets.append(_to_1000(x1, top, x2, bottom, width, height))

    # Checkboxes (small square outlines).
    edges = cv2.Canny(gray, 50, 150)
    contours, _hier = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    rects: list[tuple[int, int, int, int]] = []
    min_side = max(8, int(text_height_px * 0.55))
    max_side = max(min_side + 1, int(text_height_px * 2.2))
    for c in contours:
        peri = cv2.arcLength(c, True)
        if peri <= 0:
            continue
        approx = cv2.approxPolyDP(c, 0.03 * peri, True)
        if len(approx) != 4:
            continue
        x, y, w, h = cv2.boundingRect(approx)
        if w < min_side or h < min_side:
            continue
        if w > max_side or h > max_side:
            continue
        ar = w / max(h, 1)
        if not (0.75 <= ar <= 1.35):
            continue
        rects.append((x, y, x + w, y + h))
    rects = _dedup_rects(rects, iou_thresh=0.8)
    for x1, y1, x2, y2 in rects:
        targets.append(_to_1000(x1, y1, x2, y2, width, height))

    # Final dedup.
    px_targets = [_to_px(t, width, height) for t in targets]
    px_targets = _dedup_rects([(l, t, r, b) for (l, t, r, b) in px_targets], iou_thresh=0.85)
    return [_to_1000(l, t, r, b, width, height) for (l, t, r, b) in px_targets]


def detect_checkboxes_cv(image: Image.Image, *, text_height_px: int) -> list[list[int]]:
    """Detect small checkbox-like square outlines."""
    width, height = image.size
    gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    contours, _hier = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    rects: list[tuple[int, int, int, int]] = []
    min_side = max(8, int(text_height_px * 0.55))
    max_side = max(min_side + 1, int(text_height_px * 2.2))
    for c in contours:
        peri = cv2.arcLength(c, True)
        if peri <= 0:
            continue
        approx = cv2.approxPolyDP(c, 0.03 * peri, True)
        if len(approx) != 4:
            continue
        x, y, w, h = cv2.boundingRect(approx)
        if w < min_side or h < min_side:
            continue
        if w > max_side or h > max_side:
            continue
        ar = w / max(h, 1)
        if not (0.75 <= ar <= 1.35):
            continue
        # Confirm there is a border (dark pixels) but mostly empty inside.
        crop = gray[y : y + h, x : x + w]
        if crop.size == 0:
            continue
        _, inv = cv2.threshold(crop, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        border = max(1, int(min(w, h) * 0.18))
        inner = inv[border : h - border, border : w - border]
        if inner.size == 0:
            continue
        inner_dark = float(np.count_nonzero(inner)) / float(inner.size)
        if inner_dark > 0.18:
            # Too much ink inside: likely a picture/text box, not a checkbox.
            continue
        rects.append((x, y, x + w, y + h))
    rects = _dedup_rects(rects, iou_thresh=0.8)
    return [_to_1000(x1, y1, x2, y2, width, height) for (x1, y1, x2, y2) in rects]


def _merge_consecutive_same_label(items: list[dict], *, y_gap: int) -> list[dict]:
    if not items:
        return items
    merged: list[dict] = []
    for item in items:
        label = str(item.get("label", "") or "")
        if not merged or not label:
            merged.append(item)
            continue
        prev = merged[-1]
        prev_label = str(prev.get("label", "") or "")
        if label != prev_label:
            merged.append(item)
            continue
        y1, x1, y2, x2 = item["target_box"]
        py1, px1, py2, px2 = prev["target_box"]
        if abs(x1 - px1) > 30 or abs(x2 - px2) > 30:
            merged.append(item)
            continue
        if y1 - py2 > y_gap:
            merged.append(item)
            continue
        prev["target_box"] = [min(py1, y1), min(px1, x1), max(py2, y2), max(px2, x2)]
    return merged


def _drop_consecutive_unlabeled(items: list[dict], *, y_gap: int) -> list[dict]:
    """Drop consecutive unlabeled blank lines; keep only the topmost one."""
    if not items:
        return items
    out: list[dict] = []
    prev: dict | None = None
    for item in items:
        label = str(item.get("label", "") or "").strip()
        if prev is None:
            out.append(item)
            prev = item
            continue
        prev_label = str(prev.get("label", "") or "").strip()
        if label or prev_label:
            out.append(item)
            prev = item
            continue
        y1, x1, y2, x2 = item["target_box"]
        py1, px1, py2, px2 = prev["target_box"]
        cur_w = x2 - x1
        cur_h = y2 - y1
        prev_w = px2 - px1
        prev_h = py2 - py1
        # Only drop consecutive unlabeled *line-like* blanks, not checkboxes.
        cur_line_like = cur_w > (cur_h * 2.4)
        prev_line_like = prev_w > (prev_h * 2.4)
        if cur_line_like and prev_line_like and abs(x1 - px1) <= 30 and abs(x2 - px2) <= 30 and (y1 - py2) <= y_gap:
            continue
        out.append(item)
        prev = item
    return out


def draw_overlay(image: Image.Image, items: list[dict]) -> Image.Image:
    out = image.copy()
    draw = ImageDraw.Draw(out)
    width, height = out.size
    for item in items:
        box = item.get("target_box")
        if not box:
            continue
        y1, x1, y2, x2 = box
        left = int(round(x1 / 1000.0 * width))
        right = int(round(x2 / 1000.0 * width))
        top = int(round(y1 / 1000.0 * height))
        bottom = int(round(y2 / 1000.0 * height))
        if right <= left or bottom <= top:
            continue
        for offset in range(OVERLAY_LINE_WIDTH):
            draw.rectangle(
                [left - offset, top - offset, right + offset, bottom + offset],
                outline=VENOMOUS_GREEN,
                width=1,
            )
    return out


def process_pdf(pdf_path: Path) -> None:
    pdf_slug = slugify(pdf_path.stem)
    out_dir = MANUAL_JSONS_DIR / pdf_slug
    out_dir.mkdir(parents=True, exist_ok=True)

    doc = fitz.open(str(pdf_path))
    pages = []
    try:
        for page_index in range(doc.page_count):
            render_path = MANUAL_RENDER_DIR / pdf_slug / f"page_{page_index + 1:03d}.png"
            image = render_page_to_png(doc, page_index, render_path)
            llm_image = app.prepare_image_for_llm(image)

            response = call_gemini(llm_image)
            raw_items = response.get("items")
            if not isinstance(raw_items, list):
                raw_items = response.get("boxes", []) if isinstance(response, dict) else []

            items: list[dict] = []
            for raw in raw_items:
                if not isinstance(raw, dict):
                    continue
                target_box = _normalize_box_coords(raw.get("target_box") or raw.get("box_2d") or raw.get("bbox"))
                if not target_box:
                    continue
                anchor_box = _normalize_box_coords(raw.get("anchor_box"))
                label = str(raw.get("label", "") or "").strip()
                items.append(
                    {
                        "anchor_box": anchor_box,
                        "target_box": target_box,
                        "label": label,
                        "filled": bool(raw.get("filled", False)),
                    }
                )

            # OCR/PDF words (for anchor placement) + page text height estimate.
            words = _extract_words_for_page(pdf_path, page_index, image)
            text_height_px = _estimate_text_height_px(words, image.height)

            refined_items: list[dict] = []
            for item in items:
                raw_target = item.get("target_box")
                if not raw_target:
                    continue
                raw_anchor = item.get("anchor_box")

                refined_target, line_found = refine_target_box_cv(
                    image,
                    raw_target,
                    text_height_px=text_height_px,
                    anchor_box=raw_anchor,
                )
                left, top, right, bottom = _to_px(refined_target, image.width, image.height)
                box_w = right - left
                box_h = bottom - top
                box_like = box_w <= max(14, int(text_height_px * 2.2)) and box_h <= max(14, int(text_height_px * 2.2))
                if not line_found and not box_like:
                    continue

                item["target_box"] = refined_target
                item["filled"] = bool(item.get("filled")) or _looks_filled(image, refined_target)

                anchor_box, anchor_text = _find_anchor_box_and_label(
                    words,
                    refined_target,
                    width=image.width,
                    height=image.height,
                    text_height_px=text_height_px,
                )
                if anchor_box is not None:
                    item["anchor_box"] = anchor_box
                else:
                    ensure_anchor_box(item)

                # Prefer Gemini label when it's meaningful; otherwise fall back to anchor text.
                label = str(item.get("label", "") or "").strip()
                if not (len(label) >= 2 and any(ch.isalnum() for ch in label)):
                    label = anchor_text
                item["label"] = label

                refined_items.append(item)

            # Add checkboxes that Gemini may miss (esp. in Grammar page 1).
            checkbox_targets = detect_checkboxes_cv(image, text_height_px=text_height_px)
            existing_px = [_to_px(it["target_box"], image.width, image.height) for it in refined_items if it.get("target_box")]
            for target_box in checkbox_targets:
                px = _to_px(target_box, image.width, image.height)
                if any(_rect_iou(px, e) >= 0.45 for e in existing_px):
                    continue
                cb_item = {
                    "anchor_box": None,
                    "target_box": target_box,
                    "label": "",
                    "filled": _looks_filled(image, target_box),
                }
                ensure_anchor_box(cb_item)
                refined_items.append(cb_item)
                existing_px.append(px)

            refined_items.sort(key=lambda i: (i["target_box"][0], i["target_box"][1]))
            refined_items = _drop_consecutive_unlabeled(refined_items, y_gap=max(8, int(text_height_px * 1.2)))
            refined_items = _merge_consecutive_same_label(refined_items, y_gap=max(10, int(text_height_px * 1.5)))

            pages.append({"page": page_index + 1, "items": refined_items})

            overlay = draw_overlay(image, refined_items)
            overlay_path = out_dir / f"page_{page_index + 1:03d}.png"
            overlay.save(overlay_path, format="PNG")
            print(f"[{pdf_path.name}] page {page_index + 1}/{doc.page_count}: {len(refined_items)} items")
    finally:
        doc.close()

    MANUAL_JSONS_DIR.mkdir(parents=True, exist_ok=True)
    json_path = MANUAL_JSONS_DIR / f"{pdf_slug}.json"
    json_path.write_text(json.dumps({"pages": pages}, indent=2), encoding="utf-8")
    print(f"[{pdf_path.name}] wrote {json_path}")


def main() -> int:
    pdfs = sorted(EXAMPLES_DIR.glob("*.pdf"))
    if not pdfs:
        print("No PDFs found in examples/")
        return 1
    for pdf_path in pdfs:
        process_pdf(pdf_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
