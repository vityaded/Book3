import io
import json
import logging
import os
import re
import shutil
import subprocess
import time
import uuid
import argparse
from difflib import SequenceMatcher
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from dotenv import load_dotenv
from flask import Flask, abort, flash, jsonify, redirect, render_template, request, url_for
from PIL import Image

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

try:
    import fitz  # PyMuPDF
except Exception as exc:  # pragma: no cover - import error path
    fitz = None

try:
    from google import genai
    from google.genai import types
except Exception as exc:  # pragma: no cover - import error path
    genai = None
    types = None

try:
    import cv_tools
except Exception:
    cv_tools = None

STATIC_DIR = BASE_DIR / "static"
JOBS_DIR = BASE_DIR / "data" / "jobs"
JOB_STATIC_DIR = STATIC_DIR / "jobs"

JOBS_DIR.mkdir(parents=True, exist_ok=True)
JOB_STATIC_DIR.mkdir(parents=True, exist_ok=True)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3-flash-preview").strip()
GEMINI_DEBUG = os.getenv("GEMINI_DEBUG", "0") == "1"
GEMINI_LOG_MAX_CHARS = int(os.getenv("GEMINI_LOG_MAX_CHARS", "4000"))
GEMINI_TEMPERATURE = float(os.getenv("GEMINI_TEMPERATURE", "0.0"))
GEMINI_TOP_P = float(os.getenv("GEMINI_TOP_P", "0.1"))
GEMINI_MAX_OUTPUT_TOKENS = int(os.getenv("GEMINI_MAX_OUTPUT_TOKENS", "2048"))
GEMINI_THINKING_BUDGET = int(os.getenv("GEMINI_THINKING_BUDGET", "0"))
GEMINI_TIMEOUT_SEC = float(os.getenv("GEMINI_TIMEOUT_SEC", "60"))
GEMINI_MAX_RETRIES = int(os.getenv("GEMINI_MAX_RETRIES", "2"))
GEMINI_RETRY_BACKOFF_SEC = float(os.getenv("GEMINI_RETRY_BACKOFF_SEC", "1.5"))
GEMINI_SAFETY_BLOCK_ONLY_HIGH = os.getenv("GEMINI_SAFETY_BLOCK_ONLY_HIGH", "1") == "1"
GEMINI_FALLBACK_MODEL = os.getenv("GEMINI_FALLBACK_MODEL", "").strip()
MAX_PAGES = int(os.getenv("MAX_PAGES", "10"))
MAX_CONCURRENT_PAGES = int(os.getenv("MAX_CONCURRENT_PAGES", "2"))
MAX_IMAGE_DIM = int(os.getenv("MAX_IMAGE_DIM", "1600"))
UPLOAD_MAX_MB = int(os.getenv("UPLOAD_MAX_MB", "30"))
BOX_INSET_RATIO = float(os.getenv("BOX_INSET_RATIO", "0.04"))
FILTER_FILLED_BOXES = os.getenv("FILTER_FILLED_BOXES", "1") == "1"
BOX_COORDINATE_SCALE = os.getenv("BOX_COORDINATE_SCALE", "1000").strip().lower()
OCR_LANG = os.getenv("OCR_LANG", "eng").strip() or "eng"
OCR_TESSDATA_DIR = os.getenv("OCR_TESSDATA_DIR", "").strip()
if not OCR_TESSDATA_DIR and fitz is not None:
    try:
        detected = fitz.get_tessdata()
        if isinstance(detected, str):
            OCR_TESSDATA_DIR = detected
    except Exception:
        OCR_TESSDATA_DIR = ""

OCRMY_PDF_ENABLED = os.getenv("OCRMY_PDF_ENABLED", "1") == "1"
OCRMY_PDF_REQUIRED = os.getenv("OCRMY_PDF_REQUIRED", "1") == "1"
OCRMY_PDF_OPTIMIZE = os.getenv("OCRMY_PDF_OPTIMIZE", "0").strip() or "0"
OCRMY_PDF_SKIP_TEXT = os.getenv("OCRMY_PDF_SKIP_TEXT", "1") == "1"
OCRMY_PDF_OUTPUT_TYPE = os.getenv("OCRMY_PDF_OUTPUT_TYPE", "pdf").strip() or "pdf"
OCRMY_PDF_PDFA_IMAGE_COMPRESSION = os.getenv("OCRMY_PDF_PDFA_IMAGE_COMPRESSION", "lossless").strip() or "lossless"
OCRMY_PDF_LOG_OUTPUT = os.getenv("OCRMY_PDF_LOG_OUTPUT", "0") == "1"
OCRMY_PDF_REMOVE_VECTORS = os.getenv("OCRMY_PDF_REMOVE_VECTORS", "0") == "1"
OCRMY_PDF_ROTATE_PAGES = os.getenv("OCRMY_PDF_ROTATE_PAGES", "0") == "1"
OCRMY_PDF_DESKEW = os.getenv("OCRMY_PDF_DESKEW", "0") == "1"
OCRMY_PDF_CLEAN = os.getenv("OCRMY_PDF_CLEAN", "0") == "1"
OCRMY_PDF_REMOVE_BACKGROUND = os.getenv("OCRMY_PDF_REMOVE_BACKGROUND", "0") == "1"
OCRMY_PDF_OVERSAMPLE = os.getenv("OCRMY_PDF_OVERSAMPLE", "").strip()
OCRMY_PDF_TESSERACT_CONFIG = os.getenv("OCRMY_PDF_TESSERACT_CONFIG", "").strip()
OCRMY_PDF_CHAR_WHITELIST = os.getenv("OCRMY_PDF_CHAR_WHITELIST", "").strip()
OCRMY_PDF_TESSERACT_PSM = os.getenv("OCRMY_PDF_TESSERACT_PSM", "").strip()
OCRMY_PDF_TESSERACT_OEM = os.getenv("OCRMY_PDF_TESSERACT_OEM", "").strip()
OCRMY_PDF_TESSERACT_THRESHOLDING = os.getenv("OCRMY_PDF_TESSERACT_THRESHOLDING", "").strip()
OCRMY_PDF_USER_WORDS = os.getenv("OCRMY_PDF_USER_WORDS", "").strip()
OCRMY_PDF_USER_PATTERNS = os.getenv("OCRMY_PDF_USER_PATTERNS", "").strip()
ANCHOR_GAP = float(os.getenv("ANCHOR_GAP", "0.006"))
ANCHOR_MAX_SHIFT_X = float(os.getenv("ANCHOR_MAX_SHIFT_X", "0.25"))
ANCHOR_MAX_SHIFT_Y = float(os.getenv("ANCHOR_MAX_SHIFT_Y", "0.15"))
ANCHOR_FUZZY_RATIO = float(os.getenv("ANCHOR_FUZZY_RATIO", "0.78"))
ANCHOR_RIGHT_TOLERANCE = float(os.getenv("ANCHOR_RIGHT_TOLERANCE", "0.02"))
ANCHOR_LINE_DY_MAX = float(os.getenv("ANCHOR_LINE_DY_MAX", "0.03"))
ANCHOR_LINE_DY_MIN = float(os.getenv("ANCHOR_LINE_DY_MIN", "0.01"))
ANCHOR_LINE_DY_MULT = float(os.getenv("ANCHOR_LINE_DY_MULT", "3.0"))
ANCHOR_TARGET_PADDING_RATIO = float(os.getenv("ANCHOR_TARGET_PADDING_RATIO", "0.01") or 0.01)
DEBUG_OCR_LAYER = os.getenv("DEBUG_OCR_LAYER", "0") == "1"
DEBUG_OCR_LAYER_ALL = os.getenv("DEBUG_OCR_LAYER_ALL", "0") == "1"
OCR_MIN_WORD_HEIGHT_PX = float(os.getenv("OCR_MIN_WORD_HEIGHT_PX", "0") or 0)
# Default to NO_OCR on; set NO_OCR=0 to re-enable OCR features.
NO_OCR = os.getenv("NO_OCR", "1") == "1"

if NO_OCR:
    OCRMY_PDF_ENABLED = False
    DEBUG_OCR_LAYER = False
    DEBUG_OCR_LAYER_ALL = False

ALLOWED_EXTENSIONS = {".pdf", ".png", ".jpg", ".jpeg", ".webp", ".bmp"}

INK_THRESHOLD = 150
INK_ROW_FRACTION = 0.12
INK_ROW_RATIO_MAX = 0.35
INK_DARK_RATIO_MAX = 0.25
INK_SAMPLE_SCALE = 0.6
MIN_BOX_SIZE_PX = 12
MIN_BOX_AREA_PX = 320
CV_SCAN_ENABLED = os.getenv("CV_SCAN_ENABLED", "1") == "1"
CV_SEARCH_WIDTH_PX = int(os.getenv("CV_SEARCH_WIDTH_PX", "220"))
CV_PADDING_PX = int(os.getenv("CV_PADDING_PX", "6"))
CV_MIN_WIDTH_PX = int(os.getenv("CV_MIN_WIDTH_PX", "14"))
CV_STRIP_HALF_HEIGHT_PX = int(os.getenv("CV_STRIP_HALF_HEIGHT_PX", "6"))
CV_THRESHOLD = int(os.getenv("CV_THRESHOLD", "200"))
CV_SCAN_TOP_RATIO = float(os.getenv("CV_SCAN_TOP_RATIO", "0.30"))
CV_SCAN_BOTTOM_RATIO = float(os.getenv("CV_SCAN_BOTTOM_RATIO", "0.70"))
_cv_edge_scan_raw = int(os.getenv("CV_EDGE_SCAN_PX", "0"))
CV_EDGE_SCAN_PX = _cv_edge_scan_raw if _cv_edge_scan_raw > 0 else None
CV_DEBUG = os.getenv("CV_DEBUG", "0") == "1"
CV_UNDERLINE_ENABLED = os.getenv("CV_UNDERLINE_ENABLED", "1") == "1"
CV_UNDERLINE_BAND_TOP_RATIO = float(os.getenv("CV_UNDERLINE_BAND_TOP_RATIO", "0.62"))
CV_UNDERLINE_EXTRA_BOTTOM_PX = int(os.getenv("CV_UNDERLINE_EXTRA_BOTTOM_PX", "10"))
_cv_underline_margin_raw = int(os.getenv("CV_UNDERLINE_MARGIN_X_PX", "0"))
CV_UNDERLINE_MARGIN_X_PX = _cv_underline_margin_raw if _cv_underline_margin_raw > 0 else None
CV_UNDERLINE_ROW_RATIO_MIN = float(os.getenv("CV_UNDERLINE_ROW_RATIO_MIN", "0.02"))
CV_UNDERLINE_MIN_ROW_PIXELS = int(os.getenv("CV_UNDERLINE_MIN_ROW_PIXELS", "12"))
CV_UNDERLINE_LINE_COVERAGE_MIN = float(os.getenv("CV_UNDERLINE_LINE_COVERAGE_MIN", "0.16"))
CV_UNDERLINE_SEGMENTS_MIN = int(os.getenv("CV_UNDERLINE_SEGMENTS_MIN", "3"))
CV_UNDERLINE_GAP_CV_MAX = float(os.getenv("CV_UNDERLINE_GAP_CV_MAX", "1.4"))
CV_UNDERLINE_BELOW_SCAN_PX = int(os.getenv("CV_UNDERLINE_BELOW_SCAN_PX", "28"))
CV_UNDERLINE_BELOW_MARGIN_TOP_PX = int(os.getenv("CV_UNDERLINE_BELOW_MARGIN_TOP_PX", "2"))

PROMPT_TEMPLATE = """
Task: Identify all user-fillable blanks (underlines, dotted lines, or empty spaces meant for input) in the document. Return their bounding boxes in strict JSON format.

For each fillable blank, identify two regions:
- Anchor: The bounding box of the specific word or symbol immediately to the left of the blank (e.g., "Name:", "1.").
- Target: The bounding box of the blank space itself.

Coordinate System:
- Use [ymin, xmin, ymax, xmax] relative to the image dimensions.
- Scale: 0-1000 (integer values).

Return JSON format:
{"items": [{"anchor_box": [ymin, xmin, ymax, xmax], "target_box": [ymin, xmin, ymax, xmax], "label": "string", "filled": false}]}

Rules:
- The anchor_box and target_box must refer to the same line and the same blank.
- The anchor_box should tightly bound only the anchor text/symbol, not the blank.
- The target_box should tightly bound only the writable blank area.
- The bottom edge (ymax) of the target_box should align with the text baseline on that line.
- Use tight heights that match the cap height of the surrounding text.
- Include filled: true when the blank already contains handwriting or typed text.
- If no blanks are found, return {"items": []}.
""".strip()

RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "items": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "anchor_box": {
                        "type": "array",
                        "items": {"type": "number"},
                        "minItems": 4,
                        "maxItems": 4,
                        "description": "Anchor box [ymin, xmin, ymax, xmax] on a 0-1000 scale.",
                    },
                    "target_box": {
                        "type": "array",
                        "items": {"type": "number"},
                        "minItems": 4,
                        "maxItems": 4,
                        "description": "Target blank box [ymin, xmin, ymax, xmax] on a 0-1000 scale.",
                    },
                    "label": {"type": "string"},
                    "filled": {"type": "boolean", "description": "True if the blank is already filled."},
                },
                "required": ["target_box"],
            },
        },
        # Backward-compatible path if the model still returns boxes.
        "boxes": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "box_2d": {
                        "type": "array",
                        "items": {"type": "number"},
                        "minItems": 4,
                        "maxItems": 4,
                    },
                    "bbox": {
                        "type": "array",
                        "items": {"type": "number"},
                        "minItems": 4,
                        "maxItems": 4,
                    },
                    "label": {"type": "string"},
                    "filled": {"type": "boolean"},
                },
            },
        },
        "coord_scale": {"type": "number"},
    },
    "required": ["items"],
}

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)
CV_SCAN_AVAILABLE = CV_SCAN_ENABLED and cv_tools is not None
if CV_SCAN_ENABLED and not CV_SCAN_AVAILABLE:
    logger.warning("CV scan enabled but OpenCV is unavailable; skipping vision correction.")

app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "dev-change-me")
app.config["MAX_CONTENT_LENGTH"] = UPLOAD_MAX_MB * 1024 * 1024

_cached_client = None


def allowed_file(filename: str) -> bool:
    return Path(filename).suffix.lower() in ALLOWED_EXTENSIONS


def create_job_dirs(job_id: str) -> tuple[Path, Path]:
    job_dir = JOBS_DIR / job_id
    job_static = JOB_STATIC_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    job_static.mkdir(parents=True, exist_ok=True)
    return job_dir, job_static


def render_pdf_to_images(pdf_path: Path, output_dir: Path) -> list[Path]:
    if fitz is None:
        raise RuntimeError("PyMuPDF is not installed. Install dependencies from requirements.txt.")

    doc = fitz.open(str(pdf_path))
    image_paths = []
    page_count = min(doc.page_count, MAX_PAGES)
    zoom = 150 / 72  # 150 DPI
    matrix = fitz.Matrix(zoom, zoom)

    for page_index in range(page_count):
        page = doc.load_page(page_index)
        pix = page.get_pixmap(matrix=matrix, alpha=False)
        image_path = output_dir / f"page_{page_index + 1}.png"
        pix.save(str(image_path))
        image_paths.append(image_path)

    doc.close()
    return image_paths


def ocrmypdf_available() -> bool:
    return shutil.which("ocrmypdf") is not None


def ocr_pdf_with_ocrmypdf(input_pdf: Path, output_pdf: Path) -> Path:
    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    if output_pdf.exists():
        output_pdf.unlink()

    command = ["ocrmypdf"]
    temp_config_paths: list[Path] = []
    if OCR_LANG:
        command.extend(["-l", OCR_LANG])
    if OCRMY_PDF_OUTPUT_TYPE:
        command.extend(["--output-type", OCRMY_PDF_OUTPUT_TYPE])
        if OCRMY_PDF_OUTPUT_TYPE.startswith("pdfa") and OCRMY_PDF_PDFA_IMAGE_COMPRESSION:
            command.extend(["--pdfa-image-compression", OCRMY_PDF_PDFA_IMAGE_COMPRESSION])
    if OCRMY_PDF_SKIP_TEXT:
        command.append("--skip-text")
    if OCRMY_PDF_ROTATE_PAGES:
        command.append("--rotate-pages")
    if OCRMY_PDF_DESKEW:
        command.append("--deskew")
    if OCRMY_PDF_CLEAN:
        command.append("--clean")
    if OCRMY_PDF_REMOVE_BACKGROUND:
        command.append("--remove-background")
    if OCRMY_PDF_REMOVE_VECTORS:
        command.append("--remove-vectors")
    if OCRMY_PDF_OVERSAMPLE:
        command.extend(["--oversample", OCRMY_PDF_OVERSAMPLE])
    if OCRMY_PDF_OPTIMIZE:
        command.extend(["--optimize", OCRMY_PDF_OPTIMIZE])

    if OCRMY_PDF_CHAR_WHITELIST:
        whitelist_cfg = output_pdf.parent / "tesseract_whitelist.cfg"
        whitelist_cfg.write_text(f"tessedit_char_whitelist {OCRMY_PDF_CHAR_WHITELIST}\n", encoding="utf-8")
        temp_config_paths.append(whitelist_cfg)
        command.extend(["--tesseract-config", str(whitelist_cfg)])
    if OCRMY_PDF_TESSERACT_PSM:
        command.extend(["--tesseract-pagesegmode", OCRMY_PDF_TESSERACT_PSM])
    if OCRMY_PDF_TESSERACT_OEM:
        command.extend(["--tesseract-oem", OCRMY_PDF_TESSERACT_OEM])
    if OCRMY_PDF_TESSERACT_THRESHOLDING:
        command.extend(["--tesseract-thresholding", OCRMY_PDF_TESSERACT_THRESHOLDING])
    if OCRMY_PDF_USER_WORDS:
        command.extend(["--user-words", OCRMY_PDF_USER_WORDS])
    if OCRMY_PDF_USER_PATTERNS:
        command.extend(["--user-patterns", OCRMY_PDF_USER_PATTERNS])
    if OCRMY_PDF_TESSERACT_CONFIG:
        command.extend(["--tesseract-config", OCRMY_PDF_TESSERACT_CONFIG])

    command.extend([str(input_pdf), str(output_pdf)])

    try:
        completed = subprocess.run(
            command,
            check=True,
            text=True,
            capture_output=True,
        )
        if OCRMY_PDF_LOG_OUTPUT:
            if completed.stdout.strip():
                logger.info("ocrmypdf: %s", clip_text(completed.stdout.strip(), 800))
            if completed.stderr.strip():
                logger.info("ocrmypdf (stderr): %s", clip_text(completed.stderr.strip(), 800))
    except FileNotFoundError:
        raise RuntimeError("ocrmypdf is not installed or not on PATH.")
    except subprocess.CalledProcessError as exc:
        details = ""
        if exc.stdout:
            details += f"\nstdout: {clip_text(exc.stdout.strip(), 1200)}"
        if exc.stderr:
            details += f"\nstderr: {clip_text(exc.stderr.strip(), 1200)}"
        raise RuntimeError(f"ocrmypdf failed (exit {exc.returncode}).{details}") from exc
    finally:
        for path in temp_config_paths:
            try:
                path.unlink()
            except FileNotFoundError:
                pass

    if not output_pdf.exists():
        raise RuntimeError("ocrmypdf completed but output PDF was not created.")
    return output_pdf


def prepare_ocr_words_for_debug(ocr_words: list[dict]) -> list[dict]:
    prepared = []
    for word in ocr_words or []:
        raw = str(word.get("text", "") or "")
        display = clean_anchor_text(raw)
        if not display:
            continue
        prepared.append(
            {
                "raw": raw[:80],
                "display": display[:60],
                "x0": float(word.get("x0", 0.0)),
                "y0": float(word.get("y0", 0.0)),
                "x1": float(word.get("x1", 0.0)),
                "y1": float(word.get("y1", 0.0)),
            }
        )
    return prepared


def refine_ocr_words_bbox_from_pdf(pdf_path: Path, page_number: int, ocr_words: list[dict]) -> list[dict]:
    if fitz is None or not ocr_words:
        return ocr_words

    doc = None
    try:
        doc = fitz.open(str(pdf_path))
        page = doc.load_page(max(page_number - 1, 0))
        rect = page.rect
        width = rect.width or 1.0
        height = rect.height or 1.0

        refined = []
        for word in ocr_words:
            raw = str(word.get("text", "") or "")
            cleaned = clean_anchor_text(raw)
            if (
                not cleaned
                or " " in cleaned
                or not any(not ch.isalnum() for ch in raw)
            ):
                refined.append(word)
                continue

            word_rect = fitz.Rect(
                rect.x0 + float(word.get("x0", 0.0)) * width,
                rect.y0 + float(word.get("y0", 0.0)) * height,
                rect.x0 + float(word.get("x1", 0.0)) * width,
                rect.y0 + float(word.get("y1", 0.0)) * height,
            )
            clip = fitz.Rect(
                word_rect.x0 - width * 0.01,
                word_rect.y0 - height * 0.01,
                word_rect.x1 + width * 0.01,
                word_rect.y1 + height * 0.01,
            ) & rect

            try:
                hits = page.search_for(cleaned, clip=clip) or []
            except Exception:
                hits = []
            if not hits:
                refined.append(word)
                continue

            best_hit = None
            best_area = None
            for hit in hits:
                area = (hit & word_rect).get_area()
                if best_area is None or area > best_area:
                    best_area = area
                    best_hit = hit
            if best_hit is None:
                refined.append(word)
                continue

            word2 = dict(word)
            word2["x0"] = clamp((best_hit.x0 - rect.x0) / width, 0.0, 1.0)
            word2["y0"] = clamp((best_hit.y0 - rect.y0) / height, 0.0, 1.0)
            word2["x1"] = clamp((best_hit.x1 - rect.x0) / width, 0.0, 1.0)
            word2["y1"] = clamp((best_hit.y1 - rect.y0) / height, 0.0, 1.0)
            refined.append(word2)

        return refined
    except Exception as exc:
        logger.warning("OCR bbox refinement failed: %s", exc)
        return ocr_words
    finally:
        if doc is not None:
            doc.close()


def select_ocr_words_for_debug(ocr_words: list[dict], boxes: list[dict]) -> list[dict]:
    if not ocr_words:
        return []
    if DEBUG_OCR_LAYER_ALL:
        return ocr_words
    if not boxes:
        return ocr_words

    answer_boxes = [box for box in boxes if box.get("type") == "answer"]
    if not answer_boxes:
        return ocr_words

    lines = build_ocr_lines(ocr_words)
    if not lines:
        return ocr_words

    selected = []
    seen = set()
    for line in lines:
        line_center_y = (line["y0"] + line["y1"]) / 2
        line_height = max(line["y1"] - line["y0"], 0.0)
        allowed_line_dy = min(ANCHOR_LINE_DY_MAX, max(ANCHOR_LINE_DY_MIN, line_height * ANCHOR_LINE_DY_MULT))
        keep_line = False
        for box in answer_boxes:
            box_center_y = float(box.get("y", 0.0)) + float(box.get("h", 0.0)) / 2
            if abs(line_center_y - box_center_y) <= allowed_line_dy:
                keep_line = True
                break
        if not keep_line:
            continue

        for word in line["words"]:
            key = (word.get("block"), word.get("line"), word.get("word"), word.get("text"))
            if key in seen:
                continue
            seen.add(key)
            selected.append(word)

    return selected or ocr_words


def maybe_ocr_pdf(input_pdf: Path, job_dir: Path) -> Path:
    if NO_OCR:
        return input_pdf
    if not OCRMY_PDF_ENABLED:
        return input_pdf

    if not ocrmypdf_available():
        message = "ocrmypdf is not installed. Install it (plus Tesseract/Ghostscript) or set OCRMY_PDF_ENABLED=0."
        if OCRMY_PDF_REQUIRED:
            raise RuntimeError(message)
        logger.warning(message)
        return input_pdf

    output_pdf = job_dir / "upload_ocr.pdf"
    try:
        return ocr_pdf_with_ocrmypdf(input_pdf, output_pdf)
    except Exception as exc:
        if OCRMY_PDF_REQUIRED:
            raise
        logger.warning("Skipping OCRmyPDF due to error: %s", exc)
        return input_pdf


def convert_image_to_png(image_path: Path, output_dir: Path) -> Path:
    image = Image.open(image_path).convert("RGB")
    output_path = output_dir / "page_1.png"
    image.save(output_path, format="PNG")
    return output_path


def prepare_image_for_llm(image: Image.Image) -> Image.Image:
    width, height = image.size
    max_dim = max(width, height)
    if max_dim <= MAX_IMAGE_DIM:
        return image

    scale = MAX_IMAGE_DIM / max_dim
    new_size = (max(1, int(width * scale)), max(1, int(height * scale)))
    return image.resize(new_size, Image.LANCZOS)


def build_http_options():
    if GEMINI_TIMEOUT_SEC <= 0:
        return None
    if types and hasattr(types, "HttpOptions"):
        for timeout_value in (int(GEMINI_TIMEOUT_SEC * 1000), GEMINI_TIMEOUT_SEC):
            try:
                return types.HttpOptions(timeout=timeout_value)
            except Exception:
                continue
    return {"timeout": GEMINI_TIMEOUT_SEC}


def get_client():
    global _cached_client
    if _cached_client is not None:
        return _cached_client

    if genai is None or types is None:
        raise RuntimeError("google-genai is not installed. Install dependencies from requirements.txt.")

    if not GEMINI_API_KEY:
        raise RuntimeError("GEMINI_API_KEY is missing. Add it to your .env file.")

    http_options = build_http_options()
    try:
        _cached_client = genai.Client(api_key=GEMINI_API_KEY, http_options=http_options)
    except TypeError:
        _cached_client = genai.Client(api_key=GEMINI_API_KEY)
    return _cached_client


def strip_code_fences(text: str) -> str:
    stripped = text.strip()
    match = re.match(r"^```[a-zA-Z0-9]*\s*(.*)\s*```$", stripped, re.S)
    if match:
        return match.group(1).strip()
    return stripped


def extract_json_fragment(text: str) -> str | None:
    def extract_balanced(text_value: str) -> str | None:
        start_idx = None
        stack = []
        in_string = False
        escaped = False

        for idx, ch in enumerate(text_value):
            if start_idx is None:
                if ch in "{[":
                    start_idx = idx
                    stack.append(ch)
                continue

            if in_string:
                if escaped:
                    escaped = False
                    continue
                if ch == "\\":
                    escaped = True
                    continue
                if ch == "\"":
                    in_string = False
                continue

            if ch == "\"":
                in_string = True
                continue
            if ch in "{[":
                stack.append(ch)
                continue
            if ch in "}]":
                if not stack:
                    return None
                opener = stack.pop()
                if opener == "{" and ch != "}":
                    return None
                if opener == "[" and ch != "]":
                    return None
                if not stack:
                    return text_value[start_idx : idx + 1]

        return None

    fragment = extract_balanced(text)
    if fragment:
        return fragment

    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return text[start : end + 1]
    start = text.find("[")
    end = text.rfind("]")
    if start != -1 and end != -1 and end > start:
        return text[start : end + 1]
    return None


def remove_trailing_commas(text: str) -> str:
    return re.sub(r",\s*([}\]])", r"\1", text)


def repair_common_json_mistakes(text: str) -> str:
    if not text:
        return text

    def escape_unescaped_quotes(value: str) -> str:
        out = []
        in_string = False
        escaped = False

        def next_non_space(index: int) -> str | None:
            for j in range(index, len(value)):
                ch = value[j]
                if ch.isspace():
                    continue
                return ch
            return None

        for idx, ch in enumerate(value):
            if not in_string:
                out.append(ch)
                if ch == "\"":
                    in_string = True
                    escaped = False
                continue

            if escaped:
                out.append(ch)
                escaped = False
                continue

            if ch == "\\":
                out.append(ch)
                escaped = True
                continue

            if ch == "\"":
                nxt = next_non_space(idx + 1)
                if nxt is None or nxt in {",", ":", "}", "]"}:
                    out.append(ch)
                    in_string = False
                    continue
                out.append("\\\"")
                continue

            out.append(ch)

        return "".join(out)

    repaired = text.replace("\ufeff", "")
    repaired = repaired.replace("“", "\"").replace("”", "\"")
    repaired = repaired.replace("‘", "'").replace("’", "'")
    repaired = remove_trailing_commas(repaired)
    repaired = re.sub(r"\}(\s*)\{", r"},\1{", repaired)
    repaired = re.sub(r"\](\s*)\[", r"],\1[", repaired)
    repaired = re.sub(r"\}(\s*)\[", r"},\1[", repaired)
    repaired = re.sub(r"\](\s*)\{", r"],\1{", repaired)
    repaired = re.sub(r'(\}|\]|"|[0-9])(\s+)("[^"]+"\s*:)', r"\1,\2\3", repaired)
    repaired = re.sub(r'\b(true|false|null)(\s+)("[^"]+"\s*:)', r"\1,\2\3", repaired)
    repaired = escape_unescaped_quotes(repaired)
    return repaired


def parse_json_response(text: str) -> dict:
    raw_candidates = []
    stripped = strip_code_fences(text)
    raw_candidates.append(stripped)
    fragment = extract_json_fragment(stripped)
    if fragment:
        raw_candidates.append(fragment)

    candidates = []
    for candidate in raw_candidates:
        if not candidate:
            continue
        candidates.append(candidate)
        candidates.append(remove_trailing_commas(candidate))
        candidates.append(repair_common_json_mistakes(candidate))

    seen = set()
    last_error = None
    for candidate in candidates:
        if not candidate:
            continue
        try:
            if candidate in seen:
                continue
            seen.add(candidate)
            data = json.loads(candidate)
            if isinstance(data, list):
                return {"boxes": data}
            return data
        except json.JSONDecodeError as exc:
            last_error = exc
            continue

    if last_error:
        raise last_error
    raise json.JSONDecodeError("No JSON content found", text, 0)


def safe_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def is_truthy(value) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    if isinstance(value, str):
        return value.strip().lower() in {"true", "yes", "1", "filled"}
    return False


def inset_box(x: float, y: float, w: float, h: float, ratio: float) -> tuple[float, float, float, float]:
    if ratio <= 0:
        return x, y, w, h

    dx = w * ratio
    dy = h * ratio
    new_w = w - 2 * dx
    new_h = h - 2 * dy
    if new_w <= 0 or new_h <= 0:
        return x, y, w, h
    return x + dx, y + dy, new_w, new_h


def clip_text(value: str, limit: int) -> str:
    if not value or limit <= 0:
        return value or ""
    if len(value) <= limit:
        return value
    return f"{value[:limit]}... (truncated)"


def get_attr_or_key(obj, key: str):
    if obj is None:
        return None
    if isinstance(obj, dict):
        return obj.get(key)
    return getattr(obj, key, None)


def summarize_safety_ratings(ratings) -> list[dict]:
    summary = []
    if not ratings:
        return summary
    for rating in ratings:
        summary.append(
            {
                "category": str(get_attr_or_key(rating, "category")),
                "probability": str(get_attr_or_key(rating, "probability")),
                "blocked": bool(get_attr_or_key(rating, "blocked")),
            }
        )
    return summary


def summarize_candidate(candidate) -> dict:
    info = {
        "finish_reason": str(get_attr_or_key(candidate, "finish_reason")),
        "safety_ratings": summarize_safety_ratings(get_attr_or_key(candidate, "safety_ratings")),
    }

    content = get_attr_or_key(candidate, "content")
    parts = get_attr_or_key(content, "parts") or []
    part_summaries = []
    for part in parts:
        text = get_attr_or_key(part, "text")
        if text:
            part_summaries.append({"text": clip_text(str(text), 400)})
            continue
        inline_data = get_attr_or_key(part, "inline_data")
        if inline_data:
            data = get_attr_or_key(inline_data, "data")
            data_len = len(data) if hasattr(data, "__len__") else None
            part_summaries.append(
                {
                    "inline_data": {
                        "mime_type": str(get_attr_or_key(inline_data, "mime_type")),
                        "data_len": data_len,
                    }
                }
            )
            continue
        part_summaries.append({"type": type(part).__name__})
    info["parts"] = part_summaries
    return info


def summarize_prompt_feedback(prompt_feedback) -> dict | None:
    if not prompt_feedback:
        return None
    return {
        "block_reason": str(get_attr_or_key(prompt_feedback, "block_reason")),
        "safety_ratings": summarize_safety_ratings(get_attr_or_key(prompt_feedback, "safety_ratings")),
    }


def summarize_usage(usage) -> dict | None:
    if not usage:
        return None
    return {
        "prompt_token_count": get_attr_or_key(usage, "prompt_token_count"),
        "candidates_token_count": get_attr_or_key(usage, "candidates_token_count"),
        "total_token_count": get_attr_or_key(usage, "total_token_count"),
    }


def summarize_response(response) -> dict:
    summary = {"type": type(response).__name__}
    summary["prompt_feedback"] = summarize_prompt_feedback(get_attr_or_key(response, "prompt_feedback"))
    candidates = get_attr_or_key(response, "candidates") or []
    summary["candidate_count"] = len(candidates)
    summary["candidates"] = [summarize_candidate(candidate) for candidate in candidates]
    summary["usage_metadata"] = summarize_usage(get_attr_or_key(response, "usage_metadata"))
    return summary


def build_safety_settings():
    if not GEMINI_SAFETY_BLOCK_ONLY_HIGH or types is None:
        return None
    categories = [
        "HARM_CATEGORY_HARASSMENT",
        "HARM_CATEGORY_HATE_SPEECH",
        "HARM_CATEGORY_SEXUALLY_EXPLICIT",
        "HARM_CATEGORY_DANGEROUS_CONTENT",
    ]
    settings = []
    for category in categories:
        try:
            settings.append(types.SafetySetting(category=category, threshold="BLOCK_ONLY_HIGH"))
        except Exception:
            continue
    return settings or None


def build_generation_config():
    config_kwargs = {
        "temperature": GEMINI_TEMPERATURE,
        "top_p": GEMINI_TOP_P,
        "max_output_tokens": GEMINI_MAX_OUTPUT_TOKENS,
        "response_mime_type": "application/json",
        "response_schema": RESPONSE_SCHEMA,
    }

    safety_settings = build_safety_settings()
    if safety_settings:
        config_kwargs["safety_settings"] = safety_settings

    if GEMINI_THINKING_BUDGET >= 0 and types is not None and hasattr(types, "ThinkingConfig"):
        try:
            config_kwargs["thinking_config"] = types.ThinkingConfig(thinking_budget=GEMINI_THINKING_BUDGET)
        except Exception:
            pass

    try:
        return types.GenerateContentConfig(**config_kwargs)
    except Exception:
        return config_kwargs


def call_gemini_with_retries(client, contents, config):
    attempts = max(GEMINI_MAX_RETRIES, 0) + 1
    last_exc = None
    for attempt in range(attempts):
        try:
            return client.models.generate_content(model=GEMINI_MODEL, contents=contents, config=config)
        except Exception as exc:
            last_exc = exc
            if attempt >= attempts - 1:
                break
            backoff = GEMINI_RETRY_BACKOFF_SEC * (2**attempt)
            logger.warning(
                "Gemini request failed (model=%s, attempt %s/%s): %s",
                GEMINI_MODEL,
                attempt + 1,
                attempts,
                exc,
            )
            time.sleep(backoff)

    if GEMINI_FALLBACK_MODEL:
        for attempt in range(attempts):
            try:
                return client.models.generate_content(model=GEMINI_FALLBACK_MODEL, contents=contents, config=config)
            except Exception as exc:
                last_exc = exc
                if attempt >= attempts - 1:
                    break
                backoff = GEMINI_RETRY_BACKOFF_SEC * (2**attempt)
                logger.warning(
                    "Gemini fallback request failed (model=%s, attempt %s/%s): %s",
                    GEMINI_FALLBACK_MODEL,
                    attempt + 1,
                    attempts,
                    exc,
                )
                time.sleep(backoff)

    raise last_exc


def normalize_box_values(
    x: float,
    y: float,
    w: float,
    h: float,
    width: int | None,
    height: int | None,
    scale_override: float | None = None,
) -> tuple[float, float, float, float]:
    max_val = max(x, y, w, h)
    if scale_override and scale_override > 0 and max_val > 1.0:
        return x / scale_override, y / scale_override, w / scale_override, h / scale_override
    if max_val <= 1.0:
        return x, y, w, h

    if not width or not height:
        return x, y, w, h

    mode = BOX_COORDINATE_SCALE
    if mode in {"px", "pixel", "pixels"}:
        return x / width, y / height, w / width, h / height
    if mode in {"1000", "norm1000"}:
        if max_val <= 1.0:
            return x, y, w, h
        return x / 1000.0, y / 1000.0, w / 1000.0, h / 1000.0

    if max_val <= 1000:
        return x / 1000.0, y / 1000.0, w / 1000.0, h / 1000.0

    return x / width, y / height, w / width, h / height


def sanitize_boxes(raw_boxes, page_num: int, image_size: tuple[int, int] | None = None) -> list[dict]:
    cleaned = []
    if not isinstance(raw_boxes, list):
        return cleaned

    width = height = None
    if image_size:
        width, height = image_size

    def extract_box_coords(candidate) -> tuple[float, float, float, float] | None:
        if not isinstance(candidate, (list, tuple)) or len(candidate) != 4:
            return None
        y1 = safe_float(candidate[0])
        x1 = safe_float(candidate[1])
        y2 = safe_float(candidate[2])
        x2 = safe_float(candidate[3])
        if None in (x1, y1, x2, y2):
            return None
        if x2 <= x1 or y2 <= y1:
            return None
        return x1, y1, x2 - x1, y2 - y1

    for index, raw in enumerate(raw_boxes, start=1):
        if not isinstance(raw, dict):
            continue
        if is_truthy(raw.get("filled")) or is_truthy(raw.get("has_text")):
            continue

        x = safe_float(raw.get("x"))
        y = safe_float(raw.get("y"))
        w = safe_float(raw.get("w"))
        h = safe_float(raw.get("h"))
        coord_scale = safe_float(raw.get("coord_scale") or raw.get("coordinates_scale"))

        anchor_coords = extract_box_coords(raw.get("anchor_box"))

        if None in (x, y, w, h):
            target_candidate = (
                raw.get("target_box")
                or raw.get("box_2d")
                or raw.get("bbox")
                or raw.get("box")
                or raw.get("coordinates")
            )
            target_coords = extract_box_coords(target_candidate)
            if target_coords is not None:
                x, y, w, h = target_coords

        if None in (x, y, w, h):
            continue

        x, y, w, h = normalize_box_values(x, y, w, h, width, height, coord_scale)

        x = min(max(x, 0.0), 1.0)
        y = min(max(y, 0.0), 1.0)
        w = min(max(w, 0.0), 1.0 - x)
        h = min(max(h, 0.0), 1.0 - y)
        if w <= 0 or h <= 0:
            continue

        box_type = str(raw.get("type", "answer")).strip().lower()
        if box_type not in {"answer", "exercise"}:
            box_type = "answer"
        label = str(raw.get("label", "")).strip()[:80]
        anchor_raw = str(raw.get("anchor", raw.get("anchor_text", raw.get("after", "")))).strip()
        anchor = clean_anchor_text(anchor_raw)
        if anchor:
            anchor_lower = anchor.lower()
            if anchor_lower in {"none", "null", "nil", "na", "n/a"}:
                anchor = ""
            elif len(anchor.split()) != 1:
                anchor = ""
            else:
                anchor = anchor[:40]

        if box_type == "answer":
            x, y, w, h = inset_box(x, y, w, h, BOX_INSET_RATIO)
            if w <= 0 or h <= 0:
                continue

        # If the model returned an anchor box, enforce that the target starts after it.
        if anchor_coords is not None and width:
            ax, ay, aw, ah = anchor_coords
            ax, ay, aw, ah = normalize_box_values(ax, ay, aw, ah, width, height, coord_scale)
            ax = min(max(ax, 0.0), 1.0)
            ay = min(max(ay, 0.0), 1.0)
            aw = min(max(aw, 0.0), 1.0 - ax)
            ah = min(max(ah, 0.0), 1.0 - ay)
            if aw > 0 and ah > 0:
                # Vertical alignment: match anchor cap-height and baseline.
                anchor_bottom = min(max(ay + ah, 0.0), 1.0)
                target_h = min(max(ah, 0.0), anchor_bottom)
                if target_h > 0:
                    h = min(target_h, anchor_bottom)
                    y = clamp(anchor_bottom - h, 0.0, 1.0 - h)

                anchor_xmax = min(max(ax + aw, 0.0), 1.0)
                padding = max(0.0, ANCHOR_TARGET_PADDING_RATIO)
                x2 = x + w
                min_width_norm = max(MIN_BOX_SIZE_PX / width, 0.001)
                if x2 > min_width_norm:
                    max_x = max(0.0, x2 - min_width_norm)
                    new_x = max(x, anchor_xmax + padding)
                    x = min(max(new_x, 0.0), max_x)
                    w = x2 - x
                    if w <= 0:
                        continue

        cleaned.append(
            {
                "id": f"p{page_num}_b{index}",
                "type": box_type,
                "x": x,
                "y": y,
                "w": w,
                "h": h,
                "label": label,
                "anchor": anchor,
            }
        )

    return cleaned


def clamp(value: float, min_value: float, max_value: float) -> float:
    return max(min_value, min(value, max_value))


def normalize_word_token(value: str) -> list[str]:
    if not value:
        return []
    lower = value.lower()
    stripped = re.sub(r"^[^a-z0-9]+|[^a-z0-9]+$", "", lower)
    alnum = re.sub(r"[^a-z0-9]+", "", lower)
    variants = [lower]
    if stripped and stripped != lower:
        variants.append(stripped)
    if alnum and alnum not in variants:
        variants.append(alnum)
    return variants


def clean_anchor_text(value: str) -> str:
    if not value:
        return ""
    cleaned = "".join(ch if ch.isalnum() else " " for ch in value)
    return re.sub(r"\s+", " ", cleaned).strip()


def extract_text_words_from_image(image: Image.Image) -> list[dict]:
    if fitz is None:
        return []
    doc = None
    try:
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        img_bytes = buffer.getvalue()
        doc = fitz.open()
        page = doc.new_page(width=image.width, height=image.height)
        page.insert_image(fitz.Rect(0, 0, image.width, image.height), stream=img_bytes)
        tessdata = OCR_TESSDATA_DIR or None
        try:
            textpage = page.get_textpage_ocr(language=OCR_LANG, tessdata=tessdata)
        except TypeError:
            textpage = page.get_textpage_ocr(tessdata=tessdata)
        words = textpage.extractWORDS()
        normalized = []
        for x0, y0, x1, y1, text, block_no, line_no, word_no in words:
            if not text:
                continue
            if not any(ch.isalnum() for ch in text):
                continue
            if OCR_MIN_WORD_HEIGHT_PX > 0 and (y1 - y0) < OCR_MIN_WORD_HEIGHT_PX:
                continue
            normalized.append(
                {
                    "text": text,
                    "x0": clamp(x0 / image.width, 0.0, 1.0),
                    "y0": clamp(y0 / image.height, 0.0, 1.0),
                    "x1": clamp(x1 / image.width, 0.0, 1.0),
                    "y1": clamp(y1 / image.height, 0.0, 1.0),
                    "block": int(block_no),
                    "line": int(line_no),
                    "word": int(word_no),
                }
            )
        return normalized
    except Exception as exc:
        logger.warning("OCR word extraction failed: %s", exc)
        return []
    finally:
        if doc is not None:
            doc.close()


def extract_text_words_from_pdf_page(pdf_path: Path, page_number: int) -> list[dict]:
    if fitz is None:
        return []
    doc = None
    try:
        doc = fitz.open(str(pdf_path))
        page_index = max(page_number - 1, 0)
        page = doc.load_page(page_index)
        words = page.get_text("words") or []
        if not words:
            return []
        rect = page.rect
        width = rect.width or 1.0
        height = rect.height or 1.0
        zoom = 150.0 / 72.0  # keep in sync with render_pdf_to_images()
        normalized = []
        for x0, y0, x1, y1, text, block_no, line_no, word_no in words:
            if not text:
                continue
            if not any(ch.isalnum() for ch in text):
                continue
            if OCR_MIN_WORD_HEIGHT_PX > 0 and ((y1 - y0) * zoom) < OCR_MIN_WORD_HEIGHT_PX:
                continue
            normalized.append(
                {
                    "text": text,
                    "x0": clamp((x0 - rect.x0) / width, 0.0, 1.0),
                    "y0": clamp((y0 - rect.y0) / height, 0.0, 1.0),
                    "x1": clamp((x1 - rect.x0) / width, 0.0, 1.0),
                    "y1": clamp((y1 - rect.y0) / height, 0.0, 1.0),
                    "block": int(block_no),
                    "line": int(line_no),
                    "word": int(word_no),
                }
            )
        return normalized
    except Exception as exc:
        logger.warning("PDF word extraction failed: %s", exc)
        return []
    finally:
        if doc is not None:
            doc.close()


def refine_anchor_word_end_x(page, anchor_token: str, anchor_word: dict, box: dict, clip_margin: float = 0.01) -> float | None:
    if fitz is None or page is None:
        return None
    token = anchor_token.strip()
    if not token:
        return None

    rect = page.rect
    width = rect.width or 1.0
    height = rect.height or 1.0

    word_rect = fitz.Rect(
        rect.x0 + anchor_word["x0"] * width,
        rect.y0 + anchor_word["y0"] * height,
        rect.x0 + anchor_word["x1"] * width,
        rect.y0 + anchor_word["y1"] * height,
    )
    margin_x = width * clip_margin
    margin_y = height * clip_margin
    clip = fitz.Rect(
        word_rect.x0 - margin_x,
        word_rect.y0 - margin_y,
        word_rect.x1 + margin_x,
        word_rect.y1 + margin_y,
    )
    clip = clip & rect

    try:
        hits = page.search_for(token, clip=clip) or []
        if not hits:
            hits = page.search_for(token) or []
    except Exception:
        return None

    if not hits:
        return None

    hits_in_word = [hit for hit in hits if (hit & word_rect).get_area() > 0]
    if hits_in_word:
        hits = hits_in_word

    best_x1 = None
    best_dx = None
    for hit in hits:
        x1_norm = clamp((hit.x1 - rect.x0) / width, 0.0, 1.0)
        target_x = clamp(x1_norm + ANCHOR_GAP, 0.0, 1.0 - box["w"])
        dx = abs(target_x - box["x"])
        if best_dx is None or dx < best_dx:
            best_dx = dx
            best_x1 = x1_norm

    return best_x1


def build_ocr_lines(ocr_words: list[dict]) -> list[dict]:
    lines = {}
    for word in ocr_words:
        key = (word["block"], word["line"])
        line = lines.setdefault(
            key,
            {
                "words": [],
                "x0": 1.0,
                "y0": 1.0,
                "x1": 0.0,
                "y1": 0.0,
            },
        )
        line["words"].append(word)
        line["x0"] = min(line["x0"], word["x0"])
        line["y0"] = min(line["y0"], word["y0"])
        line["x1"] = max(line["x1"], word["x1"])
        line["y1"] = max(line["y1"], word["y1"])

    for line in lines.values():
        line["words"].sort(key=lambda item: (item["word"], item["x0"]))
        y1_values = sorted(word["y1"] for word in line["words"])
        line["baseline"] = y1_values[len(y1_values) // 2]
    return list(lines.values())

def token_similarity(word_text: str, anchor_token: str) -> float:
    word_variants = normalize_word_token(word_text)
    anchor_variants = normalize_word_token(anchor_token)
    if not word_variants or not anchor_variants:
        if word_text.strip().lower() == anchor_token.strip().lower():
            return 1.0
        return 0.0
    best = 0.0
    for word in word_variants:
        for anchor in anchor_variants:
            if word == anchor:
                return 1.0
            ratio = SequenceMatcher(None, word, anchor).ratio()
            if ratio > best:
                best = ratio
    return best


def find_anchor_match(anchor_text: str, box: dict, lines: list[dict]) -> tuple[dict, dict] | None:
    anchor_tokens = [token for token in anchor_text.split() if token]
    if not anchor_tokens:
        return None

    best = None
    best_score = None
    box_x = box["x"]
    box_y = box["y"]
    box_w = box["w"]
    box_h = box["h"]
    box_center_y = box_y + box_h / 2
    for line in lines:
        line_center_y = (line["y0"] + line["y1"]) / 2
        line_height = max(line["y1"] - line["y0"], 0.0)
        allowed_line_dy = min(ANCHOR_LINE_DY_MAX, max(ANCHOR_LINE_DY_MIN, line_height * ANCHOR_LINE_DY_MULT))
        if abs(line_center_y - box_center_y) > allowed_line_dy:
            continue
        words = line["words"]
        if len(words) < len(anchor_tokens):
            continue
        for idx in range(len(words) - len(anchor_tokens) + 1):
            similarities = []
            for offset in range(len(anchor_tokens)):
                similarity = token_similarity(words[idx + offset]["text"], anchor_tokens[offset])
                if similarity < ANCHOR_FUZZY_RATIO:
                    similarities = []
                    break
                similarities.append(similarity)
            if not similarities:
                continue
            anchor_word = words[idx + len(anchor_tokens) - 1]
            candidate_x = anchor_word["x1"] + ANCHOR_GAP
            target_x = clamp(candidate_x, 0.0, 1.0 - box_w)
            target_y = clamp(line["baseline"] - box_h, 0.0, 1.0 - box_h)
            if ANCHOR_MAX_SHIFT_X > 0 and abs(target_x - box_x) > ANCHOR_MAX_SHIFT_X:
                continue
            if ANCHOR_MAX_SHIFT_Y > 0 and abs(target_y - box_y) > ANCHOR_MAX_SHIFT_Y:
                continue
            dx = target_x - box_x
            dy = target_y - box_y
            overlap = candidate_x - box_x
            overlap_penalty = 0.0
            if overlap > ANCHOR_RIGHT_TOLERANCE:
                overlap_penalty = (overlap - ANCHOR_RIGHT_TOLERANCE) * 6
            similarity = sum(similarities) / len(similarities)
            score = abs(dx) + abs(dy) * 2 + overlap_penalty + (1.0 - similarity) * 0.35
            if best_score is None or score < best_score:
                best_score = score
                best = (line, anchor_word)
    return best


def align_boxes_to_anchor_words(
    boxes: list[dict],
    ocr_words: list[dict],
    pdf_path: Path | None = None,
    page_number: int | None = None,
) -> list[dict]:
    if not boxes or not ocr_words:
        return boxes

    doc = None
    page = None
    if pdf_path is not None and page_number is not None and fitz is not None:
        try:
            doc = fitz.open(str(pdf_path))
            page = doc.load_page(max(page_number - 1, 0))
        except Exception as exc:
            logger.warning("PDF anchor refinement unavailable: %s", exc)
            page = None

    lines = build_ocr_lines(ocr_words)
    if not lines:
        return boxes

    try:
        for box in boxes:
            if box.get("type") != "answer":
                continue
            anchor_text = clean_anchor_text(str(box.get("anchor", "")).strip())
            if not anchor_text:
                continue

            match = find_anchor_match(anchor_text, box, lines)
            if not match:
                continue
            line, anchor_word = match

            anchor_end_x = anchor_word["x1"]
            if page is not None:
                anchor_tokens = [token for token in anchor_text.split() if token]
                if anchor_tokens:
                    last_token = anchor_tokens[-1]
                    word_text = str(anchor_word.get("text", "") or "")
                    word_variants = normalize_word_token(word_text)
                    needs_refine = any(not ch.isalnum() for ch in word_text) or (
                        last_token.lower() not in word_variants
                    )
                    if needs_refine:
                        refined = refine_anchor_word_end_x(page, last_token, anchor_word, box)
                        if refined is not None:
                            anchor_end_x = refined

            target_x = clamp(anchor_end_x + ANCHOR_GAP, 0.0, 1.0 - box["w"])
            target_y = clamp(line["baseline"] - box["h"], 0.0, 1.0 - box["h"])
            if ANCHOR_MAX_SHIFT_X > 0 and abs(target_x - box["x"]) > ANCHOR_MAX_SHIFT_X:
                continue
            if ANCHOR_MAX_SHIFT_Y > 0 and abs(target_y - box["y"]) > ANCHOR_MAX_SHIFT_Y:
                continue

            box["x"] = target_x
            box["y"] = target_y

        return boxes
    finally:
        if doc is not None:
            doc.close()


def generate_boxes(image: Image.Image) -> dict:
    client = get_client()
    prompt = PROMPT_TEMPLATE
    config = build_generation_config()
    contents = [image, prompt]

    if GEMINI_DEBUG:
        logger.info(
            "Gemini request: model=%s prompt_chars=%s image_size=%sx%s config=%s",
            GEMINI_MODEL,
            len(prompt),
            image.size[0],
            image.size[1],
            config,
        )

    response = call_gemini_with_retries(client, contents, config)

    if GEMINI_DEBUG:
        response_summary = summarize_response(response)
        logger.info("Gemini response summary: %s", json.dumps(response_summary, ensure_ascii=True))
        raw_json = None
        if hasattr(response, "model_dump"):
            try:
                raw_json = json.dumps(response.model_dump(), ensure_ascii=True)
            except Exception:
                raw_json = None
        if raw_json is None and hasattr(response, "to_dict"):
            try:
                raw_json = json.dumps(response.to_dict(), ensure_ascii=True)
            except Exception:
                raw_json = None
        if raw_json is None and hasattr(response, "to_json"):
            try:
                raw_json = response.to_json()
            except Exception:
                raw_json = None
        if raw_json:
            logger.info("Gemini response raw: %s", clip_text(raw_json, GEMINI_LOG_MAX_CHARS))

    try:
        text = response.text or ""
    except Exception as exc:
        logger.error("Gemini response missing text: %s", exc)
        logger.error("Gemini response summary: %s", json.dumps(summarize_response(response), ensure_ascii=True))
        raw_json = None
        if hasattr(response, "model_dump"):
            try:
                raw_json = json.dumps(response.model_dump(), ensure_ascii=True)
            except Exception:
                raw_json = None
        if raw_json is None and hasattr(response, "to_dict"):
            try:
                raw_json = json.dumps(response.to_dict(), ensure_ascii=True)
            except Exception:
                raw_json = None
        if raw_json is None and hasattr(response, "to_json"):
            try:
                raw_json = response.to_json()
            except Exception:
                raw_json = None
        if raw_json:
            logger.error("Gemini response raw: %s", clip_text(raw_json, GEMINI_LOG_MAX_CHARS))
        raise

    if not text.strip():
        logger.error("Gemini returned empty text response.")
        logger.error("Gemini response summary: %s", json.dumps(summarize_response(response), ensure_ascii=True))
        raise ValueError("Gemini returned empty response; see logs for safety ratings.")

    return parse_json_response(text)


def looks_filled_box(image: Image.Image, box: dict) -> bool:
    if not FILTER_FILLED_BOXES:
        return False
    if box.get("type") != "answer":
        return False

    width, height = image.size
    left = max(0, int(box["x"] * width))
    top = max(0, int(box["y"] * height))
    right = min(width, int((box["x"] + box["w"]) * width))
    bottom = min(height, int((box["y"] + box["h"]) * height))

    if right <= left or bottom <= top:
        return False

    crop = image.crop((left, top, right, bottom))
    if crop.width < MIN_BOX_SIZE_PX or crop.height < MIN_BOX_SIZE_PX:
        return False
    if crop.width * crop.height < MIN_BOX_AREA_PX:
        return False

    target_w = max(10, int(crop.width * INK_SAMPLE_SCALE))
    target_h = max(10, int(crop.height * INK_SAMPLE_SCALE))
    if target_w < crop.width or target_h < crop.height:
        crop = crop.resize((target_w, target_h), Image.BILINEAR)

    gray = crop.convert("L")
    w, h = gray.size
    data = list(gray.getdata())
    dark_counts = [0] * h
    dark_total = 0

    for idx, value in enumerate(data):
        if value < INK_THRESHOLD:
            dark_total += 1
            dark_counts[idx // w] += 1

    if dark_total == 0:
        return False

    dark_ratio = dark_total / (w * h)
    ink_rows = sum(1 for count in dark_counts if count / w >= INK_ROW_FRACTION)
    ink_row_ratio = ink_rows / h

    return dark_ratio >= INK_DARK_RATIO_MAX and ink_row_ratio >= INK_ROW_RATIO_MAX


def filter_filled_boxes(image: Image.Image, boxes: list[dict]) -> list[dict]:
    if not boxes:
        return boxes

    filtered = []
    for box in boxes:
        if looks_filled_box(image, box):
            continue
        filtered.append(box)
    return filtered


def process_page(
    job_id: str,
    page_index: int,
    image_path: Path,
    pdf_path: Path | None = None,
    cv_debug: bool = False,
) -> dict:
    error = ""
    boxes = []
    ocr_debug = []
    ocr_debug_total = 0
    with Image.open(image_path) as image:
        image = image.convert("RGB")
        width, height = image.size
        llm_image = prepare_image_for_llm(image)
        llm_size = llm_image.size
        try:
            raw_response = generate_boxes(llm_image)
            raw_items = raw_response.get("items")
            if not isinstance(raw_items, list):
                raw_items = raw_response.get("boxes", [])
            boxes = sanitize_boxes(raw_items, page_index, llm_size)
            if CV_SCAN_AVAILABLE and boxes:
                try:
                    boxes = cv_tools.fix_box_overlaps_with_vision(
                        image_path,
                        boxes,
                        search_width_px=CV_SEARCH_WIDTH_PX,
                        padding_px=CV_PADDING_PX,
                        min_width_px=CV_MIN_WIDTH_PX,
                        strip_half_height_px=CV_STRIP_HALF_HEIGHT_PX,
                        threshold=CV_THRESHOLD,
                        scan_top_ratio=CV_SCAN_TOP_RATIO,
                        scan_bottom_ratio=CV_SCAN_BOTTOM_RATIO,
                        edge_scan_px=CV_EDGE_SCAN_PX,
                        underline_enabled=CV_UNDERLINE_ENABLED,
                        underline_band_top_ratio=CV_UNDERLINE_BAND_TOP_RATIO,
                        underline_extra_bottom_px=CV_UNDERLINE_EXTRA_BOTTOM_PX,
                        underline_margin_x_px=CV_UNDERLINE_MARGIN_X_PX,
                        underline_row_ratio_min=CV_UNDERLINE_ROW_RATIO_MIN,
                        underline_min_row_pixels=CV_UNDERLINE_MIN_ROW_PIXELS,
                        underline_line_coverage_min=CV_UNDERLINE_LINE_COVERAGE_MIN,
                        underline_segments_min=CV_UNDERLINE_SEGMENTS_MIN,
                        underline_gap_cv_max=CV_UNDERLINE_GAP_CV_MAX,
                        underline_below_scan_px=CV_UNDERLINE_BELOW_SCAN_PX,
                        underline_below_margin_top_px=CV_UNDERLINE_BELOW_MARGIN_TOP_PX,
                        debug=cv_debug,
                        debug_pass_name="pre_filter",
                    )
                except Exception as exc:
                    logger.warning("Vision correction (pre-filter) failed on page %s: %s", page_index, exc)
            boxes = filter_filled_boxes(image, boxes)
            if not NO_OCR and any(
                clean_anchor_text(str(box.get("anchor", "")).strip())
                for box in boxes
                if box.get("type") == "answer"
            ):
                ocr_words_for_alignment = []
                used_pdf_words = False
                if pdf_path is not None:
                    ocr_words_for_alignment = extract_text_words_from_pdf_page(pdf_path, page_index)
                    used_pdf_words = bool(ocr_words_for_alignment)
                if not ocr_words_for_alignment:
                    ocr_words_for_alignment = extract_text_words_from_image(image)
                if ocr_words_for_alignment:
                    if used_pdf_words:
                        boxes = align_boxes_to_anchor_words(
                            boxes, ocr_words_for_alignment, pdf_path=pdf_path, page_number=page_index
                        )
                    else:
                        boxes = align_boxes_to_anchor_words(boxes, ocr_words_for_alignment)
            if CV_SCAN_AVAILABLE and boxes:
                try:
                    boxes = cv_tools.fix_box_overlaps_with_vision(
                        image_path,
                        boxes,
                        search_width_px=CV_SEARCH_WIDTH_PX,
                        padding_px=CV_PADDING_PX,
                        min_width_px=CV_MIN_WIDTH_PX,
                        strip_half_height_px=CV_STRIP_HALF_HEIGHT_PX,
                        threshold=CV_THRESHOLD,
                        scan_top_ratio=CV_SCAN_TOP_RATIO,
                        scan_bottom_ratio=CV_SCAN_BOTTOM_RATIO,
                        edge_scan_px=CV_EDGE_SCAN_PX,
                        underline_enabled=CV_UNDERLINE_ENABLED,
                        underline_band_top_ratio=CV_UNDERLINE_BAND_TOP_RATIO,
                        underline_extra_bottom_px=CV_UNDERLINE_EXTRA_BOTTOM_PX,
                        underline_margin_x_px=CV_UNDERLINE_MARGIN_X_PX,
                        underline_row_ratio_min=CV_UNDERLINE_ROW_RATIO_MIN,
                        underline_min_row_pixels=CV_UNDERLINE_MIN_ROW_PIXELS,
                        underline_line_coverage_min=CV_UNDERLINE_LINE_COVERAGE_MIN,
                        underline_segments_min=CV_UNDERLINE_SEGMENTS_MIN,
                        underline_gap_cv_max=CV_UNDERLINE_GAP_CV_MAX,
                        underline_below_scan_px=CV_UNDERLINE_BELOW_SCAN_PX,
                        underline_below_margin_top_px=CV_UNDERLINE_BELOW_MARGIN_TOP_PX,
                        debug=cv_debug,
                        debug_pass_name="post_align",
                    )
                except Exception as exc:
                    logger.warning("Vision correction (post-align) failed on page %s: %s", page_index, exc)

            if DEBUG_OCR_LAYER and not NO_OCR:
                ocr_words_debug = []
                debug_used_pdf = False
                if pdf_path is not None:
                    ocr_words_debug = extract_text_words_from_pdf_page(pdf_path, page_index)
                    debug_used_pdf = bool(ocr_words_debug)
                if not ocr_words_debug:
                    ocr_words_debug = extract_text_words_from_image(image)
                ocr_debug_total = len(ocr_words_debug)
                ocr_debug_selected = select_ocr_words_for_debug(ocr_words_debug, boxes)
                if debug_used_pdf and pdf_path is not None:
                    ocr_debug_selected = refine_ocr_words_bbox_from_pdf(pdf_path, page_index, ocr_debug_selected)
                ocr_debug = prepare_ocr_words_for_debug(ocr_debug_selected)
        except Exception as exc:
            logger.exception("Gemini failed on page %s", page_index)
            error = str(exc)

    cv_debug_step_count = 0
    cv_debug_step_labels: list[str] = []
    if cv_debug:
        debug_steps_lists = [box.get("cv_debug_steps") for box in boxes if box.get("cv_debug_steps")]
        if debug_steps_lists:
            cv_debug_step_count = max(len(steps) for steps in debug_steps_lists)
            reference_steps = max(debug_steps_lists, key=len)
            for idx in range(cv_debug_step_count):
                if idx < len(reference_steps):
                    label = str(reference_steps[idx].get("step", f"step_{idx + 1}"))
                else:
                    label = f"step_{idx + 1}"
                cv_debug_step_labels.append(label)

    return {
        "page": page_index,
        "image": f"jobs/{job_id}/{image_path.name}",
        "width": width,
        "height": height,
        "boxes": boxes,
        "ocr_words": ocr_debug,
        "ocr_words_total": ocr_debug_total,
        "cv_debug": cv_debug,
        "cv_debug_step_count": cv_debug_step_count,
        "cv_debug_step_labels": cv_debug_step_labels,
        "error": error,
    }


def process_images(
    job_id: str,
    image_paths: list[Path],
    pdf_path: Path | None = None,
    cv_debug: bool = False,
) -> dict:
    pages = []

    if MAX_CONCURRENT_PAGES > 1 and len(image_paths) > 1:
        worker_count = min(MAX_CONCURRENT_PAGES, len(image_paths))
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            futures = []
            for page_index, image_path in enumerate(image_paths, start=1):
                futures.append(
                    executor.submit(
                        process_page,
                        job_id,
                        page_index,
                        image_path,
                        pdf_path,
                        cv_debug=cv_debug,
                    )
                )
            for future in as_completed(futures):
                pages.append(future.result())
        pages.sort(key=lambda page: page["page"])
    else:
        for page_index, image_path in enumerate(image_paths, start=1):
            pages.append(process_page(job_id, page_index, image_path, pdf_path, cv_debug=cv_debug))

    return {
        "job_id": job_id,
        "model": GEMINI_MODEL,
        "debug_ocr": DEBUG_OCR_LAYER and not NO_OCR,
        "debug_cv": cv_debug,
        "pages": pages,
    }


def save_result(job_dir: Path, result: dict) -> None:
    result_path = job_dir / "result.json"
    result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")


def load_result(job_id: str) -> dict:
    result_path = JOBS_DIR / job_id / "result.json"
    if not result_path.exists():
        raise FileNotFoundError
    return json.loads(result_path.read_text(encoding="utf-8"))


def resolve_cv_debug_request() -> bool:
    """Resolve CV debug mode from request args/form, falling back to env."""
    requested = request.args.get("cv_debug") or request.form.get("cv_debug")
    if requested is None or str(requested).strip() == "":
        return CV_DEBUG
    return is_truthy(requested)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload():
    if not GEMINI_API_KEY:
        flash("Missing GEMINI_API_KEY in .env.")
        return redirect(url_for("index"))

    file = request.files.get("file")
    if not file or not file.filename:
        flash("Please choose a PDF or image to upload.")
        return redirect(url_for("index"))

    if not allowed_file(file.filename):
        flash("Unsupported file type. Use PDF or image.")
        return redirect(url_for("index"))

    job_id = uuid.uuid4().hex[:10]
    job_dir, job_static = create_job_dirs(job_id)
    cv_debug = resolve_cv_debug_request()

    ext = Path(file.filename).suffix.lower()
    upload_path = job_dir / f"upload{ext}"
    file.save(upload_path)

    pdf_for_anchors = None
    try:
        if ext == ".pdf":
            pdf_for_render = maybe_ocr_pdf(upload_path, job_dir)
            if not NO_OCR:
                pdf_for_anchors = pdf_for_render
            image_paths = render_pdf_to_images(pdf_for_render, job_static)
        else:
            image_paths = [convert_image_to_png(upload_path, job_static)]
    except Exception as exc:
        flash(f"Failed to process file: {exc}")
        return redirect(url_for("index"))

    result = process_images(job_id, image_paths, pdf_path=pdf_for_anchors, cv_debug=cv_debug)
    save_result(job_dir, result)
    return redirect(url_for("result", job_id=job_id))


@app.route("/paste", methods=["POST"])
def paste():
    if not GEMINI_API_KEY:
        flash("Missing GEMINI_API_KEY in .env.")
        return redirect(url_for("index"))

    file = request.files.get("image")
    if not file or not file.filename:
        flash("Paste an image into the clipboard box first.")
        return redirect(url_for("index"))

    if not allowed_file(file.filename):
        flash("Unsupported image format from clipboard.")
        return redirect(url_for("index"))

    job_id = uuid.uuid4().hex[:10]
    job_dir, job_static = create_job_dirs(job_id)
    cv_debug = resolve_cv_debug_request()

    upload_path = job_dir / "clipboard.png"
    file.save(upload_path)

    try:
        image_paths = [convert_image_to_png(upload_path, job_static)]
    except Exception as exc:
        flash(f"Failed to read clipboard image: {exc}")
        return redirect(url_for("index"))

    result = process_images(job_id, image_paths, cv_debug=cv_debug)
    save_result(job_dir, result)
    return redirect(url_for("result", job_id=job_id))


@app.route("/result/<job_id>")
def result(job_id: str):
    try:
        job = load_result(job_id)
    except FileNotFoundError:
        abort(404)
    return render_template(
        "result.html",
        job=job,
        debug_ocr=DEBUG_OCR_LAYER and not NO_OCR,
        debug_cv=bool(job.get("debug_cv")),
    )


@app.route("/api/job/<job_id>")
def job_json(job_id: str):
    try:
        job = load_result(job_id)
    except FileNotFoundError:
        abort(404)
    return jsonify(job)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument("-debug", "--debug", action="store_true", help="Run Flask in debug mode.")
    parser.add_argument(
        "-debug-ocr",
        "--debug-ocr",
        action="store_true",
        help="Include and display OCR text layer overlay (debug only).",
    )
    parser.add_argument(
        "-debug-ocr-all",
        "--debug-ocr-all",
        action="store_true",
        help="Show the full OCR overlay (not filtered to answer lines).",
    )
    parser.add_argument(
        "--no-ocr",
        action="store_true",
        help="Disable all OCR usage (OCRmyPDF, anchor snapping, and OCR debug overlay).",
    )
    args, _unknown = parser.parse_known_args()

    if args.debug_ocr:
        DEBUG_OCR_LAYER = True
    if args.debug_ocr_all:
        DEBUG_OCR_LAYER = True
        DEBUG_OCR_LAYER_ALL = True
    if args.no_ocr:
        NO_OCR = True
        OCRMY_PDF_ENABLED = False
        DEBUG_OCR_LAYER = False
        DEBUG_OCR_LAYER_ALL = False

    host = os.getenv("FLASK_HOST", "0.0.0.0")
    port = int(os.getenv("FLASK_PORT", "5000"))
    debug = args.debug or (os.getenv("FLASK_DEBUG", "0") == "1")
    app.run(host=host, port=port, debug=debug)
