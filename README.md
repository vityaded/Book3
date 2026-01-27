# Workbook Overlay (Flask + Gemini)

Upload a PDF or image (or paste from your clipboard). The app sends each page to Gemini, gets bounding boxes for answer areas, and renders the same page with writable text boxes overlaid.

## Setup

1. Create a virtual environment and install deps:

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

2. Install OCRmyPDF system dependencies (required for PDF uploads):

- macOS (Homebrew): `brew install ocrmypdf`
- Ubuntu/Debian: `sudo apt-get install -y ocrmypdf`

3. Add your Gemini key in `.env` (copy from `.env.example` if needed):

```
GEMINI_API_KEY=your_key_here
```

## Run

```powershell
python app.py
```

Open http://127.0.0.1:5000

## Notes

- `MAX_PAGES` limits PDF pages processed per upload.
- `MAX_CONCURRENT_PAGES` enables parallel page processing to reduce total time.
- `MAX_IMAGE_DIM` downsizes images sent to Gemini for speed; overlays still align.
- `BOX_INSET_RATIO` slightly shrinks answer boxes to reduce overlap with printed text.
- `FILTER_FILLED_BOXES` drops boxes that appear to contain printed answers/examples.
- `BOX_COORDINATE_SCALE` controls how model box coordinates are normalized (`1000`, `auto`, or `px`).
- `GEMINI_DEBUG` logs request/response summaries for troubleshooting blocked or empty replies.
- `GEMINI_TEMPERATURE`, `GEMINI_TOP_P`, `GEMINI_MAX_OUTPUT_TOKENS` tune output stability and size.
- `GEMINI_THINKING_BUDGET=0` minimizes latency; raise it if accuracy drops.
- `GEMINI_TIMEOUT_SEC`, `GEMINI_MAX_RETRIES`, `GEMINI_RETRY_BACKOFF_SEC` control request timeouts and retries.
- `GEMINI_FALLBACK_MODEL` lets you auto-retry with a backup model when the primary fails.
- PDF uploads are OCR’d with `ocrmypdf` before page rendering (requires OCRmyPDF + Tesseract + Ghostscript).
- `OCRMY_PDF_ENABLED` toggles the OCR pre-pass (default `1`).
- `OCRMY_PDF_REQUIRED` fails the upload if `ocrmypdf` is missing/fails (default `1`).
- `OCRMY_PDF_SKIP_TEXT` skips OCR on pages that already have text (default `1`).
- `OCRMY_PDF_ROTATE_PAGES` enables auto-rotation based on detected text (default `0`).
- `OCRMY_PDF_DESKEW` deskews pages before OCR (default `0`).
- `OCRMY_PDF_CLEAN` cleans pages before OCR but keeps original visuals in output (default `0`).
- `OCRMY_PDF_REMOVE_BACKGROUND` whitens backgrounds before OCR (default `0`).
- `OCRMY_PDF_OVERSAMPLE` oversamples to at least this DPI for better OCR (example `300`).
- `OCRMY_PDF_REMOVE_VECTORS` masks vector lines during OCR to reduce false punctuation (default `0`).
- `OCRMY_PDF_OPTIMIZE` controls OCRmyPDF optimization level (default `0` for minimal recompression).
- `OCRMY_PDF_OUTPUT_TYPE` sets OCRmyPDF output type (default `pdf`; use `pdfa*` only if you need PDF/A).
- `OCRMY_PDF_PDFA_IMAGE_COMPRESSION` sets PDF/A image compression when `OCRMY_PDF_OUTPUT_TYPE=pdfa*` (default `lossless`).
- `OCRMY_PDF_CHAR_WHITELIST` passes a Tesseract `tessedit_char_whitelist` (example: `ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789`).
- `OCRMY_PDF_TESSERACT_PSM` sets Tesseract page segmentation mode (example `3` or `6`).
- `OCRMY_PDF_TESSERACT_OEM` sets Tesseract OCR engine mode (example `1`).
- `OCRMY_PDF_TESSERACT_THRESHOLDING` sets thresholding method (example `adaptive-otsu` or `sauvola`).
- `OCRMY_PDF_USER_WORDS` passes a `--user-words` file to Tesseract via OCRmyPDF.
- `OCRMY_PDF_USER_PATTERNS` passes a `--user-patterns` file to Tesseract via OCRmyPDF.
- `OCRMY_PDF_TESSERACT_CONFIG` passes an extra Tesseract config file path to OCRmyPDF.
- `OCRMY_PDF_LOG_OUTPUT` logs OCRmyPDF stdout/stderr (default `0`).
- `OCR_LANG` controls OCR language for OCRmyPDF + anchor matching (PDFs use the OCRmyPDF text layer; images fall back to Tesseract via PyMuPDF).
- `OCR_TESSDATA_DIR` sets the tessdata directory if Tesseract cannot find it.
- `OCR_MIN_WORD_HEIGHT_PX` drops OCR words whose bounding box height is smaller than this many pixels (helps ignore underlines/dots misread as letters; default `0`).
- `DEBUG_OCR_LAYER=1` includes an OCR text overlay on the results page (or run `python app.py --debug-ocr`).
- `DEBUG_OCR_LAYER_ALL=1` shows the full OCR overlay (default shows only OCR lines near answer boxes; or run `python app.py --debug-ocr-all`).
- `ANCHOR_GAP` offsets boxes to the right of the anchor word.
- `ANCHOR_MAX_SHIFT_X` and `ANCHOR_MAX_SHIFT_Y` cap how far a box can move when snapping to an anchor.
- `ANCHOR_FUZZY_RATIO` controls fuzzy matching strictness for anchors (higher = stricter).
- `ANCHOR_RIGHT_TOLERANCE` allows limited overlap between a detected anchor and the model box before penalizing that match (useful when the model box slightly covers the anchor).
- `ANCHOR_LINE_DY_MAX`, `ANCHOR_LINE_DY_MIN`, `ANCHOR_LINE_DY_MULT` restrict anchor matching to OCR lines near the original box (prevents snapping to the same word far away).
- If OCR fails, anchor snapping is skipped and Gemini coordinates are used as-is.
- Results are saved under `data/jobs` and images under `static/jobs`.
