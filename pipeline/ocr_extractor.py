"""OCR extraction for scanned PDF pages.

Uses PyMuPDF for rendering (no poppler/system dep) and EasyOCR for
recognition (no Tesseract/system dep).  Both are pure-Python wheels.
"""

import io
import re
from pathlib import Path

import fitz  # PyMuPDF
import numpy as np
import easyocr
from PIL import Image, ImageEnhance, ImageFilter

# Lazy-initialised reader — models (~200 MB) download once on first use.
_reader: easyocr.Reader | None = None


def models_cached() -> bool:
    """Return True if the EasyOCR detection model is already on disk."""
    return (Path.home() / ".EasyOCR" / "model" / "craft_mlt_25k.pth").exists()


def _get_reader() -> easyocr.Reader:
    global _reader
    if _reader is None:
        # 'en' covers every Latin-script language (DE/FR/ES/IT/PT/NL/PL/…).
        # gpu=False works on any machine; EasyOCR auto-uses GPU if available.
        _reader = easyocr.Reader(["en"], gpu=False, verbose=False)
    return _reader


def ocr_pages(pdf_path: str, page_numbers: list[int], dpi: int = 300) -> list[dict]:
    """
    Run OCR on specific pages of a PDF.

    Args:
        pdf_path:     Path to the PDF file.
        page_numbers: 1-indexed list of page numbers to process.
        dpi:          Render resolution. 300 is suitable for most print;
                      increase to 400+ for very small or very dense text.

    Returns:
        List of dicts with 'page' and 'text' keys (same format as extractor.py).
    """
    if not page_numbers:
        return []

    reader = _get_reader()
    zoom = dpi / 72  # PyMuPDF's native unit is 72 DPI
    matrix = fitz.Matrix(zoom, zoom)

    results = []
    doc = fitz.open(pdf_path)
    try:
        for page_num in page_numbers:
            page = doc[page_num - 1]  # fitz uses 0-based indices
            pix = page.get_pixmap(matrix=matrix, colorspace=fitz.csGRAY)
            img = Image.open(io.BytesIO(pix.tobytes("png")))
            img = _preprocess(img)

            detections = reader.readtext(np.array(img))
            text = _assemble_text(detections)
            text = _clean_ocr_text(text)

            if text and len(text.strip()) > 80:
                results.append({"page": page_num, "text": text})
    finally:
        doc.close()

    return results


# ---------------------------------------------------------------------------
# Image pre-processing
# ---------------------------------------------------------------------------


def _preprocess(image: Image.Image) -> Image.Image:
    """
    Enhance image quality before OCR.
    - Contrast +50 %: lifts faded or low-contrast prints.
    - Sharpen:        improves character-edge definition for small text.
    """
    image = ImageEnhance.Contrast(image).enhance(1.5)
    image = image.filter(ImageFilter.SHARPEN)
    return image


# ---------------------------------------------------------------------------
# Text assembly — handles multi-column layouts
# ---------------------------------------------------------------------------


def _assemble_text(detections: list) -> str:
    """
    Convert EasyOCR detections into correctly ordered text.

    EasyOCR returns (bbox, text, confidence) where
    bbox = [[x1,y1], [x2,y1], [x2,y2], [x1,y2]].

    Multi-column strategy:
    1. Record top-left (x, y) and height of every detection.
    2. Group detections sharing the same vertical band
       (within 0.6 × line-height) into a single visual line.
    3. Within each line sort left-to-right.
    4. Output lines top-to-bottom → reading order is preserved for
       two- and three-column pages alike.
    """
    if not detections:
        return ""

    items = []
    for bbox, text, _ in detections:
        x = min(pt[0] for pt in bbox)
        y = min(pt[1] for pt in bbox)
        h = max(pt[1] for pt in bbox) - y
        items.append({"x": x, "y": y, "h": h, "text": text})

    items.sort(key=lambda d: d["y"])

    # Group into visual lines
    lines: list[list[dict]] = []
    current: list[dict] = [items[0]]

    for item in items[1:]:
        line_h = max(current[-1]["h"], 1)
        if abs(item["y"] - current[-1]["y"]) < line_h * 0.6:
            current.append(item)
        else:
            lines.append(current)
            current = [item]
    lines.append(current)

    assembled = []
    for line in lines:
        line.sort(key=lambda d: d["x"])
        assembled.append(" ".join(d["text"] for d in line))

    return "\n".join(assembled)


# ---------------------------------------------------------------------------
# Text cleaning (mirrors _clean_text in extractor.py)
# ---------------------------------------------------------------------------


def _clean_ocr_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"^\s*\d{1,4}\s*$", "", text, flags=re.MULTILINE)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r"-\n([a-zäöüß])", r"\1", text)
    return text.strip()
