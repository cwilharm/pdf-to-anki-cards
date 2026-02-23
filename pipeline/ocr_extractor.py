"""OCR extraction for scanned PDF pages."""
import re

import pytesseract
from pdf2image import convert_from_path
from PIL import Image, ImageEnhance, ImageFilter


def ocr_pages(pdf_path: str, page_numbers: list[int], dpi: int = 300) -> list[dict]:
    """
    Run OCR on specific pages of a PDF.

    Args:
        pdf_path:     Path to the PDF file.
        page_numbers: 1-indexed list of page numbers to process.
        dpi:          Render resolution. 300 is good for most print;
                      increase to 400+ for very small or dense text.

    Returns:
        List of dicts with 'page' and 'text' keys (same format as extractor.py).
    """
    if not page_numbers:
        return []

    first = min(page_numbers)
    last = max(page_numbers)
    page_set = set(page_numbers)

    # Convert only the relevant page range to avoid loading the whole PDF into RAM
    images = convert_from_path(pdf_path, dpi=dpi, first_page=first, last_page=last)

    results = []
    for i, image in enumerate(images):
        page_num = first + i
        if page_num not in page_set:
            continue
        text = _ocr_page(image)
        if text and len(text.strip()) > 80:
            results.append({"page": page_num, "text": text})

    return results


def _ocr_page(image: Image.Image) -> str:
    """Preprocess and run OCR on a single page image."""
    image = _preprocess(image)

    # --psm 1  Automatic page segmentation with OSD (Orientation & Script Detection).
    #          Handles multi-column layouts and rotated/mixed-orientation pages.
    # --oem 3  LSTM neural-net engine — most accurate.
    try:
        text = pytesseract.image_to_string(image, config="--psm 1 --oem 3")
        if text.strip():
            return _clean_ocr_text(text)
    except pytesseract.TesseractError:
        pass  # OSD data not installed → fall through to simpler mode

    # Fallback: --psm 3 (fully automatic page segmentation, no OSD required)
    text = pytesseract.image_to_string(image, config="--psm 3 --oem 3")
    return _clean_ocr_text(text)


def _preprocess(image: Image.Image) -> Image.Image:
    """
    Enhance the image before OCR to improve accuracy.

    Steps:
    - Grayscale    : removes colour noise that confuses OCR.
    - Contrast +50%: lifts faded ink / low-contrast prints.
    - Sharpen      : sharpens character edges, especially helpful for small text.
    """
    image = image.convert("L")
    image = ImageEnhance.Contrast(image).enhance(1.5)
    image = image.filter(ImageFilter.SHARPEN)
    return image


def _clean_ocr_text(text: str) -> str:
    """Clean raw OCR output (mirrors _clean_text logic in extractor.py)."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # Remove lone page-number lines
    text = re.sub(r"^\s*\d{1,4}\s*$", "", text, flags=re.MULTILINE)
    # Collapse 3+ blank lines to 2
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Collapse multiple spaces / tabs
    text = re.sub(r"[ \t]{2,}", " ", text)
    # Re-join hyphenated words split across lines (e.g. "Unter-\nsuchung")
    text = re.sub(r"-\n([a-zäöüß])", r"\1", text)
    return text.strip()
