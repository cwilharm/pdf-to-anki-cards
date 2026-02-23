"""PDF text extraction module."""

import re
import pdfplumber


def extract_text_from_pdf(pdf_path: str) -> list[dict]:
    """
    Extract text from a PDF file page by page.
    Returns a list of dicts with 'page' and 'text' keys.
    Skips empty or near-empty pages.
    Falls back to OCR for pages that have embedded images but no text layer
    (i.e. scanned pages).
    """
    pages = []
    scanned_page_nums = []

    try:
        with pdfplumber.open(pdf_path) as pdf:
            for i, page in enumerate(pdf.pages):
                raw_text = page.extract_text(x_tolerance=3, y_tolerance=3)

                if raw_text:
                    text = _clean_text(raw_text)
                    if len(text.strip()) > 80:  # Skip nearly empty pages
                        pages.append({"page": i + 1, "text": text})
                        continue

                # No text (or nearly empty): if the page contains embedded images
                # it is almost certainly a scanned page — queue it for OCR.
                if page.images:
                    scanned_page_nums.append(i + 1)

    except Exception as e:
        raise RuntimeError(f"PDF konnte nicht geöffnet werden: {e}")

    # OCR fallback — only runs when scanned pages were detected
    if scanned_page_nums:
        try:
            from pipeline.ocr_extractor import ocr_pages
        except ImportError:
            raise RuntimeError(
                "Scanned pages detected but OCR dependencies are not installed.\n"
                "Run:  pip install pdf2image pytesseract Pillow\n"
                "Also: brew install tesseract poppler  (macOS)"
            )
        ocr_results = ocr_pages(pdf_path, scanned_page_nums)
        if ocr_results:
            pages.extend(ocr_results)
            pages.sort(key=lambda p: p["page"])

    return pages


def _clean_text(text: str) -> str:
    """Clean raw extracted text."""
    # Normalize line endings
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # Remove lines that are just numbers (page numbers)
    text = re.sub(r"^\s*\d{1,4}\s*$", "", text, flags=re.MULTILINE)

    # Collapse 3+ newlines to 2
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Collapse multiple spaces (but not newlines)
    text = re.sub(r"[ \t]{2,}", " ", text)

    # Remove soft hyphens / line-break hyphens
    text = re.sub(r"-\n([a-zäöüß])", r"\1", text)

    return text.strip()
