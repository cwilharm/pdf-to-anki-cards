# PDF to Anki

Convert any PDF into exam-ready Anki flashcards using OpenAI. Upload a document, pick your options, and get a ready-to-import `.apkg` file in seconds.

Supports both digital PDFs (with an embedded text layer) and scanned documents (image-only pages are processed via OCR automatically).

---

## Features

- **Automatic text extraction** — pdfplumber handles standard PDFs with no configuration
- **OCR fallback** — scanned pages are detected and processed automatically using EasyOCR; no manual pre-processing required
- **Multi-column layout support** — the OCR engine detects reading order across columns
- **Three card types** — Basic (Q&A), Cloze (fill-in-the-blank), or Both (AI decides per concept)
- **Two answer formats** — full sentences or bullet points
- **20 output languages** — all card content is written in the language you select
- **Quality filter** — short or near-duplicate cards are removed before export
- **Cost-efficient** — uses gpt-4o-mini; approximately $0.06 per 100 pages

---

## Requirements

- Python 3.11 or later
- An OpenAI API key

---

## Installation

```bash
git clone https://github.com/your-username/PDF-To-Anki.git
cd PDF-To-Anki
pip install -r requirements.txt
```

No system packages or external binaries are required. All dependencies are pure Python wheels.

---

## Configuration

Create a `.env` file in the project root:

```
OPENAI_API_KEY=sk-...
```

Alternatively, paste your key directly in the sidebar when the app is running. It is used only for the current session and never stored.

---

## Usage

```bash
streamlit run app.py
```

1. Upload a PDF (digital or scanned)
2. Set the deck name, card type, answer format, and output language
3. Click "Generate My Cards"
4. Download the `.apkg` file and import it into Anki via **File > Import**

---

## How It Works

```
PDF
 |
 v
[Extractor]   pdfplumber extracts the text layer page by page.
              Pages with no text but embedded images are queued for OCR.
              EasyOCR processes those pages at 300 DPI with contrast
              enhancement and column-aware reading order.
 |
 v
[Chunker]     Pages are grouped into ~2000-word batches to minimise
              API calls (roughly one call per 4-5 pages).
 |
 v
[Generator]   Each chunk is sent to gpt-4o-mini with a detailed
              system prompt that enforces atomicity, forbids list
              questions, and outputs structured JSON.
 |
 v
[Quality]     Cards shorter than a minimum length are removed.
              Near-duplicates are filtered using Jaccard similarity
              (threshold: 0.72).
 |
 v
[Exporter]    genanki assembles the cards into an Anki-compatible
              .apkg package. Deck IDs are derived from the deck name
              so re-importing the same deck does not create duplicates.
```

---

## Project Structure

```
PDF-To-Anki/
├── app.py                  # Streamlit UI and pipeline orchestration
├── pipeline/
│   ├── extractor.py        # pdfplumber extraction + OCR fallback routing
│   ├── ocr_extractor.py    # EasyOCR-based extraction for scanned pages
│   ├── chunker.py          # Page batching for the AI
│   ├── generator.py        # OpenAI card generation
│   ├── quality.py          # Deduplication and length filtering
│   └── exporter.py         # genanki .apkg creation
├── requirements.txt
├── .env                    # API key (not committed)
└── .gitignore
```

---

## Notes on OCR

On the first run with a scanned PDF, EasyOCR downloads its recognition models (~200 MB). This happens automatically and only once; subsequent runs use the cached models.

The OCR pipeline renders pages at 300 DPI and applies contrast enhancement and edge sharpening before recognition. For very small or dense print, accuracy improves further at 400 DPI, which can be adjusted in `ocr_extractor.py`.

EasyOCR's Latin-script model covers English, German, French, Spanish, Italian, Portuguese, Dutch, Polish, Czech, Swedish, and other languages that share the same character set.

---

## Cost Estimate

| Document length | Approximate cost |
|---|---|
| 50 pages | $0.03 |
| 100 pages | $0.06 |
| 200 pages | $0.12 |

Scanned PDFs incur no additional API cost; OCR runs locally.

---

## License

MIT
