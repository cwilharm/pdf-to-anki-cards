"""Text chunking module — groups pages into batches for the AI."""


def create_chunks(pages: list[dict], max_words_per_chunk: int = 2000) -> list[dict]:
    """
    Group pages into chunks that stay under max_words_per_chunk.
    Each chunk contains multiple consecutive pages to minimise API calls.
    Returns list of dicts with 'pages' (list of ints) and 'text' (str).
    """
    chunks: list[dict] = []
    current_pages: list[dict] = []
    current_words = 0

    for page in pages:
        word_count = len(page["text"].split())

        # If adding this page would exceed the limit and we already have content,
        # flush the current chunk first.
        if current_words + word_count > max_words_per_chunk and current_pages:
            chunks.append(_make_chunk(current_pages))
            current_pages = []
            current_words = 0

        current_pages.append(page)
        current_words += word_count

    if current_pages:
        chunks.append(_make_chunk(current_pages))

    return chunks


def _make_chunk(pages: list[dict]) -> dict:
    return {
        "pages": [p["page"] for p in pages],
        "text": "\n\n".join(p["text"] for p in pages),
    }
