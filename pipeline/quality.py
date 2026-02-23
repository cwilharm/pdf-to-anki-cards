"""Quality control — filter bad cards and remove duplicates."""
import re


def filter_and_deduplicate(cards: list[dict], similarity_threshold: float = 0.72) -> list[dict]:
    """
    Handles both basic {"front","back"} and cloze {"text"} cards.
    1. Validates structure and minimum length.
    2. Removes near-duplicate cards via Jaccard word overlap.
    """
    basic = [c for c in cards if "front" in c]
    cloze = [c for c in cards if "text" in c]

    return _dedup_basic(basic, similarity_threshold) + _dedup_cloze(cloze, similarity_threshold)


# ---------------------------------------------------------------------------
# Basic cards
# ---------------------------------------------------------------------------

def _dedup_basic(cards: list[dict], threshold: float) -> list[dict]:
    valid = []
    for card in cards:
        front = card.get("front", "").strip()
        back = card.get("back", "").strip()
        if len(front) >= 12 and len(back) >= 10:
            valid.append({"front": front, "back": back})

    seen: list[str] = []
    unique: list[dict] = []
    for card in valid:
        key = card["front"].lower()
        if not any(_jaccard(key, s) >= threshold for s in seen):
            seen.append(key)
            unique.append(card)
    return unique


# ---------------------------------------------------------------------------
# Cloze cards
# ---------------------------------------------------------------------------

_CLOZE_RE = re.compile(r"\{\{c\d+::(.+?)\}\}")


def _dedup_cloze(cards: list[dict], threshold: float) -> list[dict]:
    valid = []
    for card in cards:
        text = card.get("text", "").strip()
        if not text or not _CLOZE_RE.search(text) or len(text) < 20:
            continue
        valid.append({"text": text})

    seen: list[str] = []
    unique: list[dict] = []
    for card in valid:
        # Dedup on plain text (cloze markers stripped)
        plain = _CLOZE_RE.sub(r"\1", card["text"]).lower()
        if not any(_jaccard(plain, s) >= threshold for s in seen):
            seen.append(plain)
            unique.append(card)
    return unique


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _jaccard(a: str, b: str) -> float:
    wa, wb = set(a.split()), set(b.split())
    if not wa or not wb:
        return 0.0
    return len(wa & wb) / len(wa | wb)
