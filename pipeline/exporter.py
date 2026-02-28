"""Anki deck export via genanki — supports Basic and Cloze cards."""

import os
import tempfile
import genanki

# ---------------------------------------------------------------------------
# Shared CSS
# ---------------------------------------------------------------------------
_CSS = """
.card {
    font-family: 'Segoe UI', 'Helvetica Neue', Arial, sans-serif;
    font-size: 17px;
    line-height: 1.75;
    color: #1e2a35;
    background-color: #ffffff;
    padding: 28px 32px;
    max-width: 680px;
    margin: 0 auto;
    box-sizing: border-box;
}
.front-text {
    font-size: 19px;
    font-weight: 700;
    color: #0f1e2b;
    margin-bottom: 4px;
}
.back-text {
    font-size: 16px;
    color: #344a5e;
    margin-top: 14px;
}
hr#answer {
    border: none;
    border-top: 2px solid #3b82f6;
    margin: 18px 0;
}
/* Cloze highlight */
.cloze { color: #2563eb; font-weight: 700; }
"""

# ---------------------------------------------------------------------------
# Basic model templates
# ---------------------------------------------------------------------------
_QFMT = '<div class="front-text">{{Front}}</div>'
_AFMT = (
    '<div class="front-text">{{Front}}</div>'
    '<hr id="answer">'
    '<div class="back-text">{{Back}}</div>'
)

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def create_anki_deck(cards: list[dict], deck_name: str = "PDF Anki Cards") -> bytes:
    """
    Build a .apkg from a mixed list of:
      Basic cards  → {"front": …, "back": …}
      Cloze cards  → {"text": …}
    Returns raw .apkg bytes for download.
    """
    basic_cards = [c for c in cards if "front" in c]
    cloze_cards = [c for c in cards if "text" in c]

    deck_id = _stable_id(deck_name + "__deck__")
    deck = genanki.Deck(deck_id, deck_name)

    if basic_cards:
        basic_model = genanki.Model(
            _stable_id(deck_name + "__basic__"),
            "PDF Anki Basic",
            fields=[{"name": "Front"}, {"name": "Back"}],
            templates=[{"name": "Card", "qfmt": _QFMT, "afmt": _AFMT}],
            css=_CSS,
        )
        for card in basic_cards:
            deck.add_note(
                genanki.Note(
                    model=basic_model,
                    fields=[_to_html(card["front"]), _to_html(card["back"])],
                )
            )

    if cloze_cards:
        cloze_model = genanki.Model(
            _stable_id(deck_name + "__cloze__"),
            "PDF Anki Cloze",
            model_type=genanki.Model.CLOZE,
            fields=[{"name": "Text"}, {"name": "Extra"}],
            templates=[
                {
                    "name": "Cloze",
                    "qfmt": "{{cloze:Text}}",
                    "afmt": "{{cloze:Text}}",
                }
            ],
            css=_CSS,
        )
        for card in cloze_cards:
            deck.add_note(
                genanki.Note(
                    model=cloze_model,
                    fields=[card["text"], ""],
                )
            )

    with tempfile.NamedTemporaryFile(suffix=".apkg", delete=False) as f:
        tmp_path = f.name

    try:
        genanki.Package(deck).write_to_file(tmp_path)
        with open(tmp_path, "rb") as f:
            return f.read()
    finally:
        os.unlink(tmp_path)


def create_multi_deck_package(
    topic_decks: list[dict],
    base_name: str = "PDF Anki",
) -> bytes:
    """
    Build a single .apkg containing one sub-deck per topic.

    topic_decks: [{"topic": str, "cards": list[dict]}, ...]
      Each card is either {"front", "back"} or {"text"} (optionally with "topic").

    Sub-decks are named  base_name::topic  (Anki hierarchy notation).
    Returns raw .apkg bytes.
    """
    all_decks: list[genanki.Deck] = []

    for entry in topic_decks:
        topic = entry["topic"]
        cards = entry["cards"]
        if not cards:
            continue

        deck_full_name = f"{base_name}::{topic}"
        deck_id = _stable_id(deck_full_name + "__deck__")
        deck = genanki.Deck(deck_id, deck_full_name)

        basic_cards = [c for c in cards if "front" in c]
        cloze_cards = [c for c in cards if "text" in c]

        if basic_cards:
            basic_model = genanki.Model(
                _stable_id(deck_full_name + "__basic__"),
                "PDF Anki Basic",
                fields=[{"name": "Front"}, {"name": "Back"}],
                templates=[{"name": "Card", "qfmt": _QFMT, "afmt": _AFMT}],
                css=_CSS,
            )
            for card in basic_cards:
                deck.add_note(
                    genanki.Note(
                        model=basic_model,
                        fields=[_to_html(card["front"]), _to_html(card["back"])],
                    )
                )

        if cloze_cards:
            cloze_model = genanki.Model(
                _stable_id(deck_full_name + "__cloze__"),
                "PDF Anki Cloze",
                model_type=genanki.Model.CLOZE,
                fields=[{"name": "Text"}, {"name": "Extra"}],
                templates=[
                    {
                        "name": "Cloze",
                        "qfmt": "{{cloze:Text}}",
                        "afmt": "{{cloze:Text}}",
                    }
                ],
                css=_CSS,
            )
            for card in cloze_cards:
                deck.add_note(
                    genanki.Note(
                        model=cloze_model,
                        fields=[card["text"], ""],
                    )
                )

        all_decks.append(deck)

    with tempfile.NamedTemporaryFile(suffix=".apkg", delete=False) as f:
        tmp_path = f.name

    try:
        genanki.Package(all_decks).write_to_file(tmp_path)
        with open(tmp_path, "rb") as f:
            return f.read()
    finally:
        os.unlink(tmp_path)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _to_html(text: str) -> str:
    """Convert plain-text newlines to HTML line breaks."""
    return text.replace("\n", "<br>")


def _stable_id(seed: str) -> int:
    return abs(hash(seed)) % (10**10)
