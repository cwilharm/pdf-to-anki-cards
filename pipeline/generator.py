"""AI card generation module using OpenAI."""
import json
import re
import openai

# ---------------------------------------------------------------------------
# Single English base — the language instruction appended at the end
# steers the model's output language without duplicating prompt text.
# ---------------------------------------------------------------------------

_QUALITY = """\
━━━ QUALITY PRINCIPLES ━━━
• Only exam-relevant content — no trivial facts
• Each card tests EXACTLY ONE specific, atomic concept — never a list, never multiple aspects
• ONE question → ONE precise answer, not an enumeration of items
• Questions: specific and action-oriented (Why? How exactly? What is the difference between X and Y?)
• Prefer conceptual understanding over pure fact recall
• For formulas/algorithms: state the meaning of variables and the use case
• For comparisons: name ONE concrete difference per card, not all differences at once
• SELF-CONTAINED questions: every question must include enough context to be answerable
  by someone who has never seen the source document. Name the subject explicitly —
  never use vague references like "the process", "the institution", "the method above".
  BAD:  "What is the main role of the institution?"
  GOOD: "What is the main role of the European Central Bank (ECB)?"
• DOCUMENT-SPECIFIC questions only: every question must use the exact concepts, terms,
  models, and names from the source material — never generic placeholders. Specificity
  comes from the concrete subject matter itself, never from phrases like "in this text",
  "according to the author", or "as described above". Those phrases are strictly forbidden
  in card questions and answers.
  Ask yourself: "Could this question appear in any introductory textbook on this subject?"
  If yes, sharpen it using the specific terminology, conditions, or mechanisms from the text.
  BAD:  "What are the advantages of regularisation?" (generic, appears in every ML textbook)
  BAD:  "According to this text, when does method X outperform method Y?" (forbidden reference)
  GOOD: "Why does L2 regularisation reduce model variance without producing sparse weights,
         while L1 regularisation does?"\
"""

_FORBIDDEN = """\
━━━ FORBIDDEN ━━━
✗ Chapter headings, author names, page numbers
✗ Yes/no questions or trivial definitions ("What is X?" → "X is a …")
✗ Pure facts without learning value (dates without context)
✗ Redundant or very similar cards
✗ OVERVIEW / LIST questions — these are the most common mistake, strictly forbidden:
  - "What are the main X and their functions?"
  - "Name all Y of Z."
  - "What are the key components/institutions/elements of X?"
  - "What are the central X and what do they do?"
  - Any question where the answer would require listing 3+ separate items
  → INSTEAD: create one dedicated card per item/concept.\
"""

_ATOMICITY = """\
━━━ ATOMICITY RULE (strictly enforced) ━━━
A card is atomic when its answer contains EXACTLY ONE independent, indivisible fact.
Every independent fact must become its own card — no exceptions.

VIOLATION — do NOT create this:
  Q: "How is the European Parliament elected?"
  A: "• Directly by citizens  • Every five years  • By proportional representation"
  WHY: Three independent facts. Remove any one bullet → answer is still complete → NOT atomic.

CORRECT — create three separate cards instead:
  Q: "Who elects the European Parliament?"                     A: "Directly by EU citizens."
  Q: "How often are European Parliament elections held?"       A: "Every five years."
  Q: "What electoral system applies to the European Parliament?"  A: "Proportional representation."

ATOMICITY TEST (apply before finalising every card):
  Remove one sentence or bullet from the answer.
  • If the remaining answer is still complete → card is NOT atomic → split it.
  • If the remaining answer is incomplete/broken → card IS atomic → keep it.\
"""

# ---------------------------------------------------------------------------
# Answer-format instructions (basic cards)
# ---------------------------------------------------------------------------

_ANS_SENTENCES = (
    "Answers: 1–2 concise sentences maximum — straight to the point, no filler text. "
    "If an answer needs more than 2 sentences, split it into multiple cards."
)
_ANS_BULLETS = (
    "Answers: 2–3 tight bullet points max, each starting with '• '.\n"
    "If you need more than 3 bullets, the question is too broad — split into multiple cards.\n"
    "Example: '• Core mechanism\\n• Key consequence\\n• Exception if critical'"
)

# ---------------------------------------------------------------------------
# Card-format instructions
# ---------------------------------------------------------------------------

_FMT_BASIC = """\
━━━ CARD FORMAT: Basic (Question–Answer) ━━━
Each card: a clear question as "front" and a concise answer as "back".

GOOD  "Why is X preferred over Y in scenario Z?"  → "X is preferred because …"
GOOD  "What is the role of the X in process Y?"   → "The X is responsible for …"
GOOD  "How does mechanism X achieve Y?"            → "X achieves Y by …"

BAD   "What are the main institutions of X and their functions?"
      → Too broad. Create one card per institution instead.
BAD   "What are the key features of X?"
      → Too vague. Ask about one specific feature per card.

OUTPUT: {"cards": [{"front": "Question", "back": "Answer"}, …]}\
"""

_FMT_CLOZE = """\
━━━ CARD FORMAT: Cloze (Fill-in-the-blank) ━━━
Complete, informative sentences — mark the key term with {{c1::term}}.
Rules:
• Mark only the core term, never whole phrases
• Optional additional blanks in the same sentence: {{c2::term}}, {{c3::term}}
• The sentence must still be informative without the blank
Good:  "{{c1::Oxidative phosphorylation}} produces approximately {{c2::30}} ATP per glucose."
Bad:   "{{c1::Mitochondria are the powerhouse of the cell.}}" (too much marked)
Bad:   "X is {{c1::important}}." (too trivial)
OUTPUT: {"cards": [{"text": "Sentence with {{c1::blank}}"}, …]}\
"""

_FMT_BOTH = """\
━━━ CARD FORMAT: Mixed (auto-select) ━━━
Choose the optimal type per piece of content:
• Basic  → concepts, processes, comparisons, explanations
  {"type": "basic", "front": "Question", "back": "Answer"}
• Cloze  → key terms, definitions, formulas to memorize
  {"type": "cloze", "text": "Sentence with {{c1::blank}}"}
Cloze rules: mark only the core term, never whole phrases.
Target ratio: ~55% Basic, ~45% Cloze.
OUTPUT: {"cards": [{"type": "basic", "front": "…", "back": "…"}, {"type": "cloze", "text": "…{{c1::…}}…"}, …]}\
"""

# ---------------------------------------------------------------------------
# User message template
# ---------------------------------------------------------------------------

_USER_TMPL = """\
Create Anki flashcards from the following text (pages {pages}).
Extract all important concepts, definitions, relationships, and exam-relevant \
content — omit nothing important, skip the trivial.

<text>
{text}
</text>

Reply ONLY with the JSON object.\
"""

# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------

_ROLE = (
    "You are an expert in learning materials and exam preparation. "
    "Your sole task: create high-quality Anki flashcards from academic texts."
)


def _build_system_prompt(card_type: str, answer_format: str, language_name: str) -> str:
    """Assemble the system prompt from shared blocks + language instruction."""
    if card_type == "basic":
        fmt = _FMT_BASIC
        ans = _ANS_BULLETS if answer_format == "bullets" else _ANS_SENTENCES
        parts = [_ROLE, _QUALITY, _FORBIDDEN, _ATOMICITY, fmt, ans]

    elif card_type == "cloze":
        # Cloze cards are inherently atomic (one blank = one fact)
        parts = [_ROLE, _QUALITY, _FORBIDDEN, _FMT_CLOZE]

    else:  # both
        fmt = _FMT_BOTH
        ans = _ANS_BULLETS if answer_format == "bullets" else _ANS_SENTENCES
        parts = [_ROLE, _QUALITY, _FORBIDDEN, _ATOMICITY, fmt, f"For Basic cards: {ans}"]

    # Language instruction — appended last so it overrides any default tendency
    parts.append(f"Respond exclusively in {language_name}. "
                 f"All flashcard content must be written in {language_name}.")

    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_cards_for_chunk(
    chunk: dict,
    client: openai.OpenAI,
    language_name: str = "Deutsch",
    model: str = "gpt-4o-mini",
    card_type: str = "basic",
    answer_format: str = "sentences",
) -> list[dict]:
    """
    Call OpenAI and return a list of card dicts.
      Basic cards → {"front": …, "back": …}
      Cloze cards → {"text": …}   (contains {{c1::…}})
    Returns [] on any error so the pipeline always continues.
    """
    system = _build_system_prompt(card_type, answer_format, language_name)

    p = chunk["pages"]
    pages_str = f"{p[0]}–{p[-1]}" if len(p) > 1 else str(p[0])
    user_msg = _USER_TMPL.format(pages=pages_str, text=chunk["text"])

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user_msg},
            ],
            temperature=0.25,
            response_format={"type": "json_object"},
            max_tokens=2048,
        )
        raw = response.choices[0].message.content.strip()
        return _parse_cards(raw)
    except Exception as e:
        print(f"[generator] Error on chunk (pages {pages_str}): {e}")
        return []


# ---------------------------------------------------------------------------
# JSON parsing helpers
# ---------------------------------------------------------------------------

def _parse_cards(raw: str) -> list[dict]:
    raw = re.sub(r"```(?:json)?\s*", "", raw).strip().rstrip("```").strip()

    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if match:
            try:
                data = json.loads(match.group())
            except json.JSONDecodeError:
                return []
        else:
            return []

    if isinstance(data, dict):
        for key in ("cards", "flashcards", "karten", "anki_cards", "data", "items"):
            if key in data and isinstance(data[key], list):
                return _validate(data[key])
        for v in data.values():
            if isinstance(v, list):
                return _validate(v)
        return []

    if isinstance(data, list):
        return _validate(data)

    return []


def _validate(cards: list) -> list[dict]:
    valid = []
    for card in cards:
        if not isinstance(card, dict):
            continue
        explicit_type = card.get("type", "")

        if explicit_type == "cloze" or ("text" in card and "front" not in card):
            text = str(card.get("text", "")).strip()
            if text and "{{c" in text:
                valid.append({"text": text})

        elif "front" in card and "back" in card:
            front = str(card["front"]).strip()
            back = str(card["back"]).strip()
            if front and back:
                valid.append({"front": front, "back": back})

    return valid
