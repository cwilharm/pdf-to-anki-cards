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
• For comparisons: name ONE concrete difference per card, not all differences at once\
"""

_SELF_CONTAINED = """\
━━━ SELF-CONTAINED CARDS (HARD RULE — applies to EVERY card) ━━━
Every question must be 100% understandable WITHOUT reading the source document.
A student who has NEVER seen this PDF must be able to fully understand and answer the question.
ALL necessary context must be inside the question itself — never assume the reader has any background.

STRICTLY FORBIDDEN in questions AND answers:
✗ "the method", "the algorithm", "the institution", "the model", "the formula" — always name it explicitly
✗ "in this text", "according to the author", "as described above", "in the document"
✗ "this approach", "the process", "the concept" without stating what it is
✗ Any unnamed pronoun or reference that requires reading the source to decode

REQUIRED: Every concept, person, institution, algorithm, formula, and mechanism must be NAMED.

Self-check before writing each question: "Could someone answer this without the PDF?"
  NO  → Add the missing name/context to the question itself, then ask again
  YES → Proceed

BAD:  "What is the advantage of the method over the baseline?"
GOOD: "What is the key advantage of Transformer self-attention over LSTM recurrent connections for long sequences?"
BAD:  "What are the conditions for the theorem to hold?"
GOOD: "What are the two conditions required for the Central Limit Theorem (CLT) to apply?"
BAD:  "How does the institution regulate the market?"
GOOD: "How does the European Central Bank (ECB) regulate the money supply through open market operations?"\
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

_ANS_BULLETS = """\
━━━ ANSWER FORMAT: BULLET POINTS — MANDATORY, NO EXCEPTIONS ━━━
EVERY answer field MUST consist exclusively of bullet points starting with "• ".
Prose sentences in the answer are strictly forbidden — even for a single fact.

Rules:
• 1–3 bullets per card — if you need more than 3, the question is too broad → split it
• Each bullet: one tight fact, mechanism, or term — no filler words
• Even a one-fact answer must be a single bullet: "• [the fact]"

CORRECT:
  Q: "Who elects the members of the European Parliament?"
  A: "• Directly by EU citizens in each member state."
  Q: "What does the softmax function output?"
  A: "• A probability distribution over all output classes that sums to 1."

WRONG (forbidden prose):
  A: "The European Parliament is elected directly by EU citizens."
  A: "Softmax converts raw logits into a normalized probability distribution."\
"""

_EXHAUSTIVE = """\
━━━ CARD VOLUME: BE EXHAUSTIVE ━━━
Your goal is MAXIMUM COVERAGE — extract as many high-quality cards as possible from the text.
Systematically work through every paragraph and cover every concept, definition, mechanism,
formula, comparison, cause/effect relationship, condition, exception, and exam-relevant fact.
Do NOT skip a concept because it seems minor — if it appears in the text, it deserves a card.
A thorough pass over 4–5 pages should produce at least 15–25 cards.\
"""

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
# User message templates
# ---------------------------------------------------------------------------

_USER_TMPL = """\
Create Anki flashcards from the following text (pages {pages}).
Be EXHAUSTIVE — go through every paragraph and extract every concept, definition, \
mechanism, formula, comparison, cause/effect, condition, and exam-relevant fact. \
Omit nothing important, skip only the truly trivial.

<text>
{text}
</text>

Reply ONLY with the JSON object.\
"""

_USER_TMPL_TOPICS = """\
Create Anki flashcards from the following text (pages {pages}).
Be EXHAUSTIVE — extract every concept, definition, mechanism, formula, comparison, \
cause/effect, condition, and exam-relevant fact that belongs to one of these topics:
{topics_list}

Rules:
• Work through every paragraph — maximum coverage is required.
• Assign EXACTLY one topic from the list above to each card — add a "topic" field with the verbatim topic name.
• Only create a card if the content is genuinely about one of the listed topics.
• If the text contains no content about any of these topics, return {{"cards": []}}.
• Do NOT fabricate or invent information not present in the text.

Expected output keys per card:
  Basic:  "front", "back", "topic"
  Cloze:  "text", "topic"

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
        parts = [_ROLE, _SELF_CONTAINED, _QUALITY, _FORBIDDEN, _ATOMICITY, _EXHAUSTIVE, fmt, ans]

    elif card_type == "cloze":
        # Cloze cards are inherently atomic (one blank = one fact)
        parts = [_ROLE, _SELF_CONTAINED, _QUALITY, _FORBIDDEN, _EXHAUSTIVE, _FMT_CLOZE]

    else:  # both
        fmt = _FMT_BOTH
        ans = _ANS_BULLETS if answer_format == "bullets" else _ANS_SENTENCES
        parts = [
            _ROLE,
            _SELF_CONTAINED,
            _QUALITY,
            _FORBIDDEN,
            _ATOMICITY,
            _EXHAUSTIVE,
            fmt,
            f"For Basic cards: {ans}",
        ]

    # Language instruction — appended last so it overrides any default tendency
    parts.append(
        f"Respond exclusively in {language_name}. "
        f"All flashcard content must be written in {language_name}."
    )

    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def generate_cards_for_chunk_with_topics(
    chunk: dict,
    topics: list[str],
    client: openai.OpenAI,
    language_name: str = "Deutsch",
    model: str = "gpt-4o-mini",
    card_type: str = "basic",
    answer_format: str = "sentences",
) -> list[dict]:
    """
    Like generate_cards_for_chunk but restricts output to the given topics and
    adds a "topic" field to every returned card.

    Returns [] when the chunk contains no content for any of the listed topics.
    """
    system = _build_system_prompt(card_type, answer_format, language_name)

    p = chunk["pages"]
    pages_str = f"{p[0]}–{p[-1]}" if len(p) > 1 else str(p[0])

    topics_list = "\n".join(f"  • {t}" for t in topics)
    user_msg = _USER_TMPL_TOPICS.format(
        pages=pages_str,
        topics_list=topics_list,
        text=chunk["text"],
    )

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user_msg},
            ],
            temperature=0.25,
            response_format={"type": "json_object"},
            max_tokens=4096,
        )
        raw = response.choices[0].message.content.strip()
        return _parse_cards(raw, topic_aware=True)
    except Exception as e:
        print(f"[generator] Error on chunk (pages {pages_str}) with topics: {e}")
        return []


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
            max_tokens=4096,
        )
        raw = response.choices[0].message.content.strip()
        return _parse_cards(raw)
    except Exception as e:
        print(f"[generator] Error on chunk (pages {pages_str}): {e}")
        return []


# ---------------------------------------------------------------------------
# JSON parsing helpers
# ---------------------------------------------------------------------------


def _parse_cards(raw: str, topic_aware: bool = False) -> list[dict]:
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
                return _validate(data[key], topic_aware=topic_aware)
        for v in data.values():
            if isinstance(v, list):
                return _validate(v, topic_aware=topic_aware)
        return []

    if isinstance(data, list):
        return _validate(data, topic_aware=topic_aware)

    return []


def _validate(cards: list, topic_aware: bool = False) -> list[dict]:
    valid = []
    for card in cards:
        if not isinstance(card, dict):
            continue
        explicit_type = card.get("type", "")
        topic = str(card.get("topic", "")).strip() if topic_aware else None

        if explicit_type == "cloze" or ("text" in card and "front" not in card):
            text = str(card.get("text", "")).strip()
            if text and "{{c" in text:
                entry: dict = {"text": text}
                if topic_aware and topic:
                    entry["topic"] = topic
                valid.append(entry)

        elif "front" in card and "back" in card:
            front = str(card["front"]).strip()
            back = str(card["back"]).strip()
            if front and back:
                entry = {"front": front, "back": back}
                if topic_aware and topic:
                    entry["topic"] = topic
                valid.append(entry)

    return valid
