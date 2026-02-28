"""Scan PDF chunks to discover the main topics covered in the material."""

import json
import re
import openai

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

_SCAN_SYSTEM = """\
You are analyzing a document to identify its main topics, chapters, or subject areas.
Extract all distinct conceptual topics covered in the text.

Rules:
• List 3–15 meaningful, distinct topics
• Use the exact terminology from the document — not generic labels like "Introduction"
• Each description: 1 concise sentence explaining what aspect is covered
• Do NOT invent topics that are not present in the text

OUTPUT: {"topics": [{"name": "Topic Name", "description": "One-sentence description"}]}\
"""

_SCAN_USER = """\
Identify the main topics covered in the following text (pages {pages}).

<text>
{text}
</text>

Reply ONLY with the JSON object.\
"""

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def scan_topics_from_chunks(
    chunks: list[dict],
    client: openai.OpenAI,
    model: str = "gpt-4o-mini",
    language_name: str = "English",
) -> list[dict]:
    """
    Scan every chunk to discover the topics covered in the material.

    Returns a deduplicated list of dicts:
        [{"name": str, "description": str}, ...]
    """
    system = (
        _SCAN_SYSTEM
        + f"\n\nRespond in {language_name}. All topic names and descriptions must be in {language_name}."
    )

    all_topics: list[dict] = []

    for chunk in chunks:
        p = chunk["pages"]
        pages_str = f"{p[0]}–{p[-1]}" if len(p) > 1 else str(p[0])

        # Use a representative sample (first ~700 words) to keep scanning cheap
        words = chunk["text"].split()
        sample = " ".join(words[:700])

        user_msg = _SCAN_USER.format(pages=pages_str, text=sample)

        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user_msg},
                ],
                temperature=0.1,
                response_format={"type": "json_object"},
                max_tokens=1024,
            )
            raw = response.choices[0].message.content.strip()
            topics = _parse_topics(raw)
            all_topics.extend(topics)
        except Exception as e:
            print(f"[scanner] Error scanning pages {pages_str}: {e}")

    return _deduplicate_topics(all_topics)


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------


def _parse_topics(raw: str) -> list[dict]:
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
        for key in ("topics", "themen", "temas", "sujets", "argomenti", "items"):
            if key in data and isinstance(data[key], list):
                return _validate_topics(data[key])
        for v in data.values():
            if isinstance(v, list):
                return _validate_topics(v)

    if isinstance(data, list):
        return _validate_topics(data)

    return []


def _validate_topics(topics: list) -> list[dict]:
    valid = []
    for t in topics:
        if not isinstance(t, dict):
            continue
        name = str(
            t.get("name", t.get("topic", t.get("titel", t.get("thema", ""))))
        ).strip()
        desc = str(
            t.get(
                "description",
                t.get("beschreibung", t.get("summary", t.get("beschreibung", ""))),
            )
        ).strip()
        if name and len(name) > 2:
            valid.append({"name": name, "description": desc})
    return valid


def _deduplicate_topics(topics: list[dict], threshold: float = 0.55) -> list[dict]:
    """Remove near-duplicate topics by Jaccard similarity on lowercased names."""
    seen: list[str] = []
    unique: list[dict] = []
    for topic in topics:
        key = topic["name"].lower()
        if not any(_jaccard(key, s) >= threshold for s in seen):
            seen.append(key)
            unique.append(topic)
    return unique


def _jaccard(a: str, b: str) -> float:
    wa, wb = set(a.split()), set(b.split())
    if not wa or not wb:
        return 0.0
    return len(wa & wb) / len(wa | wb)
