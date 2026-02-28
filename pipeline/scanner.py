"""Scan PDF chunks to discover the MAIN topics covered in the material."""

import json
import re
import openai

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

_SCAN_SYSTEM = """\
You are analyzing a document to identify its main subject areas and chapter-level themes.
Your task: extract the 4–8 BROAD, HIGH-LEVEL main topics of the ENTIRE document.

Rules:
• Think at chapter level — broad subject areas, not granular sub-topics
• Each topic should cover a significant portion of the material
• Use the exact field/subject terminology from the document
• Do NOT list more than 8 topics — merge related sub-topics into one broader topic
• Do NOT invent topics that are not present in the text
• Avoid: "Introduction", "Conclusion", "Overview", "Summary" — these are not topics

GOOD examples of topic granularity:
  "Monetary Policy"         (not "The ECB's 2023 Interest Rate Decision")
  "Machine Learning"        (not "Adam Optimizer Hyperparameters")
  "EU Institutional Design" (not "The role of the European Parliament in the legislative process")

OUTPUT: {"topics": [{"name": "Topic Name", "description": "One-sentence description of what this topic covers"}]}\
"""

_SCAN_USER = """\
Identify the 4–8 main topics of the following document excerpts.
These excerpts are representative samples from across the entire document.

<document_excerpts>
{text}
</document_excerpts>

Reply ONLY with the JSON object.\
"""

# Max words to send in a single scan call (well within gpt-4o-mini context)
_MAX_WORDS_PER_BATCH = 10_000
# Words to sample from each chunk for the overview
_WORDS_PER_CHUNK_SAMPLE = 350

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
    Scan all chunks holistically to discover the document's main topics.

    Returns a deduplicated list of dicts:
        [{"name": str, "description": str}, ...]
    """
    system = (
        _SCAN_SYSTEM
        + f"\n\nRespond in {language_name}. All topic names and descriptions must be in {language_name}."
    )

    # Build representative samples from every chunk
    samples: list[str] = []
    for chunk in chunks:
        p = chunk["pages"]
        label = f"pp.{p[0]}–{p[-1]}" if len(p) > 1 else f"p.{p[0]}"
        words = chunk["text"].split()
        sample = " ".join(words[:_WORDS_PER_CHUNK_SAMPLE])
        samples.append(f"[{label}]\n{sample}")

    # Split samples into batches that fit in one API call
    batches: list[list[str]] = []
    current_batch: list[str] = []
    current_words = 0
    for s in samples:
        wc = len(s.split())
        if current_words + wc > _MAX_WORDS_PER_BATCH and current_batch:
            batches.append(current_batch)
            current_batch = []
            current_words = 0
        current_batch.append(s)
        current_words += wc
    if current_batch:
        batches.append(current_batch)

    all_topics: list[dict] = []
    for batch in batches:
        combined = "\n\n---\n\n".join(batch)
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": _SCAN_USER.format(text=combined)},
                ],
                temperature=0.1,
                response_format={"type": "json_object"},
                max_tokens=1024,
            )
            raw = response.choices[0].message.content.strip()
            topics = _parse_topics(raw)
            all_topics.extend(topics)
        except Exception as e:
            print(f"[scanner] Error scanning batch: {e}")

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
            t.get("description", t.get("beschreibung", t.get("summary", "")))
        ).strip()
        if name and len(name) > 2:
            valid.append({"name": name, "description": desc})
    return valid


def _deduplicate_topics(topics: list[dict], threshold: float = 0.45) -> list[dict]:
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
