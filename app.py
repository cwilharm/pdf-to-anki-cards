"""PDF → Anki — Streamlit app."""

import os
import re
import time
import tempfile
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
import openai

from pipeline.extractor import extract_text_from_pdf
from pipeline.chunker import create_chunks
from pipeline.generator import generate_cards_for_chunk
from pipeline.quality import filter_and_deduplicate
from pipeline.exporter import create_anki_deck

load_dotenv()

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

LANGUAGES = [
    "Deutsch",
    "English",
    "Español",
    "Français",
    "Italiano",
    "Português",
    "Nederlands",
    "Polski",
    "Čeština",
    "Русский",
    "Türkçe",
    "Svenska",
    "Norsk",
    "Dansk",
    "Suomi",
    "中文 (Mandarin)",
    "日本語",
    "한국어",
    "العربية",
    "हिन्दी",
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
_CLOZE_PATTERN = re.compile(r"\{\{c\d+::(.+?)\}\}")


def _render_cloze(text: str) -> str:
    return _CLOZE_PATTERN.sub(r'<span class="cloze-blank">\1</span>', text)


def _render_back(text: str) -> str:
    return text.replace("\n", "<br>")


def _fmt_time(secs: float) -> str:
    """Format seconds as M:SS or Xs."""
    s = max(0, int(secs))
    m, sec = divmod(s, 60)
    if m:
        return f"{m}:{sec:02d}"
    return f"{s}s"


def _mk_status(step: int, name: str, detail: str, elapsed: str, eta: str) -> str:
    """Render the rich status box as HTML."""
    dots = "".join(
        (
            '<span style="color:#22c55e;font-size:15px;letter-spacing:2px">●</span>'
            if s < step
            else (
                '<span style="color:#3b82f6;font-size:15px;letter-spacing:2px">●</span>'
                if s == step
                else '<span style="color:#cbd5e1;font-size:15px;letter-spacing:2px">○</span>'
            )
        )
        for s in range(1, 5)
    )
    timing_parts = []
    if elapsed:
        timing_parts.append(f"⏱&nbsp;{elapsed} elapsed")
    if eta and eta != "—":
        timing_parts.append(f"ETA&nbsp;~{eta}")
    timing = "&nbsp;&nbsp;·&nbsp;&nbsp;".join(timing_parts)

    detail_html = (
        f'<div style="color:#0369a1;margin-top:4px;font-size:13px">{detail}</div>'
        if detail
        else ""
    )
    timing_html = (
        f'<div style="color:#64748b;margin-top:5px;font-size:12px">{timing}</div>'
        if timing
        else ""
    )

    return (
        f'<div style="background:#f0f9ff;border:1px solid #bae6fd;border-radius:9px;'
        f'padding:13px 18px;font-size:14px;line-height:1.7">'
        f'<div style="display:flex;align-items:center;gap:10px">'
        f"<span>{dots}</span>"
        f'<span style="color:#64748b;font-size:11px;white-space:nowrap">Step {step}/4</span>'
        f"<strong>{name}</strong>"
        f"</div>{detail_html}{timing_html}</div>"
    )


# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="PDF → Anki",
    page_icon="📚",
    layout="centered",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .block-container { padding-top: 2rem; }

    .card-box {
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 14px 18px;
        margin-bottom: 8px;
        background: #f8fafc;
    }
    .card-q {
        font-weight: 700;
        font-size: 15px;
        color: #0f172a;
        margin-bottom: 5px;
    }
    .card-a {
        font-size: 14px;
        color: #475569;
        line-height: 1.65;
    }
    .badge {
        display: inline-block;
        border-radius: 4px;
        padding: 1px 7px;
        font-size: 11px;
        font-weight: 600;
        margin-right: 6px;
        vertical-align: middle;
    }
    .badge-basic { background: #3b82f6; color: #fff; }
    .badge-cloze { background: #7c3aed; color: #fff; }
    .badge-num   { background: #e2e8f0; color: #334155; }
    .cloze-blank {
        background: #fef08a;
        border-radius: 3px;
        padding: 0 3px;
        font-weight: 600;
        color: #713f12;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.title("⚙️ Settings")

    api_key = st.text_input(
        "OpenAI API Key",
        type="password",
        value=os.getenv("OPENAI_API_KEY", ""),
        help="Used only for this session — never stored.",
    )

    st.divider()

    language_name = st.selectbox(
        "Card language",
        options=LANGUAGES,
        index=1,  # English default
        help="The AI will write all card content in this language.",
    )

    st.divider()

    st.markdown("**Model:** gpt-4o-mini")
    st.info(
        "Fast, precise & incredibly cheap.\n\n"
        "**Estimated cost:**\n"
        "- 50 pages ≈ $0.03\n"
        "- 100 pages ≈ $0.06\n"
        "- 200 pages ≈ $0.12"
    )
    model = "gpt-4o-mini"

    st.divider()
    st.caption("PDF → Anki · powered by OpenAI")

# ---------------------------------------------------------------------------
# Main area
# ---------------------------------------------------------------------------
st.title("PDF → Anki Cards")
st.markdown(
    "Transform any PDF into exam-ready Anki flashcards — "
    "powered by OpenAI Gpt-4o-mini. **Let's go!** 🚀"
)

uploaded_file = st.file_uploader(
    "Drop your PDF here and let's get started!",
    type=["pdf"],
    label_visibility="collapsed",
)

if uploaded_file:
    # ── Options row ──────────────────────────────────────────────────────────
    col_name, col_type, col_fmt = st.columns([2, 2, 2])

    with col_name:
        deck_name = st.text_input(
            "Deck name",
            value=Path(uploaded_file.name).stem.replace("_", " ").replace("-", " "),
        )

    with col_type:
        card_type_label = st.selectbox(
            "Card type",
            options=["Basic (Q&A)", "Cloze (fill-in-blank)", "Both (auto)"],
            index=0,
            help=(
                "**Basic**: classic question & answer cards.\n"
                "**Cloze**: fill-in-the-blank — key term is hidden.\n"
                "**Both**: We let the AI pick the best card type."
            ),
        )
        card_type_map = {
            "Basic (Q&A)": "basic",
            "Cloze (fill-in-blank)": "cloze",
            "Both (auto)": "both",
        }
        card_type = card_type_map[card_type_label]

    with col_fmt:
        if card_type == "cloze":
            st.selectbox("Answer format", ["— (Cloze only)"], disabled=True)
            answer_format = "sentences"
        else:
            ans_label = st.selectbox(
                "Answer format",
                options=["Full sentences", "Bullet points"],
                index=0,
                help=(
                    "**Sentences**: cohesive, connected explanations.\n"
                    "**Bullets**: tight 2–3 point summaries."
                ),
            )
            answer_format = "bullets" if ans_label == "Bullet points" else "sentences"

    start = st.button("⚡ Generate My Cards!", type="primary", use_container_width=True)

    if start:
        if not api_key:
            st.error("Please add your OpenAI API key in the sidebar first.")
            st.stop()

        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name

        try:
            client = openai.OpenAI(api_key=api_key)

            progress_bar = st.progress(0.0)
            status_box = st.empty()

            # ── Step 1: Extract ──────────────────────────────────────────
            progress_bar.progress(0.04)
            status_box.markdown(
                _mk_status(1, "Reading your PDF...", "", "", ""),
                unsafe_allow_html=True,
            )

            pages = extract_text_from_pdf(tmp_path)
            if not pages:
                st.error(
                    "No text could be extracted from this PDF. "
                    "The document may be blank or in an unsupported format."
                )
                st.stop()

            progress_bar.progress(0.12)
            st.success(
                f"✅ {len(pages)} pages loaded — nice, this is going to be good!"
            )

            # ── Step 2: Chunk ────────────────────────────────────────────
            progress_bar.progress(0.14)
            status_box.markdown(
                _mk_status(2, "Slicing into chunks...", "", "", ""),
                unsafe_allow_html=True,
            )
            chunks = create_chunks(pages, max_words_per_chunk=2000)
            total_chunks = len(chunks)
            progress_bar.progress(0.16)

            # ── Step 3: Generate ─────────────────────────────────────────
            all_cards: list[dict] = []
            gen_start = time.time()

            for i, chunk in enumerate(chunks):
                p0, p1 = chunk["pages"][0], chunk["pages"][-1]
                page_label = f"Page {p0}" if p0 == p1 else f"Pages {p0}–{p1}"

                # ── Pre-call: show which chunk is starting ────────────────
                elapsed = time.time() - gen_start
                if i > 0:
                    rate = i / max(elapsed, 0.001)
                    eta_str = _fmt_time((total_chunks - i) / rate)
                else:
                    eta_str = "—"

                progress_bar.progress(0.16 + (i / total_chunks) * 0.64)
                status_box.markdown(
                    _mk_status(
                        3,
                        "AI is crafting your cards ✨",
                        f"Chunk {i + 1}/{total_chunks} &nbsp;({page_label})"
                        + (
                            f"&nbsp;·&nbsp; {len(all_cards)} cards so far"
                            if all_cards
                            else ""
                        ),
                        _fmt_time(elapsed),
                        eta_str,
                    ),
                    unsafe_allow_html=True,
                )

                # ── API call ──────────────────────────────────────────────
                cards = generate_cards_for_chunk(
                    chunk,
                    client,
                    language_name=language_name,
                    model=model,
                    card_type=card_type,
                    answer_format=answer_format,
                )
                all_cards.extend(cards)

                # ── Post-call: refresh elapsed + ETA with real timing ─────
                elapsed = time.time() - gen_start
                remaining = total_chunks - (i + 1)
                eta_str = (
                    _fmt_time(remaining / max((i + 1) / elapsed, 0.001))
                    if remaining > 0
                    else ""
                )

                progress_bar.progress(0.16 + ((i + 1) / total_chunks) * 0.64)
                status_box.markdown(
                    _mk_status(
                        3,
                        "AI is crafting your cards ✨",
                        f"Chunk {i + 1}/{total_chunks} &nbsp;({page_label})"
                        f"&nbsp;·&nbsp; {len(all_cards)} cards collected",
                        _fmt_time(elapsed),
                        eta_str,
                    ),
                    unsafe_allow_html=True,
                )

            if not all_cards:
                st.error("No cards generated. Please check your API key and try again.")
                st.stop()

            total_gen_time = _fmt_time(time.time() - gen_start)
            progress_bar.progress(0.82)
            n_basic_raw = sum(1 for c in all_cards if "front" in c)
            n_cloze_raw = sum(1 for c in all_cards if "text" in c)
            st.success(
                f"✅ {len(all_cards)} raw cards generated "
                f"({n_basic_raw} Basic, {n_cloze_raw} Cloze) in {total_gen_time} — you're on fire! 🔥"
            )

            # ── Step 4: Quality control ──────────────────────────────────
            progress_bar.progress(0.86)
            status_box.markdown(
                _mk_status(
                    4, "Quality check & dedup — keeping only the best", "", "", ""
                ),
                unsafe_allow_html=True,
            )
            filtered = filter_and_deduplicate(all_cards)
            removed = len(all_cards) - len(filtered)

            if removed:
                st.info(
                    f"ℹ️ {removed} duplicate / low-quality cards removed — quality over quantity!"
                )

            # ── Step 5: Export ───────────────────────────────────────────
            progress_bar.progress(0.94)
            status_box.markdown(
                _mk_status(4, "Packaging your Anki deck (.apkg)...", "", "", ""),
                unsafe_allow_html=True,
            )
            deck_bytes = create_anki_deck(filtered, deck_name=deck_name or "Anki Deck")

            progress_bar.progress(1.0)
            status_box.markdown(
                _mk_status(4, "Done — your cards are ready!", "", "", ""),
                unsafe_allow_html=True,
            )

            # ── Result ───────────────────────────────────────────────────
            st.balloons()
            n_basic = sum(1 for c in filtered if "front" in c)
            n_cloze = sum(1 for c in filtered if "text" in c)
            parts = []
            if n_basic:
                parts.append(f"{n_basic} Basic")
            if n_cloze:
                parts.append(f"{n_cloze} Cloze")
            st.markdown(
                f"### 🎉 {len(filtered)} cards created — {' + '.join(parts)}\n"
                f"You're going to absolutely rock your next review session!"
            )

            st.download_button(
                label="⬇️ Download Your Deck & Crush That Exam!",
                data=deck_bytes,
                file_name=f"{deck_name or 'anki_deck'}.apkg",
                mime="application/octet-stream",
                use_container_width=True,
            )

            # ── Preview ──────────────────────────────────────────────────
            with st.expander(f"👁️ Preview all {len(filtered)} cards"):
                for idx, card in enumerate(filtered, 1):
                    if "front" in card:
                        st.markdown(
                            f'<div class="card-box">'
                            f'<div class="card-q">'
                            f'<span class="badge badge-num">#{idx}</span>'
                            f'<span class="badge badge-basic">Basic</span>'
                            f"{card['front']}</div>"
                            f'<div class="card-a">{_render_back(card["back"])}</div>'
                            f"</div>",
                            unsafe_allow_html=True,
                        )
                    else:
                        st.markdown(
                            f'<div class="card-box">'
                            f'<div class="card-q">'
                            f'<span class="badge badge-num">#{idx}</span>'
                            f'<span class="badge badge-cloze">Cloze</span>'
                            f"</div>"
                            f'<div class="card-a">{_render_cloze(card["text"])}</div>'
                            f"</div>",
                            unsafe_allow_html=True,
                        )

        except RuntimeError as e:
            st.error(str(e))
        except openai.AuthenticationError:
            st.error("Invalid OpenAI API key — please double-check and try again.")
        except openai.RateLimitError:
            st.error("OpenAI rate limit hit. Wait a moment and try again.")
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

else:
    st.markdown(
        """
        ---
        **How it works — it's dead simple:**

        1. **Upload your PDF** — textbooks, lecture slides, papers, study guides, anything
        2. **Pick your options** — language, card type & answer format
        3. **Hit Generate** — the AI reads every page and builds exam-ready cards
        4. **Download the .apkg** and import it into Anki (**File → Import**)

        **Card types:**
        - **Basic** — classic question & answer
        - **Cloze** — fill-in-the-blank (key term is hidden during review)
        - **Both** — AI auto-selects the best type per concept

        > No more manual card creation. No more staring at a blank Anki deck.
        > Just drop your PDF and **go crush that exam.** 🏆
        """
    )
