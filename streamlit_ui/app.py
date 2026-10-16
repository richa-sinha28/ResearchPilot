"""ResearchPilot - Streamlit interface.

Run from the project root:
    streamlit run streamlit_ui\\app.py

Backend contract (verified against researchpilot_backend.py):
    get_papers()            -> list[dict]  paper_id, title, authors, year,
                                           arxiv_id, source_url, chunk_count
    get_paper_count()       -> int
    generate_rag_answer(question, top_k=int)
                            -> dict        question, answer, sources,
                                           retrieval_insights
    Each source dict: title, authors, year, arxiv_id, page, source_url, text,
    chunk_id, score, rrf_score, semantic_rank, tfidf_rank, retrieved_by, label

The sidebar only calls get_papers()/get_paper_count(), which read the JSONL
chunks and never load FAISS, PyTorch, or any model. Models load on the first
question and stay cached.
"""

import html
import re
import time

import streamlit as st

# Must be the first Streamlit call in the script.
st.set_page_config(
    page_title="ResearchPilot",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --------------------------------------------------------------------------
# Backend import and contract check
# --------------------------------------------------------------------------
REQUIRED_BACKEND_FUNCTIONS = (
    "load_chunks",
    "get_papers",
    "get_paper_count",
    "load_pipeline",
    "generate_rag_answer",
    "generate_rag_answer_focused",
)

backend = None
BACKEND_PROBLEM = None  # (summary, details) shown to the user if set

try:
    import researchpilot_backend as backend
except Exception as exc:  # noqa: BLE001 - includes SyntaxError in the backend
    BACKEND_PROBLEM = (
        "The ResearchPilot backend could not be imported.",
        f"{type(exc).__name__}: {exc}",
    )

if backend is not None:
    missing = [
        name
        for name in REQUIRED_BACKEND_FUNCTIONS
        if not callable(getattr(backend, name, None))
    ]
    if missing:
        BACKEND_PROBLEM = (
            "The imported backend module is missing required functions: "
            + ", ".join(missing)
            + ". Python loaded a different or outdated researchpilot_backend.py "
            "than expected.",
            f"Loaded from: {getattr(backend, '__file__', 'unknown location')}",
        )

BACKEND_OK = BACKEND_PROBLEM is None

# --------------------------------------------------------------------------
# Design system (neutral, earthy, flat)
# --------------------------------------------------------------------------
PALETTE = {
    "forest": "#0F3D26",   # headings, primary buttons (deepest, boldest)
    "pine": "#1B5E3B",     # hover states, strong accents
    "emerald": "#1F8A4C",  # bright bold accent
    "fern": "#3E9B66",     # bars, borders, citation chips
    "line": "#7FA88A",     # soft green-grey borders
    "lime": "#A8E05F",     # highlight / mark
    "mist": "#F2F8F3",     # app background
    "mint": "#E1F0E4",     # sidebar and card background
    "ink": "#0B1F16",      # body text
}

st.markdown(
    f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

html, body, .stApp {{
    font-family: 'Inter', 'Segoe UI', system-ui, sans-serif;
    color: {PALETTE['ink']};
}}
.stApp {{ background: {PALETTE['mist']}; }}
header[data-testid="stHeader"] {{ background: transparent; border-top: 4px solid {PALETTE['forest']}; }}
.block-container {{ padding-top: 1.5rem; max-width: 1480px; }}

section[data-testid="stSidebar"] {{
    background: {PALETTE['mint']};
    border-right: 2px solid {PALETTE['forest']}33;
}}
.rp-brand {{ font-size: 1.25rem; font-weight: 700; color: {PALETTE['forest']}; letter-spacing: .2px; }}
.rp-brand-sub {{ font-size: .78rem; color: {PALETTE['ink']}99; margin-top: .1rem; }}
.rp-label {{ font-size: .72rem; text-transform: uppercase; letter-spacing: .08em;
             color: {PALETTE['pine']}; font-weight: 700; margin: 1.1rem 0 .4rem; }}

.rp-title {{ font-size: 1.85rem; font-weight: 700; color: {PALETTE['forest']}; margin: 0; }}
.rp-sub {{ color: {PALETTE['ink']}cc; font-size: .95rem; margin: .25rem 0 1rem; max-width: 760px; line-height: 1.5; }}
.rp-h2 {{ font-size: 1.1rem; font-weight: 700; color: {PALETTE['pine']}; margin: 1.3rem 0 .5rem;
          padding-left: .55rem; border-left: 4px solid {PALETTE['emerald']}; }}
.rp-hint {{ font-size: .8rem; color: {PALETTE['ink']}88; margin-top: .3rem; }}

div[data-testid="stForm"] {{
    border: 1px solid {PALETTE['line']}; background: {PALETTE['mint']};
    border-radius: 8px; padding: 1rem 1.1rem;
}}
div[data-testid="stForm"] textarea {{
    background: #fff; border: 1px solid {PALETTE['line']};
    border-radius: 6px; font-size: 1rem; line-height: 1.55; color: {PALETTE['ink']};
}}
div[data-testid="stForm"] textarea:focus {{
    border-color: {PALETTE['emerald']}; box-shadow: 0 0 0 2px {PALETTE['fern']}66;
}}
[data-testid="stFormSubmitButton"] button {{
    background: {PALETTE['forest']}; color: #fff;
    border: 1px solid {PALETTE['forest']}; border-radius: 6px; font-weight: 600; padding: .45rem 1.4rem;
}}
[data-testid="stFormSubmitButton"] button:hover {{
    background: {PALETTE['emerald']}; border-color: {PALETTE['emerald']}; color: #fff;
}}
.stButton > button {{
    background: transparent; color: {PALETTE['pine']}; border: 1px solid {PALETTE['pine']};
    border-radius: 6px; font-size: .82rem; font-weight: 500; min-height: 0; padding: .25rem .7rem;
}}
.stButton > button:hover {{ border-color: {PALETTE['forest']}; background: {PALETTE['forest']}; color: #fff; }}

.rp-cite {{
    display: inline-block; background: {PALETTE['forest']}; color: {PALETTE['lime']};
    border: 1px solid {PALETTE['forest']}; border-radius: 4px; padding: 0 .35rem;
    font-size: .74rem; font-weight: 700; margin: 0 .1rem; white-space: nowrap;
}}
.rp-empty {{
    border: 1px dashed {PALETTE['fern']}; border-radius: 8px; padding: 1.2rem;
    text-align: center; color: {PALETTE['ink']}aa; font-size: .92rem;
}}

.rp-card-head {{ font-size: .72rem; text-transform: uppercase; letter-spacing: .06em; color: {PALETTE['emerald']}; font-weight: 700; }}
.rp-card-title {{ font-size: .92rem; font-weight: 600; margin: .15rem 0 .25rem; line-height: 1.35; color: {PALETTE['forest']}; }}
.rp-card-meta {{ font-size: .76rem; color: {PALETTE['ink']}99; margin-bottom: .35rem; }}
.rp-card-text {{ font-size: .86rem; line-height: 1.55; }}
.rp-card-text mark {{ background: {PALETTE['lime']}; color: {PALETTE['ink']}; padding: 0 .15rem; border-radius: 2px; }}
.rp-bar-wrap {{ background: {PALETTE['line']}44; height: 6px; border-radius: 3px; overflow: hidden; margin: .3rem 0 .15rem; }}
.rp-bar {{ height: 100%; background: {PALETTE['emerald']}; border-radius: 3px; }}
.rp-rel {{ font-size: .75rem; color: {PALETTE['pine']}; font-weight: 500; }}

.rp-paper {{ padding: .5rem .1rem; border-bottom: 1px solid {PALETTE['line']}66; font-size: .84rem; line-height: 1.35; }}
.rp-paper-meta {{ font-size: .72rem; color: {PALETTE['ink']}88; margin-top: .1rem; }}
.rp-metric {{ display: flex; justify-content: space-between; gap: 1rem; font-size: .84rem; padding: .35rem 0; border-bottom: 1px solid {PALETTE['line']}55; }}
.rp-metric span:first-child {{ color: {PALETTE['ink']}99; }}
.rp-metric span:last-child {{ font-weight: 700; color: {PALETTE['forest']}; text-align: right; }}
</style>
""",
    unsafe_allow_html=True,
)
# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
EXAMPLE_QUESTIONS = [
    "How does retrieval-augmented generation use external knowledge?",
    "What are the main limitations of the approaches described in these papers?",
    "How do the papers evaluate retrieval quality?",
]

ARXIV_ID_RE = re.compile(r"^(\d{4}\.\d{4,5}|[a-z\-]+(\.[A-Za-z]{2})?/\d{7})(v\d+)?$")
CITE_RE = re.compile(
    r"\[\s*Source\s+(\d+)\s*(?:,\s*p\.?\s*([^\]]+?))?\s*\]", re.IGNORECASE
)
STOPWORDS = set(
    "the a an of and or to in on for with is are was were be by as at from "
    "that this it its what which how why does do".split()
)


def first(data, *keys, default=None):
    """Return the first non-empty value among keys in a dictionary."""
    if isinstance(data, dict):
        for key in keys:
            value = data.get(key)
            if value not in (None, "", []):
                return value
    return default


def fmt_authors(authors, limit=70):
    if isinstance(authors, (list, tuple)):
        authors = ", ".join(str(a) for a in authors)
    text = str(authors or "").strip()
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "\u2026"


def paper_link(url, arxiv_id):
    """Use the stored URL; otherwise derive an arXiv abstract link from a
    well-formed arXiv ID. Returns None when neither is available."""
    url = str(url or "").strip()
    if url.lower().startswith(("http://", "https://")):
        return url
    arxiv_id = str(arxiv_id or "").strip()
    if arxiv_id and ARXIV_ID_RE.match(arxiv_id):
        return f"https://arxiv.org/abs/{arxiv_id}"
    return None


def describe_error(exc):
    """Translate an exception into (level, message, details_or_None)."""
    name = type(exc).__name__

    if isinstance(exc, FileNotFoundError):
        return ("error", f"A required ResearchPilot file is missing. {exc}", None)

    if isinstance(exc, ValueError):
        return ("warning", str(exc), None)

    if name == "OutOfMemoryError":
        return (
            "error",
            "The GPU ran out of memory. Close other GPU applications or "
            "lower the number of retrieved passages, then try again.",
            None,
        )

    if isinstance(exc, RuntimeError):
        # The backend raises RuntimeError with actionable text for CUDA,
        # missing models, and missing packages.
        return ("error", str(exc), None)

    return (
        "error",
        "Something went wrong while generating the answer. "
        "Please try again; if it keeps happening, check the terminal running Streamlit.",
        f"{name}: {exc}",
    )


def fetch_papers():
    """Lightweight metadata only: never initialises the GPU pipeline."""
    try:
        papers = backend.get_papers()
        try:
            count = int(backend.get_paper_count())
        except Exception:  # noqa: BLE001
            count = len(papers)
        return papers, count, None
    except Exception as exc:  # noqa: BLE001
        return [], 0, describe_error(exc)


def normalise_source(raw, rank):
    rrf = first(raw, "rrf_score")
    raw_score = first(raw, "score")
    chosen = rrf if rrf is not None else raw_score
    try:
        score = float(chosen) if chosen is not None else None
    except (TypeError, ValueError):
        score = None

    return {
        "rank": rank,
        "title": str(first(raw, "title", default="Untitled paper")),
        "arxiv": first(raw, "arxiv_id"),
        "page": first(raw, "page"),
        "url": paper_link(first(raw, "source_url"), first(raw, "arxiv_id")),
        "text": str(first(raw, "text", default="")),
        "score": score,
        "score_kind": "RRF score" if rrf is not None else "Score",
        "retrieved_by": first(raw, "retrieved_by", default=[]),
    }


def call_backend(question, top_k):
    """Run the real pipeline and normalise its result for display."""
    started = time.perf_counter()
    out = backend.generate_rag_answer(question, top_k=top_k)
    wall_seconds = time.perf_counter() - started

    if not isinstance(out, dict):
        raise RuntimeError("The backend returned an unexpected response format.")

    sources = [
        normalise_source(s, i + 1) for i, s in enumerate(out.get("sources") or [])
    ]

    # Relative relevance is within-query only; it is not a probability.
    scores = [s["score"] for s in sources if s["score"] is not None]
    top = max(scores) if scores and max(scores) > 0 else None
    for s in sources:
        s["rel"] = s["score"] / top if top and s["score"] is not None else None

    insights = out.get("retrieval_insights")

    return {
        "question": question,
        "top_k": top_k,
        "answer": str(out.get("answer") or ""),
        "sources": sources,
        "insights": insights if isinstance(insights, dict) else {},
        "wall_seconds": wall_seconds,
    }


def tokens(text):
    return {
        w
        for w in re.findall(r"[a-z0-9]+", text.lower())
        if w not in STOPWORDS and len(w) > 2
    }


def highlight_passage(text, question):
    """Mark the sentence of the real passage that best overlaps the question."""
    if not text.strip():
        return ""
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    q = tokens(question)
    best, best_score = -1, 0
    for i, sentence in enumerate(sentences):
        score = len(q & tokens(sentence))
        if score > best_score:
            best, best_score = i, score
    parts = []
    for i, sentence in enumerate(sentences):
        escaped = html.escape(sentence)
        parts.append(f"<mark>{escaped}</mark>" if i == best else escaped)
    return " ".join(parts)


def render_answer(text, source_count):
    """Escape model output, then turn valid [Source N, p. X] markers into
    badges. Markers that point to a non-existent source are dropped."""
    safe = html.escape(text, quote=False)

    def replace(match):
        number = int(match.group(1))
        if not 1 <= number <= source_count:
            return ""
        page = (match.group(2) or "").strip()
        label = f"{number}, p. {page}" if page else str(number)
        return f'<span class="rp-cite">{label}</span>'

    return CITE_RE.sub(replace, safe)


def cited_numbers(text, source_count):
    return sorted(
        {
            int(n)
            for n in (m.group(1) for m in CITE_RE.finditer(text))
            if 1 <= int(n) <= source_count
        }
    )


def select_source(number):
    st.session_state.selected = number


def use_example(question):
    st.session_state.question_input = question


def show_error(error):
    level, message, details = error
    (st.warning if level == "warning" else st.error)(message)
    if details:
        with st.expander("Technical details"):
            st.code(details, language=None)


def metric_rows(rows):
    st.markdown(
        "".join(
            f'<div class="rp-metric"><span>{html.escape(str(k))}</span>'
            f"<span>{html.escape(str(v))}</span></div>"
            for k, v in rows
        ),
        unsafe_allow_html=True,
    )


def ms_to_text(value):
    if isinstance(value, (int, float)):
        return f"{value / 1000:.2f} s" if value >= 1000 else f"{value:.0f} ms"
    return None


# --------------------------------------------------------------------------
# Session state
# --------------------------------------------------------------------------
st.session_state.setdefault("result", None)
st.session_state.setdefault("error", None)
st.session_state.setdefault("selected", None)
st.session_state.setdefault("history", [])
st.session_state.setdefault("question_input", "")
st.session_state.setdefault("top_k", 3)

# --------------------------------------------------------------------------
# Sidebar
# --------------------------------------------------------------------------
with st.sidebar:
    st.markdown(
        '<div class="rp-brand">ResearchPilot</div>'
        '<div class="rp-brand-sub">Evidence-backed literature assistant</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="rp-label">Research Papers Used</div>', unsafe_allow_html=True
    )

    if not BACKEND_OK:
        st.error("Paper list unavailable: the backend is not connected.")
    else:
        with st.spinner("Reading paper metadata..."):
            papers, paper_count, papers_error = fetch_papers()

        if papers_error:
            show_error(papers_error)
        elif not papers:
            st.info("No indexed papers were found.")
        else:
            st.caption(f"{paper_count} indexed papers")

            search = st.text_input(
                "Search papers",
                placeholder="Search by title, author, year, or arXiv ID",
                label_visibility="collapsed",
                key="paper_search",
            )
            needle = search.strip().lower()

            def searchable(paper):
                return " ".join(
                    str(first(paper, key, default=""))
                    if key != "authors"
                    else fmt_authors(first(paper, key, default=""), limit=500)
                    for key in ("title", "authors", "year", "arxiv_id", "paper_id")
                ).lower()

            shown = [p for p in papers if not needle or needle in searchable(p)]

            with st.container(height=480, border=False):
                if not shown:
                    st.caption("No papers match your search.")
                for paper in shown:
                    title = html.escape(str(first(paper, "title", default="Untitled paper")))
                    meta_parts = [
                        fmt_authors(first(paper, "authors", default="")),
                        str(first(paper, "year", default="")),
                        str(first(paper, "arxiv_id", default="")),
                    ]
                    chunks = first(paper, "chunk_count")
                    if chunks:
                        meta_parts.append(f"{chunks} passages")
                    meta = " \u00b7 ".join(p for p in meta_parts if p)
                    st.markdown(
                        f'<div class="rp-paper">{title}'
                        + (
                            f'<div class="rp-paper-meta">{html.escape(meta)}</div>'
                            if meta
                            else ""
                        )
                        + "</div>",
                        unsafe_allow_html=True,
                    )

# --------------------------------------------------------------------------
# Header
# --------------------------------------------------------------------------
st.markdown(
    '<div class="rp-title">ResearchPilot</div>'
    '<div class="rp-sub">Ask a question about the indexed research papers. '
    "Answers are generated from retrieved passages and cite the sources they use.</div>",
    unsafe_allow_html=True,
)

if not BACKEND_OK:
    summary, details = BACKEND_PROBLEM
    st.error(summary)
    st.code(details, language=None)
    st.caption(
        "Check that researchpilot_backend.py sits next to app.py and is the "
        "latest version, then restart Streamlit."
    )
    st.stop()

# --------------------------------------------------------------------------
# Workspace
# --------------------------------------------------------------------------
main, side = st.columns([2.1, 1], gap="large")

with main:
    st.markdown('<div class="rp-h2">Research question</div>', unsafe_allow_html=True)

    st.caption("Example questions")
    example_cols = st.columns(len(EXAMPLE_QUESTIONS))
    for i, example in enumerate(EXAMPLE_QUESTIONS):
        example_cols[i].button(
            example,
            key=f"example_{i}",
            on_click=use_example,
            args=(example,),
            use_container_width=True,
        )

    with st.form("ask", clear_on_submit=False, border=False):
        question = st.text_area(
            "Research question",
            key="question_input",
            height=120,
            label_visibility="collapsed",
            placeholder="Ask a research question about the indexed literature...",
        )
        top_k = st.slider(
            "Retrieved passages (Top-K)",
            min_value=1,
            max_value=5,
            key="top_k",
            help="How many source passages are given to the language model.",
        )
        submitted = st.form_submit_button("Ask ResearchPilot")

    if submitted:
        st.session_state.selected = None
        clean_question = question.strip()

        if not clean_question:
            st.session_state.result = None
            st.session_state.error = ("warning", "Please enter a research question.", None)
        else:
            with st.spinner(
                "Retrieving passages and generating the answer. "
                "The first question also loads the models and can take a minute."
            ):
                try:
                    result = call_backend(clean_question, int(top_k))
                    st.session_state.result = result
                    st.session_state.error = None
                    st.session_state.history.insert(0, result)
                    del st.session_state.history[10:]
                except Exception as exc:  # noqa: BLE001
                    st.session_state.result = None
                    st.session_state.error = describe_error(exc)

    st.markdown('<div class="rp-h2">Answer</div>', unsafe_allow_html=True)
    result = st.session_state.result

    if st.session_state.error:
        show_error(st.session_state.error)
    elif result is None:
        st.markdown(
            '<div class="rp-empty">Your answer and its supporting evidence '
            "will appear here.</div>",
            unsafe_allow_html=True,
        )
    elif not result["sources"]:
        st.warning(
            "No relevant evidence was found in the indexed papers, so no "
            "source-supported answer can be given. Try rephrasing the question."
        )
    elif not result["answer"].strip():
        st.warning("The model returned an empty answer. Please try again.")
    else:
        with st.container(border=True):
            st.markdown(
                render_answer(result["answer"], len(result["sources"])),
                unsafe_allow_html=True,
            )

        cited = cited_numbers(result["answer"], len(result["sources"]))
        if cited:
            st.caption("Inspect a cited source")
            inspect_cols = st.columns(min(len(cited), 5))
            for i, number in enumerate(cited):
                source = result["sources"][number - 1]
                label = f"Source {number}"
                if source["page"]:
                    label += f", p. {source['page']}"
                inspect_cols[i % len(inspect_cols)].button(
                    label,
                    key=f"inspect_{number}",
                    on_click=select_source,
                    args=(number,),
                    use_container_width=True,
                )

    if len(st.session_state.history) > 1:
        st.markdown('<div class="rp-h2">Session history</div>', unsafe_allow_html=True)
        for past in st.session_state.history[1:]:
            with st.expander(past["question"][:110]):
                st.markdown(
                    render_answer(past["answer"], len(past["sources"]))
                    if past["answer"].strip()
                    else "_No answer returned._",
                    unsafe_allow_html=True,
                )

with side:
    result = st.session_state.result

    st.markdown('<div class="rp-h2">Retrieved sources</div>', unsafe_allow_html=True)
    if result is None:
        st.markdown(
            '<div class="rp-empty">Retrieved passages appear here after you '
            "ask a question.</div>",
            unsafe_allow_html=True,
        )
    elif not result["sources"]:
        st.info("No passages were retrieved for this question.")
    else:
        selected = st.session_state.selected
        with st.container(height=520, border=False):
            for source in result["sources"]:
                is_selected = selected == source["rank"]
                full = source["text"]

                if is_selected:
                    body = highlight_passage(full, result["question"])
                else:
                    body = html.escape(full[:320] + ("\u2026" if len(full) > 320 else ""))

                ref = " \u00b7 ".join(
                    str(x)
                    for x in (
                        source["arxiv"],
                        f"p. {source['page']}" if source["page"] else None,
                    )
                    if x
                )

                relevance = ""
                if source["rel"] is not None:
                    pct = round(source["rel"] * 100)
                    via = (
                        " \u00b7 " + ", ".join(source["retrieved_by"])
                        if source["retrieved_by"]
                        else ""
                    )
                    relevance = (
                        f'<div class="rp-bar-wrap"><div class="rp-bar" style="width:{pct}%"></div></div>'
                        f'<div class="rp-rel">{source["score_kind"]} {source["score"]:.4f}{html.escape(via)}</div>'
                    )

                with st.container(border=True):
                    st.markdown(
                        f'<div class="rp-card-head">Source {source["rank"]}'
                        + (" \u00b7 selected" if is_selected else "")
                        + "</div>"
                        f'<div class="rp-card-title">{html.escape(source["title"])}</div>'
                        + (
                            f'<div class="rp-card-meta">{html.escape(ref)}</div>'
                            if ref
                            else ""
                        )
                        + relevance
                        + f'<div class="rp-card-text">{body}</div>',
                        unsafe_allow_html=True,
                    )
                    if source["url"]:
                        st.markdown(f"[Open paper]({source['url']})")
                    if not is_selected and len(full) > 320:
                        with st.expander("Full passage"):
                            st.text(full)

        st.caption(
            "The bar shows each passage's score relative to the top result of "
            "this query. It is a ranking signal, not a probability."
        )

    # ---- Retrieval insights --------------------------------------------
    st.markdown('<div class="rp-h2">Retrieval insights</div>', unsafe_allow_html=True)
    if result is None:
        st.caption("Insights appear after a question is processed.")
    else:
        insights = result["insights"]
        sources = result["sources"]

        unique_papers = insights.get("unique_papers")
        if unique_papers is None:
            unique_papers = len({(s["arxiv"] or s["title"]) for s in sources})

        methods = insights.get("retrieval_methods")
        rows = [
            ("Retrieved passages", insights.get("retrieved_count", len(sources))),
            ("Requested (Top-K)", result["top_k"]),
            ("Unique papers", unique_papers),
        ]
        if methods:
            rows.append(("Retrieval methods", " + ".join(str(m) for m in methods)))

        for key, label in (
            ("retrieval_latency_ms", "Retrieval latency"),
            ("generation_latency_ms", "Generation latency"),
            ("total_latency_ms", "Total backend latency"),
        ):
            text = ms_to_text(insights.get(key))
            if text:
                rows.append((label, text))

        metric_rows(rows)
        st.caption(
            "Latencies exclude one-time model loading, which happens during "
            "the first question after the app starts."
        )