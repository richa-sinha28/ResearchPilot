"""Backend for the ResearchPilot Streamlit interface.

Loads saved retrieval artifacts and a locally cached language model.
Provides paper metadata, hybrid retrieval, answer generation, and
retrieval insights without modifying the existing Jupyter notebooks.

Design notes
------------
* Light operations (load_chunks, get_papers, get_paper_count) never
  import or load FAISS, PyTorch, or any model, so the Streamlit sidebar
  renders instantly and works even if the GPU stack is unavailable.
* Heavy libraries (faiss, torch, sentence-transformers, transformers,
  scikit-learn) are imported lazily inside load_pipeline().
* The FAISS index is only ever read, never rebuilt or overwritten.
* Models are loaded with local_files_only=True; nothing is downloaded.
"""

import json
import os
import re
import time
from functools import lru_cache
from pathlib import Path

import numpy as np

# Streamlit is used ONLY for caching the heavyweight pipeline so it
# survives Streamlit's module reloads. The backend still works without it
# (for example in a plain Jupyter notebook or a unit test).
try:
    import streamlit as st

    _cache_resource = st.cache_resource(show_spinner=False)
except Exception:  # pragma: no cover - streamlit not installed

    def _cache_resource(func):
        return lru_cache(maxsize=1)(func)


# ---------------------------------------------------------
# 1. PATHS AND MODEL CONFIGURATION
# ---------------------------------------------------------

def _find_project_dir():
    """Locate the project root that contains artifacts/.

    Order of precedence:
      1. RESEARCHPILOT_PROJECT_DIR environment variable.
      2. The first folder, walking upward from this file, that contains
         artifacts/semantic_chunks.jsonl.
      3. The original assumption: one directory above this file's folder.
    """
    env_dir = os.environ.get("RESEARCHPILOT_PROJECT_DIR")
    if env_dir:
        return Path(env_dir).expanduser().resolve()

    here = Path(__file__).resolve().parent
    for candidate in (here, *here.parents):
        if (candidate / "artifacts" / "semantic_chunks.jsonl").exists():
            return candidate

    return here.parent


PROJECT_DIR = _find_project_dir()
ARTIFACTS_DIR = PROJECT_DIR / "artifacts"

CHUNKS_PATH = ARTIFACTS_DIR / "semantic_chunks.jsonl"
FAISS_PATH = ARTIFACTS_DIR / "semantic_faiss.index"

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"

RRF_K = 60  # standard reciprocal rank fusion constant
RETRIEVAL_METHODS = ["FAISS", "TF-IDF", "RRF"]

# Alternative key names accepted for the passage text. The first one that
# holds a non-empty string is normalised into chunk["text"].
_TEXT_KEYS = ("text", "passage", "content", "chunk_text")


# ---------------------------------------------------------
# 2. LOAD THE SAVED CHUNKS
# ---------------------------------------------------------

def _chunks_file_signature():
    """Return (path, mtime_ns) so the cache refreshes if the file changes."""
    try:
        return str(CHUNKS_PATH), CHUNKS_PATH.stat().st_mtime_ns
    except OSError:
        return str(CHUNKS_PATH), None


@lru_cache(maxsize=4)
def _load_chunks_cached(path_str, mtime_ns):
    """Parse the JSONL file. Cached on (path, modification time)."""
    path = Path(path_str)

    if not path.exists():
        raise FileNotFoundError(
            f"ResearchPilot chunks file not found: {path}. "
            "Expected artifacts/semantic_chunks.jsonl in the project root."
        )

    chunks = []

    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue

            try:
                chunk = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON in {path}, line {line_number}: {exc}"
                ) from exc

            if not isinstance(chunk, dict):
                raise ValueError(
                    f"Expected a JSON object in {path}, line {line_number}, "
                    f"but found {type(chunk).__name__}."
                )

            text = None
            for key in _TEXT_KEYS:
                value = chunk.get(key)
                if isinstance(value, str) and value.strip():
                    text = value
                    break

            if text is None:
                raise ValueError(
                    f"Missing or empty passage text in {path}, line "
                    f"{line_number}. Looked for keys: "
                    f"{', '.join(_TEXT_KEYS)}."
                )

            chunk["text"] = text
            chunks.append(chunk)

    if not chunks:
        raise ValueError(f"No indexed passages were found in {path}.")

    return chunks


def load_chunks():
    """Load indexed passages once, without loading GPU models.

    The returned list is shared (cached); treat it as read-only. Order is
    preserved exactly, because FAISS row i must match chunk i.
    """
    return _load_chunks_cached(*_chunks_file_signature())


# ---------------------------------------------------------
# 3. CHUNK METADATA HELPERS
# ---------------------------------------------------------

def _meta(chunk, field, *aliases):
    """Read a metadata field stored directly on the chunk or inside a
    nested "metadata" dictionary. Direct values win; the first non-empty
    value across the field name and its aliases is returned."""
    nested = chunk.get("metadata")
    if not isinstance(nested, dict):
        nested = {}

    for name in (field, *aliases):
        for container in (chunk, nested):
            value = container.get(name)
            if value is not None and value != "" and value != []:
                return value

    return None


def _clean_str(value):
    return str(value).strip() if value is not None else ""


def _paper_key(chunk):
    """Identify a paper even when some metadata is missing."""
    arxiv_id = _clean_str(_meta(chunk, "arxiv_id"))
    if arxiv_id:
        return arxiv_id

    paper_id = _clean_str(_meta(chunk, "paper_id"))
    if paper_id:
        return paper_id

    url = _clean_str(_meta(chunk, "source_url", "url"))
    if url:
        return url

    title = _clean_str(_meta(chunk, "title"))
    if title:
        return title.casefold()

    source = _clean_str(_meta(chunk, "source"))
    if source:
        return source

    return "unknown"


def _passage_key(chunk):
    """Create a key for matching the same passage across retrievers."""
    return (
        _paper_key(chunk),
        str(_meta(chunk, "page", "page_number")),
        chunk["text"].strip(),
    )


# ---------------------------------------------------------
# 4. GET PAPERS FOR THE LEFT SIDEBAR
# ---------------------------------------------------------

@lru_cache(maxsize=4)
def _get_papers_cached(path_str, mtime_ns):
    chunks = _load_chunks_cached(path_str, mtime_ns)
    papers = {}

    for chunk in chunks:
        key = _paper_key(chunk)
        title = _clean_str(_meta(chunk, "title"))

        if key not in papers:
            papers[key] = {
                "paper_id": key,
                "title": title or key,
                "authors": _meta(chunk, "authors", "author") or "",
                "year": _meta(chunk, "year") or "",
                "arxiv_id": _clean_str(_meta(chunk, "arxiv_id")),
                "source_url": _clean_str(_meta(chunk, "source_url", "url")),
                "chunk_count": 0,
            }

        paper = papers[key]
        paper["chunk_count"] += 1

        # Fill in metadata that an earlier chunk of the paper lacked.
        if not paper["title"] or paper["title"] == key:
            if title:
                paper["title"] = title

        if not paper["authors"]:
            paper["authors"] = _meta(chunk, "authors", "author") or ""

        if not paper["year"]:
            paper["year"] = _meta(chunk, "year") or ""

        if not paper["arxiv_id"]:
            paper["arxiv_id"] = _clean_str(_meta(chunk, "arxiv_id"))

        if not paper["source_url"]:
            paper["source_url"] = _clean_str(
                _meta(chunk, "source_url", "url")
            )

    return tuple(
        sorted(papers.values(), key=lambda paper: paper["title"].casefold())
    )


def get_papers():
    """Return one dictionary per unique paper in the indexed corpus.

    Keys: paper_id, title, authors, year, arxiv_id, source_url,
    chunk_count. This never loads GPU models or touches FAISS.
    Returns fresh copies, so callers may mutate the result safely.
    """
    return [dict(paper) for paper in _get_papers_cached(*_chunks_file_signature())]


def get_paper_count():
    """Return the number of unique papers in the indexed corpus."""
    return len(get_papers())


# ---------------------------------------------------------
# 5. LOAD THE RETRIEVAL PIPELINE AND LOCAL LLM
# ---------------------------------------------------------

@_cache_resource
def load_pipeline():
    """Load retrieval components and the local language model once."""

    # Fail early if an artifact is missing.
    for path in (CHUNKS_PATH, FAISS_PATH):
        if not path.exists():
            raise FileNotFoundError(
                f"Required ResearchPilot artifact was not found: {path}"
            )

    # Heavy imports are deferred so the sidebar never pays for them.
    try:
        import faiss
        import torch
        from sentence_transformers import SentenceTransformer
        from sklearn.feature_extraction.text import TfidfVectorizer
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(
            f"A required package is missing: {exc}. Install it in the "
            "same environment that runs Streamlit, e.g. "
            "'pip install faiss-cpu sentence-transformers scikit-learn "
            "transformers' plus a CUDA build of PyTorch."
        ) from exc

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is unavailable to PyTorch, but ResearchPilot is "
            "configured to use an NVIDIA GPU. Check that (1) the NVIDIA "
            "driver is installed ('nvidia-smi' works), (2) PyTorch is a "
            "CUDA build (torch.version.cuda is not None; installed "
            f"torch: {torch.__version__}), and (3) Streamlit is launched "
            "from the same virtual environment."
        )

    chunks = load_chunks()

    # Load the existing index; never rebuild or write it here.
    semantic_index = faiss.read_index(str(FAISS_PATH))

    if semantic_index.ntotal != len(chunks):
        raise ValueError(
            "FAISS index/chunk count mismatch: "
            f"{semantic_index.ntotal} vectors versus {len(chunks)} chunks. "
            "Check that semantic_faiss.index and semantic_chunks.jsonl "
            "come from the same indexing run."
        )

    # Build the lightweight TF-IDF representation in memory.
    texts = [chunk["text"] for chunk in chunks]

    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=(1, 2),
        max_df=0.98,
        sublinear_tf=True,
    )
    tfidf_matrix = vectorizer.fit_transform(texts)

    # Use only locally cached models; never download unexpectedly.
    try:
        try:
            embedding_model = SentenceTransformer(
                EMBEDDING_MODEL_NAME,
                device="cuda",
                local_files_only=True,
            )
        except TypeError:
            # Older sentence-transformers without local_files_only.
            os.environ["HF_HUB_OFFLINE"] = "1"
            embedding_model = SentenceTransformer(
                EMBEDDING_MODEL_NAME,
                device="cuda",
            )

        tokenizer = AutoTokenizer.from_pretrained(
            LLM_MODEL_NAME,
            local_files_only=True,
        )

        llm = AutoModelForCausalLM.from_pretrained(
            LLM_MODEL_NAME,
            torch_dtype=torch.float16,
            local_files_only=True,
        )
        llm.to("cuda")
        llm.eval()
    except OSError as exc:
        raise RuntimeError(
            "A model was not found in the local Hugging Face cache "
            f"({exc}). Models are loaded with local_files_only=True and "
            "are never downloaded automatically. Download "
            f"'{EMBEDDING_MODEL_NAME}' and '{LLM_MODEL_NAME}' once while "
            "online (e.g. with huggingface-cli download), then retry."
        ) from exc

    # The saved index must match the embedding model's dimension.
    model_dim = embedding_model.get_sentence_embedding_dimension()
    if model_dim is not None and semantic_index.d != model_dim:
        raise ValueError(
            f"FAISS index dimension ({semantic_index.d}) does not match "
            f"{EMBEDDING_MODEL_NAME} output dimension ({model_dim}). "
            "The index was built with a different embedding model."
        )

    return {
        "chunks": chunks,
        "semantic_index": semantic_index,
        "embedding_model": embedding_model,
        "vectorizer": vectorizer,
        "tfidf_matrix": tfidf_matrix,
        "tokenizer": tokenizer,
        "llm": llm,
    }


# ---------------------------------------------------------
# 6. SHARED SOURCE METADATA
# ---------------------------------------------------------

def format_source(chunk, score, **extra):
    """Create a consistent source dictionary for the UI."""

    source = {
        "title": _clean_str(_meta(chunk, "title")) or "Untitled paper",
        "authors": _meta(chunk, "authors", "author"),
        "year": _meta(chunk, "year"),
        "arxiv_id": _meta(chunk, "arxiv_id"),
        "page": _meta(chunk, "page", "page_number"),
        "source_url": _meta(chunk, "source_url", "url"),
        "text": chunk["text"],
        "chunk_id": _meta(chunk, "chunk_id", "id"),
        "score": float(score),
    }

    source.update(extra)
    return source


# ---------------------------------------------------------
# 7. SEMANTIC RETRIEVAL USING FAISS
# ---------------------------------------------------------

def search_semantic(query, pipeline, top_k=15):
    """Retrieve passages using the saved FAISS semantic index."""

    if top_k <= 0:
        return []

    query_embedding = pipeline["embedding_model"].encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    query_embedding = np.ascontiguousarray(
        query_embedding,
        dtype=np.float32,
    )

    index = pipeline["semantic_index"]
    count = min(top_k, index.ntotal)

    if count <= 0:
        return []

    scores, indices = index.search(query_embedding, count)

    results = []

    for score, idx in zip(scores[0], indices[0]):
        if idx < 0:  # FAISS pads with -1 when fewer hits exist
            continue

        chunk = pipeline["chunks"][int(idx)]
        results.append(format_source(chunk, score))

    return results


# ---------------------------------------------------------
# 8. KEYWORD RETRIEVAL USING TF-IDF
# ---------------------------------------------------------

def search_tfidf_diverse(
    query,
    pipeline,
    top_k=15,
    max_per_paper=15,
):
    """Retrieve keyword-relevant passages using TF-IDF.

    At most max_per_paper passages are returned per paper, so one long
    paper cannot crowd out all others.
    """

    if top_k <= 0 or max_per_paper <= 0:
        return []

    query_vector = pipeline["vectorizer"].transform([query])

    # TfidfVectorizer L2-normalises rows by default, so this dot product
    # equals cosine similarity.
    scores = (pipeline["tfidf_matrix"] @ query_vector.T).toarray().ravel()

    ranked_indices = np.argsort(scores)[::-1]

    results = []
    papers_seen = {}

    for idx in ranked_indices:
        score = float(scores[idx])

        if score <= 0:
            break

        chunk = pipeline["chunks"][int(idx)]
        paper_id = _paper_key(chunk)

        if papers_seen.get(paper_id, 0) >= max_per_paper:
            continue

        results.append(format_source(chunk, score))
        papers_seen[paper_id] = papers_seen.get(paper_id, 0) + 1

        if len(results) >= top_k:
            break

    return results


# ---------------------------------------------------------
# 9. HYBRID RETRIEVAL WITH RECIPROCAL RANK FUSION
# ---------------------------------------------------------

def build_rag_context_hybrid(
    query,
    pipeline,
    top_k=3,
    candidate_k=15,
    max_per_paper=3,
):
    """Fuse semantic and TF-IDF rankings using reciprocal rank fusion.

    Passages found by both retrievers are merged into one entry whose RRF
    score is the sum of both contributions. Each returned dictionary is a
    source dictionary plus: rrf_score, semantic_rank, tfidf_rank,
    semantic_score, tfidf_score, and retrieved_by.
    """

    if top_k <= 0 or candidate_k <= 0 or max_per_paper <= 0:
        raise ValueError(
            "top_k, candidate_k, and max_per_paper must be positive."
        )

    semantic_results = search_semantic(
        query,
        pipeline,
        top_k=candidate_k,
    )

    tfidf_results = search_tfidf_diverse(
        query,
        pipeline,
        top_k=candidate_k,
        max_per_paper=candidate_k,
    )

    fused = {}

    for method, results in (
        ("semantic", semantic_results),
        ("tfidf", tfidf_results),
    ):
        for rank, item in enumerate(results, start=1):
            key = _passage_key(item)

            if key not in fused:
                fused[key] = {
                    **item,
                    "rrf_score": 0.0,
                    "semantic_rank": None,
                    "tfidf_rank": None,
                    "semantic_score": None,
                    "tfidf_score": None,
                }

            fused[key]["rrf_score"] += 1.0 / (RRF_K + rank)
            fused[key][f"{method}_rank"] = rank
            fused[key][f"{method}_score"] = item["score"]

    candidates = sorted(
        fused.values(),
        key=lambda item: item["rrf_score"],
        reverse=True,
    )

    selected = []
    paper_counts = {}

    for item in candidates:
        paper_id = _paper_key(item)

        if paper_counts.get(paper_id, 0) >= max_per_paper:
            continue

        item["retrieved_by"] = [
            name
            for name, rank in (
                ("FAISS", item["semantic_rank"]),
                ("TF-IDF", item["tfidf_rank"]),
            )
            if rank is not None
        ]

        selected.append(item)
        paper_counts[paper_id] = paper_counts.get(paper_id, 0) + 1

        if len(selected) >= top_k:
            break

    return selected


# ---------------------------------------------------------
# 10. CITATION CHECKING
# ---------------------------------------------------------

_CITATION_RE = re.compile(
    r"\[\s*Source\s+(\d+)\s*(?:,\s*p\.?\s*[^\]]*)?\]",
    re.IGNORECASE,
)


def _normalize_citations(answer, sources):
    """Make every citation in the answer match a real retrieved source.

    * [Source N, p. X] with a valid N is rewritten using the true page
      from source N (or [Source N] when the page is unknown), so the model
      cannot misquote page numbers.
    * Citations pointing to a source number that does not exist are removed.

    Returns (clean_answer, stats_dict).
    """
    valid_numbers = set(range(1, len(sources) + 1))
    stats = {"valid": 0, "removed": 0}

    def _replace(match):
        number = int(match.group(1))

        if number not in valid_numbers:
            stats["removed"] += 1
            return ""

        stats["valid"] += 1
        page = sources[number - 1].get("page")

        if page is None or str(page).strip() == "":
            return f"[Source {number}]"

        return f"[Source {number}, p. {page}]"

    cleaned = _CITATION_RE.sub(_replace, answer)
    cleaned = re.sub(r"[ \t]+([.,;:])", r"\1", cleaned)
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)

    return cleaned.strip(), stats


# ---------------------------------------------------------
# 11. GENERATION
# ---------------------------------------------------------

def _generate_text(pipeline, messages, max_new_tokens=320):
    """Run the local Qwen model on chat messages and return the reply."""
    import torch  # deferred: only needed when actually generating

    tokenizer = pipeline["tokenizer"]
    llm = pipeline["llm"]

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = tokenizer(prompt, return_tensors="pt").to(llm.device)

    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id

    with torch.inference_mode():
        outputs = llm.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            repetition_penalty=1.0,
            pad_token_id=pad_id,
        )

    new_tokens = outputs[0][inputs["input_ids"].shape[1]:]

    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


_SYSTEM_PROMPT_STANDARD = """
You are ResearchPilot, an academic research assistant.

Answer the question using only the supplied research evidence.
If the evidence is insufficient, explicitly say so.
Never invent facts, paper titles, page numbers, or citations.

Write exactly three concise bullet points.
Each bullet must start with "- " and cite supporting evidence
using the format [Source 1, p. 2].

Use only source labels and page numbers supplied in the evidence.
If a source has an unknown page number, cite it as [Source 1].
Do not add a heading or a separate sources list.
"""

_SYSTEM_PROMPT_FOCUSED = """
You are ResearchPilot, an academic research assistant.

Answer the question using only the supplied research evidence.
Focus on the passages that directly address the question and ignore
passages that are only loosely related.
If the evidence is insufficient or does not directly answer the
question, say so explicitly instead of guessing.
Never invent facts, paper titles, page numbers, or citations.

Write two to four concise bullet points.
Each bullet must start with "- " and cite supporting evidence
using the format [Source 1, p. 2].

Use only source labels and page numbers supplied in the evidence.
If a source has an unknown page number, cite it as [Source 1].
Do not add a heading or a separate sources list.
"""

_NO_EVIDENCE_ANSWER = (
    "I could not retrieve relevant passages from the available research "
    "papers, so I cannot provide a source-supported answer."
)


def _run_rag(
    question,
    top_k,
    candidate_k,
    max_per_paper,
    system_prompt,
    max_new_tokens,
    mode,
):
    """Shared implementation behind both public answer functions."""

    if not isinstance(question, str) or not question.strip():
        raise ValueError("Please enter a research question.")

    if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k <= 0:
        raise ValueError("top_k must be a positive integer.")

    question = question.strip()

    # Load (or fetch the cached) pipeline first, so first-run model loading
    # is not counted as retrieval latency.
    pipeline = load_pipeline()
    start_time = time.perf_counter()

    # Hybrid retrieval: semantic search + TF-IDF + RRF.
    sources = build_rag_context_hybrid(
        question,
        pipeline,
        top_k=top_k,
        candidate_k=candidate_k,
        max_per_paper=max_per_paper,
    )

    retrieval_ms = round((time.perf_counter() - start_time) * 1000, 2)

    # Attach the label the model will cite, so UI and prompt agree.
    for number, source in enumerate(sources, start=1):
        source["label"] = f"Source {number}"

    base_insights = {
        "retrieved_count": len(sources),
        "unique_papers": len({_paper_key(source) for source in sources}),
        "top_k": top_k,
        "retrieval_methods": list(RETRIEVAL_METHODS),
        "retrieval_latency_ms": retrieval_ms,
        "mode": mode,
    }

    if not sources:
        return {
            "question": question,
            "answer": _NO_EVIDENCE_ANSWER,
            "sources": [],
            "retrieval_insights": {
                **base_insights,
                "generation_latency_ms": 0.0,
                "total_latency_ms": retrieval_ms,
                "citations_valid": 0,
                "citations_removed": 0,
            },
        }

    # Keep the prompt within a safe size when top_k is large.
    chars_per_source = max(800, min(2200, 9000 // len(sources)))

    evidence_parts = []

    for source in sources:
        page = source.get("page")
        page_label = page if page not in (None, "") else "unknown"

        evidence_parts.append(
            f"[{source['label']}, p. {page_label}]\n"
            f"Title: {source['title']}\n"
            f"Evidence: {source['text'][:chars_per_source]}"
        )

    evidence = "\n\n".join(evidence_parts)

    user_prompt = f"""
Question:
{question}

Retrieved research evidence:
{evidence}

Answer directly and cite the evidence supporting each bullet.
"""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    generation_start = time.perf_counter()

    raw_answer = _generate_text(
        pipeline,
        messages,
        max_new_tokens=max_new_tokens,
    )

    generation_ms = round((time.perf_counter() - generation_start) * 1000, 2)

    answer, citation_stats = _normalize_citations(raw_answer, sources)

    if not answer:
        answer = (
            "The retrieved evidence did not produce a usable answer. "
            "Please try rephrasing the question."
        )

    total_ms = round((time.perf_counter() - start_time) * 1000, 2)

    # Scores and ranks are retrieval signals, not calibrated probabilities.
    return {
        "question": question,
        "answer": answer,
        "sources": sources,
        "retrieval_insights": {
            **base_insights,
            "generation_latency_ms": generation_ms,
            "total_latency_ms": total_ms,
            "citations_valid": citation_stats["valid"],
            "citations_removed": citation_stats["removed"],
        },
    }


def generate_rag_answer(question, top_k=3):
    """Return the generated answer, retrieved sources, and insights.

    Returns a dictionary with the keys: question, answer, sources,
    retrieval_insights.
    """

    return _run_rag(
        question,
        top_k,
        candidate_k=15,
        max_per_paper=3,
        system_prompt=_SYSTEM_PROMPT_STANDARD,
        max_new_tokens=320,
        mode="standard",
    )


def generate_rag_answer_focused(question, top_k=5):
    """Notebook-compatible variant that favours directly relevant passages.

    Casts a wider candidate net and limits each paper to two passages so
    the evidence is both relevant and varied. Returns the same dictionary
    shape as generate_rag_answer().

        rag_result = generate_rag_answer_focused(
            "How does retrieval-augmented generation use external knowledge?",
            top_k=5,
        )
    """

    return _run_rag(
        question,
        top_k,
        candidate_k=max(20, 4 * top_k if isinstance(top_k, int) else 20),
        max_per_paper=2,
        system_prompt=_SYSTEM_PROMPT_FOCUSED,
        max_new_tokens=384,
        mode="focused",
    )