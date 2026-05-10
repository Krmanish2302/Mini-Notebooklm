"""
verify_pipeline.py  —  End-to-end smoke test for Mini NotebookLM.

Run from project root:
    python -m src.verify_pipeline

No real PDF, no API key, no network needed.
Uses a synthetic in-memory document to exercise the full path:
    ContentAnalyzer → PromptBuilder → MasterPipeline (ingest → query → delete)

Exit code 0  = all checks passed.
Exit code 1  = one or more checks failed (details printed above).
"""
from __future__ import annotations

import sys
import traceback
import textwrap
from typing import Callable, List, Tuple

# ─────────────────────────────────────────────────────────────────────────────
SYNTHETIC DOCUMENT  —  used for all ingest / query tests
─────────────────────────────────────────────────────────────────────────────
SYNTHETIC_TEXT = textwrap.dedent("""
    ## Introduction to Vector Databases

    Vector databases store high-dimensional embeddings and support
    approximate nearest-neighbour search.  They are the backbone of
    modern Retrieval-Augmented Generation (RAG) systems.

    ## FAISS

    FAISS (Facebook AI Similarity Search) is an open-source library
    developed by Meta AI.  It provides highly efficient similarity
    search and clustering of dense vectors.  FAISS supports both
    CPU and GPU execution and is widely used in production RAG pipelines.

    ## Chunking Strategies

    Paragraph chunking splits a document at blank-line boundaries.
    Sentence chunking produces smaller, more granular chunks.
    Page-level chunking preserves page context but may exceed token limits.
    Choosing the right strategy depends on the embedding model’s max tokens
    and the document’s structure.

    ## Reciprocal Rank Fusion

    Reciprocal Rank Fusion (RRF) is a rank aggregation method that
    combines result lists from multiple retrieval systems.  Given rank r,
    each document receives score 1 / (k + r) where k is typically 60.
    RRF is robust to score scale differences between retrieval systems.

    ## Embedding Models

    Sentence-transformers all-MiniLM-L6-v2 produces 384-dimensional
    embeddings with a 256-token context window.  MPNet produces
    768-dimensional embeddings.  OpenAI text-embedding-3-small outputs
    1536 dimensions and accepts up to 8191 tokens per chunk.
""").strip()

SYNTHETIC_QUERY = "What is FAISS and how does it relate to RAG?"


# ─────────────────────────────────────────────────────────────────────────────
TEST RUNNER
─────────────────────────────────────────────────────────────────────────────
Results: List[Tuple[str, bool, str]] = []  # (name, passed, detail)


def check(name: str, fn: Callable[[], None]) -> bool:
    """Run a single check, record result, return True if passed."""
    try:
        fn()
        Results.append((name, True, ""))
        print(f"  ✅  {name}")
        return True
    except Exception as exc:
        detail = traceback.format_exc(limit=4)
        Results.append((name, False, detail))
        print(f"  ❌  {name}")
        print(textwrap.indent(detail, "      "))
        return False


# ─────────────────────────────────────────────────────────────────────────────
SECTION 1 — Imports
─────────────────────────────────────────────────────────────────────────────
print("\n── Section 1: Imports ──")


def _import_content_analyzer():
    from src.ingestion.preprocessing.content_analyzer import ContentAnalyzer  # noqa

def _import_prompt_builder():
    from src.generation.prompt_builder import PromptBuilder, QueryRewriter, HistoryCompressor  # noqa

def _import_master_pipeline():
    from src.master_pipeline import MasterPipeline  # noqa

def _import_persona_config():
    from src.generation.persona_config import PersonaConfig  # noqa


check("Import ContentAnalyzer",   _import_content_analyzer)
check("Import PromptBuilder",      _import_prompt_builder)
check("Import MasterPipeline",     _import_master_pipeline)
check("Import PersonaConfig",      _import_persona_config)


# ─────────────────────────────────────────────────────────────────────────────
SECTION 2 — ContentAnalyzer token_stats shape
─────────────────────────────────────────────────────────────────────────────
print("\n── Section 2: ContentAnalyzer token_stats ──")


def _check_token_stats_shape():
    from src.ingestion.preprocessing.content_analyzer import ContentAnalyzer
    result = ContentAnalyzer().analyze(SYNTHETIC_TEXT, source_type="text")
    ts = result["token_stats"]
    required_strategies = {"sentence", "paragraph", "page"}
    assert required_strategies == set(ts.keys()), (
        f"Expected strategies {required_strategies}, got {set(ts.keys())}"
    )
    required_fields = {"count", "avg", "min", "max"}
    for strategy, stats in ts.items():
        assert required_fields == set(stats.keys()), (
            f"Strategy '{strategy}' missing fields: {required_fields - set(stats.keys())}"
        )
        assert stats["avg"] >= 0, f"'{strategy}' avg tokens must be >= 0"
        assert stats["min"] <= stats["avg"] <= stats["max"] or stats["count"] == 0, (
            f"'{strategy}': min/avg/max ordering violated: "
            f"{stats['min']} / {stats['avg']} / {stats['max']}"
        )


def _check_token_stats_values():
    from src.ingestion.preprocessing.content_analyzer import ContentAnalyzer
    result = ContentAnalyzer().analyze(SYNTHETIC_TEXT, source_type="pdf")
    ts = result["token_stats"]
    # Paragraph avg should be higher than sentence avg for normal text
    assert ts["paragraph"]["avg"] >= ts["sentence"]["avg"], (
        "Paragraph avg tokens should be >= sentence avg tokens"
    )
    assert result["word_count"] > 0, "word_count must be positive"
    assert result["sentence_count"] > 0, "sentence_count must be positive"
    assert result["estimated_tokens"] > 0, "estimated_tokens must be positive"


def _check_recommendation_present():
    from src.ingestion.preprocessing.content_analyzer import ContentAnalyzer
    result = ContentAnalyzer().analyze(SYNTHETIC_TEXT, source_type="pdf")
    rec = result["recommendation"]
    assert "strategy" in rec, "recommendation must include 'strategy'"
    assert "reason"   in rec, "recommendation must include 'reason'"
    assert rec["strategy"], "strategy must be non-empty string"


check("token_stats shape (all keys present)",     _check_token_stats_shape)
check("token_stats values (para >= sentence avg)", _check_token_stats_values)
check("recommendation block present",             _check_recommendation_present)


# ─────────────────────────────────────────────────────────────────────────────
SECTION 3 — QueryRewriter
─────────────────────────────────────────────────────────────────────────────
print("\n── Section 3: QueryRewriter ──")


def _rewriter_hyde():
    from src.generation.prompt_builder import QueryRewriter
    r = QueryRewriter()
    out = r.hyde(SYNTHETIC_QUERY)
    assert SYNTHETIC_QUERY in out, "HyDE output must contain original query"
    assert len(out) > len(SYNTHETIC_QUERY), "HyDE must add content to the query"


def _rewriter_expand():
    from src.generation.prompt_builder import QueryRewriter
    r = QueryRewriter()
    out = r.expand("FAISS vector search")
    assert "faiss" in out.lower() or "vector" in out.lower() or "search" in out.lower(), (
        "expand() must preserve at least one keyword"
    )


def _rewriter_pick_strategy_short():
    from src.generation.prompt_builder import QueryRewriter
    assert QueryRewriter.pick_strategy("FAISS") == "expand", (
        "Single word query should use 'expand'"
    )


def _rewriter_pick_strategy_question():
    from src.generation.prompt_builder import QueryRewriter
    assert QueryRewriter.pick_strategy("What is FAISS and how does it work?") == "hyde", (
        "Question should use 'hyde'"
    )


def _rewriter_pick_strategy_both():
    from src.generation.prompt_builder import QueryRewriter
    assert QueryRewriter.pick_strategy("vector database similarity search production") == "both", (
        "Long non-question should use 'both'"
    )


check("QueryRewriter.hyde() adds content",             _rewriter_hyde)
check("QueryRewriter.expand() preserves keywords",     _rewriter_expand)
check("pick_strategy: short query → expand",           _rewriter_pick_strategy_short)
check("pick_strategy: question → hyde",                _rewriter_pick_strategy_question)
check("pick_strategy: long non-question → both",       _rewriter_pick_strategy_both)


# ─────────────────────────────────────────────────────────────────────────────
SECTION 4 — HistoryCompressor
─────────────────────────────────────────────────────────────────────────────
print("\n── Section 4: HistoryCompressor ──")

_SHORT_HISTORY = "User: hi\nAssistant: hello"
_LONG_HISTORY = "\n\n".join(
    f"User: question {i}\nAssistant: This is a fairly long answer about topic {i}. "
    f"It covers several important points and goes into detail about the subject matter."
    for i in range(20)
)


def _compressor_short_passthrough():
    from src.generation.prompt_builder import HistoryCompressor
    out = HistoryCompressor.compress(_SHORT_HISTORY, char_limit=3000)
    assert out == _SHORT_HISTORY, "Short history must pass through unchanged"


def _compressor_long_shrinks():
    from src.generation.prompt_builder import HistoryCompressor
    out = HistoryCompressor.compress(_LONG_HISTORY, char_limit=200, keep_turns=2)
    assert len(out) < len(_LONG_HISTORY), (
        f"Compressed history ({len(out)}) should be shorter than original ({len(_LONG_HISTORY)})"
    )


def _compressor_keeps_recent():
    from src.generation.prompt_builder import HistoryCompressor
    out = HistoryCompressor.compress(_LONG_HISTORY, char_limit=200, keep_turns=2)
    # last turn should appear verbatim in the output
    assert "question 19" in out, "Most recent turn must survive compression"


def _compressor_empty_safe():
    from src.generation.prompt_builder import HistoryCompressor
    assert HistoryCompressor.compress("") == "", "Empty history must return empty string"
    assert HistoryCompressor.compress(None) == "", "None history must return empty string"


check("HistoryCompressor: short history passes through", _compressor_short_passthrough)
check("HistoryCompressor: long history shrinks",         _compressor_long_shrinks)
check("HistoryCompressor: recent turns preserved",       _compressor_keeps_recent)
check("HistoryCompressor: empty / None safe",            _compressor_empty_safe)


# ─────────────────────────────────────────────────────────────────────────────
SECTION 5 — PromptBuilder (all 3 modes)
─────────────────────────────────────────────────────────────────────────────
print("\n── Section 5: PromptBuilder ──")

_FAKE_DOCS = [
    {"content": "FAISS is a vector similarity search library by Meta AI.",
     "source": "synthetic_doc", "context_window": ""},
    {"content": "RAG systems retrieve relevant chunks before generating answers.",
     "source": "synthetic_doc", "context_window": ""},
]


def _prompt_chat_with_docs():
    from src.generation.prompt_builder import PromptBuilder
    prompt = PromptBuilder.build_chat_prompt(SYNTHETIC_QUERY, _FAKE_DOCS)
    assert "[S1]" in prompt, "Chat prompt must include numbered source [S1]"
    assert "[S2]" in prompt, "Chat prompt must include numbered source [S2]"
    assert SYNTHETIC_QUERY in prompt or "FAISS" in prompt, (
        "Query or rewritten form must appear in prompt"
    )
    assert "Q:" in prompt and "A:" in prompt, "Chat prompt must have Q:/A: markers"


def _prompt_chat_no_docs_fallback():
    from src.generation.prompt_builder import PromptBuilder
    prompt = PromptBuilder.build_chat_prompt(SYNTHETIC_QUERY, [], rewrite=False)
    assert "none" in prompt.lower() or "no relevant" in prompt.lower() or \
           "not in" in prompt.lower() or "couldn't find" in prompt.lower() or \
           "knowledge base returned no" in prompt.lower(), (
        "Empty-doc prompt must include no-results instruction, got: " + prompt[:200]
    )


def _prompt_study_with_docs():
    from src.generation.prompt_builder import PromptBuilder
    prompt = PromptBuilder.build_study_prompt(SYNTHETIC_QUERY, _FAKE_DOCS)
    assert "TOPIC:" in prompt and "EXPLAIN:" in prompt, (
        "Study prompt must have TOPIC:/EXPLAIN: markers"
    )
    assert "[S1]" in prompt, "Study prompt must include [S1]"


def _prompt_study_with_learning_path():
    from src.generation.prompt_builder import PromptBuilder
    path = [{"from": "embeddings", "to": "FAISS"}, {"from": "FAISS", "to": "RAG"}]
    prompt = PromptBuilder.build_study_prompt(SYNTHETIC_QUERY, _FAKE_DOCS, learning_path=path)
    assert "CONCEPT PATH" in prompt, "Learning path must appear in study prompt"


def _prompt_research_with_docs():
    from src.generation.prompt_builder import PromptBuilder
    prompt = PromptBuilder.build_research_prompt(SYNTHETIC_QUERY, _FAKE_DOCS)
    assert "RESEARCH Q:" in prompt and "DETAILED ANSWER:" in prompt, (
        "Research prompt must have RESEARCH Q:/DETAILED ANSWER: markers"
    )
    assert "[S1]" in prompt, "Research prompt must include [S1]"


def _prompt_research_no_docs_fallback():
    from src.generation.prompt_builder import PromptBuilder
    prompt = PromptBuilder.build_research_prompt(SYNTHETIC_QUERY, [], rewrite=False)
    assert "knowledge base returned no" in prompt.lower() or \
           "none" in prompt.lower(), (
        "Empty-doc research prompt must trigger fallback instruction"
    )


def _prompt_format_context_context_window():
    """context_window metadata must be appended when present."""
    from src.generation.prompt_builder import PromptBuilder
    docs = [{"content": "main text", "source": "s1",
             "context_window": "surrounding paragraph context"}]
    ctx = PromptBuilder.format_context(docs)
    assert "[context]" in ctx, "format_context must append [context] block"
    assert "surrounding paragraph context" in ctx


def _prompt_backward_compat_aliases():
    """Old alias methods must still work."""
    from src.generation.prompt_builder import PromptBuilder
    p1 = PromptBuilder.build_deep_research_prompt(SYNTHETIC_QUERY, _FAKE_DOCS)
    p2 = PromptBuilder.build_study_mode_prompt(SYNTHETIC_QUERY, _FAKE_DOCS)
    assert "RESEARCH Q:" in p1, "build_deep_research_prompt alias broken"
    assert "TOPIC:"      in p2, "build_study_mode_prompt alias broken"


check("Chat prompt: sources formatted as [S1][S2]",      _prompt_chat_with_docs)
check("Chat prompt: empty-docs fallback fires",          _prompt_chat_no_docs_fallback)
check("Study prompt: TOPIC/EXPLAIN markers present",     _prompt_study_with_docs)
check("Study prompt: learning path block injected",      _prompt_study_with_learning_path)
check("Research prompt: RESEARCH Q/DETAILED ANSWER",     _prompt_research_with_docs)
check("Research prompt: empty-docs fallback fires",      _prompt_research_no_docs_fallback)
check("format_context: [context] window appended",       _prompt_format_context_context_window)
check("Backward-compat aliases intact",                  _prompt_backward_compat_aliases)


# ─────────────────────────────────────────────────────────────────────────────
SECTION 6 — MasterPipeline (no LLM — stats + ingest + delete)
─────────────────────────────────────────────────────────────────────────────
print("\n── Section 6: MasterPipeline ──")

# Keep pipeline instance across checks in this section
_pipeline_ref: dict = {}


def _pipeline_instantiates():
    from src.master_pipeline import MasterPipeline
    pl = MasterPipeline(mode="chat")
    _pipeline_ref["pl"] = pl
    assert pl is not None


def _pipeline_get_stats():
    pl = _pipeline_ref.get("pl")
    if pl is None:
        from src.master_pipeline import MasterPipeline
        pl = MasterPipeline(mode="chat")
        _pipeline_ref["pl"] = pl
    stats = pl.get_stats()
    assert isinstance(stats, dict), "get_stats() must return a dict"
    # Check expected top-level keys exist (values may be 0)
    for key in ("total_sources", "total_chunks"):
        assert key in stats, f"stats missing key '{key}'"


def _pipeline_ingest_text():
    """Ingest synthetic text directly (no PDF parser needed)."""
    import tempfile, os
    pl = _pipeline_ref.get("pl")
    if pl is None:
        from src.master_pipeline import MasterPipeline
        pl = MasterPipeline(mode="chat")
        _pipeline_ref["pl"] = pl

    # Write synthetic text to a temp .txt file
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".txt", delete=False, encoding="utf-8"
    ) as f:
        f.write(SYNTHETIC_TEXT)
        tmp_path = f.name

    try:
        result = pl.ingest(
            file_path=tmp_path,
            url=None,
            source_type="text",
            chunking_strategy="paragraph",
            embedding_model="all-MiniLM-L6-v2",
        )
        assert result is not None, "ingest() must return a result dict"
        # Save source_id for the delete test
        source_id = (
            result.get("source_id")
            or result.get("id")
            or result.get("source", {}).get("id")
        )
        _pipeline_ref["source_id"] = source_id
    finally:
        os.unlink(tmp_path)


def _pipeline_stats_after_ingest():
    pl = _pipeline_ref.get("pl")
    if pl is None:
        return  # previous test failed; skip
    stats = pl.get_stats()
    total = stats.get("total_chunks", 0) or stats.get("chunks", {}).get("total_chunks", 0)
    assert total > 0, (
        f"After ingest, total_chunks must be > 0 (got {total}). "
        "Check that the embedding model loaded and FAISS index was updated."
    )


def _pipeline_delete_source():
    pl = _pipeline_ref.get("pl")
    source_id = _pipeline_ref.get("source_id")
    if pl is None or not source_id:
        raise AssertionError(
            "Skipping delete test — ingest did not produce a source_id. "
            f"pipeline={'set' if pl else 'missing'}, source_id={source_id!r}"
        )
    ok = pl.delete_source(source_id)
    assert ok is not False, f"delete_source({source_id!r}) returned False (source not found)"


def _pipeline_stats_after_delete():
    pl = _pipeline_ref.get("pl")
    source_id = _pipeline_ref.get("source_id")
    if pl is None or not source_id:
        return  # upstream failure; skip
    stats = pl.get_stats()
    sources = stats.get("total_sources", -1)
    assert sources == 0, (
        f"After delete, total_sources should be 0 (got {sources})"
    )


check("MasterPipeline instantiates",            _pipeline_instantiates)
check("MasterPipeline.get_stats() returns dict", _pipeline_get_stats)
check("MasterPipeline.ingest() text source",     _pipeline_ingest_text)
check("Chunks present after ingest",             _pipeline_stats_after_ingest)
check("MasterPipeline.delete_source() works",    _pipeline_delete_source)
check("Chunks cleared after delete",             _pipeline_stats_after_delete)


# ─────────────────────────────────────────────────────────────────────────────
FINAL REPORT
─────────────────────────────────────────────────────────────────────────────
passed = sum(1 for _, ok, _ in Results if ok)
failed = sum(1 for _, ok, _ in Results if not ok)
total  = len(Results)

print(f"\n{'='*60}")
print(f"  RESULTS:  {passed}/{total} passed", end="")
if failed:
    print(f"  —  {failed} FAILED  ⚠️")
else:
    print("  —  ALL PASSED 🎉")
print("="*60)

if failed:
    print("\nFailed checks:")
    for name, ok, detail in Results:
        if not ok:
            print(f"  ❌ {name}")
    sys.exit(1)
else:
    sys.exit(0)
