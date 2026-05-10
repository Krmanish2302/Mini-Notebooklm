"""
prompt_builder.py  —  Mode-specific prompt construction.

Chat mode uses a PersonaConfig to build the system prompt.
Study + Research modes keep the Carl Sagan persona — they are
pipeline-internal and the user doesn't customise them.

Token philosophy
----------------
System prompts are kept intentionally short.  No redundant words.
Grounding is enforced via a single clean rule block.

Gap fixes (2026-05-10)
----------------------
1. QueryRewriter   — HyDE hypothesis + keyword expansion before retrieval.
2. HistoryCompressor — collapses long chat histories to a rolling summary
                       so the context window doesn't bloat.
3. Empty-source fallback — when documents=[] every mode returns a safe
                           "nothing found" prompt instead of a blank SOURCES block.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

from src.generation.persona_config import PersonaConfig


# ── Hardcoded personas for Study / Research (not user-facing) ─────────────────
_PERSONA_STUDY = (
    "You're Carl Sagan if he were a chill classmate in teacher mode. "
    "Build intuition, use analogies, show how ideas connect.  "
    "Use ONLY the sources. Cite as [S1], [S2]… "
    "If it's not there, say: 'Not in my notes, bro.'"
)

_PERSONA_RESEARCH = (
    "You're Carl Sagan if he were a chill classmate. "
    "Go deep — structured, thorough, cite everything as [S1], [S2]…  "
    "Use ONLY the sources. Never invent facts. "
    "If it's not there, say: 'Not in my notes, bro.'"
)

_GROUNDING = (
    "Rules: "
    "(1) Use ONLY the sources.  "
    "(2) Cite inline as [S1], [S2]…  "
    "(3) Never invent facts."
)

# Shown to the LLM when retrieval returns zero chunks
_NO_SOURCES_BLOCK = (
    "SOURCES: [none — the knowledge base returned no relevant chunks for this query]\n\n"
    "INSTRUCTION: Politely tell the user you couldn't find anything relevant in the "
    "ingested documents and suggest they (a) rephrase, (b) ingest more sources, "
    "or (c) check their source filters.  Do NOT make up an answer."
)

# History turn budget — if raw history exceeds this many characters, compress it
_HISTORY_CHAR_LIMIT = 3_000
# Maximum recent turns to keep verbatim even after compression
_HISTORY_KEEP_TURNS = 4


# ── QueryRewriter ─────────────────────────────────────────────────────────────

class QueryRewriter:
    """
    Two lightweight rewriting strategies that improve retrieval quality
    without needing an extra LLM call.

    1. HyDE (Hypothetical Document Embedding)
       Prepends a one-sentence hypothetical answer to the query so the
       embedding sits closer to answer-space rather than question-space.

    2. Keyword expansion
       Appends key noun-phrases stripped from the query.  Simple but
       effective for short queries that under-specify the topic.

    Usage
    -----
        rewriter = QueryRewriter()
        hyde_q   = rewriter.hyde("What causes transformer attention to fail?")
        exp_q    = rewriter.expand("attention mechanism")
        combined = rewriter.rewrite("attention mechanism", strategy="both")
    """

    # Very small stop-word set — avoids importing NLTK
    _STOP = frozenset(
        "a an the is are was were be been being have has had do does did "
        "will would could should may might shall can cannot i me my we our "
        "you your he she it its they them their what which who whom whose "
        "when where why how all each every both few more most other some "
        "such no nor not only same so than too very just but and or if in "
        "on at to of for with by from up about into through during before "
        "after above below between out off over under again further then "
        "once here there this that these those".split()
    )

    def hyde(self, query: str) -> str:
        """
        Prepend a synthetic one-sentence answer stub so the query vector
        aligns with document answer-space rather than question-space.

        The stub is purely structural — it signals to the embedding model
        that we want answer-like content, not just keyword matches.
        """
        stub = f"In the context of the ingested documents, a relevant passage about '{query}' would state:"
        return f"{stub}\n{query}"

    def expand(self, query: str) -> str:
        """
        Append unique, non-stop noun-phrase tokens from the query so
        short queries get broader coverage in vector space.
        """
        tokens = re.findall(r"\b[a-zA-Z][a-zA-Z0-9_\-]{2,}\b", query)
        keywords = [t.lower() for t in tokens if t.lower() not in self._STOP]
        unique_kw = list(dict.fromkeys(keywords))  # preserve order, dedupe
        if not unique_kw:
            return query
        return f"{query} | keywords: {', '.join(unique_kw)}"

    def rewrite(
        self,
        query: str,
        strategy: str = "hyde",
    ) -> str:
        """
        strategy: "hyde" | "expand" | "both"

        "both" applies HyDE first, then keyword-expands the result.
        Recommended for short (< 6 words) or vague queries.
        "hyde" alone is better for full-sentence questions.
        "expand" alone is better for keyword-style queries.
        """
        if strategy == "hyde":
            return self.hyde(query)
        if strategy == "expand":
            return self.expand(query)
        if strategy == "both":
            return self.expand(self.hyde(query))
        return query

    @staticmethod
    def pick_strategy(query: str) -> str:
        """Heuristic: short queries → expand, questions → hyde, else both."""
        word_count = len(query.split())
        is_question = query.strip().endswith("?") or query.split()[0].lower() in (
            "what", "why", "how", "when", "where", "who", "explain", "describe",
            "compare", "summarise", "summarize", "list", "define",
        )
        if word_count <= 4:
            return "expand"
        if is_question:
            return "hyde"
        return "both"


# ── HistoryCompressor ─────────────────────────────────────────────────────────

class HistoryCompressor:
    """
    Keeps the chat history token-efficient by compressing old turns into
    a rolling summary while preserving the most recent N turns verbatim.

    Format expected for `history` string
    -------------------------------------
    Each turn is separated by a blank line and prefixed with
    "User:" / "Assistant:".  This matches what master_pipeline.py builds.

    Usage
    -----
        compressed = HistoryCompressor.compress(raw_history)
    """

    @staticmethod
    def _split_turns(history: str) -> List[str]:
        """Split raw history string into individual turn blocks."""
        # Turns are separated by blank lines OR by "User:" / "Assistant:" prefixes
        turns = re.split(r"\n(?=(?:User:|Assistant:))", history.strip())
        return [t.strip() for t in turns if t.strip()]

    @staticmethod
    def _summarise_turns(turns: List[str]) -> str:
        """
        Produce a compact bullet-point summary of a list of old turns.
        This is purely string-based (no LLM call) — it extracts the first
        sentence of each assistant reply as a "key point".
        """
        bullets: List[str] = []
        i = 0
        while i < len(turns):
            turn = turns[i]
            if turn.startswith("Assistant:"):
                reply = turn[len("Assistant:"):].strip()
                # First sentence only
                first_sentence = re.split(r"(?<=[.!?])\s", reply)[0]
                if first_sentence:
                    bullets.append(f"• {first_sentence}")
            i += 1
        if not bullets:
            return "[earlier conversation compressed — no key points extracted]"
        return "EARLIER CONTEXT (compressed):\n" + "\n".join(bullets)

    @classmethod
    def compress(
        cls,
        history: str,
        char_limit: int = _HISTORY_CHAR_LIMIT,
        keep_turns: int = _HISTORY_KEEP_TURNS,
    ) -> str:
        """
        If history is within char_limit, return as-is.
        Otherwise compress old turns to a summary + keep last `keep_turns`
        turns verbatim.
        """
        if not history or len(history) <= char_limit:
            return history

        turns = cls._split_turns(history)
        if len(turns) <= keep_turns:
            return history  # can't compress further, just keep all

        old_turns = turns[:-keep_turns]
        recent_turns = turns[-keep_turns:]

        summary = cls._summarise_turns(old_turns)
        recent_block = "\n\n".join(recent_turns)
        return f"{summary}\n\nRECENT CONVERSATION:\n{recent_block}"


# ── PromptBuilder ─────────────────────────────────────────────────────────────

class PromptBuilder:
    """
    Builds mode-specific prompts.

    Method contract (must match master_pipeline.py dispatch table):
        build_chat_prompt(query, documents, history="", persona_config=None)  -> str
        build_study_prompt(query, documents, history="")                       -> str
        build_research_prompt(query, documents, history="")                    -> str

    All methods:
    - Accept an optional `rewrite` bool (default True) to toggle query rewriting.
    - Handle empty `documents` list with a safe fallback prompt.
    - Compress long `history` strings automatically.
    """

    _rewriter = QueryRewriter()

    # ── Helpers ────────────────────────────────────────────────────────────────

    @staticmethod
    def format_context(documents: List[Any]) -> str:
        """
        Format retrieved chunks into a compact numbered SOURCE block.
        Appends context_window (from ContextualEnricher) when present.
        Returns empty string when documents is empty — callers check for this.
        """
        parts: List[str] = []
        for i, doc in enumerate(documents, 1):
            if hasattr(doc, "page_content"):
                content = doc.page_content
                meta = doc.metadata or {}
            else:
                content = doc.get("content", "")
                meta = {k: v for k, v in doc.items() if k != "content"}

            src = meta.get("source", meta.get("source_id", ""))
            header = f"[S{i}]{' — ' + src if src else ''}"
            block = f"{header}\n{content}"

            ctx = meta.get("context_window", "")
            if ctx:
                block += f"\n[context] {ctx}"

            parts.append(block)

        return "\n\n".join(parts)

    @staticmethod
    def _compress_history(history: str) -> str:
        return HistoryCompressor.compress(history)

    @classmethod
    def _maybe_rewrite(cls, query: str, rewrite: bool) -> str:
        if not rewrite:
            return query
        strategy = QueryRewriter.pick_strategy(query)
        return cls._rewriter.rewrite(query, strategy=strategy)

    @staticmethod
    def _sources_block(documents: List[Any]) -> Tuple[bool, str]:
        """
        Returns (has_sources: bool, formatted_block: str).
        When has_sources is False, callers should insert _NO_SOURCES_BLOCK.
        """
        if not documents:
            return False, _NO_SOURCES_BLOCK
        ctx = PromptBuilder.format_context(documents)
        return True, f"SOURCES:\n{ctx}"

    # ── Chat Mode ──────────────────────────────────────────────────────────────

    @staticmethod
    def build_chat_prompt(
        query: str,
        documents: List[Any],
        history: str = "",
        persona_config: Optional[PersonaConfig] = None,
        rewrite: bool = True,
    ) -> str:
        """
        Chat mode — persona driven by PersonaConfig.
        Falls back to the default Carl Sagan preset when no config is given.
        Grounding is ALWAYS present regardless of persona choice.
        Empty documents → polite no-results instruction.
        """
        cfg = persona_config or PersonaConfig()
        system = cfg.build_system_prompt()

        effective_query = PromptBuilder._maybe_rewrite(query, rewrite)
        hist = PromptBuilder._compress_history(history)
        hist_block = f"HISTORY:\n{hist}\n\n" if hist else ""

        has_src, src_block = PromptBuilder._sources_block(documents)

        if not has_src:
            return (
                f"{system}\n\n"
                f"{hist_block}"
                f"{src_block}\n\n"
                f"Q: {query}\nA:"
            )

        return (
            f"{system}\n\n"
            f"{hist_block}"
            f"{src_block}\n\n"
            f"Q: {effective_query}\nA:"
        )

    # ── Study Mode ─────────────────────────────────────────────────────────────

    @staticmethod
    def build_study_prompt(
        query: str,
        documents: List[Any],
        history: str = "",
        learning_path: Optional[List[Dict]] = None,
        rewrite: bool = True,
    ) -> str:
        """Study mode — fixed Sagan teacher persona, shows concept connections."""
        effective_query = PromptBuilder._maybe_rewrite(query, rewrite)
        hist = PromptBuilder._compress_history(history)
        hist_block = f"HISTORY:\n{hist}\n\n" if hist else ""

        path_block = ""
        if learning_path:
            steps = " → ".join(
                f"{s.get('from', '')} ➜ {s.get('to', '')}"
                for s in learning_path[:4]
            )
            path_block = f"CONCEPT PATH: {steps}\n\n"

        has_src, src_block = PromptBuilder._sources_block(documents)

        if not has_src:
            return (
                f"{_PERSONA_STUDY}\n\n"
                f"{path_block}"
                f"{hist_block}"
                f"{src_block}\n\n"
                f"TOPIC: {query}\nEXPLAIN:"
            )

        return (
            f"{_PERSONA_STUDY}\n\n"
            f"{path_block}"
            f"{hist_block}"
            f"{src_block}\n\n"
            f"TOPIC: {effective_query}\nEXPLAIN:"
        )

    # ── Deep Research Mode ─────────────────────────────────────────────────────

    @staticmethod
    def build_research_prompt(
        query: str,
        documents: List[Any],
        history: str = "",
        rewrite: bool = True,
    ) -> str:
        """Deep Research — fixed Sagan research persona, thorough + cited."""
        effective_query = PromptBuilder._maybe_rewrite(query, rewrite)
        hist = PromptBuilder._compress_history(history)
        hist_block = f"HISTORY:\n{hist}\n\n" if hist else ""

        has_src, src_block = PromptBuilder._sources_block(documents)

        if not has_src:
            return (
                f"{_PERSONA_RESEARCH}\n\n"
                f"{hist_block}"
                f"{src_block}\n\n"
                f"RESEARCH Q: {query}\nDETAILED ANSWER:"
            )

        return (
            f"{_PERSONA_RESEARCH}\n\n"
            f"{hist_block}"
            f"{src_block}\n\n"
            f"RESEARCH Q: {effective_query}\nDETAILED ANSWER:"
        )

    # ── Backward-compat aliases ────────────────────────────────────────────────

    @staticmethod
    def build_deep_research_prompt(
        query: str, documents: List[Any], history: str = ""
    ) -> str:
        return PromptBuilder.build_research_prompt(query, documents, history)

    @staticmethod
    def build_study_mode_prompt(
        query: str,
        documents: List[Any],
        learning_path: Optional[List[Dict]] = None,
        history: str = "",
    ) -> str:
        return PromptBuilder.build_study_prompt(query, documents, history, learning_path)
