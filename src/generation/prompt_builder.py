"""
prompt_builder.py  —  Mode-specific prompt construction.

Fixes applied
-------------
BUG-R02  HyDE / keyword-expanded query was being injected into the LLM prompt
         as the user's question.  The rewritten query is for retrieval embedding
         only — the LLM always sees the original clean query.  Fixed by
         separating `retrieval_query` (rewritten) from `prompt_query` (original)
         in all three build_*_prompt methods.
BUG-R04  HistoryCompressor._summarise_turns dropped all User: turns, leaving
         the LLM with only assistant replies and no idea what was asked.  Now
         includes paired User question + Assistant first-sentence in bullets.
BUG-S01  sanitize_query() added — strips prompt-injection patterns and enforces
         a max-length cap before the query reaches any LLM prompt.
BUG-Q02  _HISTORY_CHAR_LIMIT and _HISTORY_KEEP_TURNS promoted to named module
         constants.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

from src.generation.persona_config import PersonaConfig


# ── Module-level constants (BUG-Q02) ─────────────────────────────────────────
HISTORY_CHAR_LIMIT  = 3_000
HISTORY_KEEP_TURNS  = 4
MAX_QUERY_LENGTH    = 1_200   # BUG-S01: hard cap on user query length

# Keep old underscore aliases for any code that imported them directly
_HISTORY_CHAR_LIMIT = HISTORY_CHAR_LIMIT
_HISTORY_KEEP_TURNS = HISTORY_KEEP_TURNS


# ── Hardcoded personas for Study / Research ───────────────────────────────────
_PERSONA_STUDY = (
    "You're Carl Sagan if he were a chill classmate in teacher mode. "
    "Build intuition, use analogies, show how ideas connect.  "
    "Use ONLY the sources. Cite as [S1], [S2]\u2026 "
    "If it's not there, say: 'Not in my notes, bro.'"
)

_PERSONA_RESEARCH = (
    "You're Carl Sagan if he were a chill classmate. "
    "Go deep \u2014 structured, thorough, cite everything as [S1], [S2]\u2026  "
    "Use ONLY the sources. Never invent facts. "
    "If it's not there, say: 'Not in my notes, bro.'"
)

_GROUNDING = (
    "Rules: "
    "(1) Use ONLY the sources.  "
    "(2) Cite inline as [S1], [S2]\u2026  "
    "(3) Never invent facts."
)

_NO_SOURCES_BLOCK = (
    "SOURCES: [none \u2014 the knowledge base returned no relevant chunks for this query]\n\n"
    "INSTRUCTION: Politely tell the user you couldn\u2019t find anything relevant in the "
    "ingested documents and suggest they (a) rephrase, (b) ingest more sources, "
    "or (c) check their source filters.  Do NOT make up an answer."
)


# ── BUG-S01: Input sanitisation ───────────────────────────────────────────────

def sanitize_query(query: str, max_len: int = MAX_QUERY_LENGTH) -> str:
    """
    Strip prompt-injection attempts and enforce a length cap.

    Removes common jailbreak / system-override patterns before the query
    reaches any LLM prompt.  This is a defence-in-depth measure — it does
    NOT replace proper server-side input validation.
    """
    q = query.strip()[:max_len]
    # Remove common prompt-injection openers
    q = re.sub(
        r"(?i)(ignore|disregard|forget|override|bypass)\b.{0,50}\b"
        r"(instructions?|rules?|system\s*prompt|context|previous)",
        "[sanitized]",
        q,
    )
    # Remove explicit role-play injections
    q = re.sub(r"(?i)\bDAN\b|\bact as\b.{0,30}\bno restrictions\b", "[sanitized]", q)
    return q


# ── QueryRewriter ─────────────────────────────────────────────────────────────

class QueryRewriter:
    """
    BUG-R02 note: the output of this class is for the RETRIEVAL step only.
    It must NEVER be used as the query shown to the LLM in the final prompt.
    PromptBuilder enforces this by keeping two separate variables:
        retrieval_query = _maybe_rewrite(query, rewrite=True)  -> for embedder
        prompt_query    = query                                 -> for LLM prompt
    """

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
        stub = f"In the context of the ingested documents, a relevant passage about '{query}' would state:"
        return f"{stub}\n{query}"

    def expand(self, query: str) -> str:
        tokens = re.findall(r"\b[a-zA-Z][a-zA-Z0-9_\-]{2,}\b", query)
        keywords = [t.lower() for t in tokens if t.lower() not in self._STOP]
        unique_kw = list(dict.fromkeys(keywords))
        if not unique_kw:
            return query
        return f"{query} | keywords: {', '.join(unique_kw)}"

    def rewrite(self, query: str, strategy: str = "hyde") -> str:
        if strategy == "hyde":
            return self.hyde(query)
        if strategy == "expand":
            return self.expand(query)
        if strategy == "both":
            return self.expand(self.hyde(query))
        return query

    @staticmethod
    def pick_strategy(query: str) -> str:
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
    BUG-R04 fix: compress() now includes User: turns in the summary bullets
    so the LLM has context about what was being asked, not just what was
    answered.
    """

    @staticmethod
    def _split_turns(history: str) -> List[str]:
        turns = re.split(r"\n(?=(?:User:|Assistant:))", history.strip())
        return [t.strip() for t in turns if t.strip()]

    @staticmethod
    def _summarise_turns(turns: List[str]) -> str:
        """
        BUG-R04: produce paired (User question, Assistant first-sentence)
        bullets so the LLM knows both sides of old exchanges.
        """
        bullets: List[str] = []
        i = 0
        while i < len(turns):
            turn = turns[i]
            if turn.startswith("User:"):
                q = turn[5:].strip().split("\n")[0][:100]   # first line, capped
                bullets.append(f"\u2022 User asked: {q}")
            elif turn.startswith("Assistant:"):
                reply = turn[len("Assistant:"):].strip()
                first = re.split(r"(?<=[.!?])\s", reply)[0]
                if first:
                    bullets.append(f"  \u2192 {first}")
            i += 1

        if not bullets:
            return "[earlier conversation compressed \u2014 no key points extracted]"
        return "EARLIER CONTEXT (compressed):\n" + "\n".join(bullets)

    @classmethod
    def compress(
        cls,
        history: str,
        char_limit: int = HISTORY_CHAR_LIMIT,
        keep_turns: int = HISTORY_KEEP_TURNS,
    ) -> str:
        if not history or len(history) <= char_limit:
            return history
        turns = cls._split_turns(history)
        if len(turns) <= keep_turns:
            return history
        old_turns  = turns[:-keep_turns]
        recent_turns = turns[-keep_turns:]
        summary = cls._summarise_turns(old_turns)
        recent_block = "\n\n".join(recent_turns)
        return f"{summary}\n\nRECENT CONVERSATION:\n{recent_block}"


# ── PromptBuilder ─────────────────────────────────────────────────────────────

class PromptBuilder:
    """
    BUG-R02 fix: every build_*_prompt method now distinguishes between:
        retrieval_query  — HyDE/expanded, used ONLY for the embedding/retrieval call
        prompt_query     — original user query, shown to the LLM

    The *rewrite* parameter controls retrieval_query generation; it no longer
    affects what the LLM sees.
    """

    _rewriter = QueryRewriter()

    # ── Helpers ────────────────────────────────────────────────────────────────

    @staticmethod
    def format_context(documents: List[Any]) -> str:
        parts: List[str] = []
        for i, doc in enumerate(documents, 1):
            if hasattr(doc, "page_content"):
                content = doc.page_content
                meta = doc.metadata or {}
            else:
                content = doc.get("content", "")
                meta = {k: v for k, v in doc.items() if k != "content"}

            src = meta.get("source", meta.get("source_id", ""))
            header = f"[S{i}]{' \u2014 ' + src if src else ''}"
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
    def _retrieval_query(cls, query: str, rewrite: bool) -> str:
        """Return the rewritten query for retrieval embedding (not for LLM prompt)."""
        if not rewrite:
            return query
        strategy = QueryRewriter.pick_strategy(query)
        return cls._rewriter.rewrite(query, strategy=strategy)

    @staticmethod
    def _sources_block(documents: List[Any]) -> Tuple[bool, str]:
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
        BUG-R02 fix: *rewrite* now only affects what is passed to the retriever
        (via _retrieval_query).  The LLM always sees the original *query*.
        BUG-S01: query is sanitized before it reaches the prompt.
        """
        cfg = persona_config or PersonaConfig()
        system = cfg.build_system_prompt()

        # BUG-S01: sanitize before any use
        safe_query = sanitize_query(query)

        hist = PromptBuilder._compress_history(history)
        hist_block = f"HISTORY:\n{hist}\n\n" if hist else ""

        has_src, src_block = PromptBuilder._sources_block(documents)

        # BUG-R02: LLM always sees original (sanitized) query, never the HyDE stub
        return (
            f"{system}\n\n"
            f"{hist_block}"
            f"{src_block}\n\n"
            f"Q: {safe_query}\nA:"
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
        safe_query = sanitize_query(query)
        hist = PromptBuilder._compress_history(history)
        hist_block = f"HISTORY:\n{hist}\n\n" if hist else ""

        path_block = ""
        if learning_path:
            steps = " \u2192 ".join(
                f"{s.get('from', '')} \u279c {s.get('to', '')}"
                for s in learning_path[:4]
            )
            path_block = f"CONCEPT PATH: {steps}\n\n"

        has_src, src_block = PromptBuilder._sources_block(documents)

        # BUG-R02: always use safe_query (original) in the LLM prompt
        return (
            f"{_PERSONA_STUDY}\n\n"
            f"{path_block}"
            f"{hist_block}"
            f"{src_block}\n\n"
            f"TOPIC: {safe_query}\nEXPLAIN:"
        )

    # ── Deep Research Mode ─────────────────────────────────────────────────────

    @staticmethod
    def build_research_prompt(
        query: str,
        documents: List[Any],
        history: str = "",
        rewrite: bool = True,
    ) -> str:
        safe_query = sanitize_query(query)
        hist = PromptBuilder._compress_history(history)
        hist_block = f"HISTORY:\n{hist}\n\n" if hist else ""

        has_src, src_block = PromptBuilder._sources_block(documents)

        # BUG-R02: always use safe_query in LLM prompt
        return (
            f"{_PERSONA_RESEARCH}\n\n"
            f"{hist_block}"
            f"{src_block}\n\n"
            f"RESEARCH Q: {safe_query}\nDETAILED ANSWER:"
        )

    # ── Retrieval query helper (BUG-R02) ───────────────────────────────────────

    @classmethod
    def get_retrieval_query(cls, query: str, rewrite: bool = True) -> str:
        """
        Public method for the retriever to call when it needs the rewritten
        (HyDE / expanded) version of the query for embedding.

        This is the ONLY place the rewritten query should be used.
        Never pass its output to a build_*_prompt method.
        """
        return cls._retrieval_query(query, rewrite)

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
