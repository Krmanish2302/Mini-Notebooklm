"""
prompt_builder.py — Mode-specific prompt construction.

Key rules:
  BUG-R02: retrieval_query (HyDE/expanded) is for the RETRIEVER only.
           The LLM always sees the original clean query.
  BUG-R04: HistoryCompressor includes User: turns in summary bullets.
  BUG-S01: sanitize_query() strips prompt-injection + length cap.

History strategy: RAG-based.
  The caller (ChatGraph / pipeline) retrieves the N most-relevant past
  turn pairs from SQLite via semantic search, serialises them to a plain-
  text block, and passes it in as `history`.  No BufferWindowMemory or
  MemorySaver is used here.
"""
from __future__ import annotations
import logging
import re
from typing import Any, Dict, List, Optional

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate

from src.generation.persona_config import PersonaConfig

logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────
HISTORY_CHAR_LIMIT = 3_000
HISTORY_KEEP_TURNS = 4
MAX_QUERY_LENGTH   = 1_200
_EM_DASH           = "\u2014"

_NO_SOURCES_BLOCK = (
    "SOURCES: [none \u2014 the knowledge base returned no relevant chunks for this query]\n\n"
    "INSTRUCTION: Politely tell the user you couldn\u2019t find anything relevant in the "
    "ingested documents and suggest they (a) rephrase, (b) ingest more sources, "
    "or (c) check their source filters.  Do NOT make up an answer."
)

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


# ── BUG-S01: Input sanitization ────────────────────────────────────────────────

def sanitize_query(query: str, max_len: int = MAX_QUERY_LENGTH) -> str:
    q = query.strip()[:max_len]
    q = re.sub(
        r"(?i)(ignore|disregard|forget|override|bypass)\b.{0,50}\b"
        r"(instructions?|rules?|system\s*prompt|context|previous)",
        "[sanitized]", q,
    )
    q = re.sub(r"(?i)\bDAN\b|\bact as\b.{0,30}\bno restrictions\b", "[sanitized]", q)
    return q


# ── QueryRewriter ──────────────────────────────────────────────────────────────

class QueryRewriter:
    """
    Rewrites queries for RETRIEVAL EMBEDDING only.
    Output must NEVER be used in the LLM prompt (BUG-R02).
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

    def expand(self, query: str) -> str:
        tokens   = re.findall(r"\b[a-zA-Z][a-zA-Z0-9_\-]{2,}\b", query)
        keywords = [t.lower() for t in tokens if t.lower() not in self._STOP]
        unique_kw = list(dict.fromkeys(keywords))
        return f"{query} | keywords: {', '.join(unique_kw)}" if unique_kw else query

    def rewrite(self, query: str, strategy: str = "expand") -> str:
        if strategy == "expand": return self.expand(query)
        return query

    @staticmethod
    def pick_strategy(query: str) -> str:
        word_count  = len(query.split())
        if word_count <= 4:   return "expand"
        return "none"



# ── HistoryCompressor ──────────────────────────────────────────────────────────

class HistoryCompressor:
    """
    Compresses long RAG-retrieved history blocks.
    BUG-R04 fix: includes User: turns in bullets.
    """

    @staticmethod
    def _split_turns(history: str) -> List[str]:
        turns = re.split(r"\n(?=(?:User:|Assistant:))", history.strip())
        return [t.strip() for t in turns if t.strip()]

    @staticmethod
    def _summarise_turns(turns: List[str]) -> str:
        bullets: List[str] = []
        for turn in turns:
            if turn.startswith("User:"):
                q = turn[5:].strip().split("\n")[0][:100]
                bullets.append(f"\u2022 User asked: {q}")
            elif turn.startswith("Assistant:"):
                reply = turn[len("Assistant:"):].strip()
                first = re.split(r"(?<=[.!?])\s", reply)[0]
                if first:
                    bullets.append(f"  \u2192 {first}")
        if not bullets:
            return "[earlier conversation compressed \u2014 no key points extracted]"
        return "EARLIER CONTEXT (compressed):\n" + "\n".join(bullets)

    @classmethod
    def compress(
        cls,
        history:    str,
        char_limit: int = HISTORY_CHAR_LIMIT,
        keep_turns: int = HISTORY_KEEP_TURNS,
    ) -> str:
        if not history or len(history) <= char_limit:
            return history
        turns = cls._split_turns(history)
        if len(turns) <= keep_turns:
            return history
        old_turns    = turns[:-keep_turns]
        recent_turns = turns[-keep_turns:]
        summary      = cls._summarise_turns(old_turns)
        recent_block = "\n\n".join(recent_turns)
        return f"{summary}\n\nRECENT CONVERSATION:\n{recent_block}"


# ── Context formatter ──────────────────────────────────────────────────────────

def format_context(documents: List[Any]) -> str:
    if not documents:
        return _NO_SOURCES_BLOCK
    parts: List[str] = []
    for i, doc in enumerate(documents, 1):
        if hasattr(doc, "page_content"):
            content = doc.page_content
            meta    = doc.metadata or {}
        else:
            content = doc.get("content", "")
            meta    = {k: v for k, v in doc.items() if k != "content"}
        src = meta.get("source", meta.get("source_id", ""))
        src_suffix = (" " + _EM_DASH + " " + src) if src else ""
        header = f"[S{i}]{src_suffix}"
        block  = f"{header}\n{content}"
        if ctx := meta.get("context_window", ""):
            block += f"\n[context] {ctx}"
        parts.append(block)
    return "\n\n".join(parts)


# ── PromptBuilder ──────────────────────────────────────────────────────────────

class PromptBuilder:
    """
    Builds LangChain ChatPromptTemplate-based prompts for each mode.

    History contract (RAG-based):
      - Caller retrieves relevant past turns from SQLite/FAISS.
      - Serialises them as:  "User: ...\nAssistant: ...\n"
      - Passes the block as `history` str.
      - HistoryCompressor trims if > HISTORY_CHAR_LIMIT.
      - NO MemorySaver / ConversationBufferWindowMemory used.
    """

    _rewriter   = QueryRewriter()
    _compressor = HistoryCompressor()

    # ── Chat mode ────────────────────────────────────────────────────────

    _CHAT_TEMPLATE = ChatPromptTemplate.from_messages([
        ("system", "{system_prompt}"),
        ("human",  "HISTORY:\n{history}\n\n{sources}\n\nQ: {query}\nA:"),
    ])

    @classmethod
    def build_chat_prompt(
        cls,
        query:          str,
        documents:      List[Any],
        history:        str             = "",
        persona_config: Optional[PersonaConfig] = None,
    ) -> str:
        cfg        = persona_config or PersonaConfig()
        safe_query = sanitize_query(query)
        hist       = cls._compressor.compress(history)
        sources    = format_context(documents)
        return cls._CHAT_TEMPLATE.format(
            system_prompt=cfg.build_system_prompt(),
            history=hist,
            sources=sources,
            query=safe_query,
        )

    # ── Study mode ────────────────────────────────────────────────────────

    _STUDY_TEMPLATE = ChatPromptTemplate.from_messages([
        ("system", _PERSONA_STUDY),
        ("human",  "{path_block}{history}{sources}\n\nTOPIC: {query}\nEXPLAIN:"),
    ])

    @classmethod
    def build_study_prompt(
        cls,
        query:         str,
        documents:     List[Any],
        history:       str              = "",
        learning_path: Optional[List[Dict]] = None,
    ) -> str:
        safe_query = sanitize_query(query)
        hist       = cls._compressor.compress(history)
        hist_block = f"HISTORY:\n{hist}\n\n" if hist else ""
        sources    = format_context(documents)
        path_block = ""
        if learning_path:
            steps = " \u2192 ".join(
                "{}\u279c {}".format(s.get("from", ""), s.get("to", ""))
                for s in learning_path[:4]
            )
            path_block = f"CONCEPT PATH: {steps}\n\n"
        return cls._STUDY_TEMPLATE.format(
            path_block=path_block,
            history=hist_block,
            sources=sources,
            query=safe_query,
        )

    # ── Deep Research mode ────────────────────────────────────────────────

    _RESEARCH_TEMPLATE = ChatPromptTemplate.from_messages([
        ("system", _PERSONA_RESEARCH),
        ("human",  "{history}{sources}\n\nRESEARCH Q: {query}\nDETAILED ANSWER:"),
    ])

    @classmethod
    def build_research_prompt(
        cls,
        query:     str,
        documents: List[Any],
        history:   str = "",
    ) -> str:
        safe_query = sanitize_query(query)
        hist       = cls._compressor.compress(history)
        hist_block = f"HISTORY:\n{hist}\n\n" if hist else ""
        sources    = format_context(documents)
        return cls._RESEARCH_TEMPLATE.format(
            history=hist_block,
            sources=sources,
            query=safe_query,
        )

    # ── Retrieval query helper (BUG-R02) ──────────────────────────────────

    @classmethod
    def get_retrieval_query(cls, query: str, rewrite: bool = True) -> str:
        """
        Return the rewritten (HyDE/expanded) query for the retriever ONLY.
        Never pass this output into any build_*_prompt method.
        """
        if not rewrite:
            return query
        strategy = QueryRewriter.pick_strategy(query)
        return cls._rewriter.rewrite(query, strategy=strategy)

    # ── Backward-compat aliases ────────────────────────────────────────────

    @classmethod
    def build_deep_research_prompt(cls, query: str, documents: List[Any], history: str = "") -> str:
        return cls.build_research_prompt(query, documents, history)

    @classmethod
    def build_study_mode_prompt(cls, query: str, documents: List[Any],
                                 learning_path=None, history: str = "") -> str:
        return cls.build_study_prompt(query, documents, history, learning_path)
