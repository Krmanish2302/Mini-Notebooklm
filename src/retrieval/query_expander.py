"""
query_expander.py

Query expansion using LangChain LCEL chains.

Classes:
  SubQueryDecomposer  — break complex query into N sub-queries (LLM)
  MultiQueryExpander  — generate alternative phrasings (LangChain MultiQueryRetriever)

Both fall back to deterministic heuristic variants when no LLM is available.
"""
from __future__ import annotations
import logging
import os
import re
from typing import List, Optional

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

logger = logging.getLogger(__name__)
LLM_MODEL = os.getenv("QUERY_EXPANSION_MODEL", "gpt-4o-mini")


def _heuristic_variants(query: str, n: int = 3) -> List[str]:
    """Deterministic fallback — no LLM calls."""
    variants = [query]
    stripped = query.rstrip("?.").strip()
    if not re.match(r"^(what|how|why|when|where|who|explain|describe|list)", stripped.lower()):
        variants.append(f"What is {stripped}?")
    else:
        variants.append(f"Explain in detail: {stripped}")

    stop = {"the","a","an","is","are","was","were","of","in","on","at","to",
            "for","and","or","but","with","about","what","how","why","when","where","who"}
    keywords = " ".join(w for w in re.findall(r"\b\w+\b", stripped.lower()) if w not in stop)
    variants.append(keywords if keywords and keywords != stripped.lower() else f"Summarise: {stripped}")
    return variants[:n]


def _get_llm():
    from langchain_openai import ChatOpenAI
    return ChatOpenAI(model=LLM_MODEL, temperature=0)


class SubQueryDecomposer:
    """
    Decomposes a complex query into N focused sub-queries using an LLM LCEL chain.
    Falls back to heuristic variants when LLM unavailable.
    """

    _PROMPT = ChatPromptTemplate.from_messages([
        ("system",
         "You are a research assistant. Break the following question into "
         "{n} focused, self-contained sub-questions that together fully cover "
         "the original question. Output ONLY a numbered list, one per line."),
        ("human", "Question: {query}\n\nSub-questions:"),
    ])

    def __init__(self, n: int = 3, use_llm: bool = True):
        self.n       = n
        self.use_llm = use_llm

    def decompose(self, query: str) -> List[str]:
        if not self.use_llm:
            return _heuristic_variants(query, self.n)
        try:
            chain  = self._PROMPT | _get_llm() | StrOutputParser()
            raw    = chain.invoke({"query": query, "n": self.n})
            parsed = self._parse(raw)
            if len(parsed) >= 2:
                return parsed[:self.n]
            return (parsed + _heuristic_variants(query, self.n))[:self.n]
        except Exception as exc:
            logger.warning("[SubQueryDecomposer] LLM failed (%s) — heuristic fallback", exc)
            return _heuristic_variants(query, self.n)

    @staticmethod
    def _parse(raw: str) -> List[str]:
        lines = []
        for line in raw.splitlines():
            line = re.sub(r"^\s*\d+[.)\-]\s*", "", line).strip()
            if line and len(line) > 5:
                lines.append(line)
        return lines


class MultiQueryExpander:
    """
    Generates N alternative phrasings using LangChain MultiQueryRetriever logic.
    Falls back to heuristic variants when LLM unavailable.
    """

    _PROMPT = ChatPromptTemplate.from_messages([
        ("system",
         "Generate {n} alternative phrasings of the following question that "
         "preserve its meaning but use different words. "
         "Output ONLY the phrasings, one per line, no numbering."),
        ("human", "Original question: {query}\n\nAlternative phrasings:"),
    ])

    def __init__(self, n: int = 3, use_llm: bool = True):
        self.n       = n
        self.use_llm = use_llm

    def expand(self, query: str) -> List[str]:
        if not self.use_llm:
            return _heuristic_variants(query, self.n)
        try:
            chain    = self._PROMPT | _get_llm() | StrOutputParser()
            raw      = chain.invoke({"query": query, "n": self.n})
            variants = [l.strip() for l in raw.splitlines() if l.strip()][:self.n]
            if not variants:
                return _heuristic_variants(query, self.n)
            if query not in variants:
                variants.insert(0, query)
            return variants[:self.n]
        except Exception as exc:
            logger.warning("[MultiQueryExpander] LLM failed (%s) — heuristic fallback", exc)
            return _heuristic_variants(query, self.n)