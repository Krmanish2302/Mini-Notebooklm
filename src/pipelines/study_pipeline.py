"""
study_pipeline.py  —  Study mode pipeline.

Produces a rich study package:
  answer          : conceptual explanation (Sagan teacher persona)
  quiz_cards      : list of {question, answer, difficulty} dicts
  summary_bullets : 3-5 key takeaway bullets
  learning_path   : ordered list of concept → concept graph steps
  citations       : standard citation objects
  follow_ups      : suggested deeper-dive questions

Flow
----
1. DeepResearchPipeline.run() → rich context + base answer
2. Quiz generation   (LLM call on context_chunks)
3. Summary extraction (LLM call or heuristic)
4. Learning path     (GraphRetriever if available, else stub)
5. ResponseGenerator : assemble full result dict
"""
from __future__ import annotations

import logging
import re
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


_QUIZ_PROMPT = """\
Based ONLY on the following source passages, generate {n} quiz questions.
For each question provide:
  Q: <question>
  A: <concise answer>
  D: easy | medium | hard

Return ONLY the Q/A/D blocks — no preamble.

SOURCES:
{sources}
"""

_SUMMARY_PROMPT = """\
Based ONLY on the following source passages, write {n} concise bullet-point
takeaways (one sentence each, no more than 20 words each).
Return ONLY bullet lines starting with •.

SOURCES:
{sources}
"""


class StudyPipeline:
    """
    Parameters
    ----------
    deep_research_pipeline : DeepResearchPipeline  (handles retrieval + base answer)
    graph_retriever        : GraphRetriever or None
    graph_history          : GraphHistory or None
    llm                    : callable (str) -> str
    n_quiz_cards           : number of quiz cards to generate (default 5)
    n_summary_bullets      : number of summary bullets (default 4)
    """

    def __init__(
        self,
        deep_research_pipeline,
        graph_retriever=None,
        graph_history=None,
        llm: Optional[Callable[[str], str]] = None,
        n_quiz_cards: int = 5,
        n_summary_bullets: int = 4,
    ):
        self.research = deep_research_pipeline
        self.graph_retriever = graph_retriever
        self.graph_history   = graph_history
        self.llm             = llm
        self.n_quiz          = n_quiz_cards
        self.n_summary       = n_summary_bullets

    # ─────────────────────────────────────────────────────────────────────
    #  Main entry point
    # ─────────────────────────────────────────────────────────────────────

    def run(self, query: str) -> Dict[str, Any]:
        """
        Run a full study session turn.

        Returns
        -------
        dict with: answer, citations, follow_ups, sources_used,
                   quiz_cards, summary_bullets, learning_path,
                   retrieved_chunks, sub_queries, tokens_estimate
        """
        # 1. Get base research result (retrieval + base explanation + citations)
        base = self.research.run(query)

        # 2. Build source text block for quiz / summary generation
        ctx_chunks   = base.get("context_chunks", base.get("chunks_used", []))
        if not ctx_chunks:
            ctx_chunks = base.get("retrieved_chunks", [])[:8]
        sources_text = self._chunks_to_text(ctx_chunks)

        # 3. Generate quiz cards
        quiz_cards = self._generate_quiz(sources_text)

        # 4. Generate summary bullets
        summary_bullets = self._generate_summary(sources_text)

        # 5. Learning path from graph (best-effort)
        learning_path = self._get_learning_path(query)

        # 6. Upgrade base answer to study tone if we have an LLM
        answer = self._study_answer(query, sources_text) or base.get("answer", "")

        return {
            **base,
            "answer":           answer,
            "quiz_cards":       quiz_cards,
            "summary_bullets":  summary_bullets,
            "learning_path":    learning_path,
        }

    # ─────────────────────────────────────────────────────────────────────
    #  Generation helpers
    # ─────────────────────────────────────────────────────────────────────

    def _study_answer(self, query: str, sources_text: str) -> Optional[str]:
        """Re-generate the explanation with the Sagan study persona."""
        if not self.llm or not sources_text:
            return None

        from src.generation.prompt_builder import PromptBuilder
        # Build a fresh study prompt from the pre-assembled source text
        # We pass an ad-hoc document list with a single merged block
        fake_doc = [{"content": sources_text, "source_id": "ctx", "citation_label": "S1"}]
        prompt = PromptBuilder.build_study_prompt(query, fake_doc, rewrite=False)
        try:
            return self.llm(prompt)
        except Exception as exc:
            logger.warning("_study_answer LLM call failed: %s", exc)
            return None

    def _generate_quiz(self, sources_text: str) -> List[Dict[str, Any]]:
        """Generate quiz cards from source text via LLM."""
        if not self.llm or not sources_text.strip():
            return []

        prompt = _QUIZ_PROMPT.format(n=self.n_quiz, sources=sources_text[:4000])
        try:
            raw = self.llm(prompt)
            return self._parse_quiz(raw)
        except Exception as exc:
            logger.warning("_generate_quiz failed: %s", exc)
            return []

    def _generate_summary(self, sources_text: str) -> List[str]:
        """Generate summary bullets from source text via LLM."""
        if not self.llm or not sources_text.strip():
            return []

        prompt = _SUMMARY_PROMPT.format(n=self.n_summary, sources=sources_text[:4000])
        try:
            raw = self.llm(prompt)
            bullets = re.findall(r"^[•\-\*]\s*(.+)$", raw, re.MULTILINE)
            return [b.strip() for b in bullets if b.strip()][: self.n_summary]
        except Exception as exc:
            logger.warning("_generate_summary failed: %s", exc)
            return []

    def _get_learning_path(self, query: str) -> List[Dict[str, str]]:
        """Retrieve concept-to-concept path from KG, or return empty list."""
        if not self.graph_retriever:
            return []
        try:
            return self.graph_retriever.get_learning_path(query) or []
        except Exception as exc:
            logger.debug("_get_learning_path failed (non-fatal): %s", exc)
            return []

    # ─────────────────────────────────────────────────────────────────────
    #  Parsing helpers
    # ─────────────────────────────────────────────────────────────────────

    @staticmethod
    def _chunks_to_text(chunks: List[Dict]) -> str:
        parts = []
        for c in chunks:
            text = c.get("content", "")
            if text:
                parts.append(text)
        return "\n\n".join(parts)

    @staticmethod
    def _parse_quiz(raw: str) -> List[Dict[str, Any]]:
        """Parse Q: / A: / D: blocks from raw LLM output."""
        cards: List[Dict] = []
        # Split on blank-line-separated blocks
        blocks = re.split(r"\n{2,}", raw.strip())
        for block in blocks:
            q_match = re.search(r"Q\s*:\s*(.+)", block, re.IGNORECASE)
            a_match = re.search(r"A\s*:\s*(.+)", block, re.IGNORECASE)
            d_match = re.search(r"D\s*:\s*(easy|medium|hard)", block, re.IGNORECASE)
            if q_match and a_match:
                cards.append({
                    "question":   q_match.group(1).strip(),
                    "answer":     a_match.group(1).strip(),
                    "difficulty": d_match.group(1).lower() if d_match else "medium",
                })
        return cards
