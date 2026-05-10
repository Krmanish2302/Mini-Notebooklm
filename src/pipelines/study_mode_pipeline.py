"""
study_mode_pipeline.py  —  Study Mode Pipeline

Retrieval: Hybrid → compress → rerank → graph-augment.
Generation: flashcards + MCQ quiz + study summary.

Persona: Carl Sagan as a chill classmate. Build intuition, show connections.
Strictly grounded in retrieved sources.
"""
from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)

# ── Shared persona string ──────────────────────────────────────────────────
_PERSONA_STUDY = (
    "You're Carl Sagan if he were a chill classmate in teacher mode. "
    "Build intuition, use analogies, show how ideas connect. "
    "Use ONLY the sources. Cite as [S1], [S2]… "
    "If it’s not there, say: 'Not in my notes, bro.'"
)

# ── Generation prompts (short + precise) ───────────────────────────────
_FLASHCARD_PROMPT = """\
{persona}
From the SOURCES, generate {n} flashcards. Only use what's in the sources.

Format (repeat exactly):
FRONT: <question or term>
BACK: <answer, 1-3 sentences>

SOURCES:
{context}
"""

_QUIZ_PROMPT = """\
{persona}
From the SOURCES, generate {n} multiple-choice questions. Only use what's in the sources.

Format (repeat exactly):
Q: <question>
A) <option>  B) <option>  C) <option>  D) <option>
ANSWER: <letter>

SOURCES:
{context}
"""

_SUMMARY_PROMPT = """\
{persona}
Summarise the SOURCES into a concise study note (bullet points, max 250 words).
Focus on key concepts, definitions, and relationships. Only what's in the sources.

SOURCES:
{context}

STUDY NOTE:
"""


def _parse_flashcards(raw: str) -> List[Dict[str, str]]:
    cards, front, back = [], None, None
    for line in raw.strip().splitlines():
        line = line.strip()
        if line.startswith("FRONT:"):
            front = line[6:].strip()
        elif line.startswith("BACK:"):
            back = line[5:].strip()
            if front and back:
                cards.append({"front": front, "back": back})
                front = back = None
    return cards


def _parse_quiz(raw: str) -> List[Dict[str, Any]]:
    questions, current = [], {}
    for line in raw.strip().splitlines():
        line = line.strip()
        if line.startswith("Q:"):
            if current.get("question"):
                questions.append(current)
            current = {"question": line[2:].strip(), "options": {}}
        elif len(line) >= 2 and line[0] in "ABCD" and line[1] == ")":
            current.setdefault("options", {})[line[0]] = line[2:].strip()
        elif line.startswith("ANSWER:"):
            current["answer"] = line[7:].strip()
    if current.get("question"):
        questions.append(current)
    return questions


class StudyModePipeline:
    """
    Parameters
    ----------
    hybrid_retriever      : HybridRetriever
    contextual_compressor : ContextualCompressor
    reranker              : Reranker
    llm                   : callable(prompt) -> str
    study_mode_retriever  : StudyModeRetriever | None  (graph augmentation)
    top_k                 : retrieval count (default 12)
    score_threshold       : rerank score floor (default 0.0)
    n_flashcards          : flashcards per run (default 5)
    n_quiz_questions      : MCQ questions per run (default 5)
    generate_flashcards   : bool (default True)
    generate_quiz         : bool (default True)
    generate_summary      : bool (default True)
    """

    def __init__(
        self,
        hybrid_retriever,
        contextual_compressor,
        reranker,
        llm: Callable[[str], str],
        study_mode_retriever=None,
        top_k: int = 12,
        score_threshold: float = 0.0,
        n_flashcards: int = 5,
        n_quiz_questions: int = 5,
        generate_flashcards: bool = True,
        generate_quiz: bool = True,
        generate_summary: bool = True,
    ):
        self.retriever = hybrid_retriever
        self.compressor = contextual_compressor
        self.reranker = reranker
        self.llm = llm
        self.study_retriever = study_mode_retriever
        self.top_k = top_k
        self.score_threshold = score_threshold
        self.n_flashcards = n_flashcards
        self.n_quiz_questions = n_quiz_questions
        self.do_flashcards = generate_flashcards
        self.do_quiz = generate_quiz
        self.do_summary = generate_summary

    def run(
        self,
        query: str,
        top_k: Optional[int] = None,
        score_threshold: Optional[float] = None,
    ) -> Dict[str, Any]:
        effective_top_k = top_k if top_k is not None else self.top_k
        effective_threshold = score_threshold if score_threshold is not None else self.score_threshold

        # Step 1: Hybrid retrieve (wide)
        raw_chunks: List[Dict] = self.retriever.retrieve(query, top_k=effective_top_k * 3)

        # Step 2: Contextual compression
        try:
            compressed = self.compressor.compress(raw_chunks, query)
        except Exception as exc:
            logger.warning("StudyMode[compress] fallback: %s", exc)
            compressed = raw_chunks

        # Step 3: Rerank + threshold
        try:
            reranked = self.reranker.rerank(query, compressed, top_k=len(compressed))
            reranked = [c for c in reranked if c.get("rerank_score", 1.0) >= effective_threshold]
            reranked = reranked[:effective_top_k]
        except Exception as exc:
            logger.warning("StudyMode[rerank] fallback: %s", exc)
            reranked = compressed[:effective_top_k]

        logger.debug(
            "StudyMode: %d raw → %d compressed → %d reranked",
            len(raw_chunks), len(compressed), len(reranked),
        )

        # Step 4: Graph augmentation
        learning_path: List[Dict] = []
        if self.study_retriever is not None:
            try:
                gr = self.study_retriever.retrieve(query)
                seen = {c["id"] for c in reranked if "id" in c}
                for gc in gr.get("chunks", []):
                    if gc.get("id") not in seen:
                        reranked.append(gc)
                        seen.add(gc["id"])
                learning_path = gr.get("learning_path", [])
            except Exception as exc:
                logger.warning("StudyMode[graph_augment] fallback: %s", exc)

        # Build compact source block [S1], [S2]…
        context = "\n\n".join(
            f"[S{i+1}] {c['content']}" for i, c in enumerate(reranked) if c.get("content")
        )
        sources_used = [c["id"] for c in reranked if c.get("id")]

        # Step 5: Flashcards
        flashcards: List[Dict[str, str]] = []
        if self.do_flashcards and context:
            try:
                raw = self.llm(
                    _FLASHCARD_PROMPT.format(
                        persona=_PERSONA_STUDY, n=self.n_flashcards, context=context
                    )
                )
                flashcards = _parse_flashcards(raw)
            except Exception as exc:
                logger.warning("StudyMode[flashcards] fallback: %s", exc)

        # Step 6: Quiz
        quiz: List[Dict[str, Any]] = []
        if self.do_quiz and context:
            try:
                raw = self.llm(
                    _QUIZ_PROMPT.format(
                        persona=_PERSONA_STUDY, n=self.n_quiz_questions, context=context
                    )
                )
                quiz = _parse_quiz(raw)
            except Exception as exc:
                logger.warning("StudyMode[quiz] fallback: %s", exc)

        # Step 7: Summary
        summary = ""
        if self.do_summary and context:
            try:
                summary = self.llm(
                    _SUMMARY_PROMPT.format(persona=_PERSONA_STUDY, context=context)
                ).strip()
            except Exception as exc:
                logger.warning("StudyMode[summary] fallback: %s", exc)

        return {
            "chunks":        reranked,
            "learning_path": learning_path,
            "flashcards":    flashcards,
            "quiz":          quiz,
            "summary":       summary,
            "sources_used":  sources_used,
        }
