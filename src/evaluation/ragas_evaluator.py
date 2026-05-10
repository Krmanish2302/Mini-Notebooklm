"""
ragas_evaluator.py  —  RAGAS-style evaluation for Mini NotebookLM.

Computes 5 core RAGAS metrics LOCALLY (no external RAGAS library required).
All metrics return a float in [0.0, 1.0].

Metrics
-------
1. Faithfulness (Grounding Score)
   How well every claim in the answer is supported by the retrieved context.
   Method: sentence-level NLI — each answer sentence checked against the
   full context block using token-overlap (fallback when no NLI model).
   Score = supported_sentences / total_answer_sentences

2. Answer Relevance
   How relevant the answer is to the original question.
   Method: cosine similarity between question embedding and answer embedding.

3. Context Recall
   How much of the ground-truth information appears in the retrieved context.
   (Only available when ground_truth is provided.)
   Method: sentence-level coverage — each GT sentence checked against context.

4. Context Precision
   What fraction of retrieved chunks actually contributed to the answer.
   Method: for each chunk, check if any answer sentence overlaps ≥ threshold.

5. Answer Similarity
   Semantic similarity between the generated answer and ground truth.
   (Only available when ground_truth is provided.)
   Method: cosine similarity of sentence embeddings.

Usage
-----
    evaluator = RAGASEvaluator()
    result = evaluator.evaluate(
        question="What is RAG?",
        answer="RAG stands for Retrieval-Augmented Generation...",
        context_chunks=[{"content": "..."}, ...],
        ground_truth=None,   # optional
    )
    print(result.faithfulness)    # 0.87
    print(result.to_dict())       # full dict for JSON serialization
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
#  Result dataclass
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class RAGASResult:
    faithfulness:       float          # grounding score  — shown inline in chat
    answer_relevance:   float
    context_precision:  float
    context_recall:     Optional[float]  # None when no ground_truth provided
    answer_similarity:  Optional[float]  # None when no ground_truth provided
    # meta
    question:           str
    answer_sentences:   int
    supported_sentences: int
    chunks_evaluated:   int
    chunks_contributed: int
    has_ground_truth:   bool
    # per-chunk detail for the UI panel
    chunk_details:      List[Dict[str, Any]]

    @property
    def overall_score(self) -> float:
        """
        Weighted composite: emphasises faithfulness + relevance.
        Weights: faithfulness 0.35, answer_relevance 0.30,
                 context_precision 0.20, context_recall 0.15 (when available).
        When ground truth is absent, context_recall weight is split equally
        between faithfulness and answer_relevance.
        """
        if self.has_ground_truth and self.context_recall is not None:
            return (
                0.35 * self.faithfulness +
                0.30 * self.answer_relevance +
                0.20 * self.context_precision +
                0.15 * self.context_recall
            )
        return (
            0.40 * self.faithfulness +
            0.35 * self.answer_relevance +
            0.25 * self.context_precision
        )

    @property
    def grade(self) -> str:
        s = self.overall_score
        if s >= 0.85: return "Excellent"
        if s >= 0.70: return "Good"
        if s >= 0.55: return "Fair"
        return "Poor"

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["overall_score"]   = round(self.overall_score, 4)
        d["grade"]           = self.grade
        d["faithfulness"]    = round(self.faithfulness, 4)
        d["answer_relevance"]= round(self.answer_relevance, 4)
        d["context_precision"] = round(self.context_precision, 4)
        if self.context_recall is not None:
            d["context_recall"] = round(self.context_recall, 4)
        if self.answer_similarity is not None:
            d["answer_similarity"] = round(self.answer_similarity, 4)
        return d


# ─────────────────────────────────────────────────────────────────────────────
#  Evaluator
# ─────────────────────────────────────────────────────────────────────────────

class RAGASEvaluator:
    """
    Parameters
    ----------
    embedding_model : str   sentence-transformers model for semantic similarity
    overlap_threshold : float   min token-overlap ratio for "supported" claim
    """

    def __init__(
        self,
        embedding_model: str = "all-MiniLM-L6-v2",
        overlap_threshold: float = 0.25,
    ):
        self._emb_model_name = embedding_model
        self._embedder = None           # lazy-loaded
        self.overlap_threshold = overlap_threshold

    # ── lazy embedder ────────────────────────────────────────────────────────

    def _get_embedder(self):
        if self._embedder is None:
            from sentence_transformers import SentenceTransformer
            self._embedder = SentenceTransformer(self._emb_model_name)
        return self._embedder

    # ── public API ───────────────────────────────────────────────────────────

    def evaluate(
        self,
        question: str,
        answer: str,
        context_chunks: List[Dict[str, Any]],
        ground_truth: Optional[str] = None,
    ) -> RAGASResult:
        """
        Run all applicable metrics and return a RAGASResult.
        Fast enough to call synchronously after each LLM response.
        """
        if not answer or not answer.strip():
            return self._empty_result(question, context_chunks)

        answer_sents   = self._split_sentences(answer)
        context_text   = self._chunks_to_text(context_chunks)
        context_sents  = self._split_sentences(context_text)

        # 1. Faithfulness
        supported, faith_details = self._faithfulness(
            answer_sents, context_text, context_chunks
        )

        # 2. Answer Relevance
        relevance = self._answer_relevance(question, answer)

        # 3. Context Precision
        precision, contributed = self._context_precision(answer_sents, context_chunks)

        # 4 & 5. Ground-truth metrics (optional)
        recall = None
        similarity = None
        if ground_truth and ground_truth.strip():
            recall    = self._context_recall(ground_truth, context_text)
            similarity = self._answer_similarity(answer, ground_truth)

        faithfulness_score = (
            len(supported) / len(answer_sents) if answer_sents else 0.0
        )

        return RAGASResult(
            faithfulness        = faithfulness_score,
            answer_relevance    = relevance,
            context_precision   = precision,
            context_recall      = recall,
            answer_similarity   = similarity,
            question            = question,
            answer_sentences    = len(answer_sents),
            supported_sentences = len(supported),
            chunks_evaluated    = len(context_chunks),
            chunks_contributed  = contributed,
            has_ground_truth    = ground_truth is not None and bool(ground_truth.strip()),
            chunk_details       = faith_details,
        )

    # ── Metric 1: Faithfulness ───────────────────────────────────────────────

    def _faithfulness(
        self,
        answer_sents: List[str],
        context_text: str,
        context_chunks: List[Dict],
    ):
        """
        For each answer sentence, compute max token-overlap against all
        context chunks.  Returns (supported_indices, chunk_detail_list).
        """
        ctx_tokens = self._tokenise(context_text)
        supported  = []
        details    = []

        for i, sent in enumerate(answer_sents):
            sent_tokens = self._tokenise(sent)
            if not sent_tokens:
                continue
            overlap = len(sent_tokens & ctx_tokens) / len(sent_tokens)
            is_supported = overlap >= self.overlap_threshold
            if is_supported:
                supported.append(i)
            details.append({
                "sentence":    sent,
                "overlap":     round(overlap, 3),
                "supported":   is_supported,
            })

        # Per-chunk contribution
        chunk_details = []
        for c in context_chunks:
            c_tokens = self._tokenise(c.get("content", ""))
            contrib_count = sum(
                1 for s in answer_sents
                if c_tokens and
                   len(self._tokenise(s) & c_tokens) / max(len(self._tokenise(s)), 1)
                   >= self.overlap_threshold
            )
            chunk_details.append({
                "chunk_id":    c.get("id", c.get("source_id", "?"))[:16],
                "source":      c.get("source_id", "unknown"),
                "citation":    c.get("citation_label", ""),
                "contributed": contrib_count > 0,
                "sentences_supported": contrib_count,
                "score":       round(contrib_count / max(len(answer_sents), 1), 3),
            })

        return supported, chunk_details

    # ── Metric 2: Answer Relevance ───────────────────────────────────────────

    def _answer_relevance(self, question: str, answer: str) -> float:
        try:
            emb = self._get_embedder()
            vecs = emb.encode([question, answer], normalize_embeddings=True)
            return max(0.0, float(np.dot(vecs[0], vecs[1])))
        except Exception as exc:
            logger.warning("answer_relevance embedding failed: %s", exc)
            # Fallback: simple word-overlap
            q_tok = self._tokenise(question)
            a_tok = self._tokenise(answer)
            if not q_tok: return 0.0
            return len(q_tok & a_tok) / len(q_tok)

    # ── Metric 3: Context Precision ──────────────────────────────────────────

    def _context_precision(
        self,
        answer_sents: List[str],
        context_chunks: List[Dict],
    ):
        if not context_chunks: return 0.0, 0
        answer_tokens = set()
        for s in answer_sents:
            answer_tokens |= self._tokenise(s)

        contributed = 0
        for c in context_chunks:
            c_tok = self._tokenise(c.get("content", ""))
            if c_tok and len(c_tok & answer_tokens) / len(c_tok) >= self.overlap_threshold:
                contributed += 1

        return contributed / len(context_chunks), contributed

    # ── Metric 4: Context Recall (needs ground truth) ────────────────────────

    def _context_recall(self, ground_truth: str, context_text: str) -> float:
        gt_sents  = self._split_sentences(ground_truth)
        ctx_tokens = self._tokenise(context_text)
        if not gt_sents: return 0.0

        covered = sum(
            1 for s in gt_sents
            if len(self._tokenise(s) & ctx_tokens) / max(len(self._tokenise(s)), 1)
               >= self.overlap_threshold
        )
        return covered / len(gt_sents)

    # ── Metric 5: Answer Similarity (needs ground truth) ─────────────────────

    def _answer_similarity(self, answer: str, ground_truth: str) -> float:
        try:
            emb  = self._get_embedder()
            vecs = emb.encode([answer, ground_truth], normalize_embeddings=True)
            return max(0.0, float(np.dot(vecs[0], vecs[1])))
        except Exception as exc:
            logger.warning("answer_similarity embedding failed: %s", exc)
            a_tok  = self._tokenise(answer)
            gt_tok = self._tokenise(ground_truth)
            if not gt_tok: return 0.0
            return len(a_tok & gt_tok) / len(gt_tok)

    # ── Helpers ──────────────────────────────────────────────────────────────

    @staticmethod
    def _split_sentences(text: str) -> List[str]:
        if not text: return []
        raw = re.split(r"(?<=[.!?])\s+", text.strip())
        return [s.strip() for s in raw if len(s.strip()) > 8]

    @staticmethod
    def _tokenise(text: str):
        """Lowercase word tokens, stopwords removed."""
        STOP = {
            "a","an","the","is","it","in","on","at","to","of","for",
            "and","or","but","not","with","this","that","are","was",
            "be","by","from","as","has","have","its","their","they",
            "we","you","i","he","she","do","did","can","will","about",
        }
        tokens = re.findall(r"[a-z]+", text.lower())
        return {t for t in tokens if t not in STOP and len(t) > 2}

    @staticmethod
    def _chunks_to_text(chunks: List[Dict]) -> str:
        return " ".join(c.get("content", "") for c in chunks)

    def _empty_result(self, question: str, context_chunks: List[Dict]) -> RAGASResult:
        return RAGASResult(
            faithfulness=0.0, answer_relevance=0.0,
            context_precision=0.0, context_recall=None, answer_similarity=None,
            question=question, answer_sentences=0, supported_sentences=0,
            chunks_evaluated=len(context_chunks), chunks_contributed=0,
            has_ground_truth=False, chunk_details=[],
        )
