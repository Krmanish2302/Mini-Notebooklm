"""
ragas_bridge.py  —  Thin bridge between pipeline result dicts and RAGASEvaluator.

Call  attach_ragas(result, evaluator)  right after every pipeline .run() call.
It reads the standard keys that ALL three pipelines return and appends a
"ragas" key containing the RAGASResult dict (JSON-serialisable).

Keys read from result dict
--------------------------
  answer          : str
  question        : str   (added by this bridge if missing)
  context_chunks  : list  (preferred)
  chunks_used     : list  (fallback)
  retrieved_chunks: list  (last resort)

Keys written to result dict
---------------------------
  ragas : dict   — RAGASResult.to_dict() output
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from src.evaluation.ragas_evaluator import RAGASEvaluator, RAGASResult

logger = logging.getLogger(__name__)

# Module-level singleton so the SentenceTransformer model is loaded once.
_DEFAULT_EVALUATOR: Optional[RAGASEvaluator] = None


def get_default_evaluator() -> RAGASEvaluator:
    global _DEFAULT_EVALUATOR
    if _DEFAULT_EVALUATOR is None:
        _DEFAULT_EVALUATOR = RAGASEvaluator()
    return _DEFAULT_EVALUATOR


def attach_ragas(
    result: Dict[str, Any],
    query: str,
    evaluator: Optional[RAGASEvaluator] = None,
    ground_truth: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Evaluate the pipeline result and attach a 'ragas' key in-place.

    Parameters
    ----------
    result       : pipeline result dict (mutated in-place)
    query        : the original user question
    evaluator    : RAGASEvaluator instance (uses singleton if None)
    ground_truth : optional reference answer for recall / similarity metrics

    Returns
    -------
    The same result dict with 'ragas' key added.
    """
    if evaluator is None:
        try:
            evaluator = get_default_evaluator()
        except Exception as exc:
            logger.warning("ragas_bridge: could not load evaluator: %s", exc)
            result["ragas"] = None
            return result

    answer: str = result.get("answer", "")
    if not answer or not answer.strip():
        result["ragas"] = None
        return result

    # Resolve context chunks — prefer the deduplicated / labelled list
    ctx_chunks: List[Dict] = (
        result.get("context_chunks")
        or result.get("chunks_used")
        or result.get("retrieved_chunks", [])
    )[:20]   # cap at 20 to keep evaluation fast

    if not ctx_chunks:
        logger.debug("ragas_bridge: no context chunks found — skipping evaluation")
        result["ragas"] = None
        return result

    try:
        ragas_result: RAGASResult = evaluator.evaluate(
            question=query,
            answer=answer,
            context_chunks=ctx_chunks,
            ground_truth=ground_truth,
        )
        result["ragas"] = ragas_result.to_dict()
        logger.debug(
            "ragas_bridge: faithfulness=%.3f relevance=%.3f overall=%.3f grade=%s",
            ragas_result.faithfulness,
            ragas_result.answer_relevance,
            ragas_result.overall_score,
            ragas_result.grade,
        )
    except Exception as exc:
        logger.warning("ragas_bridge: evaluation failed: %s", exc)
        result["ragas"] = None

    return result
