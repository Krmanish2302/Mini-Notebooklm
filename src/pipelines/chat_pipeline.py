"""
chat_pipeline.py  —  Chat mode pipeline.

Flow
----
1. QueryRewriter   : HyDE / expand / both based on query shape
2. HybridRetriever : BM25 + FAISS RRF fusion
3. ContextBuilder  : dedup + token budget + citation labels
4. PromptBuilder   : assembles system + history + sources + query
5. LLM call        : invoke / stream
6. ResponseGenerator: structured output with citations + follow-ups
7. RAGChatHistory  : persist user + assistant turns
"""
from __future__ import annotations

import logging
from typing import Any, Callable, Dict, Iterator, List, Optional

from src.generation.prompt_builder import PromptBuilder, QueryRewriter
from src.generation.response_generator import ResponseGenerator
from src.retrieval.context_builder import ContextBuilder
from src.chat_history.rag_history import RAGChatHistory

logger = logging.getLogger(__name__)


class ChatPipeline:
    """
    Parameters
    ----------
    hybrid_retriever : HybridRetriever
    rag_history      : RAGChatHistory
    llm              : callable (str) -> str   (invoke, NOT the LLMClient object)
    top_k            : int   chunks to retrieve
    history_k        : int   recent turns to include in prompt
    max_ctx_tokens   : int   token budget for context block
    """

    def __init__(
        self,
        hybrid_retriever,
        rag_history: RAGChatHistory,
        llm: Callable[[str], str],
        top_k: int = 5,
        history_k: int = 3,
        max_ctx_tokens: int = 3000,
    ):
        self.retriever  = hybrid_retriever
        self.history    = rag_history
        self.llm        = llm
        self.top_k      = top_k
        self.history_k  = history_k
        self._ctx_builder = ContextBuilder(max_tokens=max_ctx_tokens)
        self._rewriter    = QueryRewriter()

    # ─────────────────────────────────────────────────────────────────────
    #  Non-streaming run
    # ─────────────────────────────────────────────────────────────────────

    def run(
        self,
        query: str,
        persona_config=None,
    ) -> Dict[str, Any]:
        """
        Run a full non-streaming chat turn.

        Returns
        -------
        dict with: answer, citations, follow_ups, sources_used,
                   chunks_used, retrieved_chunks, tokens_estimate
        """
        # 1. Retrieve
        raw_chunks = self._retrieve(query)

        # 2. Build context
        context_chunks, _sources = self._ctx_builder.build(raw_chunks, query=query)

        # 3. Build prompt
        history_str = self.history.format_for_prompt(query, k=self.history_k)
        prompt = PromptBuilder.build_chat_prompt(
            query,
            context_chunks,
            history=history_str,
            persona_config=persona_config,
            rewrite=True,
        )

        # 4. LLM call
        raw_output = self.llm(prompt)

        # 5. Structure response
        gen = ResponseGenerator(context_chunks=context_chunks)
        result = gen.assemble(raw_output, query=query, generate_follow_ups=True)
        result["retrieved_chunks"] = raw_chunks

        # 6. Persist history
        self.history.add_message("user", query)
        self.history.add_message("assistant", result["answer"])

        return result

    # ─────────────────────────────────────────────────────────────────────
    #  Internal prompt builder (used by master_pipeline streaming path)
    # ─────────────────────────────────────────────────────────────────────

    def _build_prompt(
        self,
        query: str,
        raw_chunks: List[Dict],
        history_context: str,
        persona_config=None,
    ) -> str:
        context_chunks, _ = self._ctx_builder.build(raw_chunks, query=query)
        return PromptBuilder.build_chat_prompt(
            query,
            context_chunks,
            history=history_context,
            persona_config=persona_config,
            rewrite=True,
        )

    # ─────────────────────────────────────────────────────────────────────
    #  Streaming helper
    # ─────────────────────────────────────────────────────────────────────

    def stream(
        self,
        query: str,
        persona_config=None,
        llm_stream_fn: Optional[Callable[[str], Iterator[str]]] = None,
    ) -> Iterator[str]:
        """
        Streaming variant.  Yields tokens from the LLM.
        Persists history after the stream is exhausted.

        Parameters
        ----------
        llm_stream_fn : optional alternative streaming callable.
                        Falls back to self.llm (non-streaming) if not provided.
        """
        raw_chunks = self._retrieve(query)
        context_chunks, _ = self._ctx_builder.build(raw_chunks, query=query)
        history_str = self.history.format_for_prompt(query, k=self.history_k)
        prompt = PromptBuilder.build_chat_prompt(
            query,
            context_chunks,
            history=history_str,
            persona_config=persona_config,
            rewrite=True,
        )

        stream_fn = llm_stream_fn or self.llm
        full_response = ""
        for token in stream_fn(prompt):  # type: ignore[call-arg]
            full_response += token
            yield token

        self.history.add_message("user", query)
        self.history.add_message("assistant", full_response)

    # ─────────────────────────────────────────────────────────────────────
    #  Retrieval helper
    # ─────────────────────────────────────────────────────────────────────

    def _retrieve(self, query: str) -> List[Dict]:
        """Embed-query rewrite then retrieve top_k chunks."""
        try:
            return self.retriever.retrieve(query, top_k=self.top_k)
        except Exception as exc:
            logger.warning("ChatPipeline._retrieve failed: %s", exc)
            return []
