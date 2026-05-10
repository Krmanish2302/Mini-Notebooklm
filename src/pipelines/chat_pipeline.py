"""
chat_pipeline.py  —  Chat mode pipeline.

Bug fixes applied (2026-05-10 audit):
  BUG-004: stream() no longer iterates self.llm (invoke callable) char-by-char.
           Constructor now accepts a separate llm_stream callable.
           Falls back to splitting the invoke() result by words if no stream
           callable is provided (backward compat).

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
    llm              : callable (str) -> str    (invoke, NOT the LLMClient object)
    llm_stream       : callable (str) -> Iterator[str]  (streaming tokens)
                       If None, stream() falls back to word-splitting invoke().
    top_k            : int   chunks to retrieve
    history_k        : int   recent turns to include in prompt
    max_ctx_tokens   : int   token budget for context block
    """

    def __init__(
        self,
        hybrid_retriever,
        rag_history: RAGChatHistory,
        llm: Callable[[str], str],
        llm_stream: Optional[Callable[[str], Iterator[str]]] = None,  # BUG-004
        top_k: int = 5,
        history_k: int = 3,
        max_ctx_tokens: int = 3000,
    ):
        self.retriever    = hybrid_retriever
        self.history      = rag_history
        self.llm          = llm
        self.llm_stream   = llm_stream          # BUG-004
        self.top_k        = top_k
        self.history_k    = history_k
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
        raw_chunks = self._retrieve(query)
        context_chunks, _sources = self._ctx_builder.build(raw_chunks, query=query)
        history_str = self.history.format_for_prompt(query, k=self.history_k)
        prompt = PromptBuilder.build_chat_prompt(
            query, context_chunks,
            history=history_str, persona_config=persona_config, rewrite=True,
        )
        raw_output = self.llm(prompt)
        gen = ResponseGenerator(context_chunks=context_chunks)
        result = gen.assemble(raw_output, query=query, generate_follow_ups=True)
        result["retrieved_chunks"] = raw_chunks
        result["context_chunks"]   = context_chunks   # BUG-007 fix: always set
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
            query, context_chunks,
            history=history_context, persona_config=persona_config, rewrite=True,
        )

    # ─────────────────────────────────────────────────────────────────────
    #  Streaming  (BUG-004)
    # ─────────────────────────────────────────────────────────────────────

    def stream(
        self,
        query: str,
        persona_config=None,
        llm_stream_fn: Optional[Callable[[str], Iterator[str]]] = None,
    ) -> Iterator[str]:
        """
        Streaming variant.  Yields tokens from the LLM.

        BUG-004 fix: uses self.llm_stream (a proper token-streaming callable)
        instead of self.llm (invoke, returns full str).  Iterating a str
        yields individual characters, not semantic tokens.

        Priority order for stream callable:
          1. llm_stream_fn argument (caller override)
          2. self.llm_stream (set at construction by master_pipeline)
          3. word-split fallback on self.llm (no-op graceful degradation)
        """
        raw_chunks = self._retrieve(query)
        context_chunks, _ = self._ctx_builder.build(raw_chunks, query=query)
        history_str = self.history.format_for_prompt(query, k=self.history_k)
        prompt = PromptBuilder.build_chat_prompt(
            query, context_chunks,
            history=history_str, persona_config=persona_config, rewrite=True,
        )

        # BUG-004: resolve the streaming callable with fallback chain
        _stream_fn = llm_stream_fn or self.llm_stream

        full_response = ""

        if _stream_fn is not None:
            # True token streaming
            for token in _stream_fn(prompt):
                full_response += token
                yield token
        else:
            # Graceful fallback: call invoke() and word-split
            logger.warning(
                "ChatPipeline.stream(): no llm_stream callable — "
                "falling back to word-split of invoke() result"
            )
            response = self.llm(prompt)
            for word in response.split(" "):
                token = word + " "
                full_response += token
                yield token

        self.history.add_message("user", query)
        self.history.add_message("assistant", full_response.strip())

    # ─────────────────────────────────────────────────────────────────────
    #  Retrieval helper
    # ─────────────────────────────────────────────────────────────────────

    def _retrieve(self, query: str) -> List[Dict]:
        try:
            return self.retriever.retrieve(query, top_k=self.top_k)
        except Exception as exc:
            logger.warning("ChatPipeline._retrieve failed: %s", exc)
            return []
