"""ResponseGenerator — orchestrates retrieval → prompt → LLM → parse.

This is the single entry-point for the generation half of the pipeline.
MasterPipeline calls it indirectly through generate(); in Phase 4 it will
grow to handle streaming, grounding validation, and citation injection.

Phase 3 deliverable: correct interface + full wire-up.  No logic gaps.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, Iterator, List, Optional, Union

logger = logging.getLogger(__name__)


class ResponseGenerator:
    """Orchestrates a single query through the full generation stack.

    Parameters
    ----------
    prompt_builder:  PromptBuilder instance
    llm_client:      LLMClient instance
    response_parser: ResponseParser instance
    citation_extractor: CitationExtractor instance (optional)
    """

    def __init__(
        self,
        prompt_builder,
        llm_client,
        response_parser,
        citation_extractor=None,
    ):
        self.prompt_builder = prompt_builder
        self.llm = llm_client
        self.parser = response_parser
        self.citation_extractor = citation_extractor

    def generate(
        self,
        query: str,
        documents: List[Any],
        history: str = "",
        mode: str = "chat",
        stream: bool = False,
        learning_path: Optional[List[Dict]] = None,
    ) -> Union[str, Iterator[str]]:
        """Run query through prompt → LLM → parse.  Returns str or Iterator[str].

        Parameters
        ----------
        query       : user question
        documents   : LangChain Documents (or plain dicts) from retrieval
        history     : formatted chat history string
        mode        : 'chat' | 'study' | 'research'
        stream      : if True, yield token-by-token strings
        learning_path: optional study-mode concept graph steps
        """
        # ─ Build prompt ──────────────────────────────────────────────────
        build_fn = {
            "chat":     self.prompt_builder.build_chat_prompt,
            "study":    self.prompt_builder.build_study_prompt,
            "research": self.prompt_builder.build_research_prompt,
        }.get(mode, self.prompt_builder.build_chat_prompt)

        if mode == "study" and learning_path is not None:
            prompt = build_fn(query=query, documents=documents,
                              history=history, learning_path=learning_path)
        else:
            prompt = build_fn(query=query, documents=documents, history=history)

        logger.debug("ResponseGenerator: prompt built, mode=%s stream=%s", mode, stream)

        # ─ Call LLM ──────────────────────────────────────────────────────
        if stream:
            return self._stream(prompt)

        raw = self.llm.invoke(prompt)
        parsed = self.parser.parse(raw)

        # ─ Optional citation injection ─────────────────────────────────
        if self.citation_extractor is not None and documents:
            parsed = self.citation_extractor.inject(
                response=parsed,
                documents=documents,
            )

        return parsed

    def _stream(self, prompt: str) -> Iterator[str]:
        """Yield raw tokens from LLM stream."""
        for token in self.llm.stream(prompt):
            yield token
