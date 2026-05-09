import re
from typing import List, Dict, Any, Union


class ResponseParser:
    """Parse raw LLM output.

    Contract used by master_pipeline.py:
        parse(response_text: str) -> str
            Returns the cleaned response string directly.
            master_pipeline passes this string to chat_history.add_message()
            and returns it to the caller — no dict unwrapping needed.

    For callers that need citations and metadata use:
        parse_structured(response_text: str) -> Dict[str, Any]
            Returns {"content": str, "citations": List[Dict], "has_citations": bool}
    """

    _CITATION_RE = re.compile(r'\[SOURCE_(\d+)\]')

    # ── Primary interface (used by master_pipeline) ───────────────────────────

    @staticmethod
    def parse(response_text: str) -> str:
        """Return cleaned response as a plain string.

        Citation markers ([SOURCE_X]) are preserved in the returned string
        so the UI / CitationExtractor can still process them downstream.
        Excess whitespace is normalised.
        """
        if not isinstance(response_text, str):
            # LLMClient always returns str from invoke(); guard defensively.
            response_text = str(response_text)
        return re.sub(r' {2,}', ' ', response_text).strip()

    # ── Extended interface (used by ResponseGenerator / CitationExtractor) ────

    @staticmethod
    def parse_structured(response_text: str) -> Dict[str, Any]:
        """Return content + extracted citation metadata as a dict.

        Returns
        -------
        {
            "content":      str   — response with [SOURCE_X] markers intact,
            "clean_content":str   — response with markers stripped (for display),
            "citations":    list  — [{"source_index": int, "source_id": str,
                                      "confidence": float}],
            "has_citations":bool
        }
        """
        if not isinstance(response_text, str):
            response_text = str(response_text)

        raw_indices = ResponseParser._CITATION_RE.findall(response_text)
        citations: List[Dict[str, Any]] = [
            {
                "source_index": int(idx),
                "source_id": f"source_{idx}",
                "confidence": 1.0,
            }
            for idx in sorted(set(raw_indices), key=int)
        ]

        clean = ResponseParser._CITATION_RE.sub('', response_text)
        clean = re.sub(r' {2,}', ' ', clean).strip()

        return {
            "content": response_text.strip(),
            "clean_content": clean,
            "citations": citations,
            "has_citations": bool(citations),
        }

    @staticmethod
    def validate_grounding(response: str, documents: List[Any]) -> bool:
        """Basic grounding check: response overlaps sufficiently with sources.

        Accepts both LangChain Documents (with .page_content) and plain dicts.
        Returns True if >= 30 % of response tokens appear in the combined
        context, False if the response looks potentially hallucinated.
        """
        context_text = " ".join(
            doc.page_content if hasattr(doc, "page_content") else doc.get("content", "")
            for doc in documents
        )
        context_words = set(context_text.lower().split())
        response_words = set(response.lower().split())
        if not response_words:
            return False
        overlap = len(context_words & response_words)
        return (overlap / len(response_words)) > 0.30
