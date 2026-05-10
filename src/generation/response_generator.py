"""
response_generator.py  —  Structured response assembly.

Takes a raw LLM output string and enriches it into a full response dict:
  {
    "answer"         : str,         # cleaned LLM answer
    "citations"      : list[dict],  # [{label, source_id, content_snippet}]
    "follow_ups"     : list[str],   # 2-3 suggested follow-up questions
    "sources_used"   : list[str],   # unique [S1], [S2]… labels that appear in answer
    "chunks_used"    : list[dict],  # context chunks that were cited
    "tokens_estimate": int,         # rough token count
  }

Also handles:
  - follow-up question extraction (if the LLM produced them in a known format)
  - citation label normalisation  (e.g. [s1] → [S1])
  - stripping LLM artefacts ("A:", "ANSWER:", stray markdown fences)
"""
from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Approximate chars per token
_CHARS_PER_TOKEN = 3.5

# Pattern for citation labels like [S1], [s2], [S12]
_CITE_PATTERN = re.compile(r"\[([Ss]\d{1,2})\]")

# Patterns the LLM sometimes emits at the start of its answer
_STRIP_PREFIXES = re.compile(
    r"^(?:A|ANSWER|DETAILED\s+ANSWER|EXPLAIN|RESPONSE)\s*:\s*",
    re.IGNORECASE,
)

# Follow-up questions block that some prompts elicit
_FOLLOWUP_BLOCK = re.compile(
    r"(?:follow[- ]?up|suggested|you\s+might\s+also\s+ask)[:\s\n]+(.*?)(?:\n\n|$)",
    re.IGNORECASE | re.DOTALL,
)


class ResponseGenerator:
    """
    Thin stateless utility — call assemble() after every LLM invocation.

    Parameters
    ----------
    context_chunks : list of dicts returned by ContextBuilder.build()
                     Must include 'citation_label' key added by ContextBuilder.
    """

    def __init__(self, context_chunks: Optional[List[Dict[str, Any]]] = None):
        self.context_chunks: List[Dict[str, Any]] = context_chunks or []

    def assemble(
        self,
        raw_llm_output: str,
        query: str = "",
        generate_follow_ups: bool = True,
    ) -> Dict[str, Any]:
        """
        Parse and enrich the raw LLM output.

        Returns
        -------
        dict with keys: answer, citations, follow_ups, sources_used,
                        chunks_used, tokens_estimate
        """
        cleaned   = self._clean(raw_llm_output)
        answer    = self._strip_follow_up_block(cleaned)
        follow_ups = self._extract_follow_ups(cleaned, query, generate_follow_ups)

        # Normalise citation labels
        answer    = _CITE_PATTERN.sub(lambda m: f"[{m.group(1).upper()}]", answer)

        # Which sources are actually cited in the answer?
        used_labels = list(dict.fromkeys(_CITE_PATTERN.findall(answer)))  # ordered unique

        # Build citation objects from context_chunks
        citations  = self._build_citations(used_labels)
        chunks_used = [
            c for c in self.context_chunks
            if c.get("citation_label", "") in used_labels
        ]

        return {
            "answer":          answer.strip(),
            "citations":       citations,
            "follow_ups":      follow_ups,
            "sources_used":    used_labels,
            "chunks_used":     chunks_used,
            "tokens_estimate": self._token_estimate(answer),
        }

    # ─────────────────────────────────────────────────────────────────────
    #  Private helpers
    # ─────────────────────────────────────────────────────────────────────

    @staticmethod
    def _clean(text: str) -> str:
        """Remove common LLM artefacts: markdown fences, leading answer tags."""
        # Strip markdown code fences
        text = re.sub(r"```[\s\S]*?```", "", text)
        # Strip leading answer prefix ("A:", "ANSWER:", etc.)
        text = _STRIP_PREFIXES.sub("", text.strip())
        return text.strip()

    @staticmethod
    def _strip_follow_up_block(text: str) -> str:
        """Remove follow-up question block from the end of the answer."""
        return _FOLLOWUP_BLOCK.sub("", text).strip()

    @staticmethod
    def _extract_follow_ups(
        text: str,
        query: str,
        enabled: bool,
    ) -> List[str]:
        """Extract follow-up questions if the LLM emitted them, else return []."""
        if not enabled:
            return []

        match = _FOLLOWUP_BLOCK.search(text)
        if not match:
            return []

        block = match.group(1)
        lines = re.findall(r"^\s*[-•*\d.)]+\s*(.+)$", block, re.MULTILINE)
        questions = [l.strip().rstrip(".") + "?" if not l.strip().endswith("?") else l.strip()
                     for l in lines if len(l.strip()) > 8]
        return questions[:3]

    def _build_citations(
        self, used_labels: List[str]
    ) -> List[Dict[str, Any]]:
        """Build citation objects for each used source label."""
        # Build a lookup: label -> chunk
        label_map: Dict[str, Dict] = {}
        for chunk in self.context_chunks:
            lbl = chunk.get("citation_label", "")
            if lbl and lbl not in label_map:
                label_map[lbl] = chunk

        citations = []
        for lbl in used_labels:
            chunk = label_map.get(lbl)
            if chunk:
                citations.append({
                    "label":           lbl,
                    "source_id":       chunk.get("source_id", ""),
                    "source_name":     chunk.get("source", chunk.get("source_name", "")),
                    "content_snippet": chunk.get("content", "")[:200],
                    "score":           chunk.get("rrf_score", 0.0),
                })
            else:
                citations.append({"label": lbl, "source_id": "", "source_name": "", "content_snippet": "", "score": 0.0})

        return citations

    @staticmethod
    def _token_estimate(text: str) -> int:
        return max(1, int(len(text) / _CHARS_PER_TOKEN))
