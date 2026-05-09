"""
document_profiler.py

Pre-ingestion analytics layer for PDF content.

Runs BEFORE chunking to give users a data-driven view of:
  - How many tokens each chunking strategy would produce
  - Which embedding models are compatible with each strategy
  - A top recommendation based on p95 token counts + model fit

Usage:
    from src.ingestion.preprocessing.document_profiler import DocumentProfiler

    profiler = DocumentProfiler()
    profile  = profiler.profile(text, source_type="pdf", file_path="doc.pdf")
    summary  = profiler.get_ui_summary(profile)   # ready for Streamlit
"""

import re
import math
from typing import Dict, Any, List, Optional

import numpy as np

# nltk sentence tokeniser — downloaded lazily on first use
try:
    import nltk
    _NLTK_AVAILABLE = True
except ImportError:
    _NLTK_AVAILABLE = False

# PyMuPDF for font-size based chapter detection
try:
    import fitz  # pymupdf
    _FITZ_AVAILABLE = True
except ImportError:
    _FITZ_AVAILABLE = False


# ---------------------------------------------------------------------------
# Embedding model registry
# ---------------------------------------------------------------------------

EMBEDDING_MODEL_REGISTRY: List[Dict[str, Any]] = [
    {
        "name": "all-MiniLM-L2-v2",
        "max_tokens": 256,
        "dimensions": 384,
        "cost_tier": "free",
        "note": "Fastest; best for very short chunks (sentences)",
    },
    {
        "name": "all-MiniLM-L6-v2",
        "max_tokens": 512,
        "dimensions": 384,
        "cost_tier": "free",
        "note": "Good quality/speed balance; default choice",
    },
    {
        "name": "all-mpnet-base-v2",
        "max_tokens": 514,
        "dimensions": 768,
        "cost_tier": "free",
        "note": "Best open-source SBERT quality; slower",
    },
    {
        "name": "text-embedding-3-small",
        "max_tokens": 8191,
        "dimensions": 1536,
        "cost_tier": "paid",
        "note": "OpenAI API; large context, handles pages/chapters",
    },
    {
        "name": "text-embedding-3-large",
        "max_tokens": 8191,
        "dimensions": 3072,
        "cost_tier": "paid",
        "note": "OpenAI API; highest quality, largest context",
    },
]

# Fit labels
_FIT_PERFECT   = "✅ Perfect"
_FIT_GOOD      = "✅ Good"
_FIT_RISKY     = "⚠️  Risky"
_FIT_TOO_LARGE = "❌ Too Large"
_FIT_OVERKILL  = "💡 Overkill"


class DocumentProfiler:
    """
    Analyses raw text (extracted from a PDF or any source) across four
    chunking granularities and produces a fit-matrix + top recommendation.

    Parameters
    ----------
    chunk_overlap : int
        Token overlap that will be applied during chunking.
        Used to compute *effective* token count per chunk (avg + overlap).
    precise_tokenizer : str, optional
        HuggingFace model ID to load for precise token counting.
        If None (default) the fast heuristic (words × 1.3) is used.
    """

    WORDS_TO_TOKENS = 1.3  # heuristic: 1 word ≈ 1.3 tokens

    def __init__(
        self,
        chunk_overlap: int = 50,
        precise_tokenizer: Optional[str] = None,
    ):
        self.chunk_overlap = chunk_overlap
        self._tokenizer = None
        if precise_tokenizer:
            try:
                from transformers import AutoTokenizer
                self._tokenizer = AutoTokenizer.from_pretrained(precise_tokenizer)
            except Exception:
                pass  # fall back to heuristic silently

    # -----------------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------------

    def profile(
        self,
        text: str,
        source_type: str = "pdf",
        file_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Profile *text* and return a complete analytics dict.

        Returns
        -------
        {
            source_type,
            total_words,
            total_tokens_estimated,
            strategies: {
                sentence:  {count, avg, min, max, p95, variance, total},
                paragraph: {...},
                page:      {...},
                chapter:   {...},
            },
            fit_matrix: [
                {
                    strategy,
                    avg_tokens,
                    p95_tokens,
                    effective_p95,        # p95 + overlap
                    models: [
                        {name, max_tokens, dimensions, fit, fit_label},
                        ...
                    ]
                },
                ...
            ],
            recommendation: {
                strategy, model, reason, avg_tokens, p95_tokens
            },
            warnings: [str, ...]
        }
        """
        if not text or not text.strip():
            return self._empty_profile(source_type)

        words       = text.split()
        total_words = len(words)
        total_tok   = self._estimate_tokens(text)

        # --- split into granularities ---
        sentences  = self._split_sentences(text)
        paragraphs = self._split_paragraphs(text)
        pages      = self._split_pages(text)
        chapters   = self._split_chapters(text, file_path)

        strategies: Dict[str, Dict[str, Any]] = {}
        for name, segments in [
            ("sentence",  sentences),
            ("paragraph", paragraphs),
            ("page",      pages),
            ("chapter",   chapters),
        ]:
            strategies[name] = self._compute_stats(segments)

        fit_matrix  = self._build_fit_matrix(strategies)
        recommendation = self._pick_recommendation(fit_matrix)
        warnings    = self._build_warnings(strategies, fit_matrix)

        return {
            "source_type":             source_type,
            "total_words":             total_words,
            "total_tokens_estimated":  total_tok,
            "strategies":              strategies,
            "fit_matrix":              fit_matrix,
            "recommendation":          recommendation,
            "warnings":                warnings,
        }

    def get_ui_summary(self, profile: Dict[str, Any]) -> Dict[str, Any]:
        """
        Returns a simplified, Streamlit-ready summary.

        Suitable for st.expander("📊 Document Profile") rendering.
        """
        if not profile or not profile.get("fit_matrix"):
            return {"available": False}

        rec = profile["recommendation"]
        rows = []
        for entry in profile["fit_matrix"]:
            row = {
                "Strategy":   entry["strategy"].capitalize(),
                "Avg Tokens": entry["avg_tokens"],
                "P95 Tokens": entry["p95_tokens"],
                "Eff. P95":   entry["effective_p95"],
                "Chunks":     profile["strategies"][entry["strategy"]]["count"],
            }
            for m in entry["models"]:
                row[m["name"]] = m["fit_label"]
            rows.append(row)

        return {
            "available":         True,
            "total_words":       profile["total_words"],
            "total_tokens":      profile["total_tokens_estimated"],
            "fit_table":         rows,
            "recommendation":    rec,
            "warnings":          profile["warnings"],
        }

    # -----------------------------------------------------------------------
    # Splitting helpers
    # -----------------------------------------------------------------------

    def _split_sentences(self, text: str) -> List[str]:
        if _NLTK_AVAILABLE:
            try:
                nltk.data.find("tokenizers/punkt")
            except LookupError:
                nltk.download("punkt", quiet=True)
            try:
                nltk.data.find("tokenizers/punkt_tab")
            except LookupError:
                try:
                    nltk.download("punkt_tab", quiet=True)
                except Exception:
                    pass
            try:
                sents = nltk.sent_tokenize(text)
                return [s.strip() for s in sents if s.strip()]
            except Exception:
                pass
        # Fallback: naive sentence split on . ! ?
        sents = re.split(r"(?<=[.!?])\s+", text)
        return [s.strip() for s in sents if s.strip()]

    def _split_paragraphs(self, text: str) -> List[str]:
        """Split on one or more blank lines."""
        paras = re.split(r"\n{2,}", text)
        return [p.strip() for p in paras if p.strip()]

    def _split_pages(self, text: str) -> List[str]:
        """Split on [Page N] markers inserted by PDFPipeline."""
        if "[Page" in text:
            pages = re.split(r"\[Page\s*\d+\]", text)
            return [p.strip() for p in pages if p.strip()]
        # Fallback: estimate ~500-word pages
        words = text.split()
        page_size = 500
        pages = [
            " ".join(words[i : i + page_size])
            for i in range(0, len(words), page_size)
        ]
        return [p for p in pages if p.strip()]

    def _split_chapters(self, text: str, file_path: Optional[str] = None) -> List[str]:
        """
        Detect chapter boundaries.
        Priority:
          1. PyMuPDF font-size heuristic (if file_path given + fitz available)
          2. Regex: CHAPTER N / Chapter N / ## Heading / ALL-CAPS line
        """
        # --- try PyMuPDF font-size heuristic ---
        if file_path and _FITZ_AVAILABLE:
            try:
                chapters = self._chapters_via_pymupdf(file_path, text)
                if chapters and len(chapters) > 1:
                    return chapters
            except Exception:
                pass

        # --- regex fallback ---
        pattern = re.compile(
            r"(?:^(?:CHAPTER|Chapter|Ch\.)\s+[\dIVXivx]+|^#{1,2}\s+\S|^[A-Z][A-Z\s]{8,}$)",
            re.MULTILINE,
        )
        splits = pattern.split(text)
        chapters = [s.strip() for s in splits if s.strip()]
        return chapters if len(chapters) > 1 else [text]

    def _chapters_via_pymupdf(self, file_path: str, fallback_text: str) -> List[str]:
        """Use PyMuPDF to find large-font headings as chapter boundaries."""
        doc = fitz.open(file_path)
        heading_pages: List[int] = []
        for page_num, page in enumerate(doc):
            blocks = page.get_text("dict")["blocks"]
            for block in blocks:
                for line in block.get("lines", []):
                    for span in line.get("spans", []):
                        # Font size > 14pt usually indicates a heading
                        if span.get("size", 0) > 14 and span.get("text", "").strip():
                            heading_pages.append(page_num)
                            break
        doc.close()

        if len(heading_pages) < 2:
            return []

        # Split fallback_text at detected heading pages
        # (approximate — we mark boundaries at [Page N] or proportionally)
        pages = self._split_pages(fallback_text)
        chapters: List[str] = []
        current: List[str] = []
        heading_set = set(heading_pages)
        for i, page_text in enumerate(pages):
            if i in heading_set and current:
                chapters.append("\n".join(current))
                current = [page_text]
            else:
                current.append(page_text)
        if current:
            chapters.append("\n".join(current))
        return chapters

    # -----------------------------------------------------------------------
    # Stats
    # -----------------------------------------------------------------------

    def _estimate_tokens(self, text: str) -> int:
        if self._tokenizer:
            try:
                return len(self._tokenizer.encode(text, add_special_tokens=False))
            except Exception:
                pass
        return int(len(text.split()) * self.WORDS_TO_TOKENS)

    def _estimate_tokens_list(self, segments: List[str]) -> List[int]:
        return [self._estimate_tokens(s) for s in segments]

    def _compute_stats(self, segments: List[str]) -> Dict[str, Any]:
        if not segments:
            return {"count": 0, "avg": 0, "min": 0, "max": 0,
                    "p95": 0, "variance": 0, "total": 0}
        token_counts = self._estimate_tokens_list(segments)
        arr = np.array(token_counts, dtype=float)
        return {
            "count":    len(segments),
            "avg":      int(np.mean(arr)),
            "min":      int(np.min(arr)),
            "max":      int(np.max(arr)),
            "p95":      int(np.percentile(arr, 95)),
            "variance": int(np.var(arr)),
            "total":    int(np.sum(arr)),
        }

    # -----------------------------------------------------------------------
    # Fit matrix
    # -----------------------------------------------------------------------

    def _fit_label(self, p95_effective: int, model_max: int, avg: int) -> str:
        ratio = p95_effective / model_max if model_max > 0 else 999
        if p95_effective > model_max:
            return _FIT_TOO_LARGE
        if ratio < 0.30:
            return _FIT_OVERKILL
        if ratio < 0.70:
            return _FIT_PERFECT
        if ratio < 0.90:
            return _FIT_GOOD
        return _FIT_RISKY  # 90-100% of limit

    def _build_fit_matrix(
        self, strategies: Dict[str, Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        matrix = []
        for strategy_name, stats in strategies.items():
            avg       = stats["avg"]
            p95       = stats["p95"]
            eff_p95   = p95 + self.chunk_overlap  # effective after overlap

            model_fits = []
            for model in EMBEDDING_MODEL_REGISTRY:
                label = self._fit_label(eff_p95, model["max_tokens"], avg)
                model_fits.append({
                    "name":       model["name"],
                    "max_tokens": model["max_tokens"],
                    "dimensions": model["dimensions"],
                    "cost_tier":  model["cost_tier"],
                    "fit_label":  label,
                    "note":       model["note"],
                })

            matrix.append({
                "strategy":    strategy_name,
                "count":       stats["count"],
                "avg_tokens":  avg,
                "p95_tokens":  p95,
                "effective_p95": eff_p95,
                "models":      model_fits,
            })
        return matrix

    # -----------------------------------------------------------------------
    # Recommendation
    # -----------------------------------------------------------------------

    def _pick_recommendation(self, fit_matrix: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Pick the best (strategy, model) pair:
          - Prefers free-tier models
          - Among compatible entries, picks the strategy with the
            most chunks (highest granularity = best retrieval precision)
          - Falls back to paid models if no free model fits
        """
        # Score: Perfect=4, Good=3, Risky=2, Overkill=1, TooLarge=0
        score_map = {
            _FIT_PERFECT:   4,
            _FIT_GOOD:      3,
            _FIT_RISKY:     2,
            _FIT_OVERKILL:  1,
            _FIT_TOO_LARGE: 0,
        }

        best_score   = -1
        best_entry   = None
        best_model   = None

        for entry in fit_matrix:
            if entry["count"] == 0:
                continue
            for model in entry["models"]:
                s = score_map.get(model["fit_label"], 0)
                if s >= 3:  # Perfect or Good
                    # Prefer free tier; break ties by chunk count
                    tier_bonus = 1 if model["cost_tier"] == "free" else 0
                    total = s * 10 + tier_bonus * 5 + min(entry["count"], 999)
                    if total > best_score:
                        best_score = total
                        best_entry = entry
                        best_model = model

        if best_entry is None:
            # fallback: pick smallest paid model
            for entry in fit_matrix:
                if entry["count"] == 0:
                    continue
                for model in entry["models"]:
                    if score_map.get(model["fit_label"], 0) > 0:
                        best_entry = entry
                        best_model = model
                        break
                if best_entry:
                    break

        if best_entry is None or best_model is None:
            return {"strategy": "recursive", "model": "all-MiniLM-L6-v2",
                    "reason": "Could not profile document — using safe defaults",
                    "avg_tokens": 0, "p95_tokens": 0}

        return {
            "strategy":   best_entry["strategy"],
            "model":      best_model["name"],
            "dimensions": best_model["dimensions"],
            "reason": (
                f"{best_entry['strategy'].capitalize()} chunking gives "
                f"~{best_entry['avg_tokens']} avg tokens / chunk "
                f"(p95 = {best_entry['p95_tokens']}). "
                f"Fits within {best_model['name']} limit of "
                f"{best_model['max_tokens']} tokens. "
                f"{best_model['note']}."
            ),
            "avg_tokens": best_entry["avg_tokens"],
            "p95_tokens": best_entry["p95_tokens"],
        }

    # -----------------------------------------------------------------------
    # Warnings
    # -----------------------------------------------------------------------

    def _build_warnings(self, strategies, fit_matrix) -> List[str]:
        warnings = []
        for entry in fit_matrix:
            if entry["count"] == 0:
                warnings.append(
                    f"⚠️  {entry['strategy'].capitalize()} strategy produced 0 segments — "
                    f"document may lack the required structure (e.g., no [Page N] markers)."
                )
            for m in entry["models"]:
                if m["fit_label"] == _FIT_RISKY:
                    warnings.append(
                        f"⚠️  {entry['strategy'].capitalize()} + {m['name']}: "
                        f"p95 effective tokens ({entry['effective_p95']}) is "
                        f"close to the model limit ({m['max_tokens']}). "
                        f"Some chunks may be silently truncated."
                    )
        return warnings

    # -----------------------------------------------------------------------
    # Fallback
    # -----------------------------------------------------------------------

    def _empty_profile(self, source_type: str) -> Dict[str, Any]:
        return {
            "source_type": source_type,
            "total_words": 0,
            "total_tokens_estimated": 0,
            "strategies": {},
            "fit_matrix": [],
            "recommendation": {
                "strategy": "recursive",
                "model": "all-MiniLM-L6-v2",
                "reason": "Empty document — using safe defaults",
                "avg_tokens": 0,
                "p95_tokens": 0,
            },
            "warnings": ["Document appears to be empty."],
        }
