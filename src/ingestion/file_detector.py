"""
file_detector.py — Detect the source type of a file path or URL.

Used by MasterPipeline.ingest() to resolve source_type when the caller
does not provide it explicitly.

Supported source_types (matches pipeline registry):
    pdf | image | video | audio | youtube | website | csv | text
"""
from __future__ import annotations

import mimetypes
import os
import re
from typing import Optional
from urllib.parse import urlparse


# ---------------------------------------------------------------------------
#  Extension → source_type map
# ---------------------------------------------------------------------------
_EXT_MAP: dict[str, str] = {
    # Documents
    ".pdf":  "pdf",
    # Images
    ".jpg":  "image", ".jpeg": "image", ".png": "image",
    ".gif":  "image", ".webp": "image", ".bmp": "image",
    ".tiff": "image", ".tif":  "image", ".svg": "image",
    # Video
    ".mp4":  "video", ".mkv": "video", ".avi": "video",
    ".mov":  "video", ".wmv": "video", ".flv": "video",
    ".webm": "video",
    # Audio
    ".mp3":  "audio", ".wav": "audio", ".ogg": "audio",
    ".flac": "audio", ".m4a": "audio", ".aac": "audio",
    # Tabular
    ".csv":  "csv", ".tsv": "csv",
    # Text / Markup
    ".txt":  "text", ".md":  "text", ".rst": "text",
    ".json": "text", ".xml": "text", ".yaml": "text", ".yml": "text",
}

# MIME prefix → source_type (fallback when extension is ambiguous)
_MIME_PREFIX_MAP: dict[str, str] = {
    "image/":  "image",
    "video/":  "video",
    "audio/":  "audio",
    "text/":   "text",
    "application/pdf": "pdf",
}

# YouTube URL patterns
_YOUTUBE_PATTERNS: tuple[re.Pattern, ...] = (
    re.compile(r"(https?://)?(www\.)?youtube\.com/watch", re.I),
    re.compile(r"(https?://)?(www\.)?youtube\.com/shorts/", re.I),
    re.compile(r"(https?://)?youtu\.be/", re.I),
    re.compile(r"(https?://)?(www\.)?youtube\.com/embed/", re.I),
)


class FileDetector:
    """
    Detect source_type from a file path or URL.

    Usage::

        detector = FileDetector()
        result = detector.detect(file_path="report.pdf")
        # {"source_type": "pdf", "confidence": "extension", "raw": "report.pdf"}

        result = detector.detect(url="https://www.youtube.com/watch?v=abc123")
        # {"source_type": "youtube", "confidence": "pattern", "raw": "..."}
    """

    # ------------------------------------------------------------------
    #  Public API
    # ------------------------------------------------------------------

    def detect(
        self,
        file_path: Optional[str] = None,
        url: Optional[str] = None,
    ) -> dict:
        """
        Returns a dict with at minimum::

            {
                "source_type": str,        # one of the supported types above
                "confidence":  str,        # "extension" | "mime" | "pattern" | "fallback"
                "raw":         str,        # original path / url
            }

        Raises ValueError if neither file_path nor url is provided.
        """
        if url:
            return self._detect_url(url)
        if file_path:
            return self._detect_file(file_path)
        raise ValueError("FileDetector.detect() requires file_path or url.")

    # ------------------------------------------------------------------
    #  URL detection
    # ------------------------------------------------------------------

    def _detect_url(self, url: str) -> dict:
        # 1. YouTube patterns (before generic website check)
        for pattern in _YOUTUBE_PATTERNS:
            if pattern.search(url):
                return self._result("youtube", "pattern", url)

        # 2. Generic website
        parsed = urlparse(url)
        if parsed.scheme in ("http", "https"):
            # Could still be a direct file link — check extension in path
            ext = os.path.splitext(parsed.path)[1].lower()
            if ext and ext in _EXT_MAP:
                return self._result(_EXT_MAP[ext], "extension", url)
            return self._result("website", "pattern", url)

        return self._result("website", "fallback", url)

    # ------------------------------------------------------------------
    #  File path detection
    # ------------------------------------------------------------------

    def _detect_file(self, file_path: str) -> dict:
        ext = os.path.splitext(file_path)[1].lower()

        # 1. Known extension
        if ext in _EXT_MAP:
            return self._result(_EXT_MAP[ext], "extension", file_path)

        # 2. MIME type guess
        mime, _ = mimetypes.guess_type(file_path)
        if mime:
            # Exact mime match
            if mime in _MIME_PREFIX_MAP:
                return self._result(_MIME_PREFIX_MAP[mime], "mime", file_path)
            # Prefix match
            for prefix, stype in _MIME_PREFIX_MAP.items():
                if mime.startswith(prefix):
                    return self._result(stype, "mime", file_path)

        # 3. Fallback → treat as plain text
        return self._result("text", "fallback", file_path)

    # ------------------------------------------------------------------
    #  Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _result(source_type: str, confidence: str, raw: str) -> dict:
        return {
            "source_type": source_type,
            "confidence":  confidence,
            "raw":         raw,
        }
