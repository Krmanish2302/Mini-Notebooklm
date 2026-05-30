"""
file_detector.py — detect source type from file path or URL.
Kept for backward-compat with master_pipeline.py.
"""
from __future__ import annotations
import os


def detect_source_type(file_path: str) -> str:
    """Return 'pdf' | 'csv' | 'text' | 'website' | 'youtube'."""
    if file_path.startswith(("http://", "https://")):
        if "youtube.com" in file_path or "youtu.be" in file_path:
            return "youtube"
        return "website"
    ext = os.path.splitext(file_path)[-1].lower()
    return {".pdf": "pdf", ".csv": "csv", ".txt": "text", ".md": "text", ".html": "website"}.get(ext, "text")