"""
file_detector.py

Thin wrapper kept for backward-compat with master_pipeline.py.
Detects source type from a file path or URL string.

The canonical detection logic now lives in nodes/loader_node.py.
"""
from __future__ import annotations

import os


def detect_source_type(file_path: str) -> str:
    """Return 'pdf' | 'csv' | 'text' | 'website' | 'youtube'."""
    ext = os.path.splitext(file_path)[-1].lower()
    mapping = {".pdf": "pdf", ".csv": "csv", ".txt": "text", ".md": "text", ".html": "website"}
    if file_path.startswith(("http://", "https://")):
        if "youtube.com" in file_path or "youtu.be" in file_path:
            return "youtube"
        return "website"
    return mapping.get(ext, "text")
