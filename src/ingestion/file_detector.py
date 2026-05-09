import magic
import os
from pathlib import Path
from typing import Dict, Any, Literal
from urllib.parse import urlparse
import re


class FileDetector:
    """Detects file type and returns routing information."""

    SUPPORTED_TYPES = {
        "pdf": ["application/pdf"],
        "image": ["image/jpeg", "image/png", "image/webp", "image/gif"],
        "video": ["video/mp4", "video/avi", "video/mov"],
        "audio": ["audio/mp3", "audio/wav", "audio/m4a"],
        "csv": ["text/csv", "application/vnd.ms-excel"],
        "text": ["text/plain", "text/markdown"],
    }

    @staticmethod
    def detect(file_path: str = None, url: str = None) -> Dict[str, Any]:
        """
        Detect source type from file path or URL.
        Returns: {"source_type": str, "mime_type": str, "handler": str}
        """
        if url:
            return FileDetector._detect_from_url(url)
        elif file_path:
            return FileDetector._detect_from_file(file_path)
        else:
            raise ValueError("Either file_path or url must be provided")

    @staticmethod
    def _detect_from_file(file_path: str) -> Dict[str, Any]:
        mime = magic.from_file(file_path, mime=True)
        ext = Path(file_path).suffix.lower()

        for source_type, mimes in FileDetector.SUPPORTED_TYPES.items():
            if mime in mimes:
                return {
                    "source_type": source_type,
                    "mime_type": mime,
                    "file_path": file_path,
                    "handler": f"{source_type}_pipeline",
                }

        # Fallback to extension
        if ext in [".pdf"]:
            return {
                "source_type": "pdf",
                "mime_type": "application/pdf",
                "file_path": file_path,
                "handler": "pdf_pipeline",
            }
        elif ext in [".jpg", ".jpeg", ".png", ".webp"]:
            return {
                "source_type": "image",
                "mime_type": f"image/{ext[1:]}",
                "file_path": file_path,
                "handler": "image_pipeline",
            }
        elif ext in [".mp4", ".avi", ".mov"]:
            return {
                "source_type": "video",
                "mime_type": f"video/{ext[1:]}",
                "file_path": file_path,
                "handler": "video_pipeline",
            }
        elif ext in [".mp3", ".wav", ".m4a"]:
            return {
                "source_type": "audio",
                "mime_type": f"audio/{ext[1:]}",
                "file_path": file_path,
                "handler": "audio_pipeline",
            }
        elif ext in [".csv", ".xlsx"]:
            return {
                "source_type": "csv",
                "mime_type": "text/csv",
                "file_path": file_path,
                "handler": "csv_pipeline",
            }
        elif ext in [".txt", ".md"]:
            return {
                "source_type": "text",
                "mime_type": "text/plain",
                "file_path": file_path,
                "handler": "text_pipeline",
            }

        raise IngestionError(f"Unsupported file type: {mime} / {ext}")

    @staticmethod
    def _detect_from_url(url: str) -> Dict[str, Any]:
        parsed = urlparse(url)
        domain = parsed.netloc.lower()

        # YouTube detection
        if "youtube.com" in domain or "youtu.be" in domain:
            return {
                "source_type": "youtube",
                "mime_type": "text/html",
                "url": url,
                "handler": "youtube_pipeline",
            }

        # Website detection
        return {
            "source_type": "website",
            "mime_type": "text/html",
            "url": url,
            "handler": "website_pipeline",
        }
