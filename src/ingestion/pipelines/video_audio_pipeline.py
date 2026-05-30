"""
video_audio_pipeline.py

Transcribes audio/video files using OpenAI Whisper via LangChain.
Requires: pip install openai-whisper
"""
from __future__ import annotations
import logging
from typing import List
from langchain_core.documents import Document

logger = logging.getLogger(__name__)


class VideoAudioPipeline:
    @staticmethod
    def process(file_path: str, source_id: str) -> List[Document]:
        try:
            import whisper
        except ImportError:
            logger.error("[VideoAudioPipeline] whisper not installed. pip install openai-whisper")
            return []

        model  = whisper.load_model("base")
        result = model.transcribe(file_path)
        text   = result.get("text", "").strip()

        if not text:
            logger.warning("[VideoAudioPipeline] No transcript for '%s'", file_path)
            return []

        return [Document(
            page_content=text,
            metadata={"source_id": source_id, "source_type": "audio", "source": file_path},
        )]