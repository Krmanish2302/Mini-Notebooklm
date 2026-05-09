"""
youtube_pipeline.py

Fetches YouTube transcripts using LangChain's YoutubeLoader.
No Whisper, no audio download — fast and dependency-light.
"""
from typing import Dict, Any
from langchain_community.document_loaders import YoutubeLoader


class YouTubePipeline:
    """Fetches YouTube transcripts via the YouTube Transcript API."""

    @staticmethod
    def process(url: str, source_id: str) -> Dict[str, Any]:
        """
        Load transcript from a YouTube URL.

        Args:
            url: Full YouTube URL (e.g. https://www.youtube.com/watch?v=...)
            source_id: Unique identifier for this source.

        Returns:
            dict with keys: content, metadata, modality

        Raises:
            ValueError: If transcript cannot be fetched (disabled / unavailable).
        """
        try:
            loader = YoutubeLoader.from_youtube_url(
                url,
                add_video_info=True,   # fetches title, author, publish_date
                language=["en"],        # prefer English; falls back automatically
                translation="en",
            )
            docs = loader.load()
        except Exception as exc:
            raise ValueError(
                f"Could not fetch transcript for {url}. "
                f"Make sure the video has captions enabled. Error: {exc}"
            ) from exc

        if not docs:
            raise ValueError(
                f"No transcript returned for {url}. "
                "The video may have disabled transcripts."
            )

        # Merge all transcript chunks into one string
        content = "\n".join(d.page_content for d in docs)

        # Pull video metadata from the first doc (YoutubeLoader attaches it there)
        first_meta = docs[0].metadata
        return {
            "content": content,
            "metadata": {
                "title": first_meta.get("title", ""),
                "author": first_meta.get("author", ""),
                "publish_date": first_meta.get("publish_date", ""),
                "url": url,
                "word_count": len(content.split()),
                "source_id": source_id,
            },
            "modality": "transcript",
        }
