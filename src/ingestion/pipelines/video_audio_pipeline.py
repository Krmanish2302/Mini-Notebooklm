"""
video_audio_pipeline.py  —  REMOVED

Video/audio ingestion via Whisper has been removed.
Use YouTubePipeline for YouTube URLs (transcript API, no download needed).
For local audio/video files, add a future pipeline here.
"""

class VideoAudioPipeline:
    """Placeholder — video/audio pipeline removed. Not used in production."""

    @staticmethod
    def process(*args, **kwargs):
        raise NotImplementedError(
            "VideoAudioPipeline has been removed. "
            "Use YouTubePipeline for YouTube URLs, or add a local audio pipeline here."
        )
