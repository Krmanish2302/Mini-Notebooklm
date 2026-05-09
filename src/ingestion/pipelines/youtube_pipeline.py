import yt_dlp
from faster_whisper import WhisperModel
from typing import Dict, Any
import os


class YouTubePipeline:
    """Downloads audio and transcribes YouTube videos."""

    WHISPER = WhisperModel("base", device="cpu", compute_type="int8")

    @staticmethod
    def process(url: str, source_id: str) -> Dict[str, Any]:
        # Download audio
        ydl_opts = {
            "format": "bestaudio/best",
            "postprocessors": [
                {
                    "key": "FFmpegExtractAudio",
                    "preferredcodec": "wav",
                    "preferredquality": "192",
                }
            ],
            "outtmpl": "temp_%(id)s.%(ext)s",
        }

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            audio_path = f"temp_{info['id']}.wav"

        # Transcribe
        segments, _ = YouTubePipeline.WHISPER.transcribe(audio_path, beam_size=5)
        transcript = "\n".join(
            [
                f"[{int(s.start // 60):02d}:{int(s.start % 60):02d}] {s.text}"
                for s in segments
            ]
        )

        # Cleanup
        if os.path.exists(audio_path):
            os.remove(audio_path)

        return {
            "content": transcript,
            "metadata": {
                "title": info.get("title", ""),
                "uploader": info.get("uploader", ""),
                "duration": info.get("duration", 0),
                "source_id": source_id,
            },
            "modality": "transcript",
        }
