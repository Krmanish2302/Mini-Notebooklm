import subprocess
from faster_whisper import WhisperModel
from typing import Dict, Any
import os

class VideoAudioPipeline:
    """Transcribes video/audio using faster-whisper."""
    
    MODEL = WhisperModel("base", device="cpu", compute_type="int8")
    
    @staticmethod
    def process(file_path: str, source_id: str) -> Dict[str, Any]:
        # Extract audio if video
        audio_path = file_path
        if file_path.endswith(('.mp4', '.avi', '.mov')):
            audio_path = file_path.rsplit('.', 1)[0] + '.wav'
            subprocess.run([
                'ffmpeg', '-i', file_path, '-vn', '-acodec', 'pcm_s16le', 
                '-ar', '16000', '-ac', '1', audio_path
            ], check=True, capture_output=True)
        
        segments, info = VideoAudioPipeline.MODEL.transcribe(audio_path, beam_size=5)
        
        transcript_parts = []
        for segment in segments:
            timestamp = f"[{int(segment.start // 60):02d}:{int(segment.start % 60):02d}]"
            transcript_parts.append(f"{timestamp} {segment.text}")
        
        # Cleanup temp audio
        if audio_path != file_path and os.path.exists(audio_path):
            os.remove(audio_path)
        
        return {
            "content": "\n".join(transcript_parts),
            "metadata": {
                "language": info.language,
                "duration": info.duration,
                "source_id": source_id
            },
            "modality": "transcript"
        }