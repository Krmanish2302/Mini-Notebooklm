import base64
import requests
from typing import Dict, Any

class ImagePipeline:
    """Generates captions using LLaVA via Ollama."""
    
    OLLAMA_URL = "http://localhost:11434/api/generate"
    DEFAULT_MODEL = "llava"
    
    @staticmethod
    def process(file_path: str, source_id: str) -> Dict[str, Any]:
        with open(file_path, "rb") as f:
            image_data = base64.b64encode(f.read()).decode()
        
        prompt = "Describe this image in detail. What is shown? What text is visible?"
        
        try:
            response = requests.post(
                ImagePipeline.OLLAMA_URL,
                json={
                    "model": ImagePipeline.DEFAULT_MODEL,
                    "prompt": prompt,
                    "images": [image_data],
                    "stream": False
                },
                timeout=120
            )
            response.raise_for_status()
            caption = response.json().get("response", "")
            
            return {
                "content": caption,
                "metadata": {
                    "source_file": file_path,
                    "model_used": ImagePipeline.DEFAULT_MODEL,
                    "source_id": source_id
                },
                "modality": "image_caption"
            }
        except Exception as e:
            # Fallback: return basic metadata if Ollama not available
            return {
                "content": f"[Image: {file_path}]",
                "metadata": {"error": str(e), "source_id": source_id},
                "modality": "image_caption"
            }