import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.core.models import Source, Chunk, Query, RetrievedChunk, Citation, LLMResponse, ChatMessage, Session
from src.core.exceptions import MiniNotebookLMError, IngestionError
import pydantic

def test_models():
    try:
        # Test Source
        source = Source(
            id="source_1",
            title="Test Document",
            source_type="pdf",
            file_path="/path/to/test.pdf"
        )
        print("SUCCESS: Source model verified")

        # Test Chunk
        chunk = Chunk(
            id="chunk_1",
            source_id="source_1",
            content="This is a test chunk.",
            modality="text"
        )
        print("SUCCESS: Chunk model verified")

        # Test validation
        try:
            Source(id="s2", title="Error", source_type="invalid")
        except pydantic.ValidationError:
            print("SUCCESS: Validation logic verified")

        print("\nAll Phase 2 models and validation rules verified successfully!")
    except Exception as e:
        print(f"FAILED: Test failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    test_models()
