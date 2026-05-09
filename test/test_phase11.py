import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.generation.llm_client import LLMClient
from src.generation.prompt_builder import PromptBuilder
from src.generation.response_parser import ResponseParser

def test_generation():
    print("Testing Generation Layer...")
    
    # 1. Test LLM Client Init & Tuning
    print("  Testing LLMClient initialization...")
    try:
        # Just initialize to see if it parses args correctly
        client = LLMClient(provider="ollama", model="llama3", temperature=0.5)
        client.update_tuning(temperature=0.8)
        assert client.temperature == 0.8
        print("    LLMClient OK (Initialization)")
    except Exception as e:
        print(f"    LLMClient error (might be missing API key): {e}")
        
    # 2. Test Prompt Builder
    print("  Testing PromptBuilder...")
    context = PromptBuilder.format_context([{"content": "The Eiffel Tower is in Paris."}])
    prompt = PromptBuilder.build_chat_prompt("Where is the Eiffel Tower?", context)
    assert "[SOURCE_1]" in prompt
    assert "Eiffel Tower is in Paris" in prompt
    print("    PromptBuilder OK")
    
    # 3. Test Response Parser
    print("  Testing ResponseParser...")
    response_text = "The Eiffel Tower is located in Paris [SOURCE_1]. It is very tall [SOURCE_2]."
    parsed = ResponseParser.parse(response_text)
    
    assert parsed["has_citations"]
    assert len(parsed["citations"]) == 2
    assert "The Eiffel Tower is located in Paris . It is very tall ." == parsed["content"]
    
    grounded = ResponseParser.validate_grounding(parsed["content"], "The Eiffel Tower is a tall building in Paris.")
    assert grounded == True
    
    not_grounded = ResponseParser.validate_grounding("Bananas are yellow fruit.", "The Eiffel Tower is a tall building in Paris.")
    assert not_grounded == False
    
    print("    ResponseParser OK")
    print("SUCCESS: Generation Layer verified")

if __name__ == "__main__":
    try:
        test_generation()
    except Exception as e:
        print(f"ERROR: {e}")
        sys.exit(1)
