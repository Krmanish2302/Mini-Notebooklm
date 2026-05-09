import sys
import os
sys.path.append(os.path.abspath('.'))

from src.agents.web_search_agent import WebSearchAgent

def test_web_search():
    print("Testing Web Search Agent...")
    
    agent = WebSearchAgent()
    
    # 1. Test formatting (mock search to bypass empty parser)
    print("  Testing result formatting...")
    original_search = agent.search
    agent.search = lambda q: [
        {"title": "Mock Result 1", "url": "https://example.com/1", "snippet": "This is mock 1"},
        {"title": "Mock Result 2", "url": "https://example.com/2", "snippet": "This is mock 2"}
    ]
    
    formatted = agent.search_and_format("test query")
    assert len(formatted) == 2
    assert formatted[0]["title"] == "Mock Result 1"
    assert formatted[0]["source_type"] == "website"
    assert "id" in formatted[0]
    assert formatted[0]["selected"] == False
    print("    Formatting OK")
    
    agent.search = original_search # restore
    
    # 2. Test Content Fetching
    print("  Testing Trafilatura extraction...")
    content = agent.fetch_content("https://example.com")
    print(f"Extracted length: {len(content)}")
    print(f"Content snippet: {content[:200]}")
    assert len(content) > 0, "No content extracted"
    assert "domain" in content.lower() or "example" in content.lower(), "Unexpected content from example.com"
    print("    Extraction OK")
    
    print("SUCCESS: Web Search Agent verified")

if __name__ == "__main__":
    try:
        test_web_search()
    except Exception as e:
        print(f"ERROR: {e}")
        sys.exit(1)
