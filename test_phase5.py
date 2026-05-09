import sys
import os
sys.path.append(os.path.abspath('.'))

from src.ingestion.preprocessing.adaptive_preprocessor import AdaptivePreprocessor

def test_preprocessing():
    print("Testing AdaptivePreprocessor...")
    preprocessor = AdaptivePreprocessor()
    
    # PDF Test
    pdf_content = "[Page 1]\nThis is some text with hy-\nphenation. [Page 2]\nMore text here."
    res = preprocessor.process(pdf_content, "pdf")
    print("  PDF cleaning: OK")
    assert "hyphenation" in res['cleaned_content']
    assert "## Page 1" in res['cleaned_content']
    
    # Website Test
    web_content = "Main article content. All rights reserved. Subscribe to our newsletter."
    res = preprocessor.process(web_content, "website")
    print("  Website cleaning: OK")
    assert "All rights reserved" not in res['cleaned_content']
    
    # YouTube Test
    yt_content = "[00:15:30] Hello everyone. ♪ music playing ♪ [00:16:00] Next topic."
    res = preprocessor.process(yt_content, "youtube")
    print("  YouTube cleaning: OK")
    assert "music playing" not in res['cleaned_content']
    
    print("SUCCESS: Preprocessing verified")

if __name__ == "__main__":
    try:
        test_preprocessing()
    except Exception as e:
        print(f"ERROR: {e}")
        sys.exit(1)
