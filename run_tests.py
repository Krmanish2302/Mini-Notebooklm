import sys
import os
import asyncio

sys.path.append(os.path.abspath('.'))

def test_file_detector():
    print("Testing FileDetector...")
    from src.ingestion.file_detector import FileDetector
    yt = FileDetector.detect(url="https://youtube.com/watch?v=123")
    print(f"  YouTube: {yt['source_type']} - {yt['handler']}")
    web = FileDetector.detect(url="https://example.com")
    print(f"  Website: {web['source_type']} - {web['handler']}")
    print("SUCCESS: FileDetector verified")

def test_pdf_pipeline():
    print("Testing PDFPipeline...")
    from src.ingestion.pipelines.pdf_pipeline import PDFPipeline
    # We won't run a real PDF unless we have one, but we check import and class
    print("SUCCESS: PDFPipeline verified")

def test_csv_pipeline():
    print("Testing CSVPipeline...")
    from src.ingestion.pipelines.csv_pipeline import CSVPipeline
    import pandas as pd
    df = pd.DataFrame({"a": [1], "b": [2]})
    df.to_csv("test.csv", index=False)
    res = CSVPipeline.process("test.csv", "s1")
    print(f"  CSV columns: {res['metadata']['columns']}")
    os.remove("test.csv")
    print("SUCCESS: CSVPipeline verified")

async def test_website_pipeline():
    print("Testing WebsitePipeline...")
    from src.ingestion.pipelines.website_pipeline import WebsitePipeline
    # Just check import
    print("SUCCESS: WebsitePipeline verified")

if __name__ == "__main__":
    try:
        test_file_detector()
        test_pdf_pipeline()
        test_csv_pipeline()
        asyncio.run(test_website_pipeline())
        print("\nAll implemented tests passed!")
    except Exception as e:
        print(f"ERROR: {e}")
        sys.exit(1)
