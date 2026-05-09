import sys
import os
import shutil
import csv
sys.path.append(os.path.abspath('.'))

from src.master_pipeline import MasterPipeline

class MockLLM:
    def invoke(self, prompt):
        # Dummy response with citation
        return "This is a mock response based on: Neural Networks. [SOURCE_1]"
    def stream(self, prompt):
        yield "This "
        yield "is a "
        yield "mock."

def test_master_pipeline():
    print("Testing Master Pipeline...")
    
    # Cleanup past tests to prevent false positives
    for p in ["test_master_faiss", "test_master.db", "test_master_graph", "config.yaml", "data"]:
        if os.path.exists(p):
            if os.path.isdir(p): shutil.rmtree(p)
            else: os.remove(p)
            
    # Need a config.yaml for Config loader if it doesn't default properly
    # Actually, Config uses default dict if missing.
    
    pipeline = MasterPipeline(mode="chat")
    
    # 1. Test Ingestion
    print("  Testing ingestion (CSV)...")
    csv_path = "test_dummy.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Concept", "Definition"])
        writer.writerow(["Machine Learning", "A subfield of AI that focuses on building systems that learn from data."])
        writer.writerow(["Neural Networks", "Computing systems inspired by biological neural networks."])
        
    source_id = pipeline.ingest(file_path=csv_path, source_type="csv")
    assert source_id is not None
    print("    Ingestion OK")
    
    # 2. Test Generation & Modes
    print("  Testing generation and modes...")
    pipeline.set_llm(provider="ollama", model="llama3", api_key="")
    pipeline.llm = MockLLM() # override with mock
    
    pipeline.set_mode("chat")
    resp_chat = pipeline.generate("Tell me about Neural Networks")
    assert "Neural Networks" in resp_chat
    
    pipeline.set_mode("study")
    resp_study = pipeline.generate("How do Neural Networks relate to AI?")
    assert "mock response" in resp_study
    
    print("    Generation OK")
    
    # 3. Test stats
    print("  Testing stats...")
    stats = pipeline.get_stats()
    assert stats["sources"] == 1
    assert stats["mode"] == "study"
    print("    Stats OK")
    
    # Cleanup
    if os.path.exists(csv_path): os.remove(csv_path)
    print("SUCCESS: Master Pipeline verified")

if __name__ == "__main__":
    try:
        test_master_pipeline()
    except Exception as e:
        print(f"ERROR: {e}")
        sys.exit(1)
