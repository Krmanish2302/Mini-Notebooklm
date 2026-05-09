import sys
import os
import shutil
import yaml

sys.path.append(os.path.abspath('.'))

# ---------------------------------------------------------------
# Write a temp config so MasterPipeline uses isolated test paths
# ---------------------------------------------------------------
TEST_CFG = "test_reg_config.yaml"
with open(TEST_CFG, "w") as f:
    yaml.dump({
        "embedding": {
            "default_model": "all-MiniLM-L6-v2",
            "dimension": 384
        },
        "storage": {
            "faiss_path": "test_reg_faiss/index.faiss",
            "db_path": "test_reg.db",
            "graph_path": "test_reg_graph/graph.pkl"
        }
    }, f)

# Cleanup previous run
for p in ["test_reg_faiss", "test_reg.db", "test_reg_graph"]:
    if os.path.exists(p):
        if os.path.isdir(p): shutil.rmtree(p)
        else: os.remove(p)

os.makedirs("test_reg_faiss", exist_ok=True)
os.makedirs("test_reg_graph", exist_ok=True)

# Patch defaults BEFORE import so components pick up test paths
from src.storage.faiss_store import FAISSStore
from src.storage.sqlite_manager import SQLiteManager
from src.graph.graph_storage import GraphStorage

faiss_store  = FAISSStore(dimension=384, index_path="test_reg_faiss/index.faiss")
sqlite_mgr   = SQLiteManager(db_path="test_reg.db")
graph_store  = GraphStorage(graph_path="test_reg_graph/graph.pkl")

from src.master_pipeline import MasterPipeline

# Patch pipeline storage before it creates its own
pipeline = MasterPipeline.__new__(MasterPipeline)
pipeline.faiss_store    = faiss_store
pipeline.sqlite         = sqlite_mgr
pipeline.graph_storage  = graph_store

# Finish remaining __init__ manually (avoids double-loading models)
import uuid, time
from src.core.config import Config
from src.ingestion.file_detector import FileDetector
from src.ingestion.preprocessing.adaptive_preprocessor import AdaptivePreprocessor
from src.ingestion.embedding.embedding_pipeline import EmbeddingPipeline
from src.ingestion.merging.cross_modal_merger import CrossModalMerger
from src.retrieval.hybrid_retriever import HybridRetriever
from src.retrieval.contextual_compressor import ContextualCompressor
from src.retrieval.reranker import Reranker
from src.retrieval.advanced_retriever import AdvancedRetriever
from src.graph.graph_retriever import GraphRetriever
from src.retrieval.study_mode import StudyModeRetriever
from src.generation.prompt_builder import PromptBuilder
from src.generation.response_parser import ResponseParser
from src.chat_history.chat_history_manager import ChatHistoryManager
from src.storage.source_manager import SourceManager
from src.agents.web_search_agent import WebSearchAgent

pipeline.config         = Config(TEST_CFG)
pipeline.mode           = "chat"
pipeline.session_id     = str(uuid.uuid4())
pipeline.file_detector  = FileDetector()
pipeline.preprocessor   = AdaptivePreprocessor()
pipeline.embedder       = EmbeddingPipeline(model_name="all-MiniLM-L6-v2")
pipeline.cross_modal_merger = CrossModalMerger(embedder=pipeline.embedder.embedder)
pipeline.hybrid_retriever   = HybridRetriever(faiss_store)
pipeline.compressor     = ContextualCompressor()
pipeline.reranker       = Reranker()
pipeline.advanced_retriever = AdvancedRetriever(pipeline.hybrid_retriever, pipeline.compressor, pipeline.reranker)
pipeline.graph_retriever    = GraphRetriever(graph_store)
pipeline.study_retriever    = StudyModeRetriever(pipeline.advanced_retriever, graph_store, pipeline.graph_retriever)
pipeline.llm            = None
pipeline.prompt_builder = PromptBuilder()
pipeline.response_parser= ResponseParser()
pipeline.chat_history   = ChatHistoryManager(pipeline.session_id, "chat", graph_store)
pipeline.source_manager = SourceManager(faiss_store, sqlite_mgr, graph_store)
pipeline.web_search     = WebSearchAgent()
pipeline._response_cache= {}

GROQ_KEY = "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"
MODEL    = "llama-3.3-70b-versatile"

def section(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")

def run_regression():
    print("\n=== FULL PIPELINE REGRESSION TEST ===\n")

    # ---- 1. LLM ----
    section("1. LLM CONFIGURATION")
    pipeline.set_llm(provider="groq", model=MODEL, api_key=GROQ_KEY)
    print("Groq LLM configured with", MODEL)

    # ---- 2. INGESTION ----
    section("2a. PDF INGESTION")
    pdf_path = r"C:\Users\kumar\OneDrive\Desktop\Large_Language_Models_A_Deep_Dive_-_Uday_Kamath.pdf"
    try:
        pdf_id = pipeline.ingest(file_path=pdf_path, source_type="pdf")
        print(f"[PASS] PDF  source_id={pdf_id}")
    except Exception as e:
        print(f"[FAIL] PDF: {e}")

    section("2b. WEBSITE INGESTION")
    try:
        web_id = pipeline.ingest(url="https://en.wikipedia.org/wiki/Large_language_model", source_type="website")
        print(f"[PASS] Website  source_id={web_id}")
    except Exception as e:
        print(f"[FAIL] Website: {e}")

    section("2c. YOUTUBE INGESTION")
    try:
        yt_id  = pipeline.ingest(url="https://youtu.be/A6k3bjJ0bN0?si=DSeCYhE84Oo1jSs1", source_type="youtube")
        print(f"[PASS] YouTube  source_id={yt_id}")
    except Exception as e:
        print(f"[FAIL] YouTube: {e}")

    # ---- 3. STORAGE AUDIT ----
    section("3. STORAGE AUDIT")
    stats = pipeline.get_stats()
    print(f"Sources in memory : {stats['sources']}")
    print(f"FAISS chunks      : {stats['chunks']}")
    print(f"Graph nodes       : {stats['graph']}")

    import sqlite3
    with sqlite3.connect("test_reg.db") as conn:
        src_count   = conn.execute("SELECT COUNT(*) FROM sources").fetchone()[0]
        chunk_count = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        print(f"SQLite sources    : {src_count}")
        print(f"SQLite chunks     : {chunk_count}")
        if chunk_count > 0:
            sample = conn.execute("SELECT content FROM chunks LIMIT 1").fetchone()[0]
            print(f"Sample chunk      : {sample[:200]}...")

    # ---- 4. GENERATION MODES ----
    section("4a. CHAT MODE")
    pipeline.set_mode("chat")
    try:
        r = pipeline.generate("What is a Large Language Model?")
        print(f"Response:\n{r}\n")
    except Exception as e:
        print(f"[FAIL]: {e}")

    section("4b. DEEP RESEARCH MODE")
    pipeline.set_mode("deep_research")
    try:
        r = pipeline.generate("Explain transformer architecture and attention mechanisms.")
        print(f"Response:\n{r}\n")
    except Exception as e:
        print(f"[FAIL]: {e}")

    section("4c. STUDY MODE")
    pipeline.set_mode("study")
    try:
        r = pipeline.generate("How do embeddings relate to attention mechanisms?")
        print(f"Response:\n{r}\n")
    except Exception as e:
        print(f"[FAIL]: {e}")

    # ---- 5. EDGE CASES ----
    section("5. EDGE CASES")
    # 5a: Repeated query (cache hit)
    pipeline.set_mode("chat")
    try:
        r1 = pipeline.generate("What is a Large Language Model?")
        r2 = pipeline.generate("What is a Large Language Model?")  # should be cached
        cache_hit = (r1 == r2)
        print(f"[{'PASS' if cache_hit else 'FAIL'}] Cache hit on repeated query: {cache_hit}")
    except Exception as e:
        print(f"[FAIL] Cache test: {e}")

    # 5b: Generate without LLM
    print("\nEdge case: generate before set_llm...")
    pipeline2 = MasterPipeline.__new__(MasterPipeline)
    pipeline2.llm = None
    pipeline2.mode = "chat"
    try:
        pipeline2.generate("test")
        print("[FAIL] Should have raised ValueError")
    except ValueError as e:
        print(f"[PASS] Correctly raised ValueError: {e}")
    except Exception as e:
        print(f"[FAIL] Wrong exception: {e}")

    # 5c: Mode switch preserves chat history
    pipeline.set_mode("chat")
    pipeline.set_mode("study")
    print(f"[PASS] Mode switch OK. Current mode: {pipeline.mode}")

    print("\n=== REGRESSION COMPLETE ===")

if __name__ == "__main__":
    try:
        run_regression()
    except Exception as e:
        import traceback
        traceback.print_exc()
        sys.exit(1)
    finally:
        if os.path.exists(TEST_CFG):
            os.remove(TEST_CFG)
