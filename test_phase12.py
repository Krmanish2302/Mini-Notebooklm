import sys
import os
sys.path.append(os.path.abspath('.'))

from src.chat_history.chat_history_manager import ChatHistoryManager
from src.graph.graph_storage import GraphStorage

def test_chat_history():
    print("Testing Chat History Management...")
    
    # 1. Test RAG History (Chat Mode)
    print("  Testing RAGChatHistory (Chat Mode)...")
    rag_manager = ChatHistoryManager(session_id="session_1", mode="chat")
    
    rag_manager.add_message("user", "What is photosynthesis?")
    rag_manager.add_message("assistant", "Photosynthesis is the process by which plants use sunlight to synthesize foods from carbon dioxide and water.")
    rag_manager.add_message("user", "What is the capital of France?")
    rag_manager.add_message("assistant", "The capital of France is Paris.")
    
    # Retrieve relevant history
    relevant = rag_manager.get_history_context("Tell me more about plants.", max_messages=10)
    assert "photosynthesis" in relevant.lower()
    print("    RAGChatHistory OK")
    
    # 2. Test Graph History (Study Mode)
    print("  Testing GraphChatHistory (Study Mode)...")
    
    graph_path = "test_history_graph/graph.pkl"
    if os.path.exists("test_history_graph"):
        import shutil
        shutil.rmtree("test_history_graph")
        
    graph_storage = GraphStorage(graph_path=graph_path)
    graph_manager = ChatHistoryManager(session_id="session_2", mode="study", graph_storage=graph_storage)
    
    graph_manager.add_message("user", "I want to learn about Neural Networks.", concepts=["Neural Networks"])
    graph_manager.add_message("assistant", "Neural Networks are computing systems inspired by the biological neural networks that constitute animal brains.", concepts=["Neural Networks", "Biological Brains"])
    
    context = graph_manager.get_history_context()
    assert "Neural Networks" in context
    
    print("    GraphChatHistory OK")
    
    # Cleanup
    import shutil
    shutil.rmtree("test_history_graph")
    print("SUCCESS: Chat History verified")

if __name__ == "__main__":
    try:
        test_chat_history()
    except Exception as e:
        print(f"ERROR: {e}")
        sys.exit(1)
