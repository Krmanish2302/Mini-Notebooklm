#!/usr/bin/env python3
"""
Backend orchestrator for Mini NotebookLM.
Run this first, then run Streamlit UI.
"""

import time
import sys
from src.master_pipeline import MasterPipeline

def main():
    print("🚀 Starting Mini NotebookLM Backend...")
    
    # Initialize pipeline (loads models, indexes, etc.)
    pipeline = MasterPipeline(mode="chat")
    
    print("✅ Backend initialized successfully!")
    print(f"📊 Stats: {pipeline.get_stats()}")
    print("📡 Ready to serve requests...")
    print("\nPress Ctrl+C to stop\n")
    
    try:
        while True:
            time.sleep(60)
    except KeyboardInterrupt:
        print("\n🛑 Shutting down...")
        sys.exit(0)

if __name__ == "__main__":
    main()