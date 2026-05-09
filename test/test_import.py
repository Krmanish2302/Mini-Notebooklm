import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
try:
    import app
    import backend
    from src.master_pipeline import MiniNotebookLM
    print('Test successful: All modules imported without errors')
except Exception as e:
    print(f'Test failed with error: {e}')
    sys.exit(1)
