"""
RAG Knowledge and FAISS Indexing Service
Manages FAISS semantic search engine, history metadata engine, and index rebuilding.
"""
import os
import threading
from config import APP_ROOT
from rag_engine import tomato_rag, plant_rag
from rag_faiss_engine import PlantFAISSRAG
from history_faiss_engine import HistoryFAISSMetaEngine

# Initialize FAISS engines eagerly so they are ready before first request
faiss_rag = PlantFAISSRAG()
history_faiss = HistoryFAISSMetaEngine()

def ask_rebuild_index():
    """
    Ask the user in the terminal whether to rebuild the FAISS index, with a 5s timeout.
    Only executed in the __main__ startup path.
    """
    index_path = os.path.join(APP_ROOT, "local_faiss_index")
    if not os.path.exists(index_path):
        print(" [FAISS Startup] No existing index found. Building fresh index from CSV files...")
        return True
    print("\n" + "="*60)
    print(" [FAISS Startup] Existing FAISS index found.")
    print(" Type 'y' + Enter to REBUILD from CSVs, or press Enter / wait 5s to use cached index.")
    print("="*60, flush=True)
    answer_holder = [""]
    def _read_input():
        try:
            answer_holder[0] = input("> ").strip().lower()
        except Exception:
            pass
    t = threading.Thread(target=_read_input, daemon=True)
    t.start()
    t.join(timeout=5)
    if answer_holder[0] == "y":
        print(" [FAISS Startup] Rebuilding index from CSV files...")
        return True
    print(" [FAISS Startup] Using cached FAISS index.")
    return False
