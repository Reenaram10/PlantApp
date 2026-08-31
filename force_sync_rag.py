from app import app, faiss_rag, PlantKnowledge, DiseaseKnowledge

def force_sync():
    with app.app_context():
        print(" [Force Sync] Starting manual re-indexing...")
        faiss_rag.sync_from_db(PlantKnowledge, DiseaseKnowledge)
        print(" [Force Sync] Done.")

if __name__ == "__main__":
    force_sync()
