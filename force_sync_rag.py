from create_app import create_app
from services.rag_service import faiss_rag
from models.catalog import PlantKnowledge, DiseaseKnowledge

app = create_app()

def force_sync():
    with app.app_context():
        print(" [Force Sync] Starting manual re-indexing...")
        faiss_rag.sync_from_db(PlantKnowledge, DiseaseKnowledge)
        print(" [Force Sync] Done.")

if __name__ == "__main__":
    force_sync()
