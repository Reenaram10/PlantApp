import os
from flask import Flask
from config import APP_ROOT, SQLALCHEMY_DATABASE_URI, JWT_SECRET
from extensions import db, cors
from routes import register_routes
from services.rag_service import faiss_rag, history_faiss

def create_app() -> Flask:
    """Application factory for PlantApp.
    Initializes Flask with static_folder=<APP_ROOT>/static,
    loads config from config.py, initializes db and cors,
    registers all 14 Blueprints (static_bp last), and ensures
    faiss_rag and history_faiss are ready before the first request.
    """
    app = Flask(__name__, static_folder=os.path.join(APP_ROOT, "static"))
    app.config["SQLALCHEMY_DATABASE_URI"] = SQLALCHEMY_DATABASE_URI
    app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
    app.config["SECRET_KEY"] = JWT_SECRET

    db.init_app(app)
    cors.init_app(app)
    register_routes(app)

    # Ensure faiss_rag and history_faiss are ready before the first request
    if not hasattr(faiss_rag, "vector_db") or faiss_rag.vector_db is None:
        try:
            faiss_rag.load_index()
        except Exception:
            pass

    if not hasattr(history_faiss, "vector_db") or history_faiss.vector_db is None:
        try:
            history_faiss.load_index()
        except Exception:
            pass

    return app
