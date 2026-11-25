import os
from typing import Any
from flask_sqlalchemy import SQLAlchemy
from flask_cors import CORS

db = SQLAlchemy()
cors = CORS()

class BaseModel(db.Model):  # type: ignore
    __abstract__ = True

    def __init__(self, **kwargs: Any) -> None:
        for k, v in kwargs.items():
            setattr(self, k, v)

# --- LANGFUSE LOGGING & OBSERVABILITY CLIENT ---
try:
    from langfuse import Langfuse
    langfuse_client = Langfuse(
        public_key=os.getenv("LANGFUSE_PUBLIC_KEY", "pk-lf-demo"),
        secret_key=os.getenv("LANGFUSE_SECRET_KEY", "sk-lf-demo"),
        host=os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")
    )
    LANGFUSE_ENABLED = True
    print(" [Langfuse] Telemetry & logging client initialized.")
except Exception as lf_err:
    langfuse_client = None
    LANGFUSE_ENABLED = False
    print(f" [Langfuse] Telemetry client info: {lf_err}")
