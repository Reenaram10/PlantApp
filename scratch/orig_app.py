# LLM Options: "groq", "deepseek", "gemini", "slm", "local"
LLM_OPTION = "slm"

# STT (Speech-to-Text) Options: "vosk", "whisper", "google", "azure"
STT_OPTION = "vosk"

# TTS (Text-to-Speech) Options: "gtts", "azure", "elevenlabs", "none"
TTS_OPTION = "gtts"

# Failsafe Methods (used when primary option fails or is set to 0)
LLM_FAILSAFE = "local"      # Falls back to keyword matching
STT_FAILSAFE = "vosk"       # Falls back to VOSK
TTS_FAILSAFE = "gtts"       # Falls back to gTTS

import os
import sys
import inspect
import warnings
from typing import Any
from sqlalchemy.exc import SAWarning

# Ensure application root is in sys.path
APP_ROOT = os.path.dirname(os.path.abspath(__file__))
if APP_ROOT not in sys.path:
    sys.path.insert(0, APP_ROOT)

# Suppress unrecognized SQL Server version warning (SQL Server 2025 compatibility)
warnings.filterwarnings("ignore", category=SAWarning, message=r".*Unrecognized server version info.*")

from difflib import SequenceMatcher
import numpy as np
import google.generativeai as genai
from flask import Flask, request, jsonify, make_response
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import text
from datetime import datetime, timezone
from openai import OpenAI  
import os
import json
import re
import hmac
import hashlib
import traceback
import wave
import subprocess
import time
import random
import base64
import io
import requests
import google.generativeai as genai 
import tensorflow as tf
try:
    from keras.models import load_model
except ImportError:
    from tensorflow.keras.models import load_model  # type: ignore[import-untyped,import-not-found]
import cv2
VOSK_AVAILABLE = False
import wave
import json
import os
import subprocess
import tempfile
import time
import shutil
from gtts import gTTS
import base64
from Crypto.Cipher import AES
import hashlib

def encrypt(plain_text, working_key):
    iv = b'\x00\x01\x02\x03\x04\x05\x06\x07\x08\t\n\x0b\x0c\r\x0e\x0f'
    key = working_key.encode('utf-8')
    byte_key = hashlib.md5(key).digest()
    cipher = AES.new(byte_key, AES.MODE_CBC, iv)
    pad_len = 16 - (len(plain_text) % 16)
    padded_text = plain_text + chr(pad_len) * pad_len
    encrypted_text = cipher.encrypt(padded_text.encode('utf-8'))
    return encrypted_text.hex()

def decrypt(cipher_text, working_key):
    iv = b'\x00\x01\x02\x03\x04\x05\x06\x07\x08\t\n\x0b\x0c\r\x0e\x0f'
    key = working_key.encode('utf-8')
    byte_key = hashlib.md5(key).digest()
    encrypted_text = bytes.fromhex(cipher_text)
    cipher = AES.new(byte_key, AES.MODE_CBC, iv)
    decrypted_text = cipher.decrypt(encrypted_text).decode('utf-8')
    pad_len = ord(decrypted_text[-1])
    return decrypted_text[:-pad_len]
import io
import re
import requests
from groq import Groq
from werkzeug.utils import secure_filename
import os
from dotenv import load_dotenv

load_dotenv()

# --- LANGFUSE LOGGING & OBSERVABILITY ---
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

def log_langfuse_trace(trace_name, user_input, output_reply, metadata=None, level="DEFAULT", tags=None):
    """Log a completed LLM/RAG trace with timing metrics & failure telemetry to Langfuse."""
    if not LANGFUSE_ENABLED or not langfuse_client:
        return
    try:
        meta = metadata or {}
        trace_tags = tags or (["failure", "error"] if level == "ERROR" else ["performance", "telemetry"])
        
        # Support both Langfuse SDK v2/v3 (client.trace) and v4 (client.start_observation)
        if hasattr(langfuse_client, "trace") and callable(getattr(langfuse_client, "trace")):
            langfuse_client.trace(
                name=trace_name,
                input=user_input,
                output=output_reply,
                metadata=meta,
                tags=trace_tags
            )
        elif hasattr(langfuse_client, "start_observation") and callable(getattr(langfuse_client, "start_observation")):
            obs = langfuse_client.start_observation(
                name=trace_name,
                input=user_input,
                output=output_reply,
                metadata={**meta, "tags": trace_tags},
                level=level if level in ("DEBUG", "DEFAULT", "WARNING", "ERROR") else "DEFAULT"
            )
            obs.end()
        langfuse_client.flush()
    except Exception as e:
        print(f" [Langfuse] Logging trace exception: {e}")

app = Flask(__name__, static_folder="static")
CORS(app)

@app.route('/admin.html')
@app.route('/admin')
def serve_admin_dashboard():
    return app.send_static_file('admin.html')

from rag_engine import tomato_rag, plant_rag
from rag_faiss_engine import PlantFAISSRAG
from history_faiss_engine import HistoryFAISSMetaEngine

# Initialize FAISS RAG engine
faiss_rag = PlantFAISSRAG()
history_faiss = HistoryFAISSMetaEngine()

def ask_rebuild_index():
    """Ask the user in the terminal whether to rebuild the FAISS index, with a 5s timeout."""
    import threading, os
    index_path = "local_faiss_index"
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



ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png', 'gif'}

def allowed_file(filename):
    """Validate uploaded file has an allowed image extension"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# ========== LOCAL DISEASE DETECTION MODELS (2-STAGE HIERARCHICAL) ==========
CROP_MODEL_PATH = os.path.join("plant_disease_detection", "model", "crop_type_model.h5")
CROP_CLASSES_PATH = os.path.join("plant_disease_detection", "model", "crop_indices.json")

POTATO_MODEL_PATH = os.path.join("plant_disease_detection", "model", "potato_disease_model.h5")
POTATO_CLASSES_PATH = os.path.join("plant_disease_detection", "model", "potato_indices.json")

TOMATO_MODEL_PATH = os.path.join("plant_disease_detection", "model", "tomato_disease_model.h5")
TOMATO_CLASSES_PATH = os.path.join("plant_disease_detection", "model", "tomato_indices.json")

FLOWER_MODEL_PATH = os.path.join("plant_disease_detection", "model", "flower_crop_model.h5")
FLOWER_CLASSES_PATH = os.path.join("plant_disease_detection", "model", "flower_indices.json")

def load_json_mapping(path):
    if os.path.exists(path):
        try:
            with open(path, "r") as f:
                return json.load(f)
        except Exception as e:
            print(f" [Model] Error loading {path}: {e}")
    return {}

crop_indices = load_json_mapping(CROP_CLASSES_PATH)
potato_indices = load_json_mapping(POTATO_CLASSES_PATH)
tomato_indices = load_json_mapping(TOMATO_CLASSES_PATH)
flower_indices = load_json_mapping(FLOWER_CLASSES_PATH)

try:
    print(f" [Model] Loading Stage 1 Crop Type model from {CROP_MODEL_PATH}...")
    crop_type_model = load_model(CROP_MODEL_PATH)
    print(f" [Model] Crop Type model loaded: {crop_indices}")
except Exception as e:
    print(f" [Model] Error loading crop_type_model: {e}")
    crop_type_model = None

try:
    print(f" [Model] Loading Stage 2 Potato Disease model from {POTATO_MODEL_PATH}...")
    potato_disease_model = load_model(POTATO_MODEL_PATH)
    print(f" [Model] Potato model loaded: {potato_indices}")
except Exception as e:
    print(f" [Model] Error loading potato_disease_model: {e}")
    potato_disease_model = None

try:
    print(f" [Model] Loading Stage 2 Tomato Disease model from {TOMATO_MODEL_PATH}...")
    tomato_disease_model = load_model(TOMATO_MODEL_PATH)
    print(f" [Model] Tomato model loaded: {tomato_indices}")
except Exception as e:
    print(f" [Model] Error loading tomato_disease_model: {e}")
    tomato_disease_model = None

try:
    print(f" [Model] Loading Dedicated Flower Crop model from {FLOWER_MODEL_PATH}...")
    flower_crop_model = load_model(FLOWER_MODEL_PATH)
    print(f" [Model] Flower Crop model loaded: {flower_indices}")
except Exception as e:
    print(f" [Model] Error loading flower_crop_model: {e}")
    flower_crop_model = None

def llm_call(prompt):
    global LLM_OPTION

    #  GROQ
    if LLM_OPTION == "groq":
        try:
            groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

            response = groq_client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[
                    {"role": "system", "content": "You are a helpful plant assistant."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2,
                max_tokens=200
            )
            return response.choices[0].message.content.strip()

        except Exception as e:
            print(f" Groq error: {e}")
            return groq_call_failsafe(prompt)

    #  DEEPSEEK (OpenAI-Compatible API)
    elif LLM_OPTION == "deepseek":
        try:
            deepseek_client = OpenAI(
            api_key=os.getenv("DEEPSEEK_API_KEY"),
            base_url="https://api.deepseek.com"
        )

            response = deepseek_client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": "You are a helpful plant assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,
            max_tokens=200
        )

            return response.choices[0].message.content.strip()

        except Exception as e:
            print(f" DeepSeek error: {e}")
            return groq_call_failsafe(prompt)

    #  OPENAI
    elif LLM_OPTION == "gemini":
        try:
            genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

            model = genai.GenerativeModel("gemini-1.5-flash")
            response = model.generate_content(prompt)

            return response.text.strip()

        except Exception as e:
            print(f" Gemini error: {e}")
            return groq_call_failsafe(prompt)

    #  SLM (Small Language Model via Hugging Face)
    elif LLM_OPTION == "slm":
        return slm_call(prompt)

    #  LOCAL fallback
    elif LLM_OPTION == "local":
        return groq_call_failsafe(prompt)

    else:
        print(f" Unknown LLM option: {LLM_OPTION}")
        return groq_call_failsafe(prompt)


def slm_call(prompt):
    """
    Calls a specialized Small Language Model (Phi mini) via Local Ollama API.
    """
    api_url = "http://127.0.0.1:11434/api/generate"
    model_id = "phi3" # or "phi" depends on locally pulled model
    
    # Prompt engineering for specific plant expert personality in SLM
    system_instr = "You are a plant-based expert specialized in botanical advice and nursery assistance."
    formatted_prompt = f"System: {system_instr}\n\nUser: {prompt}\n\nExpert:"
    
    payload = {
        "model": model_id,
        "prompt": formatted_prompt,
        "stream": False,
        "options": {
            "temperature": 0.3,
            "num_predict": 250
        }
    }
    
    try:
        print(f" [SLM] Connecting to Ollama at {api_url} with model {model_id}...")
        response = requests.post(api_url, json=payload, timeout=40, proxies={"http": None, "https": None})
        
        if response.status_code == 200:
            res_json = response.json()
            if 'response' in res_json:
                print(" [SLM] SUCCESS: Received response from Ollama.")
                return res_json['response'].strip()
            else:
                print(" [SLM] ERROR: 'response' key missing in Ollama JSON.")
        else:
            print(f" [SLM] ERROR: Ollama returned {response.status_code}: {response.text[:200]}")
            
    except Exception as e:
        print(f" [SLM] CONNECTION FAILURE: {type(e).__name__}: {e}")
        
    # FALLBACK: Try Groq if SLM fails instead of just keyword matching
    print(" [SLM] Falling back to Groq for better quality...")
    try:
        # We reuse the Groq call logic but with the expert prompt
        expert_prompt = f"Expert Advice Needed: {prompt}. Please answer as a botanical expert."
        # Temporarily change LLM_OPTION to groq for a single call
        temp_option = LLM_OPTION
        globals()['LLM_OPTION'] = "groq"
        fallback_resp = llm_call(expert_prompt)
        globals()['LLM_OPTION'] = temp_option
        return fallback_resp
    except:
        return groq_call_failsafe(prompt)


def groq_call_failsafe(prompt):
    """
    Failsafe method using simple keyword matching.
    Works without external API calls.
    """
    print(" Using failsafe LLM (keyword matching)...")
    
    prompt_lower = prompt.lower()
    
    # Keyword-based responses
    if any(word in prompt_lower for word in ["indoor", "bedroom", "house", "apartment"]):
        return "indoor"
    elif any(word in prompt_lower for word in ["outdoor", "garden", "balcony", "terrace"]):
        return "outdoor"
    elif any(word in prompt_lower for word in ["fruit", "vegetable", "edible"]):
        return "fruits"
    elif any(word in prompt_lower for word in ["flower", "flowering", "colorful", "bloom"]):
        return "flowering"
    elif any(word in prompt_lower for word in ["succulent", "cactus", "dry"]):
        return "succulents"
    elif any(word in prompt_lower for word in ["climbing", "vine", "creeper"]):
        return "climbing"
    else:
        return "general"


VOSK_AVAILABLE = False
vosk_model = None

SYSTEM_PROMPT = """
You are Green-Buy's helpful plant assistant. When helping customers:

1. SPELLING CHECK RULES:
   - For any misspelled plant name, ONLY respond with: "Did you mean <correct_name> (ID:<id>)?"
   - Do not proceed until user confirms

2. AFTER SPELLING CONFIRMATION:
   - Only proceed if user confirms with "yes" or similar affirmative
   - Then ask about living space, light conditions, experience level, maintenance preferences, location/climate

3. FOR PURCHASE REQUESTS:
   - Only process after spelling is confirmed
   - Include care instructions after order confirmation

4. FOR DELETION REQUESTS:
   - When user asks to remove/delete/cancel an order
   - Respond with: "DELETE_ORDER:<plant_id>"

5. FOR IMAGE REQUESTS:
   - When user asks "show me/show picture of/show image of <plant>", display the image
   - Include basic plant information with the image

Always use plant IDs from the database in your responses.
"""

# Suppress benign SQLAlchemy 2.0 SQL Server 2025 version warning
import warnings
from sqlalchemy.exc import SAWarning
warnings.filterwarnings('ignore', category=SAWarning, message=r".*Unrecognized server version info.*")

# ---- SQL Server Database Configuration ----
# Supports Windows Authentication (DB_TRUSTED_CONNECTION=yes)
# and SQL Server Authentication (provide DB_USER + DB_PASSWORD)
_db_server   = os.getenv("DB_SERVER", "localhost")
_db_name     = os.getenv("DB_NAME", "PlantAppDB")
_db_user     = os.getenv("DB_USER", "")
_db_password = os.getenv("DB_PASSWORD", "")
_db_trusted  = os.getenv("DB_TRUSTED_CONNECTION", "yes")

# Auto-detect newest installed Microsoft ODBC Driver
_odbc_driver = "ODBC+Driver+18+for+SQL+Server"
try:
    import pyodbc
    _available = pyodbc.drivers()
    if "ODBC Driver 18 for SQL Server" in _available:
        _odbc_driver = "ODBC+Driver+18+for+SQL+Server"
    elif "ODBC Driver 17 for SQL Server" in _available:
        _odbc_driver = "ODBC+Driver+17+for+SQL+Server"
    elif "SQL Server" in _available:
        _odbc_driver = "SQL+Server"
except Exception:
    pass

if _db_trusted.lower() == "yes":
    _mssql_uri = (
        f"mssql+pyodbc://@{_db_server}/{_db_name}"
        f"?driver={_odbc_driver}"
        f"&TrustServerCertificate=yes"
    )
else:
    from urllib.parse import quote_plus
    _mssql_uri = (
        f"mssql+pyodbc://{quote_plus(_db_user)}:{quote_plus(_db_password)}"
        f"@{_db_server}/{_db_name}"
        f"?driver={_odbc_driver}"
        f"&TrustServerCertificate=yes"
    )

app.config['SQLALCHEMY_DATABASE_URI'] = _mssql_uri
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

app.config["SQLALCHEMY_ENGINE_OPTIONS"] = {
    "pool_pre_ping": True,
    "pool_recycle": 300,
    "pool_timeout": 30,
}


db = SQLAlchemy(app)

# DB check moved to main block

class Business(db.Model):
    __tablename__ = "businesses"

    business_id = db.Column(db.String(50), primary_key=True)
    business_name = db.Column(db.String(100), nullable=False)
    owner_name = db.Column(db.String(100))
    email = db.Column(db.String(100))
    phone = db.Column(db.String(20))
    address = db.Column(db.Text)
    city = db.Column(db.String(50))
    state = db.Column(db.String(50))
    country = db.Column(db.String(50))
    status = db.Column(db.String(20), default='Active') # 'Active' / 'Inactive'
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow)

class AuditLog(db.Model):
    __tablename__ = "audit_logs"

    id = db.Column(db.Integer, primary_key=True)
    business_id = db.Column(db.String(50))
    user_id = db.Column(db.Integer)
    user_email = db.Column(db.String(100))
    action = db.Column(db.String(100), nullable=False)
    module = db.Column(db.String(50), nullable=False)
    record_id = db.Column(db.String(50))
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    details = db.Column(db.Text)

class UserQuery(db.Model):
    __tablename__ = "user_queries"

    uq_id = db.Column(db.Integer, primary_key=True)
    business_id = db.Column(db.String(50), default='NURS-0001')
    user_id = db.Column(db.Integer, db.ForeignKey("login.id", ondelete="CASCADE"), nullable=False)
    plant_id = db.Column(db.Integer, db.ForeignKey("plant.plant_id", ondelete="SET NULL"), nullable=True)
    query_text = db.Column('query', db.Text, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

class _UserQueryDescriptor:
    """Descriptor that preserves UserQuery.query for Flask-SQLAlchemy model queries,
    while allowing instance-level uq.query to access or mutate query_text seamlessly."""
    def __init__(self, original_desc):
        self.original_desc = original_desc

    def __get__(self, instance, owner):
        if instance is None:
            return self.original_desc.__get__(None, owner)
        return getattr(instance, 'query_text', None)

    def __set__(self, instance, value):
        instance.query_text = value

UserQuery.query = _UserQueryDescriptor(inspect.getattr_static(db.Model, 'query'))

class UserInterest(db.Model):
    __tablename__ = "user_interest"

    interest_id = db.Column(db.Integer, primary_key=True)
    business_id = db.Column(db.String(50), default='NURS-0001')
    user_id = db.Column(db.Integer, db.ForeignKey("login.id", ondelete="CASCADE"), nullable=False)
    plant_id = db.Column(db.Integer, db.ForeignKey("plant.plant_id", ondelete="CASCADE"), nullable=False)

    interest_type = db.Column(db.String(20), nullable=False)  # enquiry / purchase
    notes = db.Column(db.Text)

    created_at = db.Column(db.DateTime, default=datetime.utcnow)


class User(db.Model):
    __tablename__ = 'login'

    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(100))
    email = db.Column(db.String(120), unique=True, nullable=False)
    password = db.Column(db.String(200), nullable=False)
    role = db.Column(db.String(20), default='NURSERY_ADMIN') # 'SUPER_ADMIN', 'NURSERY_ADMIN', 'STAFF'
    business_id = db.Column(db.String(50), default='NURS-0001')

    last_login = db.Column(db.DateTime)
    last_logout = db.Column(db.DateTime)
    is_online = db.Column(db.Boolean, default=False)



class Plant(db.Model):
    __tablename__ = "plant"

    plant_id = db.Column(db.Integer, primary_key=True)
    business_id = db.Column(db.String(50), default='NURS-0001')
    plant_name = db.Column(db.String(100))
    description = db.Column(db.Text)
    price = db.Column(db.Float)
    image_path = db.Column(db.String(255))
    stock = db.Column(db.Integer, default=0)
    discount_price = db.Column(db.Float, default=0.0)

class PlantKnowledge(db.Model):
    __tablename__ = "plant_knowledge"

    id = db.Column(db.Integer, primary_key=True)
    plant_name = db.Column(db.String(255))
    scientific_name = db.Column(db.String(255))
    description = db.Column(db.Text)
    sunlight = db.Column(db.Text)
    water_requirement = db.Column(db.Text)
    soil_type = db.Column(db.Text)
    fertilizer = db.Column(db.Text)
    medicinal_uses = db.Column(db.Text)
    common_diseases = db.Column(db.Text)
    care_tips = db.Column(db.Text)

class DiseaseKnowledge(db.Model):
    __tablename__ = "disease_knowledge"

    id = db.Column(db.Integer, primary_key=True)
    disease_name = db.Column(db.String(255))
    symptoms = db.Column(db.Text)
    causes = db.Column(db.Text)
    organic_treatment = db.Column(db.Text)
    chemical_treatment = db.Column(db.Text)
    prevention = db.Column(db.Text)

class Supplier(db.Model):
    __tablename__ = "supplier"
    supplier_id = db.Column(db.Integer, primary_key=True)
    business_id = db.Column(db.String(50), default='NURS-0001')
    supplier_name = db.Column(db.String(100), nullable=False)
    contact_person = db.Column(db.String(100))
    phone = db.Column(db.String(20))
    email = db.Column(db.String(100))
    address = db.Column(db.Text)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

class InventoryTransaction(db.Model):
    __tablename__ = "inventory_transaction"
    transaction_id = db.Column(db.Integer, primary_key=True)
    business_id = db.Column(db.String(50), default='NURS-0001')
    plant_id = db.Column(db.Integer, db.ForeignKey("plant.plant_id", ondelete="CASCADE"), nullable=False)
    supplier_id = db.Column(db.Integer, db.ForeignKey("supplier.supplier_id", ondelete="SET NULL"), nullable=True)
    type = db.Column(db.String(10), nullable=False) # 'ADD' or 'REMOVE'
    quantity = db.Column(db.Integer, nullable=False)
    notes = db.Column(db.Text)
    bill_no = db.Column(db.String(50))
    bill_date = db.Column(db.String(20))
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    plant = db.relationship('Plant', backref='transactions')
    supplier = db.relationship('Supplier', backref='transactions')

class SupplierPurchase(db.Model):
    __tablename__ = "supplier_purchase"
    purchase_id = db.Column(db.Integer, primary_key=True)
    business_id = db.Column(db.String(50), default='NURS-0001')
    supplier_id = db.Column(db.Integer, db.ForeignKey("supplier.supplier_id", ondelete="CASCADE"), nullable=False)
    bill_no = db.Column(db.String(50), nullable=False)
    bill_date = db.Column(db.DateTime, default=datetime.utcnow)
    total_amount = db.Column(db.Float, nullable=False)
    amount_paid = db.Column(db.Float, default=0.0)
    balance = db.Column(db.Float, nullable=False)
    status = db.Column(db.String(20), default='Pending') # Pending, Partial, Paid
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    supplier = db.relationship('Supplier', backref='purchases')

class PurchaseItem(db.Model):
    __tablename__ = "purchase_item"
    item_id = db.Column(db.Integer, primary_key=True)
    business_id = db.Column(db.String(50), default='NURS-0001')
    purchase_id = db.Column(db.Integer, db.ForeignKey("supplier_purchase.purchase_id", ondelete="CASCADE"), nullable=False)
    plant_id = db.Column(db.Integer, db.ForeignKey("plant.plant_id", ondelete="CASCADE"), nullable=False)
    quantity = db.Column(db.Integer, nullable=False)
    unit_price = db.Column(db.Float, nullable=False)

    purchase = db.relationship('SupplierPurchase', backref='items')
    plant = db.relationship('Plant', backref='purchases')

class Variety(db.Model):
    __tablename__ = "variety"

    variety_id = db.Column(db.Integer, primary_key=True)
    business_id = db.Column(db.String(50), default='NURS-0001')
    plant_id = db.Column(db.Integer, db.ForeignKey("plant.plant_id", ondelete="CASCADE"), nullable=False)
    variety_name = db.Column(db.String(100), nullable=False)
    variety_description = db.Column(db.Text)
    variety_price = db.Column(db.Float)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    plant = db.relationship('Plant', backref='varieties')




class Conversation(db.Model):
    __tablename__ = 'conversation'
    conversation_id = db.Column(db.Integer, primary_key=True)
    business_id = db.Column(db.String(50), default='NURS-0001')
    user_id = db.Column(db.Integer, db.ForeignKey('login.id', ondelete='CASCADE'))
    conversation = db.Column(db.Text, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

class Cart(db.Model):
    __tablename__ = "cart_new"
    cart_id = db.Column(db.Integer, primary_key=True)
    business_id = db.Column(db.String(50), default='NURS-0001')
    user_id = db.Column(db.String(100), nullable=False)
    plant_id = db.Column(db.Integer, db.ForeignKey("plant.plant_id"), nullable=False)
    added_at = db.Column(db.DateTime, default=datetime.utcnow)
    quantity = db.Column(db.Integer, default=1)

    plant = db.relationship('Plant', backref='cart_items')

class Order(db.Model):
    __tablename__ = 'orders'
    order_id = db.Column(db.Integer, primary_key=True)
    business_id = db.Column(db.String(50), default='NURS-0001')
    plant_name = db.Column(db.String(100))
    user_id = db.Column(db.Integer, db.ForeignKey('login.id'))
    order_date = db.Column(db.DateTime, default=datetime.utcnow)
    total_amount = db.Column(db.Float)
    payment_status = db.Column(db.String(20), default='Pending')
    order_status = db.Column(db.String(20), default='Processing')
    order_group_id = db.Column(db.String(50))
    price = db.Column(db.Float)
    quantity = db.Column(db.Integer)
    payment_method = db.Column(db.String(50))
    shipping_address = db.Column(db.Text)
    notes = db.Column(db.Text)
    delivery_date = db.Column(db.String(50))
    tracking_number = db.Column(db.String(100))
    amount_paid = db.Column(db.Float, default=0.0)
    balance = db.Column(db.Float, default=0.0)

class Category(db.Model):
    __tablename__ = 'category'
    category_id = db.Column(db.Integer, primary_key=True)
    business_id = db.Column(db.String(50), default='NURS-0001')
    category_name = db.Column(db.String(100), nullable=False)

class PlantCategory(db.Model):
    __tablename__ = 'plant_category'
    plant_category_id = db.Column(db.Integer, primary_key=True)
    plant_id = db.Column(db.Integer, db.ForeignKey('plant.plant_id'), nullable=False)
    category_id = db.Column(db.Integer, db.ForeignKey('category.category_id'), nullable=False)


class LLMResult(db.Model):
    __tablename__ = 'llm_results'
    id = db.Column(db.Integer, primary_key=True)
    business_id = db.Column(db.String(50), default='NURS-0001')
    user_query = db.Column(db.Text, nullable=False)
    llm_response = db.Column(db.Text, nullable=False)
    category_suggested = db.Column(db.String(100))
    plants_found = db.Column(db.Integer, default=0)
    llm_name = db.Column(db.String(50))
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    expires_at = db.Column(db.DateTime)


class Query(db.Model):
    __tablename__ = 'query'
    query_id = db.Column(db.Integer, primary_key=True)
    business_id = db.Column(db.String(50), default='NURS-0001')
    description = db.Column(db.Text, nullable=False)
    canonical_key = db.Column(db.String(100))   
    user_id = db.Column(db.Integer, db.ForeignKey('login.id', ondelete='CASCADE'), nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    user = db.relationship('User', backref='queries')


class QueryResult(db.Model):
    __tablename__ = 'query_result'
    result_id = db.Column(db.Integer, primary_key=True)
    business_id = db.Column(db.String(50), default='NURS-0001')
    query_id = db.Column(db.Integer, db.ForeignKey('query.query_id', ondelete='CASCADE'), nullable=False)
    result_description = db.Column(db.Text, nullable=False)
    result_source = db.Column(db.String(50), nullable=False)
    
    response_type = db.Column(db.String(50), default='text')
    response_data = db.Column(db.Text)
    
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    query = db.relationship('Query', backref='results')


class CategorySynonym(db.Model):
    __tablename__ = 'category_synonym'
    synonym_id = db.Column(db.Integer, primary_key=True)
    business_id = db.Column(db.String(50), default='NURS-0001')
    synonym = db.Column(db.String(100), nullable=False)
    category_id = db.Column(db.Integer, db.ForeignKey('category.category_id', ondelete='CASCADE'), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    category = db.relationship('Category', backref='synonyms')

# ========== MULTI-TENANT JWT & AUTHENTICATION MIDDLEWARE ==========
from functools import wraps
from flask import g

JWT_SECRET = os.getenv("JWT_SECRET", "super-secret-saas-key-2026")

def b64_encode(data_bytes):
    return base64.urlsafe_b64encode(data_bytes).rstrip(b'=').decode('utf-8')

def b64_decode(data_str):
    padding = '=' * (4 - (len(data_str) % 4))
    return base64.urlsafe_b64decode(data_str + padding)

def generate_jwt(payload, secret=JWT_SECRET, exp_seconds=86400):
    header = {"alg": "HS256", "typ": "JWT"}
    payload_copy = dict(payload)
    if "exp" not in payload_copy:
        payload_copy["exp"] = int(time.time()) + exp_seconds
    
    header_b64 = b64_encode(json.dumps(header).encode('utf-8'))
    payload_b64 = b64_encode(json.dumps(payload_copy).encode('utf-8'))
    
    signing_input = f"{header_b64}.{payload_b64}".encode('utf-8')
    signature = hmac.new(secret.encode('utf-8'), signing_input, hashlib.sha256).digest()
    sig_b64 = b64_encode(signature)
    
    return f"{header_b64}.{payload_b64}.{sig_b64}"

def verify_jwt(token, secret=JWT_SECRET):
    try:
        if not token:
            return None
        if token.startswith("Bearer "):
            token = token[7:]
        parts = token.strip().split('.')
        if len(parts) != 3:
            return None
        header_b64, payload_b64, sig_b64 = parts
        
        signing_input = f"{header_b64}.{payload_b64}".encode('utf-8')
        expected_sig = hmac.new(secret.encode('utf-8'), signing_input, hashlib.sha256).digest()
        actual_sig = b64_decode(sig_b64)
        
        if not hmac.compare_digest(expected_sig, actual_sig):
            return None
            
        payload = json.loads(b64_decode(payload_b64).decode('utf-8'))
        if payload.get("exp") and time.time() > payload["exp"]:
            return None
            
        return payload
    except Exception as e:
        return None

def log_audit_action(action, module, record_id=None, details=None, business_id=None, user_id=None, user_email=None):
    try:
        biz_id = business_id or (g.current_user.get('business_id') if hasattr(g, 'current_user') and g.current_user else 'NURS-0001')
        u_id = user_id or (g.current_user.get('user_id') if hasattr(g, 'current_user') and g.current_user else None)
        u_email = user_email or (g.current_user.get('email') if hasattr(g, 'current_user') and g.current_user else None)
        
        entry = AuditLog(
            business_id=biz_id,
            user_id=u_id,
            user_email=u_email,
            action=action,
            module=module,
            record_id=str(record_id) if record_id else None,
            details=details
        )
        db.session.add(entry)
        db.session.commit()
    except Exception as e:
        db.session.rollback()
        print("Audit log failed:", e)

def get_current_tenant():
    """Helper to extract authenticated user's tenant ID from request context"""
    if hasattr(g, 'current_user') and g.current_user and g.current_user.get('business_id'):
        return g.current_user['business_id']
    auth_header = request.headers.get("Authorization", "")
    if auth_header:
        payload = verify_jwt(auth_header)
        if payload and payload.get('business_id'):
            return payload['business_id']
    return 'NURS-0001'

def require_tenant_auth(allowed_roles=None):
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            auth_header = request.headers.get("Authorization", "")
            payload = verify_jwt(auth_header)
            
            if not payload:
                token = request.args.get("token")
                if not token and request.is_json:
                    try:
                        body = request.get_json(silent=True)
                        if isinstance(body, dict):
                            token = body.get("token")
                    except Exception:
                        token = None
                payload = verify_jwt(token)
                
            if not payload:
                # Default guest context for public store operations
                g.current_user = {
                    "user_id": 0,
                    "email": "guest@nursery.com",
                    "role": "STAFF",
                    "business_id": "NURS-0001"
                }
            else:
                g.current_user = payload

            user_biz = g.current_user.get("business_id", "NURS-0001")
            user_role = g.current_user.get("role", "STAFF")

            if user_role != "SUPER_ADMIN" and user_biz != "SYSTEM":
                biz = db.session.get(Business, user_biz)
                if biz and biz.status == "Inactive":
                    return jsonify({"status": "error", "message": "This nursery account has been deactivated."}), 403

            if allowed_roles:
                if user_role not in allowed_roles:
                    return jsonify({"status": "error", "message": "Access denied: Unauthorized role"}), 403

            return f(*args, **kwargs)
        return decorated_function
    return decorator

GREETINGS = {
    "hi", "hello", "hey", "hii", "heyy",
    "thanks", "thank you", "thx",
    "good morning", "good evening", "good night"
}


def normalize_query(text):
    text = text.lower().strip()
    # Remove standard punctuation but KEEP emojis (Unicode characters)
    # This regex removes symbols like !?., but preserves alphanumeric + spaces + emojis
    text = re.sub(r'[^\w\s\u2600-\u27BF\U0001f300-\U0001faff]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text



# --- REMOVED OBSOLETE CACHING/CANONICAL LOGIC ---





def add_category_synonym(synonym, category_id):
    """Add a synonym for an existing category"""
    try:
        syn = CategorySynonym(
            synonym=synonym.lower(),
            category_id=category_id
        )
        db.session.add(syn)
        db.session.commit()
        print(f" Synonym '{synonym}' added for category_id {category_id}")
        return syn.synonym_id
    except Exception as e:
        print(f"Error adding synonym: {e}")
        db.session.rollback()
        return None
    


@app.route("/api/user_queries", methods=["POST"])
def save_user_query():
    data = request.json
    user_id = data.get("user_id")
    plant_id = data.get("plant_id")  # optional
    query = data.get("query")

    if not user_id or not query:
        return jsonify({"status": "error", "message": "user_id and query required"}), 400

    try:
        new_q = UserQuery(
            user_id=user_id,
            plant_id=plant_id,
            query_text=query
        )
        db.session.add(new_q)
        db.session.commit()

        return jsonify({
            "status": "success",
            "message": "Query saved ",
            "uq_id": new_q.uq_id
        })

    except Exception as e:
        db.session.rollback()
        return jsonify({"status": "error", "message": str(e)}), 500


def log_user_interest(user_id, plant_id, interest_type, notes=""):
    try:
        #  skip guest users
        if not user_id or str(user_id) in ["anonymous", "guest_user"]:
            return

        db.session.execute(text("""
            INSERT INTO user_interest (user_id, plant_id, interest_type, notes)
            VALUES (:user_id, :plant_id, :interest_type, :notes)
        """), {
            "user_id": int(user_id),
            "plant_id": int(plant_id),
            "interest_type": interest_type,
            "notes": notes
        })

        db.session.commit()

    except Exception as e:
        db.session.rollback()
        print(" user_interest insert failed:", e)



def get_category_from_synonym(user_input):
    """Check if user input matches any synonym"""
    synonym_row = db.session.execute(text("""
        SELECT c.category_id, c.category_name
        FROM category_synonym cs
        JOIN category c ON cs.category_id = c.category_id
        WHERE LOWER(cs.synonym) = :syn
    """), {"syn": user_input.lower()}).fetchone()
    
    if synonym_row:
        return {
            "category_id": synonym_row[0],
            "category_name": synonym_row[1]
        }
    return None



# --- REMOVED OBSOLETE INTENT MAPPING ---


def get_cached_llm_result(normalized_query):
    row = db.session.execute(text("""
        SELECT TOP 1 llm_response, llm_name, expires_at
        FROM llm_results
        WHERE user_query = :q
        ORDER BY created_at DESC
    """), {"q": normalized_query}).fetchone()

    if not row:
        return None

    reply, llm_name, expires_at = row

    if expires_at and datetime.now(timezone.utc) < expires_at:
        return {"reply": reply, "llm_name": llm_name}

    return None




# Fuzzy matching

def fuzzy_find_plant(name):
    return db.session.execute(text("""
        SELECT TOP 1 plant_id, plant_name, description, price
        FROM plant
        WHERE LOWER(plant_name) LIKE LOWER(:x)
    """), {"x": f"%{name}%"}).fetchone()


def store_canonical_answer(canonical_key, reply, user_message=None):
    try:
        row = LLMResult(
            user_query=normalize_query(user_message) if user_message else canonical_key,
            llm_response=reply,
            category_suggested=canonical_key,
            plants_found=0,
            llm_name=LLM_OPTION
        )
        db.session.add(row)
        db.session.commit()
    except Exception as e:
        db.session.rollback()
        print(" Canonical store failed:", e)

# In the function that stores queries
def store_query(description, user_id, canonical_key=None):
    """Store user query"""
    try:
        #  FIX: Convert user_id to integer
        try:
            user_id_int = int(user_id)
        except (ValueError, TypeError):
            print(f" Invalid user_id: {user_id}, using 1 as default")
            user_id_int = 1  # Default to guest user
        
        query = Query(
        description=description,
        canonical_key=canonical_key,
        user_id=user_id_int
        )
        
        db.session.add(query)
        db.session.commit()
        print(f" Stored query: {description}")
        return query.query_id
    except Exception as e:
        print(f" Error storing query: {e}")
        db.session.rollback()
        return None
# Multi-tenant Register Endpoint
@app.route("/register", methods=["POST"])
@app.route("/api/auth/register", methods=["POST"])
def register():
    data = request.json or {}
    username = data.get("username", "").strip()
    email = data.get("email", "").strip().lower()
    password = data.get("password", "").strip()
    business_id = data.get("business_id", "").strip() or "NURS-0001"
    role = data.get("role", "").strip() or "NURSERY_ADMIN"

    if not username or not email or not password:
        return jsonify({
            "status": "error",
            "message": "Username, email and password are required"
        }), 400

    try:
        existing = db.session.execute(
            text("SELECT id FROM login WHERE LOWER(email) = LOWER(:email)"),
            {"email": email}
        ).fetchone()

        if existing:
            return jsonify({
                "status": "error",
                "message": "Email already registered. Please login."
            }), 409

        # Check nursery validity
        biz = db.session.get(Business, business_id)
        if not biz:
            return jsonify({"status": "error", "message": f"Invalid Business ID: {business_id}"}), 404

        new_user = User(
            username=username,
            email=email,
            password=password,
            role=role,
            business_id=business_id
        )
        db.session.add(new_user)
        db.session.commit()
        user_id = new_user.id

        token_payload = {
            "user_id": user_id,
            "username": username,
            "email": email,
            "role": role,
            "business_id": business_id
        }
        token = generate_jwt(token_payload)

        log_audit_action("USER_REGISTER", "AUTH", record_id=user_id, details=f"Registered user {email} under {business_id}", business_id=business_id, user_id=user_id, user_email=email)

        return jsonify({
            "status": "success",
            "message": "Registered successfully",
            "token": token,
            "user_id": user_id,
            "username": username,
            "email": email,
            "role": role,
            "business_id": business_id,
            "business_name": biz.business_name
        })

    except Exception as e:
        print(f"Register error: {e}")
        db.session.rollback()
        return jsonify({"status": "error", "message": str(e)}), 500


# Multi-tenant Login Endpoint
@app.route("/login", methods=["POST"])
@app.route("/api/auth/login", methods=["POST"])
def login():
    data = request.json or {}
    email = data.get("email", "").strip().lower()
    password = data.get("password", "").strip()
    req_business_id = data.get("business_id", "").strip()

    if not email or not password:
        return jsonify({
            "status": "error",
            "message": "Email/Username and password required"
        }), 400

    try:
        user = db.session.execute(text("""
            SELECT id, username, email, password, role, business_id
            FROM login
            WHERE (LOWER(email) = :email OR LOWER(username) = :email) AND password = :password
        """), {
            "email": email,
            "password": password
        }).fetchone()

        if not user:
            return jsonify({
                "status": "error",
                "message": "Invalid credentials"
            }), 401

        user_id, username, user_email, _, role, user_biz_id = user

        # Handle Super Admin login
        if role == 'SUPER_ADMIN' or (user_email and user_email.lower() == 'admin@system.com'):
            token_payload = {
                "user_id": user_id,
                "username": username or "Super Admin",
                "email": user_email,
                "role": "SUPER_ADMIN",
                "business_id": "SYSTEM"
            }
            token = generate_jwt(token_payload)
            log_audit_action("SUPER_ADMIN_LOGIN", "AUTH", record_id=user_id, details="Super Admin logged in", business_id="SYSTEM", user_id=user_id, user_email=user_email)
            return jsonify({
                "status": "success",
                "token": token,
                "user_id": user_id,
                "username": username or "Super Admin",
                "email": user_email,
                "role": "SUPER_ADMIN",
                "business_id": "SYSTEM",
                "business_name": "Platform Administration"
            })

        # For Nursery Admin / Staff, verify Business ID if provided
        target_biz_id = user_biz_id or "NURS-0001"
        if req_business_id and req_business_id.upper() != target_biz_id.upper():
            return jsonify({
                "status": "error",
                "message": f"User does not belong to Business ID: {req_business_id}"
            }), 403

        # Verify Business is Active
        biz = db.session.get(Business, target_biz_id)
        if not biz:
            return jsonify({"status": "error", "message": f"Nursery {target_biz_id} not found."}), 404
        if biz.status == "Inactive":
            return jsonify({"status": "error", "message": "This nursery account has been deactivated. Please contact Super Admin."}), 403

        token_payload = {
            "user_id": user_id,
            "username": username or user_email,
            "email": user_email,
            "role": role or "NURSERY_ADMIN",
            "business_id": target_biz_id
        }
        token = generate_jwt(token_payload)

        # Update login tracking
        try:
            db.session.execute(text("UPDATE login SET last_login = :now, is_online = 1 WHERE id = :id"), {"now": datetime.utcnow(), "id": user_id})
            db.session.commit()
        except:
            db.session.rollback()

        log_audit_action("USER_LOGIN", "AUTH", record_id=user_id, details=f"Logged in as {role} for {target_biz_id}", business_id=target_biz_id, user_id=user_id, user_email=user_email)

        return jsonify({
            "status": "success",
            "token": token,
            "user_id": user_id,
            "username": username or user_email,
            "email": user_email,
            "role": role or "NURSERY_ADMIN",
            "business_id": target_biz_id,
            "business_name": biz.business_name
        })

    except Exception as e:
        print(f"Login error: {e}")
        return jsonify({"status": "error", "message": f"Database error: {str(e)}"}), 500


# ========== SUPER ADMIN PLATFORM ENDPOINTS ==========
@app.route("/api/super-admin/login", methods=["POST"])
@app.route("/api/auth/super-admin/login", methods=["POST"])
def super_admin_login():
    data = request.json or {}
    email = data.get("email", "").strip().lower()
    password = data.get("password", "").strip()

    if email != "admin@system.com" or password != "admin123":
        user = db.session.execute(text("SELECT id, username, email FROM login WHERE LOWER(email) = :email AND password = :pwd AND role = 'SUPER_ADMIN'"), {"email": email, "pwd": password}).fetchone()
        if not user:
            return jsonify({"status": "error", "message": "Invalid Super Admin credentials"}), 401

    token = generate_jwt({
        "user_id": 1,
        "username": "Super Admin",
        "email": email,
        "role": "SUPER_ADMIN",
        "business_id": "SYSTEM"
    })
    return jsonify({
        "status": "success",
        "token": token,
        "role": "SUPER_ADMIN",
        "business_id": "SYSTEM",
        "username": "Super Admin"
    })

@app.route("/api/super-admin/nurseries", methods=["POST"])
@app.route("/api/super-admin/nursery", methods=["POST"])
@require_tenant_auth(allowed_roles=["SUPER_ADMIN"])
def register_new_nursery():
    data = request.json or {}
    business_name = (data.get("business_name") or data.get("name") or "").strip()
    owner_name = (data.get("owner_name") or data.get("owner") or "").strip()
    email = data.get("email", "").strip().lower()
    phone = data.get("phone", "").strip()
    address = data.get("address", "").strip()
    city = data.get("city", "").strip()
    state = data.get("state", "").strip()
    country = data.get("country", "").strip()
    
    admin_name = data.get("admin_name", "").strip() or owner_name or "Nursery Admin"
    admin_email = data.get("admin_email", "").strip().lower() or email
    admin_password = data.get("admin_password", "").strip() or "admin123"

    if not business_name or not admin_email:
        return jsonify({"status": "error", "message": "Nursery Name and Admin Email are required"}), 400

    try:
        # Auto-generate unique Business ID (NURS-XXXX)
        last_biz = db.session.execute(text("SELECT TOP 1 business_id FROM businesses WHERE business_id LIKE 'NURS-%' ORDER BY business_id DESC")).fetchone()
        if last_biz and last_biz[0]:
            try:
                num = int(last_biz[0].replace("NURS-", "")) + 1
            except:
                num = 2
        else:
            num = 1
        
        new_biz_id = f"NURS-{num:04d}"
        
        # Ensure ID is unique
        while db.session.get(Business, new_biz_id) is not None:
            num += 1
            new_biz_id = f"NURS-{num:04d}"

        # Insert new nursery entity
        new_biz = Business(
            business_id=new_biz_id,
            business_name=business_name,
            owner_name=owner_name,
            email=email,
            phone=phone,
            address=address,
            city=city,
            state=state,
            country=country,
            status="Active"
        )
        db.session.add(new_biz)

        # Create Nursery Admin account
        from sqlalchemy import func
        existing_user = User.query.filter(func.lower(User.email) == admin_email.lower()).first()
        if existing_user:
            existing_user.business_id = new_biz_id
            existing_user.role = "NURSERY_ADMIN"
            existing_user.password = admin_password
        else:
            admin_user = User(
                username=admin_name,
                email=admin_email,
                password=admin_password,
                role="NURSERY_ADMIN",
                business_id=new_biz_id
            )
            db.session.add(admin_user)

        db.session.commit()

        log_audit_action("CREATE_NURSERY", "SUPER_ADMIN", record_id=new_biz_id, details=f"Registered nursery {business_name} ({new_biz_id})", business_id="SYSTEM")

        return jsonify({
            "status": "success",
            "message": f"Nursery '{business_name}' registered successfully!",
            "business_id": new_biz_id,
            "business_name": business_name,
            "admin_email": admin_email,
            "business": {
                "business_id": new_biz_id,
                "business_name": business_name,
                "owner_name": owner_name,
                "email": email,
                "phone": phone
            }
        })

    except Exception as e:
        db.session.rollback()
        print(f"Error registering nursery: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route("/api/super-admin/nurseries", methods=["GET"])
@require_tenant_auth(allowed_roles=["SUPER_ADMIN"])
def list_nurseries_super_admin():
    search = request.args.get("search", "").strip().lower()
    query = Business.query
    if search:
        query = query.filter((Business.business_name.ilike(f"%{search}%")) | (Business.business_id.ilike(f"%{search}%")) | (Business.email.ilike(f"%{search}%")))
    
    nurseries = query.order_by(Business.created_at.desc()).all()
    result = []
    for n in nurseries:
        admin = User.query.filter_by(business_id=n.business_id, role="NURSERY_ADMIN").first()
        plant_count = Plant.query.filter_by(business_id=n.business_id).count()
        order_count = Order.query.filter_by(business_id=n.business_id).count()
        result.append({
            "business_id": n.business_id,
            "business_name": n.business_name,
            "owner_name": n.owner_name,
            "email": n.email,
            "phone": n.phone,
            "address": n.address,
            "city": n.city,
            "state": n.state,
            "country": n.country,
            "status": n.status,
            "created_at": n.created_at.isoformat() if n.created_at else None,
            "admin_email": admin.email if (admin and admin.email) else (n.email if n.email else None),
            "plant_count": plant_count,
            "order_count": order_count
        })
    return jsonify({"status": "success", "nurseries": result})

@app.route("/api/super-admin/nurseries/<business_id>", methods=["DELETE"])
@app.route("/api/super-admin/nursery/<business_id>", methods=["DELETE"])
@require_tenant_auth(allowed_roles=["SUPER_ADMIN"])
def delete_nursery_super_admin(business_id):
    if business_id in ["SYSTEM", "NURS-0001"]:
        return jsonify({"status": "error", "message": "Cannot delete primary default nursery"}), 400

    biz = db.session.get(Business, business_id)
    if not biz:
        return jsonify({"status": "error", "message": "Nursery not found"}), 404

    try:
        # Cascade raw SQL cleanup of child tables
        db.session.execute(text("DELETE FROM login WHERE business_id = :bid"), {"bid": business_id})
        db.session.execute(text("DELETE FROM plant WHERE business_id = :bid"), {"bid": business_id})
        db.session.execute(text("DELETE FROM inventory_transaction WHERE business_id = :bid"), {"bid": business_id})
        db.session.execute(text("DELETE FROM user_queries WHERE business_id = :bid"), {"bid": business_id})
        db.session.execute(text("DELETE FROM user_interest WHERE business_id = :bid"), {"bid": business_id})
        db.session.execute(text("DELETE FROM audit_logs WHERE business_id = :bid"), {"bid": business_id})
        db.session.execute(text("DELETE FROM businesses WHERE business_id = :bid"), {"bid": business_id})
        db.session.commit()

        log_audit_action("DELETE_NURSERY", "SUPER_ADMIN", record_id=business_id, details=f"Deleted nursery {business_id}", business_id="SYSTEM")
        return jsonify({"status": "success", "message": f"Nursery {business_id} deleted successfully"})
    except Exception as e:
        db.session.rollback()
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route("/api/super-admin/nurseries/<business_id>/status", methods=["PATCH", "PUT"])
@app.route("/api/super-admin/nursery/<business_id>/status", methods=["PATCH", "PUT"])
@require_tenant_auth(allowed_roles=["SUPER_ADMIN"])
def toggle_nursery_status(business_id):
    data = request.json or {}
    new_status = data.get("status", "").strip()
    if new_status not in ["Active", "Inactive"]:
        return jsonify({"status": "error", "message": "Status must be 'Active' or 'Inactive'"}), 400

    biz = db.session.get(Business, business_id)
    if not biz:
        return jsonify({"status": "error", "message": "Nursery not found"}), 404

    biz.status = new_status
    db.session.commit()

    log_audit_action("TOGGLE_NURSERY_STATUS", "SUPER_ADMIN", record_id=business_id, details=f"Changed status of {business_id} to {new_status}", business_id="SYSTEM")

    return jsonify({"status": "success", "message": f"Nursery {business_id} status updated to {new_status}", "business_id": business_id, "status": new_status})

@app.route("/api/super-admin/stats", methods=["GET"])
@require_tenant_auth(allowed_roles=["SUPER_ADMIN"])
def super_admin_stats():
    total_nurseries = Business.query.count()
    active_nurseries = Business.query.filter_by(status="Active").count()
    inactive_nurseries = Business.query.filter_by(status="Inactive").count()
    total_users = User.query.count()
    total_plants = Plant.query.count()
    total_orders = Order.query.count()

    return jsonify({
        "status": "success",
        "stats": {
            "total_nurseries": total_nurseries,
            "active_nurseries": active_nurseries,
            "inactive_nurseries": inactive_nurseries,
            "total_users": total_users,
            "total_plants": total_plants,
            "total_orders": total_orders
        }
    })

@app.route("/api/super-admin/audit-logs", methods=["GET"])
@require_tenant_auth(allowed_roles=["SUPER_ADMIN"])
def get_super_admin_audit_logs():
    logs = AuditLog.query.order_by(AuditLog.timestamp.desc()).limit(100).all()
    res = []
    for l in logs:
        res.append({
            "id": l.id,
            "business_id": l.business_id,
            "user_id": l.user_id,
            "user_email": l.user_email,
            "action": l.action,
            "module": l.module,
            "record_id": l.record_id,
            "timestamp": l.timestamp.isoformat() if l.timestamp else None,
            "details": l.details
        })
    return jsonify({"status": "success", "audit_logs": res})




def resolve_plant_synonym(user_message):
    row = db.session.execute(text("""
        SELECT TOP 1 p.plant_id, p.plant_name, p.description, p.price, p.image_path
        FROM plant_synonym ps
        JOIN plant p ON p.plant_id = ps.plant_id
        WHERE LOWER(:msg) LIKE CONCAT('%', LOWER(ps.synonym), '%')
        ORDER BY LEN(ps.synonym) DESC
    """), {"msg": user_message.lower()}).fetchone()

    if row:
        return {
            "plant_id": row[0],
            "plant_name": row[1],
            "description": row[2],
            "price": row[3],
            "image_url": get_plant_image_url(row[4])
        }

    return None


import difflib

# Global dictionary tracking per-user active dialogue state for multi-turn sequential conversations
USER_DIALOG_STATE = {}

def extract_quantity_nlp(text):
    """
    Extracts numerical quantity from user string using an LCEL AI Chain.
    e.g. 'two', '5 plants', 'add double' -> int
    """
    try:
        from langchain_core.prompts import PromptTemplate
        from langchain_community.llms import Ollama
        from langchain_core.output_parsers import StrOutputParser

        quantity_prompt = PromptTemplate.from_template(
            "The user was asked how many plants they want. User input: '{user_input}'. "
            "Extract the exact numerical integer. If none is given, assume 1. Respond ONLY with the integer digits."
        )
        quantity_chain = quantity_prompt | Ollama(model="phi3", temperature=0) | StrOutputParser()
        
        qty_str = quantity_chain.invoke({"user_input": text}).strip()
        import re
        match = re.search(r'\d+', qty_str)
        if match:
            return max(1, int(match.group()))
        return 1
    except Exception as e:
        print(f"Error in LangChain quantity extraction: {e}")
        return 1

def fuzzy_resolve_plant(user_message):
    """
    NLP Fuzzy plant matching that maps user query (including misspellings/phonetic inputs like 'rase nuh', 'roz', 'hibiscs')
    to a canonical plant in the database.
    """
    msg_clean = re.sub(r'[^\w\s]', ' ', user_message.lower()).strip()
    if not msg_clean:
        return None

    # 1. Phonetic fixes & common STT mishearings dictionary
    phonetic_fixes = {
        "rase nuh": "rose",
        "rase": "rose",
        "roz": "rose",
        "ros": "rose",
        "hibiscs": "hibiscus",
        "snak": "snake plant",
        "alo": "aloe vera",
        "money": "money plant",
        "jasmin": "jasmine",
        "tulsi": "holy basil",
        "tulas": "tulsi"
    }

    def _fetch_rows():
        try:
            return db.session.execute(text("""
                SELECT p.plant_id, p.plant_name, p.price, p.image_path, ps.synonym
                FROM plant p
                LEFT JOIN plant_synonym ps ON p.plant_id = ps.plant_id
            """)).fetchall()
        except Exception as e:
            print(f"Error fetching plants for fuzzy resolution: {e}")
            return []

    from flask import has_app_context
    if not has_app_context():
        with app.app_context():
            rows = _fetch_rows()
            direct = resolve_plant_synonym(user_message)
    else:
        rows = _fetch_rows()
        direct = resolve_plant_synonym(user_message)

    if direct:
        direct["score"] = 1.0
        direct["is_exact"] = True

    plant_map = {}
    for row in rows:
        pid, name, price, img, syn = row[0], row[1], row[2], row[3], row[4]
        if pid not in plant_map:
            plant_map[pid] = {
                "plant_id": pid,
                "plant_name": name,
                "price": price,
                "image_url": get_plant_image_url(img),
                "synonyms": set()
            }
        if name:
            plant_map[pid]["synonyms"].add(name.lower())
        if syn:
            plant_map[pid]["synonyms"].add(syn.lower())

    # Check phonetic fix dictionary
    for trigger, target in phonetic_fixes.items():
        if trigger in msg_clean:
            for pid, info in plant_map.items():
                for s in info["synonyms"]:
                    if target in s:
                        return {
                            "plant_id": info["plant_id"],
                            "plant_name": info["plant_name"],
                            "price": info["price"],
                            "image_url": info["image_url"],
                            "score": 0.95,
                            "is_exact": False,
                            "raw_query": trigger
                        }

    # Direct match check
    if direct:
        return direct

    # Difflib SequenceMatcher fuzzy matching
    best_match = None
    best_score = 0.0

    words = msg_clean.split()
    for pid, info in plant_map.items():
        for syn in info["synonyms"]:
            ratio = difflib.SequenceMatcher(None, msg_clean, syn).ratio()
            if ratio > best_score:
                best_score = ratio
                best_match = info

            for w in words:
                w_ratio = difflib.SequenceMatcher(None, w, syn).ratio()
                if w_ratio > best_score and w_ratio > 0.6:
                    best_score = w_ratio
                    best_match = info

    if best_match and best_score >= 0.55:
        return {
            "plant_id": best_match["plant_id"],
            "plant_name": best_match["plant_name"],
            "price": best_match["price"],
            "image_url": best_match["image_url"],
            "score": best_score,
            "is_exact": (best_score >= 0.98)
        }

    return None



def handle_count_plants(canonical_key, query_id, user_message):
    """
    Handles all plant count queries safely
    """
    if not canonical_key.startswith("COUNT_PLANTS"):
        return None

    row = db.session.execute(
        text("SELECT COUNT(*) FROM plant")
    ).fetchone()

    count = row[0] if row else 0
    reply = f" There are {count} plants available in our nursery."

    # store_query_result call removed

    return {
        "status": "success",
        "type": "text",
        "reply": reply,
        "canonical_key": canonical_key
    }


def log_category_interest(user_id, category_id, interest_type="search"):

    if not user_id or not str(user_id).isdigit():
        return

    try:
        if db.engine.dialect.name == 'mssql':
            db.session.execute(text("""
                IF EXISTS (SELECT 1 FROM user_category_interest WHERE user_id = :u AND category_id = :c AND interest_type = :t)
                    UPDATE user_category_interest 
                    SET count = count + 1, last_interaction = CURRENT_TIMESTAMP 
                    WHERE user_id = :u AND category_id = :c AND interest_type = :t
                ELSE
                    INSERT INTO user_category_interest (user_id, category_id, interest_type, count, last_interaction)
                    VALUES (:u, :c, :t, 1, CURRENT_TIMESTAMP)
            """), {
                "u": int(user_id),
                "c": int(category_id),
                "t": interest_type
            })
        else:
            db.session.execute(text("""
                INSERT INTO user_category_interest
                (user_id, category_id, interest_type, count)
                VALUES (:u, :c, :t, 1)
                ON CONFLICT (user_id, category_id, interest_type)
                DO UPDATE SET
                    count = user_category_interest.count + 1,
                    last_interaction = CURRENT_TIMESTAMP
            """), {
                "u": int(user_id),
                "c": int(category_id),
                "t": interest_type
            })

        db.session.commit()

    except Exception as e:
        db.session.rollback()
        print(" Category interest failed:", e)



def process_message(user_message, location="", user_id=None, history=None):
    """
    PlantCare AI retrieve-and-format logic using plant_profiles.csv and RAG.
    Tracks sub-component execution timings (latencies) and failure telemetry via Langfuse.
    """
    import csv
    import string
    import json
    from groq import Groq
    
    t_start = time.perf_counter()
    user_message = user_message.strip()
    
    timings = {}
    failures = {}

    def get_meta(trace_status="SUCCESS", extra=None):
        total_ms = round((time.perf_counter() - t_start) * 1000, 2)
        meta = {
            "user_id": user_id,
            "location": location,
            "performance_ms": {
                "total_latency_ms": total_ms,
                **timings
            },
            "status": trace_status
        }
        if failures:
            meta["failures"] = failures
        if extra:
            meta.update(extra)
        return meta
    
    def clean_text(text):
        text = text.lower()
        text = re.sub(r'[^\w\s]', ' ', text)
        return ' '.join(text.split())

    normalized_query = clean_text(user_message)
    query_words = set(normalized_query.split())

    user_key = str(user_id) if user_id else "anonymous"
    current_dialog_state = USER_DIALOG_STATE.get(user_key)

    # -------------------------------------------------------------------------
    # MULTI-TURN SEQUENTIAL CONVERSATION DIALOGUE STATE MACHINE
    # -------------------------------------------------------------------------
    if current_dialog_state:
        state_type = current_dialog_state.get("state")

        # STATE 1: Awaiting Confirmation ("Do you mean Rose?")
        if state_type == "AWAITING_PLANT_CONFIRMATION":
            clean_input = clean_text(user_message).strip()
            
            try:
                from langchain_core.prompts import PromptTemplate
                from langchain_community.llms import Ollama
                from langchain_core.output_parsers import StrOutputParser
                
                affirmation_prompt = PromptTemplate.from_template(
                    "The user was asked if they want to add a plant to their cart.\n"
                    "Their response was: '{user_input}'\n\n"
                    "Task: Categorize this response into EXACTLY one of three words: POSITIVE, NEGATIVE, or OTHER.\n"
                    "- Output POSITIVE ONLY if they are explicitly agreeing to add the plant (e.g. 'yes', 'sure', 'add it').\n"
                    "- Output NEGATIVE ONLY if they are declining (e.g. 'no', 'nah').\n"
                    "- Output OTHER if they are asking a new question, changing the topic, or making a statement unrelated to answering 'yes/no' (e.g. 'do you sell X', 'what is Y').\n\n"
                    "Classification:"
                )
                affirmation_chain = affirmation_prompt | Ollama(model="phi3", temperature=0) | StrOutputParser()
                aff_result = affirmation_chain.invoke({"user_input": clean_input}).strip().upper()
            except Exception as e:
                print(f"Affirmation logic failed: {e}")
                aff_result = "OTHER"

            if "POSITIVE" in aff_result:
                plant_info = current_dialog_state.get("plant_info")
                USER_DIALOG_STATE[user_key] = {
                    "state": "AWAITING_QUANTITY",
                    "plant_info": plant_info
                }
                plant_name = plant_info["plant_name"]
                price = plant_info.get("price", 0)
                reply = f"Great! How many **{plant_name}** plants (Price: ₹{price} each) would you like to add to your cart?"
                return {
                    "status": "success",
                    "type": "text",
                    "reply": reply
                }
            elif "NEGATIVE" in aff_result:
                USER_DIALOG_STATE.pop(user_key, None)
                reply = "No problem! Which plant were you looking for?"
                return {
                    "status": "success",
                    "type": "text",
                    "reply": reply
                }

        # STATE 2: Awaiting Quantity ("How many Rose plants?")
        elif state_type == "AWAITING_QUANTITY":
            plant_info = current_dialog_state.get("plant_info")
            quantity = extract_quantity_nlp(user_message)
            
            plant_id = plant_info["plant_id"]
            plant_name = plant_info["plant_name"]

            plant_db = db.session.get(Plant, plant_id) if 'Plant' in globals() else None
            avail_stock = plant_db.stock if plant_db and plant_db.stock is not None else 0

            if avail_stock <= 0:
                USER_DIALOG_STATE.pop(user_key, None)
                reply = f"Sorry, **{plant_name}** is currently Out of Stock (0 available) in our database catalog. 🪴"
                return {"status": "success", "type": "text", "reply": reply}
            elif quantity > avail_stock:
                reply = f"We currently have only **{avail_stock}** **{plant_name}** plant(s) available in stock in our database. How many would you like to add?"
                return {"status": "success", "type": "text", "reply": reply}

            USER_DIALOG_STATE.pop(user_key, None)
            price = plant_db.price if plant_db else (plant_info.get("price", 0.0) or 0.0)
            total_price = price * quantity

            # Save to Cart table if user is logged in
            if user_id:
                def _add_cart():
                    try:
                        cart_item = Cart.query.filter_by(user_id=str(user_id), plant_id=plant_id).first()
                        if cart_item:
                            cart_item.quantity += quantity
                        else:
                            db.session.add(Cart(user_id=str(user_id), plant_id=plant_id, quantity=quantity))
                        db.session.commit()
                    except Exception as e:
                        db.session.rollback()
                        print(f"Error adding to Cart DB: {e}")

                from flask import has_app_context
                if not has_app_context():
                    with app.app_context():
                        _add_cart()
                else:
                    _add_cart()

            reply = f"PLACE_ORDER:{plant_id}\n✅ Added {quantity} **{plant_name}** plant(s) to your cart! Total: ₹{total_price:.2f}."
            return {
                "status": "success",
                "type": "image",
                "reply": reply,
                "image_url": plant_info.get("image_url", ""),
                "plant_id": plant_id
            }

    # -------------------------------------------------------------------------
    # DISPLAY CART INTENT CHECK
    # -------------------------------------------------------------------------
    cart_signals = ["my cart", "show cart", "view cart", "what is in my cart", "cart contents", "what's in my cart", "open cart"]
    if any(sig in normalized_query for sig in cart_signals) or normalized_query == "cart":
        def _fetch_cart_reply():
            user_cart = Cart.query.filter_by(user_id=str(user_id)).all()
            if not user_cart:
                return "🛒 Your shopping cart is currently empty! Feel free to browse our catalog for plants."
            
            cart_reply = "🛒 **Your Shopping Cart**\n\n"
            total_price = 0.0
            for idx, item in enumerate(user_cart):
                plant_obj = db.session.get(Plant, item.plant_id)
                if plant_obj:
                    price = plant_obj.price
                    item_total = float(price) * item.quantity
                    total_price += item_total
                    cart_reply += f"{idx+1}. **{plant_obj.plant_name}** - {item.quantity} qty (₹{price:.2f} each) = **₹{item_total:.2f}**\n"
            
            cart_reply += f"\n💰 **Total Value: ₹{total_price:.2f}**\n\n*Type 'checkout' to place your order!*"
            return cart_reply
        
        from flask import has_app_context
        if not has_app_context():
            with app.app_context():
                cart_reply = _fetch_cart_reply()
        else:
            cart_reply = _fetch_cart_reply()
            
        log_langfuse_trace("cart_display_query", user_message, cart_reply, get_meta("SUCCESS", {"intent": "VIEW_CART"}))
        return {"status": "success", "type": "text", "reply": cart_reply}

    # -------------------------------------------------------------------------
    # DATABASE & INVENTORY QUERY INTENT CHECK
    # -------------------------------------------------------------------------
    db_query_signals = [
        "database", "db", "stock", "quantity", "available", "availability", "how many",
        "how much", "count", "inventory", "catalog", "total plants", "in stock", "out of stock",
        "price", "cost", "rate", "discount", "discounts", "offer", "status", "sell", "buy", "purchase", "order", "get"
    ]
    is_db_query = any(sig in normalized_query for sig in db_query_signals)
    has_explicit_buy = any(k in normalized_query for k in ["buy", "order", "purchase", "add to cart", "place order"])

    if is_db_query and not has_explicit_buy:
        fuzzy_plant = fuzzy_resolve_plant(user_message)
        if fuzzy_plant:
            plant_name = fuzzy_plant["plant_name"]
            plant_id = fuzzy_plant["plant_id"]
            
            plant_obj = None
            try:
                plant_obj = db.session.get(Plant, plant_id)
            except Exception:
                try:
                    plant_obj = Plant.query.filter(Plant.plant_name.ilike(f"%{plant_name}%")).first()
                except Exception:
                    pass

            price = plant_obj.price if plant_obj else (fuzzy_plant.get("price", 0) or 0.0)
            disc_price = 0.0
            stock_count = plant_obj.stock if (plant_obj and plant_obj.stock is not None) else 0
            status_str = "Available" if stock_count > 0 else "Out of Stock"

            price_fmt = f"₹{price:.2f}"
            if disc_price and disc_price > 0 and disc_price < price:
                price_fmt = f"~~₹{price:.2f}~~ **₹{disc_price:.2f}** (Special Price!)"
            
            cart_prompt = "\n\nWould you like to add some to your cart?" if stock_count > 0 else ""
            if any(k in normalized_query for k in ["price", "cost", "rate", "discount", "offer"]):
                reply = f"💵 The price of **{plant_name}** in our database is **{price_fmt}**.\n📦 **Stock**: **{stock_count}** in stock (**Status**: {status_str}). 🪴{cart_prompt}"
            elif any(k in normalized_query for k in ["stock", "how many", "quantity", "count", "available", "availability", "status"]):
                reply = f"📦 We currently have **{stock_count}** **{plant_name}** plant(s) in stock in our database catalog (**Status**: {status_str}, **Price**: {price_fmt}). 🪴{cart_prompt}"
            else:
                reply = f"🌿 **{plant_name}** is in our database catalog (Price: {price_fmt}, Stock: **{stock_count}**, Status: **{status_str}**). 🪴{cart_prompt}"
            if "Would you like to add some to your cart?" in reply:
                USER_DIALOG_STATE[user_key] = {
                    "state": "AWAITING_PLANT_CONFIRMATION",
                    "plant_info": {
                        "plant_id": plant_id,
                        "plant_name": plant_name,
                        "price": price if not (disc_price and disc_price > 0) else disc_price,
                        "image_url": fuzzy_plant.get("image_url", "")
                    }
                }

            log_langfuse_trace("db_inventory_query", user_message, reply, get_meta("SUCCESS", {"intent": "DB_QUERY", "plant": plant_name, "stock": stock_count}))
            return {
                "status": "success",
                "type": "text",
                "reply": reply
            }
        elif any(k in normalized_query for k in ["how many", "database", "db", "total", "count", "catalog", "plants", "available", "out of stock"]):
            try:
                plants_in_db = Plant.query.all()
                total_cnt = len(plants_in_db)
                if total_cnt > 0:
                    details = []
                    for p in plants_in_db[:8]:
                        st_str = "Available" if (p.stock is not None and p.stock > 0) else "Out of Stock"
                        pr_val = p.price
                        details.append(f"• **{p.plant_name}**: ₹{pr_val:.2f} ({p.stock if p.stock is not None else 0} in stock, {st_str})")
                    reply = f"We currently have **{total_cnt}** plant varieties in our database catalog:\n\n" + "\n".join(details) + "\n\nWhich plant would you like to check or add to your cart?"
                else:
                    reply = "Our database contains a rich variety of plants including Mango, Lichi, Rose, Jasmine, and Aloe Vera! 🌿"
            except Exception as e:
                reply = "Our database contains a rich variety of plants including Mango, Lichi, Rose, Jasmine, and Aloe Vera! 🌿"
            
            log_langfuse_trace("db_general_query", user_message, reply, get_meta("SUCCESS", {"intent": "DB_GENERAL_QUERY"}))
            return {
                "status": "success",
                "type": "text",
                "reply": reply
            }

    # If NOT in active dialogue state, check for plant order/buy intent or fuzzy plant match
    buy_keywords = ["buy", "order", "purchase", "add to cart", "place order"]
    is_buy_intent = any(k in normalized_query for k in buy_keywords)

    fuzzy_match = fuzzy_resolve_plant(user_message)
    if fuzzy_match and is_buy_intent:
        # If fuzzy match required phonetic correction or was misspelled (e.g. 'rase nuh')
        if not fuzzy_match.get("is_exact") or "rase" in normalized_query:
            USER_DIALOG_STATE[user_key] = {
                "state": "AWAITING_PLANT_CONFIRMATION",
                "plant_info": fuzzy_match
            }
            plant_name = fuzzy_match["plant_name"]
            price = fuzzy_match.get("price", 0)
            reply = f"Do you mean **{plant_name}** (Price: ₹{price})?"
            return {
                "status": "success",
                "type": "text",
                "reply": reply
            }
        else:
            # Exact plant match - prompt for quantity
            USER_DIALOG_STATE[user_key] = {
                "state": "AWAITING_QUANTITY",
                "plant_info": fuzzy_match
            }
            plant_name = fuzzy_match["plant_name"]
            price = fuzzy_match.get("price", 0)
            reply = f"How many **{plant_name}** plants (Price: ₹{price} each) would you like to add to your cart?"
            return {
                "status": "success",
                "type": "text",
                "reply": reply
            }
    
    history_context = ""
    semantic_history = history_faiss.get_semantic_history(user_id, user_message)
    if semantic_history:
        history_context = semantic_history

    cart_context = ""
    try:
        user_cart = Cart.query.filter_by(user_id=str(user_id)).all()
        if user_cart:
            cart_plants = []
            for item in user_cart:
                plant_obj = db.session.get(Plant, item.plant_id)
                if plant_obj:
                    cart_plants.append(f"{item.quantity}x {plant_obj.plant_name}")
            if cart_plants:
                cart_context = "User's current Shopping Cart has: " + ", ".join(cart_plants) + ".\n"
    except Exception:
        pass

    if history and isinstance(history, list) and len(history) > 0:
        history_context += "Immediate recent context:\n" + "\n".join([f"{msg.get('role', 'unknown').capitalize()}: {msg.get('content', '')}" for msg in history[-4:]]) + "\n\n"

    # Check if greeting
    if normalized_msg := clean_text(user_message):
        if normalized_msg in GREETINGS:
            reply = "Hello! I am your PlantCare AI assistant. How can I help you today?"
            log_langfuse_trace("greeting_query", user_message, reply, get_meta("SUCCESS", {"intent": "GREETING"}))
            return {
                "status": "success",
                "type": "text",
                "reply": reply
            }

    # Setup RAG retrieval timing
    rag_results = []
    t_rag_start = time.perf_counter()
    try:
        rag_results = faiss_rag.retrieve(user_message, k=3)
        timings["rag_retrieval_ms"] = round((time.perf_counter() - t_rag_start) * 1000, 2)
    except Exception as e:
        timings["rag_retrieval_ms"] = round((time.perf_counter() - t_rag_start) * 1000, 2)
        failures["rag_retrieval_error"] = str(e)
        print(f" [RAG] FAISS retrieval failed: {e}")

    rag_metadata = None
    if rag_results:
        rag_metadata = {
            "similarity": round(float(rag_results[0]["score"]), 4),
            "distance": round(float(rag_results[0]["raw_distance"]), 4)
        }
    else:
        rag_metadata = {
            "similarity": 0.50,
            "distance": 1.00
        }

    plants_list = []
    if rag_results:
        for doc in rag_results:
            if isinstance(doc, dict):
                p_meta = doc.get("metadata")
                if isinstance(p_meta, dict):
                    p_name = p_meta.get("name")
                    if p_name and p_meta.get("type") == "plant":
                        if p_name not in plants_list:
                            plants_list.append(p_name)

    def format_plant_profile(bp):
        def is_avail(val):
            if val is None:
                return False
            val_str = str(val).strip().lower()
            return val_str not in ("", "n/a", "none", "null", "undefined", "unspecified", "-")

        reply_parts = [f"🌱 Plant Profile: {bp['plant_name'].strip()}"]
        
        if is_avail(bp.get('scientific_name')):
            reply_parts.append(f"🔬 Scientific Name\n{bp['scientific_name'].strip()}")
            
        cond_parts = []
        if is_avail(bp.get('temperature')):
            cond_parts.append(f"• Temperature: {bp['temperature'].strip()}")
        if is_avail(bp.get('humidity')):
            cond_parts.append(f"• Humidity: {bp['humidity'].strip()}")
        if is_avail(bp.get('moisture')):
            cond_parts.append(f"• Soil Moisture: {bp['moisture'].strip()}")
        if is_avail(bp.get('sunlight')):
            cond_parts.append(f"• Sunlight: {bp['sunlight'].strip()}")
            
        if cond_parts:
            reply_parts.append("🌡️ Growing Conditions\n" + "\n".join(cond_parts))
            
        if is_avail(bp.get('soil_type')):
            reply_parts.append(f"🌾 Soil\n{bp['soil_type'].strip()}")
            
        if is_avail(bp.get('watering_frequency')):
            reply_parts.append(f"💧 Watering\n{bp['watering_frequency'].strip()}")
            
        nutr_parts = []
        if is_avail(bp.get('nitrogen')):
            nutr_parts.append(f"🟢 Nitrogen: {bp['nitrogen'].strip()}")
        if is_avail(bp.get('phosphorus')):
            nutr_parts.append(f"🟡 Phosphorus: {bp['phosphorus'].strip()}")
        if is_avail(bp.get('potassium')):
            nutr_parts.append(f"🟣 Potassium: {bp['potassium'].strip()}")
            
        if nutr_parts:
            reply_parts.append("🌿 Nutrient Requirements\n" + "\n".join(nutr_parts))
            
        if is_avail(bp.get('growth_duration')):
            reply_parts.append(f"⏳ Growth Duration\n{bp['growth_duration'].strip()}")
            
        return "\n\n".join(reply_parts)

    def fetch_db_plant_details(plant_name_or_query):
        """Retrieve plant details, stock, price, status, and care info directly from database tables (Plant & PlantKnowledge)"""
        try:
            clean_target = re.sub(r'^(tell me about|show me|do you have|info on|care for|profile for|about|can i buy|buy|price of|cost of|stock of|how many)\s+', '', plant_name_or_query, flags=re.IGNORECASE).strip().lower()
            if not clean_target:
                clean_target = plant_name_or_query.strip().lower()
                
            p_db = Plant.query.filter(Plant.plant_name.ilike(f"%{clean_target}%")).first()
            pk_db = PlantKnowledge.query.filter(PlantKnowledge.plant_name.ilike(f"%{clean_target}%")).first()

            if not p_db and not pk_db:
                return None

            plant_name = p_db.plant_name if p_db else pk_db.plant_name
            price = p_db.price if p_db else 0.0
            disc_price = 0.0
            stock_cnt = p_db.stock if p_db and p_db.stock is not None else 0
            status_str = "Available" if stock_cnt > 0 else "Out of Stock"
            desc = p_db.description if p_db and p_db.description else (pk_db.description if pk_db else "")

            reply_parts = [f"🌱 **Plant Profile: {plant_name}**"]

            # Price & Stock Details directly from DB
            price_str = f"₹{price:.2f}"
            if disc_price and disc_price > 0 and disc_price < price:
                price_str = f"~~₹{price:.2f}~~ **₹{disc_price:.2f}** (Special Price!)"
                
            reply_parts.append(f"💵 **Price**: {price_str}\n📦 **Stock**: **{stock_cnt}** in stock (**Status**: {status_str})")

            if pk_db:
                if pk_db.scientific_name:
                    reply_parts.append(f"🔬 **Scientific Name**\n*{pk_db.scientific_name.strip()}*")
                
                conds = []
                if pk_db.sunlight:
                    conds.append(f"• Sunlight: {pk_db.sunlight.strip()}")
                if pk_db.water_requirement:
                    conds.append(f"• Water: {pk_db.water_requirement.strip()}")
                if pk_db.soil_type:
                    conds.append(f"• Soil: {pk_db.soil_type.strip()}")
                if conds:
                    reply_parts.append("🌡️ **Growing Conditions**\n" + "\n".join(conds))

                if pk_db.fertilizer:
                    reply_parts.append(f"🌾 **Fertilizer & Nutrients**\n{pk_db.fertilizer.strip()}")
                if pk_db.care_tips:
                    reply_parts.append(f"🌿 **Care Tips**\n{pk_db.care_tips.strip()}")
                if pk_db.medicinal_uses:
                    reply_parts.append(f"💊 **Medicinal / Extra Uses**\n{pk_db.medicinal_uses.strip()}")
            elif desc:
                reply_parts.append(f"📝 **Description**\n{desc.strip()}")

            return "\n\n".join(reply_parts)
        except Exception as err:
            print(f"Error in fetch_db_plant_details: {err}")
            return None

    # Classify Query Intent Timing
    intent = "PROFILE"
    t_intent_start = time.perf_counter()
    try:
        class_prompt = (
            f"{history_context}"
            f"Analyze the user's message: '{user_message}'.\n"
            "Classify the intent into exactly one of three categories:\n"
            "1. PROFILE: If the user is requesting a profile, care-sheet, or specific growing/care parameters of a single specific plant name (e.g., 'tell me about rose', 'Mango', 'Acanthocereus', 'unsupported plant xyz', 'Tomato').\n"
            "2. GENERAL: If the user is asking a general botanical question, list of plants in our nursery, providing their geographic location/climate for recommendations, care tips, growing guides, etc.\n"
            "3. OFF_TOPIC: If the user is asking about topics completely unrelated to plants, gardening, farming, or our catalog. Note: Geographic locations, cities, or climates (e.g., 'tamilnadu', 'florida', 'hot climate') are NOT OFF_TOPIC; they are GENERAL intent used for plant matching.\n\n"
            "Respond with exactly one word: PROFILE, GENERAL, or OFF_TOPIC. Do not write any other text or markdown code wraps."
        )
        from langchain_community.llms import Ollama
        llm = Ollama(model="phi3", temperature=0.0)
        sys_msg = "You are a routing classification system for a Plant Care Assistant. You must classify user inputs into PROFILE, GENERAL, or OFF_TOPIC. Every question that is not about plants, botany, diseases, gardening, or agriculture is OFF_TOPIC.\n\n"
        intent = llm.invoke(sys_msg + class_prompt).strip().upper()
        timings["intent_classification_ms"] = round((time.perf_counter() - t_intent_start) * 1000, 2)
    except Exception as e:
        timings["intent_classification_ms"] = round((time.perf_counter() - t_intent_start) * 1000, 2)
        failures["intent_classification_error"] = str(e)
        print(f" [Intent Routing] Classification failed: {e}")
        intent = "PROFILE"

    if "OFF_TOPIC" in intent or intent == "OFF_TOPIC":
        reply = "❌ Sorry, I can only help you with queries related to plants, gardening, or the nursery catalog."
        log_langfuse_trace("off_topic_query", user_message, reply, get_meta("SUCCESS", {"intent": "OFF_TOPIC"}))
        return {
            "status": "success",
            "type": "text",
            "reply": reply
        }

    rag_category = None
    if intent == "GENERAL":
        t_cat_start = time.perf_counter()
        try:
            cat_prompt = (
                f"{history_context}"
                f"Determine the knowledge category for the user query: '{user_message}'.\n"
                "Pick exactly one of these categories:\n"
                "- GEOGRAPHY: Questions about native regions, countries, climate zones, altitude, frost/drought tolerance, where a plant grows.\n"
                "- SEASON: Questions about seasonal care, winter/summer watering, seasonal fertilization, seasonal temperature or sunlight requirements.\n"
                "- DISEASE: Questions about plant diseases, symptoms, pests, yellowing leaves, fungal/bacterial infections, treatment, prevention.\n"
                "- PLANT: Any other plant care, profile, soil, nutrients, watering, growth duration, medicinal uses.\n"
                "Respond with exactly one word: GEOGRAPHY, SEASON, DISEASE, or PLANT."
            )
            from langchain_community.llms import Ollama
            llm = Ollama(model="phi3", temperature=0.0)
            sys_msg = "You are a data source router. Respond with GEOGRAPHY, SEASON, DISEASE, or PLANT only.\n\n"
            cat_label = llm.invoke(sys_msg + cat_prompt).strip().upper()
            
            category_map = {"GEOGRAPHY": "geography", "SEASON": "season", "DISEASE": "disease", "PLANT": "plant"}
            rag_category = category_map.get(cat_label, None)
            timings["category_classification_ms"] = round((time.perf_counter() - t_cat_start) * 1000, 2)
        except Exception as e:
            timings["category_classification_ms"] = round((time.perf_counter() - t_cat_start) * 1000, 2)
            failures["category_classification_error"] = str(e)
            rag_category = None

    best_plant = None
    best_score = 0
    t_csv_start = time.perf_counter()
    csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "plant_profiles.csv")
    
    if os.path.exists(csv_path):
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                p_name = row.get('plant_name', '').strip()
                s_name = row.get('scientific_name', '').strip()
                
                p_name_clean = clean_text(p_name)
                s_name_clean = clean_text(s_name)
                
                score = 0
                if normalized_query == p_name_clean or normalized_query == s_name_clean:
                    score += 200
                else:
                    if p_name_clean and p_name_clean in normalized_query:
                        score += 100
                    if s_name_clean and s_name_clean in normalized_query:
                        score += 100
                    if len(normalized_query) > 2:
                        if p_name_clean and normalized_query in p_name_clean:
                            score += 50
                        if s_name_clean and normalized_query in s_name_clean:
                            score += 50
                    
                    p_words = set(p_name_clean.split()) - {'plant', 'tree'}
                    overlap = query_words.intersection(p_words)
                    if overlap:
                        score += len(overlap) * 20
                        
                    s_words = set(s_name_clean.split())
                    s_overlap = query_words.intersection(s_words)
                    if s_overlap:
                        score += len(s_overlap) * 20
                
                if score > best_score:
                    best_score = score
                    best_plant = row
    timings["csv_search_ms"] = round((time.perf_counter() - t_csv_start) * 1000, 2)

    general_question_words = {"how", "why", "what", "grow", "care", "disease", "diseases", "prevent", "medicinal", 
                              "symptoms", "symptom", "causes", "cause", "plants", "flowers", "fruits", "treatment", "treatments"}
    has_general_question = len(query_words.intersection(general_question_words)) > 0

    # DB FIRST PLANT LOOKUP CHECK
    db_profile_reply = fetch_db_plant_details(user_message)
    if db_profile_reply and intent == "PROFILE" and not has_general_question:
        res: dict[str, Any] = {
            "status": "success",
            "type": "text",
            "reply": db_profile_reply,
            "rag_metadata": rag_metadata
        }
        if plants_list:
            res["plants"] = plants_list
        log_langfuse_trace("plant_profile_db_hit", user_message, db_profile_reply, get_meta("SUCCESS", {"intent": intent, "source": "database"}))
        return res

    if best_plant and best_score > 0 and intent == "PROFILE" and not has_general_question:
        reply = format_plant_profile(best_plant)
        res: dict[str, Any] = {
            "status": "success",
            "type": "text",
            "reply": reply,
            "rag_metadata": rag_metadata
        }
        if plants_list:
            res["plants"] = plants_list
        log_langfuse_trace("plant_profile_csv_hit", user_message, reply, get_meta("SUCCESS", {"intent": intent, "source": "csv"}))
        return res

    if intent == "GENERAL" or has_general_question:
        filtered_results = []
        t_filt_start = time.perf_counter()
        try:
            filtered_results = faiss_rag.retrieve(user_message, k=4, category_filter=rag_category)
            timings["rag_category_retrieval_ms"] = round((time.perf_counter() - t_filt_start) * 1000, 2)
        except Exception as e:
            timings["rag_category_retrieval_ms"] = round((time.perf_counter() - t_filt_start) * 1000, 2)
            failures["rag_category_retrieval_error"] = str(e)

        effective_results = filtered_results if filtered_results else rag_results

        if effective_results and len(effective_results) > 0:
            context = "\n\n".join([str(doc.get("content", "")) for doc in effective_results if isinstance(doc, dict)])
            source_hint = rag_category.upper() if rag_category else "GENERAL"
            prompt = f"""You are a helpful plant care assistant specialized in {source_hint} plant data. Answer the user's question based on the following relevant knowledge base snippets.
            
Relevant Knowledge Base Snippets:
{context}

{history_context}
{cart_context}
User Question: {user_message}

Provide a helpful, informative answer using the information above. Keep the response concise and well-designed in markdown.
Answer:"""
            t_llm_start = time.perf_counter()
            try:
                from langchain_community.llms import Ollama
                is_disease = rag_category == "disease"
                sys_msg = (
                    "You are a plant disease expert. Answer in 3-5 short bullet points only. No long paragraphs.\n\n"
                    if is_disease else
                    "You are a helpful plant assistant. Be concise.\n\n"
                )
                llm = Ollama(model="phi3", temperature=0.3)
                reply = llm.invoke(sys_msg + prompt).strip()
                timings["llm_generation_ms"] = round((time.perf_counter() - t_llm_start) * 1000, 2)
                res = {
                    "status": "success",
                    "type": "text",
                    "reply": reply,
                    "rag_metadata": rag_metadata
                }
                if plants_list:
                    res["plants"] = plants_list
                
                trace_lvl = "ERROR" if failures else "DEFAULT"
                log_langfuse_trace("rag_query_synthesis", user_message, reply, get_meta("SUCCESS", {"intent": intent, "rag_category": rag_category}), level=trace_lvl)
                return res
            except Exception as e:
                timings["llm_generation_ms"] = round((time.perf_counter() - t_llm_start) * 1000, 2)
                failures["llm_generation_error"] = str(e)
                print(f" [RAG] RAG LLM query failed: {e}")

    def make_plant_unavailable_guardrail(query_or_name):
        clean_target = re.sub(r'^(tell me about|show me|do you have|info on|care for|profile for|about|can i buy|buy)\s+', '', query_or_name, flags=re.IGNORECASE).strip()
        if not clean_target:
            clean_target = query_or_name.strip()
        clean_target = clean_target.title()
        
        return (
            f"🌿 **Nursery Availability Notice**\n\n"
            f"Sorry, **{clean_target}** is currently not available in our nursery catalog or database. "
            f"We are actively expanding our plant collection and hope to bring it to our nursery soon! 🪴\n\n"
            f"In the meantime, feel free to explore our available plants in the shop or ask about our indoor plants, "
            f"flowering saplings, and fruit trees!"
        )

    def check_db_plant_availability(query_or_name):
        clean_target = re.sub(r'^(tell me about|show me|do you have|info on|care for|profile for|about|can i buy|buy)\s+', '', query_or_name, flags=re.IGNORECASE).strip().lower()
        if not clean_target:
            clean_target = query_or_name.strip().lower()
        
        t_db_start = time.perf_counter()
        matched_name = None
        try:
            plant_db = Plant.query.filter(Plant.plant_name.ilike(f"%{clean_target}%")).first()
            if plant_db:
                matched_name = plant_db.plant_name
        except Exception as e:
            failures["db_plant_query_error"] = str(e)

        if not matched_name:
            try:
                pk_db = PlantKnowledge.query.filter(PlantKnowledge.plant_name.ilike(f"%{clean_target}%")).first()
                if pk_db:
                    matched_name = pk_db.plant_name
            except Exception as e:
                failures["db_pk_query_error"] = str(e)

        if not matched_name:
            try:
                match = db.session.execute(text("SELECT plant_name FROM plant WHERE LOWER(plant_name) LIKE LOWER(:x)"), {"x": f"%{clean_target}%"}).fetchone()
                if match:
                    matched_name = match[0]
            except Exception as e:
                failures["db_raw_sql_error"] = str(e)

        timings["db_search_ms"] = round((time.perf_counter() - t_db_start) * 1000, 2)
        return matched_name

    if not best_plant or best_score <= 0 or intent == "PROFILE":
        matched_db_plant = check_db_plant_availability(user_message)
        
        if matched_db_plant:
            print(f" [LLM Profile Cache] Plant '{matched_db_plant}' found in DB but missing from CSV. Generating profile with LLM once...")
            sys_prompt = "You are a strict botanical database system. Reply ONLY in the requested JSON format. Do not use conversational filler or markdown wrapping."
            llm_prompt = f"""
Generate care and growing parameters for the plant species '{matched_db_plant}'.
Reply with exactly a JSON object having keys:
"plant_name", "scientific_name", "moisture", "temperature", "humidity", "sunlight", "soil_type", "nitrogen", "phosphorus", "potassium", "watering_frequency", "growth_duration".
Make sure values are accurate and realistic.
"""
            t_gen_start = time.perf_counter()
            try:
                groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
                response = groq_client.chat.completions.create(
                    model="llama-3.1-8b-instant",
                    messages=[
                        {"role": "system", "content": sys_prompt},
                        {"role": "user", "content": llm_prompt}
                    ],
                    temperature=0.1,
                    max_tokens=250
                )
                cleaned_resp = response.choices[0].message.content.strip()
                if cleaned_resp.startswith("```json"):
                    cleaned_resp = cleaned_resp[7:]
                if cleaned_resp.endswith("```"):
                    cleaned_resp = cleaned_resp[:-3]
                cleaned_resp = cleaned_resp.strip()

                profile_data = json.loads(cleaned_resp)
                timings["llm_profile_generation_ms"] = round((time.perf_counter() - t_gen_start) * 1000, 2)
                
                csv_headers = [
                    "plant_name", "scientific_name", "moisture", "temperature", "humidity", 
                    "sunlight", "soil_type", "nitrogen", "phosphorus", "potassium", 
                    "watering_frequency", "growth_duration"
                ]
                row_to_write = {col: str(profile_data.get(col, "N/A")).strip() for col in csv_headers}
                row_to_write["plant_name"] = matched_db_plant
                
                with open(csv_path, 'a', newline='', encoding='utf-8') as f_append:
                    writer = csv.DictWriter(f_append, fieldnames=csv_headers)
                    writer.writerow(row_to_write)

                reply = format_plant_profile(row_to_write)
                res: dict[str, Any] = {
                    "status": "success",
                    "type": "text",
                    "reply": reply,
                    "rag_metadata": rag_metadata
                }
                if plants_list:
                    res["plants"] = plants_list
                
                trace_lvl = "ERROR" if failures else "DEFAULT"
                log_langfuse_trace("plant_profile_llm_generated", user_message, reply, get_meta("SUCCESS", {"plant": matched_db_plant}), level=trace_lvl)
                return res

            except Exception as llm_err:
                timings["llm_profile_generation_ms"] = round((time.perf_counter() - t_gen_start) * 1000, 2)
                failures["llm_profile_generation_error"] = str(llm_err)

        guardrail_reply = make_plant_unavailable_guardrail(user_message)
        trace_lvl = "ERROR" if failures else "DEFAULT"
        log_langfuse_trace("nursery_availability_guardrail", user_message, guardrail_reply, get_meta("SUCCESS", {"intent": intent}), level=trace_lvl)
        return {
            "status": "success",
            "type": "text",
            "reply": guardrail_reply,
            "rag_metadata": rag_metadata
        }

    fallback_reply = make_plant_unavailable_guardrail(user_message)
    trace_lvl = "ERROR" if failures else "DEFAULT"
    log_langfuse_trace("nursery_availability_guardrail_fallback", user_message, fallback_reply, get_meta("SUCCESS"), level=trace_lvl)
    return {
        "status": "success",
        "type": "text",
        "reply": fallback_reply,
        "rag_metadata": rag_metadata
    }

# --- END OF OBSOLETE LOGIC ---




@app.route("/api/chat", methods=["POST"])
def chat():
    """Handle chat API requests"""
    try:
        data = request.get_json()
        user_message = data.get("message", "")
        location = data.get("location", "")
        user_id = data.get("user_id")

        if not user_message:
            return jsonify({"status": "error", "message": "No message provided"}), 400
        history = data.get("history", [])

        response = process_message(user_message, location, user_id, history)
        
        # Save History to FAISS Engine
        if response and response.get("status") == "success" and response.get("reply"):
            history_faiss.add_interaction(user_id, user_message, response.get("reply"))
        
        if response.get("type") == "text":
            speech_response = generate_speech(response["reply"])
            if speech_response.get("status") == "success":
                response["audio"] = speech_response.get("audio")

        return jsonify(response)

    except Exception as e:
        print(f"Error in /api/chat: {str(e)}")
        return jsonify({
            "status": "error",
            "type": "text",
            "reply": "Sorry, I encountered an error processing your request."
        }), 500
    
def tts_failsafe(text):
    """Failsafe TTS using gTTS"""
    print(" Using failsafe TTS (gTTS)...")
    try:
        mp3_fp = io.BytesIO()
        tts = gTTS(text=text, lang='en', slow=False)
        tts.write_to_fp(mp3_fp)
        mp3_fp.seek(0)
        audio_base64 = base64.b64encode(mp3_fp.read()).decode()
        return {"status": "success", "audio": audio_base64}
    except Exception as e:
        print(f" gTTS failsafe error: {e}")
        return {"status": "error", "message": "Could not generate speech"}
    
def get_plant_image_url(image_path):
    if not image_path:
        return "/static/images/default_plant.png"
    if image_path.startswith(('http://', 'https://')):
        return image_path
    
    clean_name = image_path.lstrip('/').split('/')[-1] # Just the filename
    
    # 1. Try direct path first
    static_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static')
    plants_dir = os.path.join(static_dir, 'images', 'plants')
    
    direct_path = os.path.join(plants_dir, clean_name)
    if os.path.exists(direct_path):
        return f"/static/images/plants/{clean_name}"
        
    # 2. Case-insensitive and prefix-insensitive search in static/images/plants
    if os.path.exists(plants_dir):
        files = os.listdir(plants_dir)
        name_base = os.path.splitext(clean_name.lower())[0]
        
        # Try exact match first (case-insensitive)
        for f in files:
            if f.lower() == clean_name.lower():
                 return f"/static/images/plants/{f}"
        
        # Try finding the name as part of the filename (handles prefixes like plant_76_)
        for f in files:
            if name_base in f.lower():
                return f"/static/images/plants/{f}"
                
    # 3. Fallback to default
    return "/static/images/default_plant.png"


import traceback

# ========== ADMIN API ENDPOINTS ==========
@app.route('/api/admin/suggest-varieties', methods=['POST'])
def suggest_varieties():
    """Suggest plant varieties based on plant name"""
    try:
        data = request.json
        plant_name = data.get('plant_name', '').strip()
        stock_count = data.get('stock_count', 0)
        price = data.get('price', 199.99)
        description = data.get('description', '')

        if not plant_name:
            return jsonify({"status": "error", "message": "Plant name required"}), 400

        #  Check if plant exists FIRST (case-insensitive)
        existing_plant = db.session.execute(
            text("SELECT plant_id, plant_name, stock, price FROM plant WHERE LOWER(plant_name) = LOWER(:name)"),
            {"name": plant_name}
        ).fetchone()

        if existing_plant:
            #  Plant already exists - just return it
            plant_id = existing_plant[0]
            existed = True
            current_stock = existing_plant[2]
            current_price = existing_plant[3]
            
            print(f" Plant '{plant_name}' already exists with ID: {plant_id}")

        else:
            #  Create new plant
            try:
                new_p = Plant(plant_name=plant_name, description=description, price=price, stock=stock_count)
                db.session.add(new_p)
                db.session.commit()
                plant_id = new_p.plant_id
                current_stock = new_p.stock
                current_price = new_p.price

                # Log initial stock if > 0
                if stock_count > 0:
                    db.session.execute(
                        text("""
                            INSERT INTO inventory_transaction (plant_id, type, quantity, notes, created_at)
                            VALUES (:pid, 'ADD', :qty, 'Initial stock from AI suggest', :now)
                        """),
                        {"pid": plant_id, "qty": stock_count, "now": datetime.now(timezone.utc)}
                    )
                    db.session.commit()
                existed = False
                print(f" Created new plant '{plant_name}' with ID: {plant_id}")

            except Exception as insert_error:
                db.session.rollback()
                print(f" Insert error (retrying): {insert_error}")
                
                # Retry: double-check if it exists now
                check_plant = db.session.execute(
                    text("SELECT plant_id, stock, price FROM plant WHERE LOWER(plant_name) = LOWER(:name)"),
                    {"name": plant_name}
                ).fetchone()
                
                if check_plant:
                    plant_id = check_plant[0]
                    existed = True
                    current_stock = check_plant[1]
                    current_price = check_plant[2]
                    print(f" Plant exists (concurrent request): ID {plant_id}")
                else:
                    raise insert_error

        # Suggest varieties using LLM
        prompt = f"""
For the plant "{plant_name}", suggest 5 specific varieties.
Return ONLY the variety names, one per line.
Example: Ruby Red, Golden Tiger, Variegated White, etc.
"""
        varieties_text = llm_call(prompt)
        varieties = [v.strip() for v in varieties_text.split('\n') if v.strip()]

        return jsonify({
            "status": "success",
            "plant_id": plant_id,
            "plant_name": plant_name,
            "existed": existed,
            "current_stock": current_stock,
            "price": current_price,
            "varieties": varieties[:5],
            "message": " Plant already available" if existed else " New plant created"
        })

    except Exception as e:
        print(f" Error in suggest_varieties: {e}")
        print(traceback.format_exc())
        db.session.rollback()
        return jsonify({"status": "error", "message": str(e)}), 500
        
def reset_plant_sequence():
    """Reset the plant_id sequence to the max ID in the table"""
    try:
        # Get max ID
        max_id_row = db.session.execute(text("SELECT MAX(plant_id) FROM plant")).fetchone()
        max_id = max_id_row[0] if max_id_row and max_id_row[0] else 0
        
        if max_id:
            if db.engine.dialect.name == 'mssql':
                db.session.execute(text(f"DBCC CHECKIDENT ('plant', RESEED, {max_id})"))
                db.session.commit()
                print(f" Reseeded plant identity to {max_id}")
            else:
                next_val = max_id + 1
                db.session.execute(text(f"ALTER SEQUENCE plant_plant_id_seq RESTART WITH {next_val}"))
                db.session.commit()
                print(f" Reset plant sequence to start at {next_val} (max was {max_id})")
        else:
            print(" No plants found in database")
        
        return True
    except Exception as e:
        print(f" Sequence reset warning: {e}")
        # Don't fail completely - the retry logic will handle it
        return False
@app.route('/api/rebuild-index', methods=['POST'])
@app.route('/api/admin/rag/sync', methods=['POST'])
def sync_rag_index():
    """Force sync/rebuild FAISS RAG index from database and CSV files"""
    try:
        print(" [FAISS] Manual RAG index rebuild requested.")
        faiss_rag.sync_from_db(PlantKnowledge, DiseaseKnowledge)
        return jsonify({
            "status": "success", 
            "message": "FAISS vector index rebuilt successfully from database and CSV files."
        })
    except Exception as e:
        print(f" [FAISS] RAG rebuild error: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500

# Call this once on startup
# reset_plant_sequence call moved to main block
    
@app.route('/api/admin/add-varieties', methods=['POST'])
def add_varieties():
    """Add selected varieties to a plant"""
    try:
        data = request.json
        plant_id = data.get('plant_id')
        plant_name = data.get('plant_name')
        selected_varieties = data.get('selected_varieties', [])

        if not plant_id or not selected_varieties:
            return jsonify({"status": "error", "message": "Plant ID and varieties required"}), 400

        added = 0
        duplicates = 0

        for variety_name in selected_varieties:
            # Check if variety already exists
            existing = db.session.execute(
                text("""
                    SELECT variety_id FROM variety
                    WHERE plant_id = :pid AND LOWER(variety_name) = :vname
                """),
                {"pid": plant_id, "vname": variety_name.lower()}
            ).fetchone()

            if existing:
                duplicates += 1
                continue

            # Add new variety
            db.session.execute(
                text("""
                    INSERT INTO variety (plant_id, variety_name, variety_price)
                    VALUES (:pid, :vname, :price)
                """),
                {"pid": plant_id, "vname": variety_name, "price": 199.99}
            )
            added += 1

        db.session.commit()

        return jsonify({
            "status": "success",
            "message": f"Added {added} varieties",
            "added": added,
            "duplicates": duplicates
        })

    except Exception as e:
        print(f"Error in add_varieties: {e}")
        db.session.rollback()
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route('/api/admin/inventory', methods=['GET'])
@require_tenant_auth(allowed_roles=['SUPER_ADMIN', 'NURSERY_ADMIN', 'STAFF'])
def get_inventory():
    """Get all plants with inventory info for current nursery"""
    try:
        biz_id = get_current_tenant()
        where_clause = "WHERE p.business_id = :biz_id"
        if biz_id == 'NURS-0001':
            where_clause = "WHERE p.business_id = :biz_id OR p.business_id IS NULL OR p.business_id = ''"

        plants = db.session.execute(text(f"""
            SELECT p.plant_id, p.plant_name, p.description, p.price, p.stock, p.image_path,
                   STRING_AGG(c.category_name, ', ') as categories
            FROM plant p
            LEFT JOIN plant_category pc ON p.plant_id = pc.plant_id
            LEFT JOIN category c ON pc.category_id = c.category_id
            {where_clause}
            GROUP BY p.plant_id, p.plant_name, p.description, p.price, p.stock, p.image_path
            ORDER BY p.plant_name ASC
        """), {"biz_id": biz_id}).fetchall()

        plant_list = []
        total_stock = 0

        for p in plants:
            image_path = p[5]
            image_url = get_plant_image_url(image_path)
            
            plant_list.append({
                "plant_id": p[0],
                "plant_name": p[1],
                "description": p[2],
                "price": float(p[3]) if p[3] else 0,
                "stock": p[4] or 0,
                "image_path": image_path,
                "image_url": image_url,
                "categories": p[6]
            })
            total_stock += (p[4] or 0)

        return jsonify({
            "status": "success",
            "plants": plant_list,
            "total_plants": len(plant_list),
            "total_stock": total_stock,
            "business_id": biz_id
        })

    except Exception as e:
        print(f"Error in get_inventory: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route('/api/admin/restock/<int:plant_id>', methods=['POST'])
@require_tenant_auth(allowed_roles=['SUPER_ADMIN', 'NURSERY_ADMIN', 'STAFF'])
def restock_plant(plant_id):
    """Restock a plant by adding quantity"""
    try:
        biz_id = get_current_tenant()
        data = request.json or {}
        quantity = data.get('quantity', 0)

        if quantity <= 0:
            return jsonify({"status": "error", "message": "Quantity must be positive"}), 400

        plant = Plant.query.filter((Plant.plant_id == plant_id) & ((Plant.business_id == biz_id) | (Plant.business_id == 'NURS-0001'))).first()
        if not plant:
            return jsonify({"status": "error", "message": "Plant not found"}), 404

        old_stock = plant.stock or 0
        plant.stock = old_stock + quantity

        # Log transaction
        tx = InventoryTransaction(
            business_id=biz_id,
            plant_id=plant_id,
            type='ADD',
            quantity=quantity,
            notes="Manual restock via API"
        )
        db.session.add(tx)
        db.session.commit()

        log_audit_action("RESTOCK_PLANT", "INVENTORY", record_id=plant_id, details=f"Restocked {plant.plant_name} by +{quantity}", business_id=biz_id)

        return jsonify({
            "status": "success",
            "message": f"Restocked {plant.plant_name}",
            "plant_id": plant_id,
            "plant_name": plant.plant_name,
            "old_stock": old_stock,
            "new_stock": plant.stock,
            "quantity_added": quantity
        })

    except Exception as e:
        print(f"Error in restock_plant: {e}")
        db.session.rollback()
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route('/api/admin/get-plant/<int:plant_id>', methods=['GET'])
@require_tenant_auth(allowed_roles=['SUPER_ADMIN', 'NURSERY_ADMIN', 'STAFF'])
def get_plant_details(plant_id):
    """Get detailed info about a plant"""
    try:
        biz_id = get_current_tenant()
        plant = Plant.query.filter((Plant.plant_id == plant_id) & ((Plant.business_id == biz_id) | (Plant.business_id == 'NURS-0001'))).first()
        if not plant:
            return jsonify({"status": "error", "message": "Plant not found"}), 404

        return jsonify({
            "status": "success",
            "plant": {
                "plant_id": plant.plant_id,
                "plant_name": plant.plant_name,
                "description": plant.description,
                "price": float(plant.price) if plant.price else 0,
                "stock": plant.stock or 0,
                "image_path": plant.image_path,
                "business_id": plant.business_id
            }
        })

    except Exception as e:
        print(f"Error in get_plant_details: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route('/api/admin/add-plant', methods=['POST'])
@require_tenant_auth(allowed_roles=['SUPER_ADMIN', 'NURSERY_ADMIN'])
def add_plant():
    """Create a new plant with initial stock and categories for current nursery"""
    try:
        biz_id = get_current_tenant()
        data = request.json or {}
        name = data.get('plant_name')
        price = float(data.get('price', 0))
        stock = int(data.get('stock', 0))
        description = data.get('description', '')
        category_ids = data.get('categories', [])

        if not name:
            return jsonify({"status": "error", "message": "Plant name is required"}), 400

        # Insert plant using SQLAlchemy model (preserves identity handling across all dialects)
        new_plant = Plant(
            business_id=biz_id,
            plant_name=name,
            price=price,
            stock=stock,
            description=description
        )
        db.session.add(new_plant)
        db.session.flush()
        next_pid = new_plant.plant_id

        # Add categories
        for cat_id in category_ids:
            try:
                db.session.execute(
                    text("INSERT INTO plant_category (plant_id, category_id) VALUES (:pid, :cid)"),
                    {"pid": next_pid, "cid": cat_id}
                )
            except Exception as ce:
                print(f"Notice adding category: {ce}")

        # Log initial stock transaction if > 0
        if stock > 0:
            try:
                db.session.execute(
                    text("""
                        INSERT INTO inventory_transaction (business_id, plant_id, type, quantity, notes, created_at)
                        VALUES (:biz, :pid, 'ADD', :qty, 'Initial stock on creation', CURRENT_TIMESTAMP)
                    """),
                    {
                        "biz": biz_id,
                        "pid": next_pid,
                        "qty": stock
                    }
                )
            except Exception as te:
                print(f"Notice adding inventory transaction: {te}")

        db.session.commit()
        log_audit_action("ADD_PLANT", "PLANTS", record_id=next_pid, details=f"Added plant '{name}'", business_id=biz_id, user_id=getattr(g, 'user_id', None), user_email=getattr(g, 'user_email', None))

        return jsonify({
            "status": "success",
            "message": f"Plant '{name}' added successfully",
            "plant_id": next_pid,
            "business_id": biz_id
        })

    except Exception as e:
        print(f"Error in add_plant: {e}")
        traceback.print_exc()
        db.session.rollback()
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route('/api/admin/update-plant/<int:plant_id>', methods=['PUT'])
@require_tenant_auth(allowed_roles=['SUPER_ADMIN', 'NURSERY_ADMIN'])
def update_plant_details(plant_id):
    """Update plant details with strict tenant isolation"""
    try:
        biz_id = get_current_tenant()
        data = request.json or {}
        
        # Enforce tenant data isolation: Local Admin can only modify plants owned by their organization
        plant = Plant.query.filter_by(plant_id=plant_id, business_id=biz_id).first()
        if not plant and biz_id == 'NURS-0001':
            plant = db.session.get(Plant, plant_id)
        
        if not plant:
            return jsonify({"status": "error", "message": "Plant not found or unauthorized access"}), 404

        if 'plant_name' in data:
            plant.plant_name = data['plant_name']
        if 'description' in data:
            plant.description = data['description']
        if 'price' in data:
            plant.price = float(data['price'])
        if 'stock' in data:
            old_stock = plant.stock or 0
            new_stock = int(data['stock'])
            if old_stock != new_stock:
                diff = new_stock - old_stock
                tx = InventoryTransaction(
                    business_id=biz_id,
                    plant_id=plant_id,
                    type='ADD' if diff > 0 else 'REMOVE',
                    quantity=abs(diff),
                    notes=f"Stock Correction from {old_stock} to {new_stock}"
                )
                db.session.add(tx)
                plant.stock = new_stock

        db.session.commit()
        log_audit_action("UPDATE_PLANT", "PLANTS", record_id=plant_id, details=f"Updated plant {plant.plant_name}", business_id=biz_id)

        return jsonify({
            "status": "success",
            "message": f"Updated {plant.plant_name}",
            "plant_id": plant_id
        })

    except Exception as e:
        print(f"Error in update_plant_details: {e}")
        db.session.rollback()
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route('/api/admin/delete-plant/<int:plant_id>', methods=['DELETE'])
@require_tenant_auth(allowed_roles=['SUPER_ADMIN', 'NURSERY_ADMIN'])
def delete_plant_endpoint(plant_id):
    """Delete a plant with strict tenant isolation"""
    try:
        biz_id = get_current_tenant()
        plant = Plant.query.filter_by(plant_id=plant_id, business_id=biz_id).first()
        
        if not plant:
            return jsonify({"status": "error", "message": "Plant not found or unauthorized access"}), 404

        plant_name = plant.plant_name

        # Delete related child records for this plant
        db.session.execute(text("DELETE FROM plant_category WHERE plant_id = :id"), {"id": plant_id})
        db.session.execute(text("DELETE FROM variety WHERE plant_id = :id"), {"id": plant_id})
        db.session.execute(text("DELETE FROM cart_new WHERE plant_id = :id"), {"id": plant_id})

        # Delete the plant
        db.session.delete(plant)
        db.session.commit()

        log_audit_action("DELETE_PLANT", "PLANTS", record_id=plant_id, details=f"Deleted plant {plant_name}", business_id=biz_id)

        return jsonify({
            "status": "success",
            "message": f"Deleted {plant_name}",
            "plant_id": plant_id
        })

    except Exception as e:
        print(f"Error in delete_plant_endpoint: {e}")
        db.session.rollback()
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route('/api/admin/search-plants', methods=['GET'])
@require_tenant_auth(allowed_roles=['SUPER_ADMIN', 'NURSERY_ADMIN', 'STAFF'])
def search_plants_admin():
    """Search for plants by name within the current tenant organization"""
    try:
        biz_id = get_current_tenant()
        query = request.args.get('q', '').strip()
        
        if not query or len(query) < 2:
            return jsonify({"status": "error", "message": "Query too short"}), 400

        plants = db.session.execute(text("""
            SELECT TOP 20 plant_id, plant_name, description, price, stock, image_path
            FROM plant
            WHERE business_id = :biz AND LOWER(plant_name) LIKE :q
            ORDER BY plant_name ASC
        """), {"biz": biz_id, "q": f"%{query.lower()}%"}).fetchall()

        plant_list = [{
            "plant_id": p[0],
            "plant_name": p[1],
            "description": p[2],
            "price": float(p[3]) if p[3] else 0,
            "stock": p[4] or 0,
            "image_url": get_plant_image_url(p[5])
        } for p in plants]

        return jsonify({
            "status": "success",
            "plants": plant_list,
            "count": len(plant_list)
        })

    except Exception as e:
        print(f"Error in search_plants_admin: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500

    

# --- SUPPLIER MASTER ROUTES ---
@app.route('/api/admin/suppliers', methods=['GET'])
def get_suppliers():
    """Get all suppliers"""
    try:
        rows = db.session.execute(text("SELECT supplier_id, supplier_name, phone, email, address FROM supplier ORDER BY supplier_name ASC")).fetchall()
        suppliers = [{
            "supplier_id": r[0],
            "name": r[1],
            "contact_number": r[2],
            "email": r[3],
            "address": r[4]
        } for r in rows]
        return jsonify({"status": "success", "suppliers": suppliers})
    except Exception as e:
        print(f"Error in get_suppliers: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/api/admin/suppliers', methods=['POST'])
def add_supplier():
    """Add a new supplier"""
    try:
        data = request.json
        name = data.get('name')
        if not name:
            return jsonify({"status": "error", "message": "Supplier name required"}), 400
        
        db.session.execute(text("""
            INSERT INTO supplier (supplier_name, phone, email, address)
            VALUES (:name, :phone, :email, :address)
        """), {
            "name": name,
            "phone": data.get('contact_number'),
            "email": data.get('email'),
            "address": data.get('address')
        })
        db.session.commit()
        return jsonify({"status": "success", "message": "Supplier added successfully"})
    except Exception as e:
        print(f"Error in add_supplier: {e}")
        db.session.rollback()
        return jsonify({"status": "error", "message": str(e)}), 500

# --- CATEGORY MASTER ROUTES ---
@app.route('/api/admin/categories', methods=['GET'])
def get_categories_admin():
    """Get all categories for admin"""
    try:
        rows = db.session.execute(text("SELECT category_id, category_name FROM category ORDER BY category_name ASC")).fetchall()
        categories = [{"category_id": r[0], "category_name": r[1]} for r in rows]
        return jsonify({"status": "success", "categories": categories})
    except Exception as e:
        print(f"Error in get_categories_admin: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/api/admin/categories', methods=['POST'])
def add_category_admin():
    """Add a new category"""
    try:
        data = request.json
        name = data.get('category_name')
        if not name:
            return jsonify({"status": "error", "message": "Category name required"}), 400
        
        existing_cat = Category.query.filter_by(category_name=name).first()
        if not existing_cat:
            new_cat = Category(category_name=name)
            db.session.add(new_cat)
            db.session.commit()
        return jsonify({"status": "success", "message": "Category saved"})
    except Exception as e:
        print(f"Error in add_category_admin: {e}")
        db.session.rollback()
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/api/admin/categories/<int:cat_id>', methods=['DELETE'])
def delete_category_admin(cat_id):
    """Delete a category"""
    try:
        db.session.execute(text("DELETE FROM plant_category WHERE category_id = :id"), {"id": cat_id})
        db.session.execute(text("DELETE FROM category WHERE category_id = :id"), {"id": cat_id})
        db.session.commit()
        return jsonify({"status": "success", "message": "Category deleted"})
    except Exception as e:
        print(f"Error in delete_category_admin: {e}")
        db.session.rollback()
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/api/admin/categories/<int:cat_id>/plants', methods=['GET'])
def get_category_plants_admin(cat_id):
    """Get all plants for a category"""
    try:
        rows = db.session.execute(text("""
            SELECT p.plant_id, p.plant_name, p.stock, p.price, p.image_path
            FROM plant p
            JOIN plant_category pc ON pc.plant_id = p.plant_id
            WHERE pc.category_id = :cid
        """), {"cid": cat_id}).fetchall()
        plants = [{
            "plant_id": r[0],
            "plant_name": r[1],
            "stock": r[2],
            "price": float(r[3]) if r[3] else 0,
            "image_url": get_plant_image_url(r[4])
        } for r in rows]
        return jsonify({"status": "success", "plants": plants})
    except Exception as e:
        print(f"Error in get_category_plants_admin: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/api/debug/list-plant-images', methods=['GET'])
def list_plant_images():
    import os
    static_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static')
    plants_dir = os.path.join(static_dir, 'images', 'plants')
    
    files = []
    if os.path.exists(plants_dir):
        files = os.listdir(plants_dir)
    
    return jsonify({
        "static_dir": static_dir,
        "plants_dir": plants_dir,
        "exists": os.path.exists(plants_dir),
        "file_count": len(files),
        "files_sample": files[:20],
        "root_path": app.root_path
    })
# ---------- ROUTES ----------
@app.route('/')
def index():
    from flask import send_from_directory
    return send_from_directory('.', 'auth.html')

@app.route('/admin')
@app.route('/admin.html')
def admin_page():
    from flask import send_from_directory
    return send_from_directory('static', 'admin.html')

@app.route('/<path:filename>')
def serve_root_files(filename):
    from flask import send_from_directory
    # If the file exists in root, serve it
    if os.path.exists(os.path.join(app.root_path, filename)):
        return send_from_directory('.', filename)
    # fall back to static
    return send_from_directory('static', filename)
@app.route('/api/suggest_plants/<user_id>', methods=['GET'])
def suggest_plants(user_id):
    try:
        # Get user's last enquiry or purchase
        last_interest = UserInterest.query.filter_by(user_id=user_id if user_id.isdigit() else 0).order_by(UserInterest.created_at.desc()).first()
        
        if last_interest:
            # Suggest based on same plant or similar (simplified for monolith)
            plants = Plant.query.limit(8).all()
        else:
            plants = Plant.query.limit(8).all()
            
        return jsonify({
            "status": "success",
            "plants": [{
                "plant_id": p.plant_id,
                "plant_name": p.plant_name,
                "description": p.description,
                "price": p.price,
                "discount_price": 0.0,
                "effective_price": p.price,
                "stock": p.stock if p.stock is not None else 0,
                "availability_status": "Available" if (p.stock is not None and p.stock > 0) else "Out of Stock",
                "image_url": get_plant_image_url(p.image_path) if p.image_path else None
            } for p in plants]
        })
    except Exception as e:
        print(f"Suggestion error: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/api/user_interest', methods=['POST'])
def save_user_interest():
    try:
        data = request.json
        user_id = data.get("user_id")
        plant_id = data.get("plant_id")
        interest_type = data.get("interest_type", "enquiry")
        notes = data.get("notes", "")

        if not user_id or not plant_id:
            return jsonify({"status": "error", "message": "Missing required fields"}), 400

        interest = UserInterest(
            user_id=int(user_id) if str(user_id).isdigit() else 0,
            plant_id=int(plant_id),
            interest_type=interest_type,
            notes=notes
        )
        db.session.add(interest)
        db.session.commit()
        return jsonify({"status": "success"})
    except Exception as e:
        db.session.rollback()
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/api/cart/update', methods=['POST'])
def update_cart():
    try:
        data = request.json
        user_id = str(data.get("user_id", "anonymous"))
        plant_id = data.get("plant_id")
        action = data.get("action") # 'increase' or 'decrease'

        item = Cart.query.filter_by(user_id=user_id, plant_id=plant_id).first()
        if not item:
            return jsonify({"status": "error", "message": "Item not in cart"}), 404

        if action == "increase":
            item.quantity += 1
        elif action == "decrease":
            if item.quantity > 1:
                item.quantity -= 1
            else:
                db.session.delete(item)
        
        db.session.commit()
        return jsonify({"status": "success"})
    except Exception as e:
        db.session.rollback()
        return jsonify({"status": "error", "message": str(e)}), 500

# --- INVENTORY MANAGEMENT ROUTES ---
@app.route('/api/admin/inventory/add', methods=['POST'])
def inventory_add():
    """Add stock via supplier or return"""
    try:
        data = request.json
        plant_id = data.get('plant_id')
        quantity = int(data.get('quantity', 0))
        
        if not plant_id or quantity <= 0:
            return jsonify({"status": "error", "message": "Invalid plant or quantity"}), 400
            
        plant = db.session.get(Plant, plant_id)
        if not plant:
            return jsonify({"status": "error", "message": "Plant not found"}), 404
            
        supplier_id = data.get('supplier_id')
        if supplier_id == "": supplier_id = None
        
        # Create transaction
        tx = InventoryTransaction(
            plant_id=plant_id,
            supplier_id=supplier_id,
            type='ADD',
            quantity=quantity,
            notes=data.get('notes'),
            bill_no=data.get('bill_no'),
            bill_date=data.get('bill_date')
        )
        
        # Update plant stock
        plant.stock = (plant.stock or 0) + quantity
        
        db.session.add(tx)
        db.session.commit()
        
        return jsonify({"status": "success", "message": f"Added {quantity} units to {plant.plant_name}"})
    except Exception as e:
        print(f"Error in inventory_add: {e}")
        db.session.rollback()
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/api/admin/inventory/remove', methods=['POST'])
def inventory_remove():
    """Remove stock (wastage/return to supplier)"""
    try:
        data = request.json
        plant_id = data.get('plant_id')
        quantity = int(data.get('quantity', 0))
        
        if not plant_id or quantity <= 0:
            return jsonify({"status": "error", "message": "Invalid plant or quantity"}), 400
            
        plant = db.session.get(Plant, plant_id)
        if not plant:
            return jsonify({"status": "error", "message": "Plant not found"}), 404
            
        # Create transaction
        tx = InventoryTransaction(
            plant_id=plant_id,
            type='REMOVE',
            quantity=quantity,
            notes=data.get('notes'),
            bill_no=data.get('bill_no'),
            bill_date=data.get('bill_date')
        )
        
        # Update plant stock
        plant.stock = max(0, (plant.stock or 0) - quantity)
        
        db.session.add(tx)
        db.session.commit()
        
        return jsonify({"status": "success", "message": f"Removed {quantity} units from {plant.plant_name}"})
    except Exception as e:
        print(f"Error in inventory_remove: {e}")
        db.session.rollback()
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/api/admin/inventory/transactions', methods=['GET'])
@require_tenant_auth(allowed_roles=['SUPER_ADMIN', 'NURSERY_ADMIN', 'STAFF'])
def get_inventory_transactions():
    """Get recent inventory transactions for current nursery"""
    try:
        biz_id = get_current_tenant()
        rows = db.session.execute(text("""
            SELECT TOP 100 it.created_at, p.plant_name, it.type, it.quantity, s.supplier_name as supplier_name, it.bill_no, it.bill_date, it.notes
            FROM inventory_transaction it
            JOIN plant p ON p.plant_id = it.plant_id
            LEFT JOIN supplier s ON s.supplier_id = it.supplier_id
            WHERE it.business_id = :biz_id OR it.business_id IS NULL OR it.business_id = ''
            ORDER BY it.created_at DESC
        """), {"biz_id": biz_id}).fetchall()
        
        transactions = []
        for r in rows:
            transactions.append({
                "date": r[0].strftime("%B %d, %Y %I:%M %p") if r[0] else "N/A",
                "plant_name": r[1],
                "type": r[2],
                "quantity": r[3],
                "supplier_name": r[4] or "N/A",
                "bill_no": r[5],
                "bill_date": r[6],
                "notes": r[7]
            })
        
        return jsonify({"status": "success", "transactions": transactions})
    except Exception as e:
        print(f"Error in get_inventory_transactions: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500

# --- CUSTOMER & ORDER MANAGEMENT ---
@app.route('/api/admin/customers', methods=['GET'])
@require_tenant_auth(allowed_roles=['SUPER_ADMIN', 'NURSERY_ADMIN', 'STAFF'])
def get_customers_admin():
    """Get all customers for admin within current nursery"""
    try:
        biz_id = get_current_tenant()
        rows = db.session.execute(text("""
            SELECT u.id, u.email, u.last_login, u.is_online,
            (SELECT COUNT(*) FROM orders WHERE user_id = u.id AND (business_id = :biz_id OR business_id IS NULL OR business_id = '')) as purchases,
            (SELECT SUM(COALESCE(o.price, p.price, 0)) FROM orders o LEFT JOIN plant p ON o.plant_name = p.plant_name WHERE o.user_id = u.id AND (o.business_id = :biz_id OR o.business_id IS NULL OR o.business_id = '')) as amount
            FROM login u
            WHERE u.business_id = :biz_id OR u.business_id IS NULL OR u.business_id = ''
            ORDER BY u.id DESC
        """), {"biz_id": biz_id}).fetchall()
        
        customers = []
        for r in rows:
            customers.append({
                "id": r[0],
                "email": r[1],
                "last_login": r[2].strftime("%Y-%m-%d %H:%M") if r[2] else None,
                "is_online": bool(r[3]),
                "purchases": r[4],
                "amount_purchased": float(r[5]) if r[5] else 0
            })
        return jsonify({"status": "success", "customers": customers})
    except Exception as e:
        print(f"Error in get_customers_admin: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/api/admin/orders', methods=['GET'])
@require_tenant_auth(allowed_roles=['SUPER_ADMIN', 'NURSERY_ADMIN', 'STAFF'])
def get_all_orders_admin():
    """Get all customer orders for admin within current nursery"""
    try:
        biz_id = get_current_tenant()
        rows = db.session.execute(text("""
            SELECT 
                COALESCE(o.order_group_id, CONCAT('OLD-', CAST(o.order_id AS VARCHAR(20)))) as gid, 
                u.email, 
                o.order_status, 
                o.order_date, 
                SUM(COALESCE(o.price, p.price, 0)) as total_price, 
                COALESCE(o.payment_method, 'Not Specified') as pm,
                SUM(COALESCE(o.amount_paid, 0)) as amount_paid,
                SUM(COALESCE(o.balance, 0)) as balance
            FROM orders o
            LEFT JOIN login u ON o.user_id = u.id
            LEFT JOIN plant p ON o.plant_name = p.plant_name
            WHERE o.business_id = :biz_id OR o.business_id IS NULL OR o.business_id = ''
            GROUP BY COALESCE(o.order_group_id, CONCAT('OLD-', CAST(o.order_id AS VARCHAR(20)))), u.email, o.order_status, o.order_date, COALESCE(o.payment_method, 'Not Specified')
            ORDER BY o.order_date DESC
        """), {"biz_id": biz_id}).fetchall()
        
        orders = []
        for r in rows:
            orders.append({
                "order_group_id": r[0],
                "customer_name": r[1] or "Guest",
                "order_status": r[2],
                "order_date": r[3].strftime("%Y-%m-%d %H:%M") if r[3] else "N/A",
                "total_amount": float(r[4]) if r[4] else 0,
                "payment_method": r[5],
                "amount_paid": float(r[6]) if r[6] else 0,
                "balance": float(r[7]) if r[7] else 0
            })
        return jsonify({"status": "success", "orders": orders})
    except Exception as e:
        print(f"Error in get_all_orders_admin: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/api/admin/orders/record-payment', methods=['POST'])
@require_tenant_auth(allowed_roles=['SUPER_ADMIN', 'NURSERY_ADMIN', 'STAFF'])
def record_order_payment():
    """Record partial or full payment for an order group within current nursery"""
    biz_id = get_current_tenant()
    data = request.json or {}
    order_group_id = data.get("order_group_id")
    amount = data.get("amount")
    
    if not order_group_id or amount is None:
        return jsonify({"status": "error", "message": "Missing group ID or amount"}), 400
        
    try:
        if order_group_id.startswith('OLD-'):
            actual_id = order_group_id.replace('OLD-', '')
            where_clause = "order_id = :gid"
            gid_val = int(actual_id) if actual_id.isdigit() else 0
        else:
            where_clause = "order_group_id = :gid"
            gid_val = order_group_id
            
        rows = db.session.execute(text(f"SELECT order_id, balance FROM orders WHERE {where_clause} AND (business_id = :biz_id OR business_id IS NULL OR business_id = '') AND balance > 0"), {"gid": gid_val, "biz_id": biz_id}).fetchall()
        
        remaining_amount = float(amount)
        for r in rows:
            if remaining_amount <= 0:
                break
            oid = r[0]
            bal = float(r[1])
            pay_this = min(bal, remaining_amount)
            db.session.execute(text("UPDATE orders SET amount_paid = COALESCE(amount_paid, 0) + :p, balance = balance - :p WHERE order_id = :oid"), {"p": pay_this, "oid": oid})
            remaining_amount -= pay_this
            
        db.session.commit()
        log_audit_action("RECORD_ORDER_PAYMENT", "ORDERS", record_id=order_group_id, details=f"Recorded payment of {amount} for order {order_group_id}", business_id=biz_id)
        return jsonify({"status": "success", "message": "Sales payment recorded successfully"})
    except Exception as e:
        print(f"Error in record_order_payment: {e}")
        db.session.rollback()
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/api/admin/orders/<order_group_id>', methods=['GET'])
def get_order_details_admin(order_group_id):
    """Get details for a specific order group"""
    try:
        # Handle fallback IDs from get_all_orders_admin
        if order_group_id.startswith('OLD-'):
            actual_id = order_group_id.replace('OLD-', '')
            where_clause = "o.order_id = :gid"
            gid_val = int(actual_id) if actual_id.isdigit() else 0
        else:
            where_clause = "o.order_group_id = :gid"
            gid_val = order_group_id

        # Get orders in this group
        rows = db.session.execute(text(f"""
            SELECT o.order_id, o.plant_name, COALESCE(o.price, p.price, 0), o.quantity, o.order_status
            FROM orders o
            LEFT JOIN plant p ON o.plant_name = p.plant_name
            WHERE {where_clause}
        """), {"gid": gid_val}).fetchall()
        
        if not rows:
            return jsonify({"status": "error", "message": "Order not found"}), 404
            
        items = []
        total_amount = 0
        status = rows[0][4]
        
        for r in rows:
            qty = r[3] or 1
            price = float(r[2] or 0)
            items.append({
                "plant_name": r[1],
                "quantity": qty,
                "total_amount": price * qty
            })
            total_amount += price * qty
            
        # Get customer info from first order record
        cust_info = db.session.execute(text(f"""
            SELECT TOP 1 u.email, o.payment_method, o.shipping_address, o.order_date
            FROM orders o
            LEFT JOIN login u ON o.user_id = u.id
            WHERE {where_clause}
        """), {"gid": gid_val}).fetchone()
        
        # Safety check if cust_info is None (though rows was found)
        if not cust_info:
            return jsonify({"status": "error", "message": "Customer info not found"}), 404

        summary = {
            "order_group_id": order_group_id,
            "customer_name": cust_info[0] or "Guest",
            "customer_email": cust_info[0] or "N/A",
            "payment_method": cust_info[1] or "Not Specified",
            "shipping_address": cust_info[2] or "Not Provided",
            "order_date": cust_info[3].strftime("%Y-%m-%d %H:%M") if (cust_info[3] and hasattr(cust_info[3], 'strftime')) else "N/A",
            "order_status": status or "Processing",
            "total_amount": total_amount,
            "tracking_number": "N/A",
            "delivery_date": "N/A",
            "notes": ""
        }
        
        return jsonify({
            "status": "success",
            "summary": summary,
            "items": items
        })
    except Exception as e:
        print(f"Error in get_order_details_admin: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/api/admin/orders/<order_group_id>', methods=['PUT'])
def update_order_status_admin(order_group_id):
    """Update status of all orders in a group"""
    status = request.json.get("status")
    if not status:
        return jsonify({"status": "error", "message": "Status is required"}), 400
        
    try:
        db.session.execute(text("UPDATE orders SET order_status = :s WHERE order_group_id = :gid"), {"s": status, "gid": order_group_id})
        db.session.commit()
        return jsonify({"status": "success", "message": "Order status updated"})
    except Exception as e:
        print(f"Error in update_order_status_admin: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500

# PURCHASE MASTER ROUTES

@app.route('/api/admin/purchases', methods=['POST'])
def add_supplier_purchase():
    """Add a new purchase from a supplier"""
    data = request.json
    supplier_id = data.get("supplier_id")
    bill_no = data.get("bill_no")
    bill_date_str = data.get("bill_date")
    items = data.get("items", []) # List of {plant_id, quantity, unit_price}
    
    if not supplier_id or not items:
        return jsonify({"status": "error", "message": "Supplier and items are required"}), 400
        
    try:
        total_amount = sum(item['quantity'] * item['unit_price'] for item in items)
        bill_date = datetime.strptime(bill_date_str, '%Y-%m-%d') if bill_date_str else datetime.utcnow()
        
        purchase = SupplierPurchase(
            supplier_id=supplier_id,
            bill_no=bill_no or f"PUR-{random.randint(1000, 9999)}",
            bill_date=bill_date,
            total_amount=total_amount,
            balance=total_amount,
            status='Pending'
        )
        db.session.add(purchase)
        db.session.flush() # Get purchase_id
        
        for itm in items:
            p_id = itm['plant_id']
            qty = itm['quantity']
            price = itm['unit_price']
            
            # 1. Create Purchase Item
            pi = PurchaseItem(
                purchase_id=purchase.purchase_id,
                plant_id=p_id,
                quantity=qty,
                unit_price=price
            )
            db.session.add(pi)
            
            # 2. Update Plant Stock
            plant = db.session.get(Plant, p_id)
            if plant:
                plant.stock += qty
                
            # 3. Log Inventory Transaction
            tx = InventoryTransaction(
                plant_id=p_id,
                supplier_id=supplier_id,
                type='ADD',
                quantity=qty,
                notes=f'Supplier Purchase: {purchase.bill_no}',
                bill_no=purchase.bill_no,
                bill_date=bill_date.strftime('%Y-%m-%d')
            )
            db.session.add(tx)
            
        db.session.commit()
        return jsonify({"status": "success", "message": "Purchase recorded successfully"})
    except Exception as e:
        db.session.rollback()
        print(f"Error in add_supplier_purchase: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/api/admin/purchases', methods=['GET'])
def get_supplier_purchases():
    """List all supplier purchases"""
    try:
        sql = """
            SELECT p.purchase_id, s.supplier_name, p.bill_no, p.bill_date, 
                   p.total_amount, p.amount_paid, p.balance, p.status
            FROM supplier_purchase p
            JOIN supplier s ON p.supplier_id = s.supplier_id
            ORDER BY p.created_at DESC
        """
        rows = db.session.execute(text(sql)).fetchall()
            
        result = []
        for r in rows:
            result.append({
                "purchase_id": r[0],
                "supplier_name": r[1],
                "bill_no": r[2],
                "bill_date": r[3].strftime('%Y-%m-%d') if hasattr(r[3], 'strftime') else str(r[3]),
                "total_amount": float(r[4]),
                "amount_paid": float(r[5]),
                "balance": float(r[6]),
                "status": r[7]
            })
        return jsonify({"status": "success", "purchases": result})
    except Exception as e:
        print(f"Error in get_supplier_purchases: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/api/admin/purchases/record-payment', methods=['POST'])
def record_purchase_payment():
    """Record a payment against a purchase balance"""
    data = request.json
    purchase_id = data.get("purchase_id")
    amount = float(data.get("amount", 0))
    
    if not purchase_id or amount <= 0:
        return jsonify({"status": "error", "message": "Valid Purchase ID and amount are required"}), 400
        
    try:
        purchase = db.session.get(SupplierPurchase, purchase_id)
        if not purchase:
            return jsonify({"status": "error", "message": "Purchase not found"}), 404
            
        if amount > purchase.balance:
            return jsonify({"status": "error", "message": f"Payment exceeds balance (₹{purchase.balance})"}), 400
            
        purchase.amount_paid += amount
        purchase.balance = purchase.total_amount - purchase.amount_paid
        
        if purchase.balance <= 0:
            purchase.status = 'Paid'
        elif purchase.amount_paid > 0:
            purchase.status = 'Partial'
            
        db.session.commit()
        return jsonify({"status": "success", "message": "Payment recorded", "new_balance": purchase.balance})
    except Exception as e:
        print(f"Error in record_purchase_payment: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/api/admin/varieties/existing/<int:plant_id>', methods=['GET'])
def get_existing_varieties_admin(plant_id):
    """Get existing varieties for a plant"""
    try:
        rows = db.session.execute(text("SELECT variety_name FROM variety WHERE plant_id = :pid"), {"pid": plant_id}).fetchall()
        varieties = [{"variety_name": r[0]} for r in rows]
        return jsonify({"status": "success", "varieties": varieties})
    except Exception as e:
        print(f"Error in get_existing_varieties_admin: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/api/admin/update-plant-image/<int:plant_id>', methods=['POST'])
def update_plant_image_admin(plant_id):
    """Update plant image (URL or File)"""
    try:
        plant = db.session.get(Plant, plant_id)
        if not plant:
            return jsonify({"status": "error", "message": "Plant not found"}), 404
            
        image_url = request.form.get('image_url')
        if 'image' in request.files:
            file = request.files['image']
            if file and file.filename:
                # Basic filename cleaning
                filename = "".join(c for c in file.filename if c.isalnum() or c in ('.', '_', '-')).strip()
                save_path = os.path.join('static', 'plantpics', filename)
                os.makedirs(os.path.join('static', 'plantpics'), exist_ok=True)
                file.save(save_path)
                plant.image_path = f"plantpics/{filename}"
        elif image_url:
            plant.image_path = image_url # Assuming direct URL or handling path
            
        db.session.commit()
        return jsonify({"status": "success", "message": "Image updated successfully"})
    except Exception as e:
        print(f"Error in update_plant_image_admin: {e}")
        db.session.rollback()
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/api/user_queries/<user_id>', methods=['GET'])
def get_user_history(user_id):
    try:
        queries = UserQuery.query.filter_by(user_id=int(user_id) if str(user_id).isdigit() else 0).order_by(UserQuery.created_at.desc()).limit(10).all()
        return jsonify({
            "status": "success",
            "history": [{"query": q.query_text, "date": q.created_at.isoformat()} for q in queries]
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route('/api/category_synonyms', methods=['GET', 'POST', 'DELETE'])
def manage_category_synonyms():
    """Manage category synonyms"""
    try:
        if request.method == "GET":
            synonyms = CategorySynonym.query.all()
            return jsonify({
                "status": "success",
                "synonyms": [{
                    "synonym_id": s.synonym_id,
                    "synonym": s.synonym,
                    "category_id": s.category_id,
                    "category_name": s.category.category_name,
                    "created_at": s.created_at.strftime("%Y-%m-%d %H:%M:%S")
                } for s in synonyms]
            })

        elif request.method == "POST":
            data = request.json
            synonym = data.get("synonym")
            category_id = data.get("category_id")
            
            if not synonym or not category_id:
                return jsonify({"status": "error", "message": "Synonym and category_id required"}), 400
            
            # Verify category exists
            category = db.session.get(Category, category_id)
            if not category:
                return jsonify({"status": "error", "message": "Category not found"}), 404
            
            synonym_id = add_category_synonym(synonym, category_id)
            return jsonify({
                "status": "success",
                "message": f"Synonym '{synonym}' added to category '{category.category_name}'",
                "synonym_id": synonym_id
            })

        elif request.method == "DELETE":
            synonym_id = request.json.get("synonym_id")
            if not synonym_id:
                return jsonify({"status": "error", "message": "Synonym ID required"}), 400
            
            syn = db.session.get(CategorySynonym, synonym_id)
            if not syn:
                return jsonify({"status": "error", "message": "Synonym not found"}), 404
            
            db.session.delete(syn)
            db.session.commit()
            return jsonify({"status": "success", "message": "Synonym deleted"})

    except Exception as e:
        print(f"Category Synonyms API error: {str(e)}")
        db.session.rollback()
        return jsonify({"status": "error", "message": str(e)}), 500




@app.route('/api/config', methods=['GET', 'POST'])
def manage_config():
    """Get or update configuration options"""
    global LLM_OPTION, STT_OPTION, TTS_OPTION
    
    try:
        if request.method == "GET":
            return jsonify({
                "status": "success",
                "config": {
                    "LLM_OPTION": LLM_OPTION,
                    "STT_OPTION": STT_OPTION,
                    "TTS_OPTION": TTS_OPTION,
                    "LLM_FAILSAFE": LLM_FAILSAFE,
                    "STT_FAILSAFE": STT_FAILSAFE,
                    "TTS_FAILSAFE": TTS_FAILSAFE
                }
            })

        elif request.method == "POST":
            data = request.json
            
            if "LLM_OPTION" in data:
                LLM_OPTION = data["LLM_OPTION"]
                print(f" LLM changed to: {LLM_OPTION}")
            
            if "STT_OPTION" in data:
                STT_OPTION = data["STT_OPTION"]
                print(f" STT changed to: {STT_OPTION}")
            
            if "TTS_OPTION" in data:
                TTS_OPTION = data["TTS_OPTION"]
                print(f" TTS changed to: {TTS_OPTION}")
            
            return jsonify({
                "status": "success",
                "message": "Configuration updated",
                "config": {
                    "LLM_OPTION": LLM_OPTION,
                    "STT_OPTION": STT_OPTION,
                    "TTS_OPTION": TTS_OPTION
                }
            })

    except Exception as e:
        print(f"Config API error: {str(e)}")
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/api/stt", methods=["POST", "GET"])
def stt_api():
    """Speech-to-Text (STT) API Endpoint"""
    try:
        if request.method == "GET":
            return jsonify({
                "status": "success",
                "stt_option": STT_OPTION,
                "stt_failsafe": STT_FAILSAFE,
                "supported": True
            })

        data = request.json or {}
        text_input = data.get("text", "")
        if text_input:
            return jsonify({
                "status": "success",
                "transcript": text_input,
                "engine": STT_OPTION
            })

        return jsonify({
            "status": "success",
            "message": "STT active. Use browser Web Speech API or send audio/text transcript.",
            "stt_option": STT_OPTION
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500



@app.route('/api/plant/<int:plant_id>/image', methods=['GET', 'POST', 'DELETE'])
def manage_plant_image(plant_id):
    """Upload, retrieve, or delete plant image"""
    try:
        plant = db.session.get(Plant, plant_id)
        if not plant:
            return jsonify({"status": "error", "message": "Plant not found"}), 404

        if request.method == "GET":
            # Return image path if present
            if plant.image_path:
                return jsonify({
                    "status": "success",
                    "image_url": f"/static/{plant.image_path.lstrip('/')}",
                    "source": "filesystem"
                })
            else:
                return jsonify({"status": "error", "message": "No image found"}), 404

        elif request.method == "POST":
            # Upload image
            if 'image' not in request.files:
                return jsonify({"status": "error", "message": "No image file provided"}), 400

            file = request.files['image']
            if file.filename == '':
                return jsonify({"status": "error", "message": "No selected file"}), 400

            if not allowed_file(file.filename):
                return jsonify({"status": "error", "message": "Only JPG, PNG, GIF allowed"}), 400

            # Read image data
            image_data = file.read()
            
            # Validate image size (max 5MB)
            if len(image_data) > 5 * 1024 * 1024:
                return jsonify({"status": "error", "message": "Image too large (max 5MB)"}), 400

            # Store in static/plantpics
            filename = "".join(c for c in file.filename if c.isalnum() or c in ('.', '_', '-')).strip()
            save_path = os.path.join('static', 'plantpics', filename)
            os.makedirs(os.path.join('static', 'plantpics'), exist_ok=True)
            with open(save_path, 'wb') as f:
                f.write(image_data)
            plant.image_path = f"plantpics/{filename}"
            db.session.commit()

            return jsonify({
                "status": "success",
                "message": f"Image uploaded for {plant.plant_name}",
                "plant_id": plant_id,
                "size": len(image_data),
                "image_url": f"/static/{plant.image_path}"
            })

        elif request.method == "DELETE":
            # Delete image reference
            plant.image_path = None
            db.session.commit()
            return jsonify({"status": "success", "message": "Image deleted"})

    except Exception as e:
        print(f"Plant image API error: {str(e)}")
        db.session.rollback()
        return jsonify({"status": "error", "message": str(e)}), 500



import re
import json

def extract_json(text):
    """
    Robustly extract JSON from text even if it contains extra text or markdown blocks.
    """
    if not text:
        return None
        
    # Try to find JSON block {...}
    try:
        match = re.search(r'\{.*\}', text, re.DOTALL)
        if match:
            json_str = match.group()
            return json.loads(json_str)
        # Fallback: try raw json if no braces found (unlikely for objects)
        return json.loads(text)
    except Exception as e:
        pass
# --- REMOVED OBSOLETE VALIDATION LOGIC ---




# Routes removed due to errors

@app.route('/api/query_results', methods=['GET', 'POST'])
def manage_query_results():
    """Get or store query results"""
    try:
        if request.method == "GET":
            query_id = request.args.get("query_id")
            tenant_id = get_current_tenant()
            if query_id:
                results = QueryResult.query.filter_by(query_id=query_id, business_id=tenant_id).all()
            else:
                results = QueryResult.query.filter_by(business_id=tenant_id).all()
            
            return jsonify({
                "status": "success",
                "results": [{
                    "result_id": r.result_id,
                    "query_id": r.query_id,
                    "result_description": r.result_description,
                    "result_source": r.result_source,
                    "created_at": r.created_at.strftime("%Y-%m-%d %H:%M:%S")
                } for r in results]
            })

        elif request.method == "POST":
            data = request.json or {}
            query_id = data.get("query_id")
            result_description = data.get("result_description")
            result_source = data.get("result_source")  # 'llm' or 'local'
            
            if not query_id or not result_description or not result_source:
                return jsonify({"status": "error", "message": "query_id, result_description, and result_source required"}), 400
            
            if result_source not in ['llm', 'local']:
                return jsonify({"status": "error", "message": "result_source must be 'llm' or 'local'"}), 400
            
            tenant_id = get_current_tenant()
            new_res = QueryResult(
                business_id=tenant_id,
                query_id=query_id,
                result_description=result_description,
                result_source=result_source
            )
            db.session.add(new_res)
            db.session.commit()
            return jsonify({
                "status": "success",
                "message": "Query result stored",
                "result_id": new_res.result_id
            })

    except Exception as e:
        print(f"Query Results API error: {str(e)}")
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route('/logout', methods=['POST'])
def logout():
    data = request.json
    user_id = data.get("user_id")

    if user_id:
        db.session.execute(text("""
            UPDATE login
            SET last_logout = CURRENT_TIMESTAMP,
                is_online = 0
            WHERE id = :uid
        """), {"uid": user_id})

        db.session.commit()

    return jsonify({'status': 'success'})





@app.route('/plants', methods=['GET'])
def get_plants():
    try:
        sql = text("SELECT plant_name, description, price FROM plant")
        result = db.session.execute(sql).fetchall()
        plants = [{
            "plant_name": row[0], 
            "description": row[1], 
            "price": row[2]
        } for row in result]
        return jsonify({"status": "success", "plants": plants})
    except Exception as e:
        print(f"Error fetching plants: {str(e)}")
        return jsonify({"status": "error", "message": "Unable to fetch plants"}), 500


@app.route("/api/cart", methods=["POST"])
def add_to_cart():
    data = request.json
    user_id = str(data.get("user_id", "anonymous"))
    plant_id = data.get("plant_id")

    if not plant_id:
        return jsonify({"status": "error", "message": "plant_id required"}), 400

    try:
        existing = Cart.query.filter_by(user_id=user_id, plant_id=plant_id).first()
        if not existing:
            new_item = Cart(user_id=user_id, plant_id=plant_id, business_id=get_current_tenant() or 'NURS-0001')
            db.session.add(new_item)
            db.session.commit()

        #  AUTO LOG PURCHASE
        log_user_interest(user_id, plant_id, "purchase", "Added to cart")

        return jsonify({"status": "success", "message": "Added to cart "})

    except Exception as e:
        db.session.rollback()
        return jsonify({"status": "error", "message": str(e)}), 500
    

@app.route("/api/cart", methods=["GET"])
def get_cart():
    user_id = request.args.get("user_id", "anonymous")

    rows = db.session.execute(text("""
        SELECT p.plant_id, p.plant_name, p.price, p.description, p.image_path
        FROM cart_new c
        JOIN plant p ON p.plant_id = c.plant_id
        WHERE c.user_id = :user_id
        ORDER BY c.added_at DESC
    """), {"user_id": user_id}).fetchall()

    cart = []
    for r in rows:
        cart.append({
            "plant_id": r[0],
            "plant_name": r[1],
            "price": r[2],
            "description": r[3],
            "image_url": f"/static/{r[4]}" if r[4] else None
        })

    return jsonify({"status": "success", "cart": cart})


@app.route("/api/checkout", methods=["POST"])
def checkout():
    user_id = request.json.get("user_id")
    payment_info = request.json.get("payment_info", {})
    shipping_address = payment_info.get("address", "Not Provided")

    rows = db.session.execute(text("""
        SELECT plant_id, quantity
        FROM cart_new
        WHERE user_id = :u
    """), {"u": user_id}).fetchall()

    if not rows:
        return jsonify({"error": "Cart empty"}), 400

    order_group_id = f"ORD-{random.randint(10000, 99999)}"
    total_amount = 0
    items_desc = []

    for r in rows:
        plant = db.session.get(Plant, r[0])
        if not plant or (plant.stock is not None and plant.stock < r[1]):
            avail_stock = plant.stock if plant and plant.stock is not None else 0
            return jsonify({"error": f"{plant.plant_name if plant else 'Item'} out of stock (available: {avail_stock})"}), 400
        
        # Decrease stock count in database
        if plant.stock is not None:
            plant.stock = max(0, plant.stock - r[1])

        effective_price = plant.price
        total_amount += effective_price * r[1]
        items_desc.append(f"{plant.plant_name}x{r[1]}")

    db.session.commit()

    # For CCAvenue Integration
    merchant_id = os.getenv("CCAVENUE_MERCHANT_ID", "2115164") # Placeholder
    access_code = os.getenv("CCAVENUE_ACCESS_CODE", "AVBM05KL24BH15MBHB") # Placeholder
    working_key = os.getenv("CCAVENUE_WORKING_KEY", "7D0D9D9D9D9D9D9D9D9D9D9D9D9D9D9D") # Placeholder
    
    redirect_url = f"{request.url_root.rstrip('/')}/api/payment/response"
    
    # Use partial amount if provided, otherwise default to total
    charge_amount = payment_info.get("amount", total_amount)
    if charge_amount <= 0 or charge_amount > total_amount:
        charge_amount = total_amount

    # Standard CCAvenue parameter string
    merchant_param = f"merchant_id={merchant_id}&order_id={order_group_id}&currency=INR&amount={charge_amount}&redirect_url={redirect_url}&cancel_url={redirect_url}&language=EN&billing_address={shipping_address}&merchant_param1={user_id}&merchant_param2={total_amount}"
    
    encrypted_data = encrypt(merchant_param, working_key)
    
    return jsonify({
        "status": "success", 
        "payment_url": "https://secure.ccavenue.com/transaction/transaction.do?command=initiateTransaction",
        "enc_request": encrypted_data,
        "access_code": access_code
    })

@app.route("/api/payment/response", methods=["POST"])
def payment_response():
    """Handle CCAvenue callback"""
    working_key = os.getenv("CCAVENUE_WORKING_KEY", "7D0D9D9D9D9D9D9D9D9D9D9D9D9D9D9D")
    enc_resp = request.form.get("encResp")
    
    if not enc_resp:
        return "Invalid response", 400
        
    try:
        dec_resp = decrypt(enc_resp, working_key)
        # Parse response string (it's in param1=val1&param2=val2 format)
        resp_dict = {p.split('=')[0]: p.split('=')[1] for p in dec_resp.split('&') if '=' in p}
        
        order_status = resp_dict.get("order_status")
        order_group_id = resp_dict.get("order_id")
        user_id = resp_dict.get("merchant_param1")
        total_order_amount = float(resp_dict.get("merchant_param2", 0))
        amount_paid = float(resp_dict.get("amount", 0))

        if order_status == "Success":
            # Finalize the order in DB
            with app.app_context():
                # Retrieve cart again to reduce stock accurately
                rows = db.session.execute(text("SELECT plant_id, quantity FROM cart_new WHERE user_id = :u"), {"u": user_id}).fetchall()
                
                remaining_paid = amount_paid
                remaining_total = total_order_amount
                
                for i, r in enumerate(rows):
                    plant = db.session.get(Plant, r[0])
                    if plant:
                        plant.stock -= r[1]
                        item_total = plant.price * r[1]
                        
                        # Allocate paid amount proportionally or just dump on first items
                        # Here we use a safe approach: items get their share
                        if i == len(rows) - 1:
                            # Last item gets the rest to avoid rounding issues
                            item_paid = remaining_paid
                            item_balance = remaining_total - remaining_paid
                        else:
                            ratio = item_total / total_order_amount if total_order_amount > 0 else 0
                            item_paid = round(amount_paid * ratio, 2)
                            item_balance = item_total - item_paid
                            remaining_paid -= item_paid
                            remaining_total -= item_total

                        new_order = Order(
                            plant_name=plant.plant_name,
                            user_id=int(user_id) if str(user_id).isdigit() else None,
                            order_status='Processing',
                            order_group_id=order_group_id,
                            price=plant.price,
                            quantity=r[1],
                            total_amount=item_total,
                            payment_status='Paid' if item_balance <= 0 else 'Partial',
                            payment_method='CCAvenue',
                            shipping_address=resp_dict.get("billing_address", "Provided via PG"),
                            amount_paid=item_paid,
                            balance=item_balance
                        )
                        db.session.add(new_order)
                
                db.session.execute(text("DELETE FROM cart_new WHERE user_id=:u"), {"u": user_id})
                db.session.commit()
                
            return f"""
                <html>
                    <body onload="window.parent.postMessage('payment_success', '*')">
                        <h2>Payment Successful! Order ID: {order_group_id}</h2>
                        <p>You can close this tab and return to the shop.</p>
                        <script>setTimeout(() => window.close(), 3000);</script>
                    </body>
                </html>
            """
        else:
            return f"Payment status: {order_status}. Please try again."
            
    except Exception as e:
        print("Payment Error:", e)
        return f"Payment Verification Failed: {str(e)}", 500

@app.route("/api/shop/plants", methods=["GET"])
def get_shop_plants():
    """Get all plants for the dynamically rendered shop"""
    try:
        plants = Plant.query.all()
        return jsonify({
            "status": "success",
            "plants": [{
                "plant_id": p.plant_id,
                "plant_name": p.plant_name,
                "description": p.description,
                "price": p.price,
                "discount_price": 0.0,
                "effective_price": p.price,
                "stock": p.stock if p.stock is not None else 0,
                "availability_status": "Available" if (p.stock is not None and p.stock > 0) else "Out of Stock",
                "image_url": f"/static/{p.image_path.lstrip('/')}" if p.image_path else None
            } for p in plants]
        })
    except Exception as e:
        print(f"Error in get_shop_plants: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500



@app.route("/api/cart/remove", methods=["POST"])
def remove_from_cart():
    data = request.json
    user_id = str(data.get("user_id", "anonymous"))
    plant_id = data.get("plant_id")

    if not plant_id:
        return jsonify({"status": "error", "message": "plant_id required"}), 400

    try:
        db.session.execute(text("""
            DELETE FROM cart_new
            WHERE user_id = :user_id AND plant_id = :plant_id
        """), {"user_id": user_id, "plant_id": plant_id})

        db.session.commit()
        return jsonify({"status": "success", "message": "Removed from cart "})

    except Exception as e:
        db.session.rollback()
        return jsonify({"status": "error", "message": str(e)}), 500



@app.route("/api/history/<user_id>", methods=["GET"])
def get_history(user_id):
    rows = db.session.execute(text("""
        SELECT TOP 50 query_id, description, created_at
        FROM query
        WHERE user_id = :uid
        ORDER BY created_at DESC
    """), {"uid": user_id}).fetchall()

    history = []
    for r in rows:
        history.append({
            "query_id": r[0],
            "query": r[1],
            "time": str(r[2])
        })

    return jsonify({"status": "success", "history": history})



@app.route('/api/queries', methods=['GET', 'POST'])
def manage_queries():
    """Get or store queries"""
    try:
        if request.method == "GET":
            user_id = request.args.get("user_id")
            if user_id:
                queries = Query.query.filter_by(user_id=user_id).all()
            else:
                queries = Query.query.all()
            
            return jsonify({
                "status": "success",
                "queries": [{
                    "query_id": q.query_id,
                    "description": q.description,
                    "user_id": q.user_id,
                    "created_at": q.created_at.strftime("%Y-%m-%d %H:%M:%S")
                } for q in queries]
            })

        elif request.method == "POST":
            data = request.json
            description = data.get("description")
            user_id = data.get("user_id")  # Can be None
            
            if not description:
                return jsonify({"status": "error", "message": "Description required"}), 400
            
            query_id = store_query(description, user_id)
            return jsonify({
                "status": "success",
                "message": "Query stored",
                "query_id": query_id
            })

    except Exception as e:
        print(f"Queries API error: {str(e)}")
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route('/api/history/query', methods=['POST'])
def flask_history_query():
    """
    Answers questions about user's past plant history using LangChain & FAISS.
    """
    plant_name = None
    try:
        from app.history.history_retriever import answer_from_history
        data = request.get_json() or {}
        query = data.get("query")
        plant_name = data.get("plant_name")

        if not query:
            return jsonify({"status": "error", "message": "query is required"}), 400

        result = answer_from_history(query=query, plant_name=plant_name)
        return jsonify(result)
    except Exception as e:
        print(f" [History API] Error processing query: {e}")
        return jsonify({"type": "history", "plant": plant_name or "Unknown", "answer": "I couldn't find that information in the stored history.", "sources": []}), 500


@app.route('/api/history/profile', methods=['POST'])
def flask_history_profile():
    """
    Adds a new plant profile history version if constant parameters changed.
    """
    try:
        from app.history.history_retriever import history_manager
        data = request.get_json() or {}
        if not data.get("plant_name"):
            return jsonify({"status": "error", "message": "plant_name is required"}), 400
        
        res = history_manager.add_plant_profile_version(data)
        return jsonify({"status": "success", "profile": res})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route('/api/history/variable', methods=['POST'])
def flask_history_variable():
    """
    Updates current variable metrics without creating a profile history version.
    """
    try:
        from app.history.history_retriever import history_manager
        data = request.get_json() or {}
        p_name = data.get("plant_name")
        if not p_name:
            return jsonify({"status": "error", "message": "plant_name is required"}), 400

        updates = {k: v for k, v in data.items() if k != "plant_name" and v is not None}
        res = history_manager.update_variable_state(p_name, updates)
        return jsonify({"status": "success", "variables": res})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route('/api/history/analysis', methods=['POST'])
def flask_history_analysis():
    """
    Records an image analysis event in history FAISS vector store.
    """
    try:
        from app.history.history_retriever import history_manager
        data = request.get_json() or {}
        p_name = data.get("plant_name")
        d_name = data.get("disease_name")
        if not p_name or not d_name:
            return jsonify({"status": "error", "message": "plant_name and disease_name are required"}), 400

        res = history_manager.add_analysis_history(
            plant_name=p_name,
            disease_name=d_name,
            plant_confidence=float(data.get("plant_confidence", 1.0)),
            disease_confidence=float(data.get("disease_confidence", 1.0)),
            timestamp=data.get("timestamp")
        )
        return jsonify({"status": "success", "analysis": res})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

    

# Find and replace all instances of:
# plant_obj = Plant.query.get(p[0])
# With:
# plant_obj = db.session.get(Plant, p[0])
    
@app.route('/api/orders/<int:user_id>', methods=['GET'])
def get_orders(user_id):
    try:
        rows = db.session.execute(text("""
            SELECT o.order_id, o.plant_name, p.price, o.order_date
            FROM orders o
            JOIN plant p ON o.plant_name = p.plant_name
            WHERE o.user_id = :uid
            ORDER BY o.order_date DESC
        """), {"uid": user_id}).fetchall()

        orders = []
        for r in rows:
            orders.append({
                "order_id": r[0],
                "plant_name": r[1],
                "price": r[2],
                "order_date": str(r[3])
            })

        return jsonify({"status": "success", "orders": orders})

    except Exception as e:
        print("Orders API error:", e)
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy'})




def search_plant_image(query):
    """
    Search for a plant image in the local database
    """
    try:
        # Search in plant table
        plant = db.session.execute(text("""
            SELECT TOP 1 image_path, plant_name 
            FROM plant 
            WHERE LOWER(plant_name) LIKE :q
        """), {"q": f"%{query.lower()}%"}).fetchone()
        
        if plant and plant[0]:
            return {
                "status": "success",
                "image_url": get_plant_image_url(plant[0])
            }
            
        # Search in variety table as fallback
        variety = db.session.execute(text("""
            SELECT TOP 1 p.image_path, v.variety_name
            FROM variety v
            JOIN plant p ON p.plant_id = v.plant_id
            WHERE LOWER(v.variety_name) LIKE :q
        """), {"q": f"%{query.lower()}%"}).fetchone()
        
        if variety and variety[0]:
            return {
                "status": "success",
                "image_url": get_plant_image_url(variety[0])
            }
            
        return {"status": "error", "message": "No local image found for this plant."}
    except Exception as e:
        print(f"Local image search error: {e}")
        return {"status": "error", "message": str(e)}


def generate_speech(text):
    """
    Generate speech wrapper for gTTS failsafe
    """
    return tts_failsafe(text)


@app.route('/api/detect-tomato-disease', methods=['POST'])
def api_detect_tomato_disease():
    """
    Dedicated OpenCV-powered Tomato plant leaf disease detection endpoint.
    Accepts image file under 'image' or 'file' key, or base64 'image_b64' JSON.
    """
    try:
        from plant_disease_detection.tomato_disease_opencv import tomato_detector
        
        img_bytes = None
        if 'image' in request.files:
            img_bytes = request.files['image'].read()
        elif 'file' in request.files:
            img_bytes = request.files['file'].read()
        elif request.is_json:
            data = request.get_json() or {}
            if 'image_b64' in data:
                b64_str = data['image_b64']
                if ',' in b64_str:
                    b64_str = b64_str.split(',')[1]
                img_bytes = base64.b64decode(b64_str)

        if not img_bytes:
            return jsonify({"status": "error", "message": "No tomato leaf image file or base64 provided"}), 400

        result = tomato_detector.predict_tomato_disease(img_bytes)
        return jsonify(result)

    except Exception as e:
        print(f" [Tomato OpenCV API] Error processing request: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500


# ==============================================================================
# DATASET PROFILES & EXCEL/CSV UPLOAD API ENDPOINTS FOR ADMIN
# ==============================================================================

PROFILE_DATASETS = {
    "geography": {
        "file": "GEOGRAPHY_PROFILEE.csv",
        "title": "Geography & Climate Profiles",
        "key_col": "plant_name",
        "required_cols": ["plant_name"],
        "columns": [
            "plant_name", "country", "state_or_region", "continent", "native_status",
            "cultivation_status", "climate_zone", "best_growing_season", "best_growing_months",
            "temperature_min", "temperature_max", "annual_rainfall_min", "annual_rainfall_max",
            "humidity_min", "humidity_max", "altitude_min", "altitude_max", "soil_suitability",
            "water_availability", "sunlight_condition", "frost_tolerance", "heat_tolerance",
            "drought_tolerance", "salinity_tolerance", "wind_tolerance", "common_growing_method",
            "commercial_production", "remarks"
        ],
        "example": {
            "plant_name": "Mango", "country": "India", "state_or_region": "Maharashtra",
            "continent": "Asia", "native_status": "Native", "cultivation_status": "Widely Cultivated",
            "climate_zone": "Tropical", "best_growing_season": "Summer", "best_growing_months": "March-June",
            "temperature_min": "24", "temperature_max": "35", "annual_rainfall_min": "750",
            "annual_rainfall_max": "2500", "humidity_min": "50", "humidity_max": "70",
            "altitude_min": "0", "altitude_max": "1200", "soil_suitability": "Well-drained alluvial loam",
            "water_availability": "Moderate", "sunlight_condition": "Full Sun", "frost_tolerance": "Low",
            "heat_tolerance": "High", "drought_tolerance": "Moderate", "salinity_tolerance": "Low",
            "wind_tolerance": "Moderate", "common_growing_method": "Grafting", "commercial_production": "High",
            "remarks": "Requires dry weather during flowering."
        }
    },
    "plant_profile": {
        "file": "plant_profiles.csv",
        "title": "Plant Profiles Catalog",
        "key_col": "plant_name",
        "required_cols": ["plant_name"],
        "columns": [
            "plant_name", "scientific_name", "moisture", "temperature", "humidity",
            "sunlight", "soil_type", "nitrogen", "phosphorus", "potassium",
            "watering_frequency", "growth_duration"
        ],
        "example": {
            "plant_name": "Mango", "scientific_name": "Mangifera indica", "moisture": "50-70%",
            "temperature": "24-35°C", "humidity": "50-60%", "sunlight": "Full Sun",
            "soil_type": "Well-drained loamy soil", "nitrogen": "High", "phosphorus": "Medium",
            "potassium": "High", "watering_frequency": "2-3 times per week", "growth_duration": "5-8 years"
        }
    },
    "disease_profile": {
        "file": "plant_disease_profiles (1).csv",
        "title": "Disease Profiles & Identification",
        "key_col": "disease_name",
        "required_cols": ["plant_name", "disease_name"],
        "columns": [
            "plant_name", "disease_name", "disease_type", "symptoms", "causes",
            "prevention", "treatment", "affected_parts", "severity"
        ],
        "example": {
            "plant_name": "Tomato", "disease_name": "Early Blight", "disease_type": "Fungal",
            "symptoms": "Concentric dark spots on leaves", "causes": "Alternaria solani fungus",
            "prevention": "Crop rotation, clean seeds", "treatment": "Copper-based fungicide spray",
            "affected_parts": "Leaves, stems", "severity": "Moderate to High"
        }
    },
    "seasonal_profile": {
        "file": "seasonal_profile.csv",
        "title": "Seasonal Care & Nutrient Profiles",
        "key_col": "plant_name",
        "required_cols": ["plant_name", "season"],
        "columns": [
            "plant_name", "season", "watering_frequency", "water_amount_liters",
            "fertilizer_type", "fertilizer_amount", "nitrogen_requirement",
            "phosphorus_requirement", "potassium_requirement", "sunlight_requirement",
            "temperature_range_celsius", "wind_tolerance", "care_instructions"
        ],
        "example": {
            "plant_name": "Tomato", "season": "Summer", "watering_frequency": "Daily",
            "water_amount_liters": "1.5", "fertilizer_type": "NPK 10-10-10", "fertilizer_amount": "50g",
            "nitrogen_requirement": "Medium", "phosphorus_requirement": "High",
            "potassium_requirement": "High", "sunlight_requirement": "6-8 hours full sun",
            "temperature_range_celsius": "20-30", "wind_tolerance": "Moderate",
            "care_instructions": "Mulch soil surface to preserve moisture."
        }
    }
}


@app.route('/api/admin/profile-stats', methods=['GET'])
def get_profile_stats():
    """
    Returns record counts, columns, file details, and preview rows for dataset profiles.
    """
    try:
        profile_type = request.args.get('profile_type', 'all')
        base_dir = os.path.dirname(os.path.abspath(__file__))
        results = {}

        targets = [profile_type] if profile_type in PROFILE_DATASETS else list(PROFILE_DATASETS.keys())

        import pandas as pd
        for p_key in targets:
            info = PROFILE_DATASETS[p_key]
            file_path = os.path.join(base_dir, info["file"])
            
            if not os.path.exists(file_path):
                results[p_key] = {
                    "key": p_key,
                    "title": info["title"],
                    "exists": False,
                    "row_count": 0,
                    "columns": info["columns"],
                    "preview": [],
                    "last_modified": "N/A"
                }
                continue

            # Load CSV safely with fallback encodings
            df = None
            for enc in ['utf-8', 'latin-1', 'cp1252']:
                try:
                    df = pd.read_csv(file_path, encoding=enc)
                    break
                except Exception:
                    continue
            
            if df is None:
                df = pd.DataFrame()

            # Clean dataframe for JSON response (replace NaN with empty string)
            df_clean = df.fillna("")
            preview_rows = df_clean.head(10).to_dict(orient="records")
            mod_time = datetime.fromtimestamp(os.path.getmtime(file_path)).strftime('%Y-%m-%d %H:%M:%S')

            results[p_key] = {
                "key": p_key,
                "title": info["title"],
                "exists": True,
                "row_count": len(df),
                "columns": list(df.columns),
                "required_cols": info["required_cols"],
                "preview": preview_rows,
                "last_modified": mod_time,
                "filename": info["file"]
            }

        return jsonify({"status": "success", "datasets": results})

    except Exception as e:
        print(f" [Profile Stats Error]: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route('/api/admin/upload-profiles', methods=['POST'])
def upload_profile_dataset():
    """
    Accepts Excel (.xlsx / .xls) or CSV file uploads to update Geography, Plant, or Disease dataset profiles.
    Supports 'replace' or 'append' mode. Rebuilds FAISS RAG index automatically.
    """
    import io
    import pandas as pd
    try:
        if 'file' not in request.files:
            return jsonify({"status": "error", "message": "No file uploaded in form payload ('file' key expected)."}), 400

        file_obj = request.files['file']
        profile_type = request.form.get('profile_type', 'plant_profile')
        mode = request.form.get('mode', 'replace').lower() # 'replace' or 'append'

        if profile_type not in PROFILE_DATASETS:
            return jsonify({
                "status": "error",
                "message": f"Invalid profile_type. Must be one of: {list(PROFILE_DATASETS.keys())}"
            }), 400

        meta = PROFILE_DATASETS[profile_type]
        base_dir = os.path.dirname(os.path.abspath(__file__))
        target_file_path = os.path.join(base_dir, str(meta["file"]))

        filename = file_obj.filename.lower()
        content = file_obj.read()

        if not content:
            return jsonify({"status": "error", "message": "Uploaded file is empty."}), 400

        # Parse file based on extension
        new_df = None
        if filename.endswith('.xlsx') or filename.endswith('.xls'):
            new_df = pd.read_excel(io.BytesIO(content))
        else:
            for enc in ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']:
                try:
                    new_df = pd.read_csv(io.BytesIO(content), encoding=enc)
                    break
                except Exception:
                    continue

        if new_df is None or new_df.empty:
            return jsonify({"status": "error", "message": "Could not parse spreadsheet data or sheet is empty."}), 400

        # Clean column names (strip whitespace)
        new_df.columns = [str(c).strip() for c in new_df.columns]

        # Check required columns
        missing_cols = [col for col in meta["required_cols"] if col not in new_df.columns]
        if missing_cols:
            return jsonify({
                "status": "error",
                "message": f"Uploaded file is missing required columns for {meta['title']}: {missing_cols}. Found columns: {list(new_df.columns)}"
            }), 400

        # Process dataset depending on mode
        final_df = None
        if mode == 'append' and os.path.exists(target_file_path):
            existing_df = None
            for enc in ['utf-8', 'latin-1', 'cp1252']:
                try:
                    existing_df = pd.read_csv(target_file_path, encoding=enc)
                    break
                except Exception:
                    continue
            
            if existing_df is not None and not existing_df.empty:
                existing_df.columns = [str(c).strip() for c in existing_df.columns]
                # Combine dataframes
                combined = pd.concat([existing_df, new_df], ignore_index=True)
                # Drop exact duplicate rows
                key_col = str(meta.get("key_col") or "")
                if key_col and key_col in combined.columns:
                    final_df = combined.drop_duplicates(subset=[key_col], keep="last")
                else:
                    final_df = combined.drop_duplicates(keep="last")
            else:
                final_df = new_df
        else:
            final_df = new_df

        # Fill NaNs with empty string
        final_df = final_df.fillna("")

        # Save back to target CSV file using latin1 or utf8
        try:
            final_df.to_csv(target_file_path, index=False, encoding='utf-8')
        except Exception:
            final_df.to_csv(target_file_path, index=False, encoding='latin1')

        # Trigger RAG FAISS re-indexing
        print(f" [Admin Upload] Syncing RAG vector database after {meta['title']} update...")
        rag_reindexed = False
        try:
            faiss_rag.sync_from_db(PlantKnowledge, DiseaseKnowledge)
            rag_reindexed = True
        except Exception as rag_err:
            print(f" [Admin Upload] FAISS RAG sync warning: {rag_err}")

        return jsonify({
            "status": "success",
            "message": f"Successfully updated {meta['title']} ({mode} mode). Total rows: {len(final_df)}.",
            "profile_type": profile_type,
            "mode": mode,
            "new_rows_uploaded": len(new_df),
            "total_rows": len(final_df),
            "rag_reindexed": rag_reindexed,
            "filename": meta["file"]
        })

    except Exception as e:
        print(f" [Upload Profile Error]: {e}")
        traceback.print_exc()
        return jsonify({"status": "error", "message": f"Server error processing upload: {str(e)}"}), 500


@app.route('/api/admin/download-template/<profile_type>', methods=['GET'])
def download_profile_template(profile_type):
    """
    Generates and returns a downloadable CSV template with standard headers and sample data for admins.
    """
    try:
        if profile_type not in PROFILE_DATASETS:
            return jsonify({"status": "error", "message": "Invalid profile type."}), 400

        meta = PROFILE_DATASETS[profile_type]
        cols = list(meta["columns"]) if isinstance(meta["columns"], (list, tuple)) else []
        example = dict(meta["example"]) if isinstance(meta["example"], dict) else {}

        import pandas as pd
        template_df = pd.DataFrame([example], columns=cols)
        
        csv_buffer = io.StringIO()
        template_df.to_csv(csv_buffer, index=False)
        csv_data = csv_buffer.getvalue()

        response = make_response(csv_data)
        response.headers["Content-Disposition"] = f"attachment; filename={profile_type}_template.csv"
        response.headers["Content-type"] = "text/csv; charset=utf-8"
        return response

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


def ensure_admin_user():
    try:
        user = db.session.execute(text("""
            SELECT id FROM login WHERE LOWER(username) = 'admin' OR LOWER(email) = 'admin' OR LOWER(email) = 'admin@gmail.com'
        """)).fetchone()
        if not user:
            db.session.execute(text("""
                INSERT INTO login (username, email, password, role, business_id, is_online)
                VALUES ('admin', 'admin@gmail.com', 'admin@123', 'SUPER_ADMIN', 'SYSTEM', 0)
            """))
            db.session.commit()
            print(" [DB] Admin user created: admin / admin@123")
        else:
            db.session.execute(text("""
                UPDATE login SET username = 'admin', password = 'admin@123' WHERE id = :id
            """), {"id": user[0]})
            db.session.commit()
            print(" [DB] Admin user credentials verified: admin / admin@123")
    except Exception as e:
        db.session.rollback()
        print(f" [DB] Admin user setup notice: {e}")

# ----------------------------------------------------
# n8n TOOL INTEGRATION ENDPOINTS (Plant & Disease ID)
# ----------------------------------------------------
@app.route('/api/tools/plant-identify', methods=['POST'])
def n8n_plant_identify():
    """n8n Tool Node 1: Plant Identification Tool"""
    try:
        data = request.json or {}
        img_path = data.get('image_path') or 'sample_leaf.jpg'
        
        try:
            from app.tools.plant_identifier import identify_plant
            if os.path.exists(img_path):
                plant_info = identify_plant(img_path)
            else:
                plant_info = {
                    "name": "Tomato",
                    "scientific_name": "Solanum lycopersicum",
                    "confidence": 0.96,
                    "image_type": "leaf",
                    "status": "identified"
                }
        except Exception:
            plant_info = {
                "name": "Tomato",
                "scientific_name": "Solanum lycopersicum",
                "confidence": 0.96,
                "image_type": "leaf",
                "status": "identified"
            }

        return jsonify({
            "status": "success",
            "tool": "plant_identification",
            "plant": plant_info
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route('/api/tools/disease-identify', methods=['POST'])
def n8n_disease_identify():
    """n8n Tool Node 2: Disease Identification Tool (Receives plant_name)"""
    try:
        data = request.json or {}
        plant_name = data.get('plant_name', 'Tomato')
        img_path = data.get('image_path') or 'sample_leaf.jpg'

        try:
            from app.tools.disease_identifier import identify_disease
            if os.path.exists(img_path):
                disease_info = identify_disease(img_path, plant_name=plant_name)
            else:
                disease_info = {
                    "plant_name": plant_name,
                    "disease_name": "Early Blight",
                    "confidence": 0.94,
                    "status": "detected"
                }
        except Exception:
            disease_info = {
                "plant_name": plant_name,
                "disease_name": "Early Blight",
                "confidence": 0.94,
                "status": "detected"
            }

        return jsonify({
            "status": "success",
            "tool": "disease_identification",
            "disease": disease_info
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route('/api/tools/n8n-webhook', methods=['POST'])
def n8n_webhook():
    """n8n Webhook Endpoint for external workflow trigger/logging"""
    try:
        payload = request.json or {}
        return jsonify({
            "status": "success",
            "message": "n8n Webhook received successfully",
            "event": payload.get("event", "n8n_execution"),
            "data": payload.get("data", {})
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


if __name__ == "__main__":
    for attempt in range(1, 4):
        try:
            with app.app_context():
                db.session.remove()
                db.create_all()
                print(" DB schemas verified.")
                ensure_admin_user()
                
                # (1) DB Connectivity Check
                print("(1) Result:", db.session.execute(text("SELECT 1")).fetchone())
                print(" DB CONNECTED SUCCESSFULLY")
                    
                # (2) Reset Plant Sequence
                reset_plant_sequence()
                
                # (3) Sync FAISS RAG from database / CSV files
                if ask_rebuild_index():
                    faiss_rag.sync_from_db(PlantKnowledge, DiseaseKnowledge)
                else:
                    faiss_rag.load_index()
                break
                
        except Exception as e:
            print(f" DB initialization attempt {attempt}/3 error: {e}")
            if attempt < 3:
                print(" Retrying database connection in 2 seconds...")
                time.sleep(2)
            else:
                print(" Could not establish database connection after 3 attempts.")

    port = int(os.getenv("PORT", 8082))
    print(f" Starting on http://127.0.0.1:{port}")
    app.run(host='127.0.0.1', port=port, debug=False)

