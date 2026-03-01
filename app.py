# LLM Options: "groq", "deepseek", "gemini", "local"
LLM_OPTION = "groq"

# STT (Speech-to-Text) Options: "vosk", "whisper", "google", "azure"
STT_OPTION = "vosk"

# TTS (Text-to-Speech) Options: "gtts", "azure", "elevenlabs", "none"
TTS_OPTION = "gtts"

# Failsafe Methods (used when primary option fails or is set to 0)
LLM_FAILSAFE = "local"      # Falls back to keyword matching
STT_FAILSAFE = "vosk"       # Falls back to VOSK
TTS_FAILSAFE = "gtts"       # Falls back to gTTS

from difflib import SequenceMatcher
import numpy as np
import google.generativeai as genai
from flask import Flask, request, jsonify, make_response
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import text
from datetime import datetime, timezone
from openai import OpenAI   
try:
    from vosk import Model, KaldiRecognizer
    VOSK_AVAILABLE = True
except ImportError:
    VOSK_AVAILABLE = False
    print("⚠️ VOSK not available - voice features disabled")
import wave
import json
import os
import subprocess
import tempfile
import time
import shutil
from gtts import gTTS
import base64
import io
import re
import requests
from groq import Groq
from werkzeug.utils import secure_filename
import os
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__, static_folder="static")
CORS(app)


ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png', 'gif'}


PEXELS_API_KEY = "PHbyGGYo79TaO88Oh4g8MDSkzcg04YFYvAz94YfdWHrmjmrIKGTGnAnq"
PEXELS_API_URL = "https://api.pexels.com/v1/search"

def llm_call(prompt):
    global LLM_OPTION

    # ✅ GROQ
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
            print(f"❌ Groq error: {e}")
            return groq_call_failsafe(prompt)

    # ✅ DEEPSEEK (OpenAI-Compatible API)
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
            print(f"❌ DeepSeek error: {e}")
            return groq_call_failsafe(prompt)

    # ✅ OPENAI
    elif LLM_OPTION == "gemini":
        try:
            genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

            model = genai.GenerativeModel("gemini-1.5-flash")
            response = model.generate_content(prompt)

            return response.text.strip()

        except Exception as e:
            print(f"❌ Gemini error: {e}")
            return groq_call_failsafe(prompt)

    # ✅ LOCAL fallback
    elif LLM_OPTION == "local":
        return groq_call_failsafe(prompt)

    else:
        print(f"❌ Unknown LLM option: {LLM_OPTION}")
        return groq_call_failsafe(prompt)


def groq_call_failsafe(prompt):
    """
    Failsafe method using simple keyword matching.
    Works without external API calls.
    """
    print("🔄 Using failsafe LLM (keyword matching)...")
    
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


if VOSK_AVAILABLE:
    try:
        vosk_model = Model("vosk-model-small-en-us-0.15")
    except Exception as e:
        print(f"⚠️ Failed to load VOSK model: {e}")
        VOSK_AVAILABLE = False
else:
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

# Database Configuration
default_db_uri = "postgresql://neondb_owner:npg_KBStXxq52HPZ@ep-gentle-grass-adnpzd0p-pooler.c-2.us-east-1.aws.neon.tech/neondb?sslmode=require"
app.config['SQLALCHEMY_DATABASE_URI'] = os.getenv("DATABASE_URL", default_db_uri)
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False


app.config["SQLALCHEMY_ENGINE_OPTIONS"] = {
    "pool_pre_ping": True,    
    "pool_recycle": 1800     
}


db = SQLAlchemy(app)

with app.app_context():
    try:
        print(db.session.execute(text("SELECT 1")).fetchone())
        print("✅ DB CONNECTED SUCCESSFULLY")
    except Exception as e:
        print("❌ DB CONNECTION FAILED:", e)

class UserQuery(db.Model):
    __tablename__ = "user_queries"

    uq_id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("login.id", ondelete="CASCADE"), nullable=False)
    plant_id = db.Column(db.Integer, db.ForeignKey("plant.plant_id", ondelete="SET NULL"), nullable=True)
    query = db.Column(db.Text, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

class UserInterest(db.Model):
    __tablename__ = "user_interest"

    interest_id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("login.id", ondelete="CASCADE"), nullable=False)
    plant_id = db.Column(db.Integer, db.ForeignKey("plant.plant_id", ondelete="CASCADE"), nullable=False)

    interest_type = db.Column(db.String(20), nullable=False)  # enquiry / purchase
    notes = db.Column(db.Text)

    created_at = db.Column(db.DateTime, default=datetime.utcnow)


class User(db.Model):
    __tablename__ = 'login'

    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password = db.Column(db.String(200), nullable=False)

    last_login = db.Column(db.DateTime)
    last_logout = db.Column(db.DateTime)
    is_online = db.Column(db.Boolean, default=False)



class Plant(db.Model):
    __tablename__ = "plant"

    plant_id = db.Column(db.Integer, primary_key=True)
    plant_name = db.Column(db.String(100))
    description = db.Column(db.Text)
    price = db.Column(db.Float)
    image_path = db.Column(db.String(255))
    stock = db.Column(db.Integer, default=0)

class Variety(db.Model):
    __tablename__ = "variety"

    variety_id = db.Column(db.Integer, primary_key=True)
    plant_id = db.Column(db.Integer, db.ForeignKey("plant.plant_id", ondelete="CASCADE"), nullable=False)
    variety_name = db.Column(db.String(100), nullable=False)
    variety_price = db.Column(db.Float)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    # Relationship back to Plant
    plant = db.relationship('Plant', backref='varieties')




class Conversation(db.Model):
    __tablename__ = 'conversation'
    conversation_id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('login.id', ondelete='CASCADE'))
    conversation = db.Column(db.Text, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

class Cart(db.Model):
    __tablename__ = "cart_new"
    cart_id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.String(100), nullable=False)
    plant_id = db.Column(db.Integer, db.ForeignKey("plant.plant_id"), nullable=False)
    added_at = db.Column(db.DateTime, default=datetime.utcnow)

    plant = db.relationship('Plant', backref='cart_items')

class Order(db.Model):
    __tablename__ = 'orders'
    order_id = db.Column(db.Integer, primary_key=True)
    plant_name = db.Column(db.String(100), db.ForeignKey('plant.plant_name'))
    user_id = db.Column(db.Integer, db.ForeignKey('login.id'))
    order_date = db.Column(db.DateTime, default=datetime.utcnow)

class Category(db.Model):
    __tablename__ = 'category'
    category_id = db.Column(db.Integer, primary_key=True)
    category_name = db.Column(db.String(100), unique=True, nullable=False)

class PlantCategory(db.Model):
    __tablename__ = 'plant_category'
    plant_id = db.Column(db.Integer, db.ForeignKey('plant.plant_id'), primary_key=True)
    category_id = db.Column(db.Integer, db.ForeignKey('category.category_id'), primary_key=True)


class LLMResult(db.Model):
    __tablename__ = 'llm_results'
    id = db.Column(db.Integer, primary_key=True)
    user_query = db.Column(db.Text, nullable=False)
    llm_response = db.Column(db.Text, nullable=False)
    category_suggested = db.Column(db.String(100))
    plants_found = db.Column(db.Integer, default=0)
    llm_name = db.Column(db.String(50))   # ✅ ADD THIS
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    expires_at = db.Column(db.DateTime)


class Query(db.Model):
    __tablename__ = 'query'
    query_id = db.Column(db.Integer, primary_key=True)
    description = db.Column(db.Text, nullable=False)
    canonical_key = db.Column(db.String(100))   
    user_id = db.Column(db.Integer, db.ForeignKey('login.id', ondelete='CASCADE'), nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    user = db.relationship('User', backref='queries')


class QueryResult(db.Model):
    __tablename__ = 'query_result'
    result_id = db.Column(db.Integer, primary_key=True)
    query_id = db.Column(db.Integer, db.ForeignKey('query.query_id', ondelete='CASCADE'), nullable=False)
    result_description = db.Column(db.Text, nullable=False)
    result_source = db.Column(db.String(50), nullable=False)  # 'llm' or 'local'
    
    # ✅ NEW: Store response type
    response_type = db.Column(db.String(50), default='text')  # 'text', 'plants_with_images', 'image'
    response_data = db.Column(db.Text)  # JSON: stores plant IDs, image URLs, etc.
    
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    query = db.relationship('Query', backref='results')


class CategorySynonym(db.Model):
    __tablename__ = 'category_synonym'
    synonym_id = db.Column(db.Integer, primary_key=True)
    synonym = db.Column(db.String(100), nullable=False, unique=True)
    category_id = db.Column(db.Integer, db.ForeignKey('category.category_id', ondelete='CASCADE'), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    category = db.relationship('Category', backref='synonyms')

GREETINGS = {
    "hi", "hello", "hey", "hii", "heyy",
    "thanks", "thank you", "thx",
    "good morning", "good evening", "good night"
}


def normalize_query(text):
    text = text.lower().strip()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text

SIMILARITY_THRESHOLD = 0.75  # 75% match = same query

def calculate_similarity(query1, query2):
    """Calculate similarity between two queries (0-1)"""
    # Normalize queries
    q1 = normalize_query(query1).lower()
    q2 = normalize_query(query2).lower()
    
    # Calculate sequence similarity
    similarity = SequenceMatcher(None, q1, q2).ratio()
    return similarity

def find_similar_query(user_message):
    """Find similar queries with keyword extraction"""
    try:
        normalized = normalize_query(user_message)
        
        stop_words = {"the", "a", "an", "is", "are", "for", "in", "on", "at", "of", "to", "and", "or"}
        user_words = set(w for w in normalized.split() if w not in stop_words and len(w) > 2)
        
        print(f"🔍 Looking for queries similar to: {user_message}")
        print(f"📌 Key words: {user_words}")
        
        recent_queries = db.session.execute(text("""
            SELECT query_id, description
            FROM query
            ORDER BY created_at DESC
            LIMIT 200
        """)).fetchall()
        
        best_matches = []
        
        for query_id, description in recent_queries:
            similarity = calculate_similarity(user_message, description)
            
            norm_desc = normalize_query(description)
            desc_words = set(w for w in norm_desc.split() if w not in stop_words and len(w) > 2)
            
            if user_words and desc_words:
                word_overlap = len(user_words & desc_words) / len(user_words | desc_words)
            else:
                word_overlap = 0
            
            combined_score = (similarity * 0.7) + (word_overlap * 0.3)
            
            if combined_score >= SIMILARITY_THRESHOLD:
                best_matches.append((query_id, description, combined_score))
        
        best_matches.sort(key=lambda x: x[2], reverse=True)
        
        if best_matches:
            query_id, original_query, score = best_matches[0]
            print(f"✅ Best match: {score:.0%} - '{original_query}'")
            
            # ✅ GET RESPONSE TYPE AND DATA
            results = db.session.execute(text("""
                SELECT result_description, result_source, response_type, response_data
                FROM query_result
                WHERE query_id = :qid
                ORDER BY created_at DESC
                LIMIT 1
            """), {"qid": query_id}).fetchone()
            
            if results:
                result_desc, result_source, response_type, response_data = results
                
                return {
                    "type": "cached_similar",
                    "original_query": original_query,
                    "similarity_score": score,
                    "result": result_desc,
                    "source": result_source,
                    "response_type": response_type,  # ✅ PASS TYPE
                    "response_data": json.loads(response_data) if response_data else None  # ✅ PASS DATA
                }
        
        return None
        
    except Exception as e:
        print(f"⚠️ Similarity matching error: {e}")
        return None    
    
def get_varieties_by_plant(name):
    rows = db.session.execute(text("""
        SELECT v.variety_id,
               p.plant_name,
               v.variety_name,
               v.variety_price,
               p.image_path
        FROM variety v
        JOIN plant p ON p.plant_id = v.plant_id
        WHERE p.plant_name ILIKE :x
    """), {"x": f"%{name}%"}).fetchall()

    return [{
        "variety_id": r[0],
        "plant_name": r[1],
        "variety_name": r[2],
        "price": float(r[3]),
        "image_url": f"http://localhost:5001/static/{r[4]}"
    } for r in rows]



# CATEGORY + COLOR + LLM HELPER FUNCTIONS

def ask_llm_for_plants(query):
    prompt = f"""
Suggest 5 REAL plant names suitable for:
"{query}"

Return ONLY plant names.
One plant per line.
"""

    text = llm_call(prompt)
    if not text:
        return []

    return [p.strip() for p in text.split("\n") if p.strip()]

def create_category(category_name):
    row = db.session.execute(
        text("""
            INSERT INTO category (category_name)
            VALUES (:c)
            RETURNING category_id
        """),
        {"c": category_name}
    ).fetchone()

    db.session.commit()
    return row[0]

import json

def store_query_result(query_id, result_description, result_source, response_type='text', response_data=None):
    """Store query result in database with response type"""
    try:
        result = QueryResult(
            query_id=query_id,
            result_description=result_description,
            result_source=result_source,
            response_type=response_type,
            response_data=json.dumps(response_data) if response_data else None
        )
        db.session.add(result)
        db.session.commit()
        print(f"✅ Query result stored: ID {result.result_id} (Type: {response_type})")
        return result.result_id

    except Exception as e:
        print(f"Error storing query result: {e}")
        db.session.rollback()
        return None


def map_plant_to_category(plant_id, category_id):
    db.session.execute(
        text("""
            INSERT INTO plant_category (plant_id, category_id)
            VALUES (:p, :c)
            ON CONFLICT DO NOTHING
        """),
        {"p": plant_id, "c": category_id}
    )


def search_normal_query(user_message):
    normalized = normalize_query(user_message)

    row = db.session.execute(text("""
        SELECT q.query_id, qr.result_description
        FROM query q
        JOIN query_result qr ON qr.query_id = q.query_id
        WHERE q.description = :q
        ORDER BY q.created_at DESC
        LIMIT 1
    """), {"q": normalized}).fetchone()

    return row[1] if row else None

# Add this near your other utility functions

CANONICAL_CLUSTERS = {
    "COUNT_PLANTS": [
        "how many plants",
        "total plants", 
        "plant count",
        "how many plants in garden",
        "total number of plants",
        "how many plants do we have",
        "what is the count",
        "how many plants in nursery",  # ✅ ADD THIS
        "total plants in nursery"      # ✅ ADD THIS
    ],
    "LIST_INDOOR_PLANTS": [
        "show indoor plants",
        "indoor plants",
        "plants for indoor",
        "best indoor plants",
        "display all indoor plants"    # ✅ ADD THIS
    ],
    "LIST_OUTDOOR_PLANTS": [
        "outdoor plants",
        "show outdoor plants",
        "plants for outside"
    ],
    "SHOW_PLANT_IMAGE": [
        "show me image",
        "show picture",
        "show plant photo",
        "picture of",
        "display all the plants"       # ❌ REMOVE - too generic
    ]
}

def get_canonical_intent_cluster(user_message):
    """Match user message to predefined intent clusters"""
    normalized = normalize_query(user_message).lower()
    
    # ✅ Check COUNT_PLANTS FIRST (before generic queries)
    for example in CANONICAL_CLUSTERS.get("COUNT_PLANTS", []):
        similarity = calculate_similarity(user_message, example)
        if similarity >= 0.65:  # Lower threshold for count queries
            print(f"✅ Matched COUNT_PLANTS: {similarity:.0%} match to '{example}'")
            return "COUNT_PLANTS"
    
    # Check other clusters
    for canonical_key, examples in CANONICAL_CLUSTERS.items():
        if canonical_key == "COUNT_PLANTS":
            continue  # Already checked
            
        for example in examples:
            similarity = calculate_similarity(user_message, example)
            if similarity >= 0.7:
                print(f"✅ Matched to cluster: {canonical_key} (70%+ match to example)")
                return canonical_key
    
    return None

def get_canonical_from_llm(user_message):
    """Fallback: use LLM only if cluster matching fails"""
    
    # First try cluster matching
    cluster = get_canonical_intent_cluster(user_message)
    if cluster:
        return cluster
    
    # Fallback to LLM
    prompt = f"""
Convert the following user query into a SINGLE canonical intent.
Return ONLY the canonical key (no explanation).

Examples:
- "how many plants are there" → COUNT_PLANTS
- "show indoor plants" → LIST_INDOOR_PLANTS
- "image of rose" → SHOW_PLANT_IMAGE

User query: "{user_message}"
"""
    
    key = llm_call(prompt)
    return key.strip().upper().replace(" ", "_") if key else "UNKNOWN"

def search_by_canonical(canonical_key):
    row = db.session.execute(text("""
        SELECT llm_response
        FROM llm_results
        WHERE user_query = :ck
          AND (expires_at IS NULL OR expires_at > NOW())
        ORDER BY created_at DESC
        LIMIT 1
    """), {"ck": canonical_key}).fetchone()

    return row[0] if row else None

def store_llm_result(user_query, raw_text, inferred_category, matched_count):
    """Store LLM inference result"""
    try:
        # ✅ FIX: Remove expires_at and other invalid fields
        result = LLMResult(
    user_query=normalize_query(user_query),
    llm_response=raw_text,
    category_suggested=inferred_category,
    plants_found=matched_count,
    llm_name=LLM_OPTION
        )

        
        db.session.add(result)
        db.session.commit()
        print(f"✅ Stored LLM result for: {user_query}")
        return result.id
    except Exception as e:
        print(f"⚠️ Error storing LLM result: {e}")
        db.session.rollback()
        return None




def add_category_synonym(synonym, category_id):
    """Add a synonym for an existing category"""
    try:
        syn = CategorySynonym(
            synonym=synonym.lower(),
            category_id=category_id
        )
        db.session.add(syn)
        db.session.commit()
        print(f"✅ Synonym '{synonym}' added for category_id {category_id}")
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
            query=query
        )
        db.session.add(new_q)
        db.session.commit()

        return jsonify({
            "status": "success",
            "message": "Query saved ✅",
            "uq_id": new_q.uq_id
        })

    except Exception as e:
        db.session.rollback()
        return jsonify({"status": "error", "message": str(e)}), 500


def log_user_interest(user_id, plant_id, interest_type, notes=""):
    try:
        # ✅ skip guest users
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
        print("⚠️ user_interest insert failed:", e)



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


def map_intent_to_category(user_message):
    """
    1) Check if user input matches a SYNONYM first
    2) Try to infer category using LLM
    3) If category exists in DB → return {"type": "category", "value": category}
    4) Else → ask LLM for plant names
    5) Store results in llm_results table
    """

    text_lower = user_message.lower().strip()

    # ✅ NEW: Check SYNONYMS FIRST
    synonym_result = get_category_from_synonym(text_lower)
    if synonym_result:
        print(f"✅ Matched synonym: {text_lower} → {synonym_result['category_name']}")
        store_llm_result(user_message, f"Matched synonym: {text_lower}", synonym_result['category_name'], 0)
        return {
            "type": "category",
            "value": synonym_result['category_name']
        }

    # ✅ DYNAMICALLY fetch ALL categories from database
    all_categories = db.session.execute(
        text("SELECT LOWER(category_name) FROM category")
    ).fetchall()
    VALID_CATEGORIES = [row[0] for row in all_categories]

    # Ask LLM for category
    prompt = f"""
User request: "{user_message}"

Extract ONLY the plant category name (max 2-3 words).
Examples: "indoor", "outdoor", "flowering", "succulents"

Return ONLY the category name, nothing else.
"""

    try:
        llm_text = llm_call(prompt)
    except Exception as e:
        print("Groq category error:", e)
        llm_text = ""

    llm_text = llm_text.lower().strip() if llm_text else ""
    
    # ✅ SANITIZE: Extract only first line (remove explanations)
    llm_text = llm_text.split('\n')[0].strip()
    
    # ✅ TRUNCATE to 100 chars max
    llm_text = llm_text[:100]
    
    print(f"📝 LLM Category: '{llm_text}'")

    # ✅ Check against ALL categories in database
    for c in VALID_CATEGORIES:
        if llm_text.startswith(c) or c in llm_text:
            row = db.session.execute(
                text("""
                    SELECT category_id
                    FROM category
                    WHERE LOWER(category_name) = :c
                """),
                {"c": c}
            ).fetchone()

            if row:
                # ✅ Store result in llm_results
                store_llm_result(user_message, llm_text, c, 0)
                return {
                    "type": "category",
                    "value": c
                }

    # No category found → ask LLM for plant names
    plant_prompt = f"""
Suggest 5 real plant names suitable for:
"{user_message}"

Return ONLY plant names, one per line.
"""

    try:
        raw_text = llm_call(plant_prompt)
        if not raw_text:
            return None
        plant_names = [p.strip() for p in raw_text.split("\n") if p.strip()]
    except Exception as e:
        print("Groq plant error:", e)
        plant_names = []

    # Match plants with DB
    matched_plants = []

    for name in plant_names:
        row = db.session.execute(
            text("""
                SELECT plant_id, plant_name, description, price
                FROM plant
                WHERE plant_name ILIKE :x
            """),
            {"x": f"%{name}%"}
        ).fetchone()

        if row:
            matched_plants.append({
                "plant_id": row[0],
                "plant_name": row[1],
                "description": row[2],
                "price": row[3]
            })

    if matched_plants:
        # ✅ NEW: Auto-create category if plants were found
        inferred_category = llm_text.strip() if llm_text else "suggested"
        
        # ✅ TRUNCATE category name to 100 chars
        inferred_category = inferred_category[:100]
        
        # Check if category already exists
        existing_cat = db.session.execute(
            text("SELECT category_id FROM category WHERE LOWER(category_name) = :c"),
            {"c": inferred_category}
        ).fetchone()
        
        if not existing_cat:
            new_cid = create_category(inferred_category)
            print(f"✨ Auto-created category: {inferred_category}")
        else:
            new_cid = existing_cat[0]
        
        # Map all matched plants to this category
        for plant in matched_plants:
            map_plant_to_category(plant["plant_id"], new_cid)
        
        db.session.commit()
        
        # ✅ Store result in llm_results
        store_llm_result(user_message, raw_text, inferred_category, len(matched_plants))
        
        return {
            "type": "plants",
            "value": matched_plants,
            "category_name": inferred_category
        }

    # ✅ Store failed result
    store_llm_result(user_message, plant_prompt, "no_match", 0)
    return None


def get_cached_llm_result(normalized_query):
    row = db.session.execute(text("""
        SELECT llm_response, llm_name, expires_at
        FROM llm_results
        WHERE user_query = :q
        ORDER BY created_at DESC
        LIMIT 1
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
        SELECT plant_id, plant_name, description, price
        FROM plant
        WHERE plant_name ILIKE :x
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
        print("❌ Canonical store failed:", e)

# In the function that stores queries
def store_query(description, user_id, canonical_key=None):
    """Store user query"""
    try:
        # ✅ FIX: Convert user_id to integer
        try:
            user_id_int = int(user_id)
        except (ValueError, TypeError):
            print(f"⚠️ Invalid user_id: {user_id}, using 1 as default")
            user_id_int = 1  # Default to guest user
        
        query = Query(
        description=description,
        canonical_key=canonical_key,
        user_id=user_id_int
        )
        
        db.session.add(query)
        db.session.commit()
        print(f"✅ Stored query: {description}")
        return query.query_id
    except Exception as e:
        print(f"⚠️ Error storing query: {e}")
        db.session.rollback()
        return None

@app.route("/register", methods=["POST"])
def register():
    data = request.json
    username = data.get("username", "").strip()
    email = data.get("email", "").strip().lower()
    password = data.get("password", "").strip()

    if not username or not email or not password:
        return jsonify({
            "status": "error",
            "message": "Username, email and password are required"
        }), 400

    try:
        existing = db.session.execute(
            text("SELECT id FROM login WHERE email = :email"),
            {"email": email}
        ).fetchone()

        if existing:
            return jsonify({
                "status": "error",
                "message": "Email already registered. Please login."
            }), 409

        row = db.session.execute(
            text("""
                INSERT INTO login (username, email, password)
                VALUES (:username, :email, :password)
                RETURNING id
            """),
            {"username": username, "email": email, "password": password}
        ).fetchone()

        db.session.commit()

        return jsonify({
            "status": "success",
            "message": "Registered successfully ✅",
            "user_id": row[0],
            "username": username,
            "email": email
        })

    except Exception as e:
        print("Register error:", e)
        db.session.rollback()
        return jsonify({"status": "error", "message": "Registration failed"}), 500


@app.route("/login", methods=["POST"])
def login():
    data = request.json
    email = data.get("email", "").strip().lower()
    password = data.get("password", "").strip()

    if not email or not password:
        return jsonify({
            "status": "error",
            "message": "Email and password required"
        }), 400

    try:
        user = db.session.execute(text("""
            SELECT id, username, email
            FROM login
            WHERE email = :email AND password = :password
        """), {
            "email": email,
            "password": password
        }).fetchone()

        if not user:
            return jsonify({
                "status": "error",
                "message": "Invalid credentials"
            }), 401

        return jsonify({
            "status": "success",
            "user_id": user[0],
            "username": user[1],
            "email": user[2]
        })

    except Exception as e:
        print("Login error:", e)
        return jsonify({
            "status": "error",
            "message": "Database error"
        }), 500


def resolve_plant_synonym(user_message):
    row = db.session.execute(text("""
        SELECT p.plant_id, p.plant_name, p.description, p.price, p.image_path
        FROM plant_synonym ps
        JOIN plant p ON p.plant_id = ps.plant_id
        WHERE :msg ILIKE '%' || ps.synonym || '%'
        ORDER BY LENGTH(ps.synonym) DESC
        LIMIT 1
    """), {"msg": user_message.lower()}).fetchone()

    if row:
        return {
            "plant_id": row[0],
            "plant_name": row[1],
            "description": row[2],
            "price": row[3],
            "image_url": get_plant_image_url(db.session.get(Plant, row[0]))
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
    reply = f"🌱 There are {count} plants available in our nursery."

    store_query_result(query_id, reply, "local")
    store_canonical_answer(canonical_key, reply, user_message)

    return {
        "status": "success",
        "type": "text",
        "reply": reply,
        "canonical_key": canonical_key
    }


def log_category_interest(user_id, category_id, interest_type="search"):

    if not user_id:
        return

    try:
        db.session.execute(text("""
            INSERT INTO user_category_interest
            (user_id, category_id, interest_type, count)

            VALUES (:u, :c, :t, 1)

            ON CONFLICT (user_id, category_id, interest_type)
            DO UPDATE SET
                count = user_category_interest.count + 1,
                last_interaction = NOW()
        """), {
            "u": int(user_id),
            "c": int(category_id),
            "t": interest_type
        })

        db.session.commit()

    except Exception as e:
        db.session.rollback()
        print("⚠️ Category interest failed:", e)



def process_message(user_message, location="", user_id=None):
    """Process user message with improved caching"""
    
    user_message = user_message.strip()
    normalized = normalize_query(user_message)
    text_lower = user_message.lower()
    
    query_id = None
    if normalized in GREETINGS:
        return {
            "status": "success",
            "type": "text",
            "reply": "🌿 Hello! How can I help you with plants today?"
        }


    # --------------------------------------------------
    # ✅ NEW: LLM-BASED QUERY VALIDATION
    # --------------------------------------------------
    validation = validate_user_query(user_message)
    
    if validation["status"] == "invalid":
        print(f"⚠️ Invalid query reason: {validation.get('reason')}")
        query_id = store_query(user_message, user_id)
        if query_id:
            store_query_result(query_id, validation["reply"], "invalid_query")
        
        return validation


    # --------------------------------------------------
    # 1️⃣ EXACT CACHE MATCH (skip for image queries)
    # --------------------------------------------------
    is_image_query = any(w in normalized for w in ["show", "display", "list", "image", "picture", "photo"])
    
    if not is_image_query:
        normal_answer = search_normal_query(normalized)
        if normal_answer:
            print("✅ Exact query match found")
            return {
                "status": "success",
                "type": "text",
                "reply": normal_answer,
                "source": "cache_exact"
            }

    # --------------------------------------------------
    # 2️⃣ SIMILAR QUERY MATCH
    # --------------------------------------------------
    if not is_image_query:
        similar_match = find_similar_query(user_message)

        if similar_match and similar_match.get("result"):
            print(f"🎯 Similar query found: {similar_match['similarity_score']:.0%} match")
            
            query_id = store_query(user_message, user_id)
            
            response = {
                "status": "success",
                "type": similar_match.get("response_type", "text"),
                "reply": similar_match['result'],
                "source": "cache_similar",
                "similarity_score": similar_match['similarity_score'],
                "original_query": similar_match['original_query']
            }
            
            if similar_match.get("response_data"):
                if similar_match["response_type"] == "plants_with_images":
                    response["plants"] = similar_match["response_data"].get("plants", [])
                    response["count"] = len(response["plants"])
                    response["category"] = similar_match["response_data"].get("category")
                elif similar_match["response_type"] == "image":
                    response["image_url"] = similar_match["response_data"].get("image_url")
                    response["image_credit"] = similar_match["response_data"].get("image_credit")
            
            if query_id:
                store_query_result(query_id, response["reply"], "cache_similar", 
                                 response_type=response["type"], 
                                 response_data=similar_match.get("response_data"))
            
            return response
    plant = resolve_plant_synonym(user_message)

    if plant:
        reply = f"🌱 Found {plant['plant_name']}"

        query_id = store_query(user_message, user_id)

        if query_id:
            store_query_result(
            query_id,
            reply,
            "local",
            response_type="plants_with_images",
            response_data={"plants": [plant]}
        )

        return {
        "status": "success",
        "type": "plants_with_images",
        "reply": reply,
        "plants": [plant]
    }

    
    # --------------------------------------------------
    # 3️⃣ CANONICAL KEY DETECTION
    # --------------------------------------------------
    canonical_key = get_query_synonym(user_message)

    if not canonical_key:
        canonical_key = get_canonical_from_llm(user_message)


# ONLY force image if user explicitly asks for image
    if any(w in normalized for w in ["image", "photo", "picture"]) and "variet" not in normalized:
        canonical_key = "SHOW_PLANT_IMAGE"


    print("🧠 Canonical:", canonical_key)
    # --------------------------------------------------
# 🌼 GENERAL VARIETY HANDLER (ANY PLANT)
# --------------------------------------------------
    if canonical_key.startswith("LIST_") and canonical_key.endswith("_VARIETIES"):

    # Extract plant name from canonical
    # LIST_ROSE_VARIETIES → rose
        plant_name = canonical_key.replace("LIST_", "").replace("_VARIETIES", "").lower()

        print("🌿 Variety requested for:", plant_name)

        rows = db.session.execute(text("""
    SELECT v.variety_id,
           p.plant_name,
           v.variety_name,
           v.variety_price,
           p.image_path
    FROM variety v
    JOIN plant p ON p.plant_id = v.plant_id
    WHERE LOWER(p.plant_name) = :plant
"""), {"plant": plant_name}).fetchall()

        if rows:

            varieties = []

            for r in rows:
                varieties.append({
                "plant_id": r[0],   # reuse field
                "plant_name": r[2],  # variety name
                "description": r[1],
                "price": float(r[3]),
                "image_url": f"{request.host_url}static/{r[4].lstrip('/')}"
            })

            reply = f"🌸 Found {len(varieties)} {plant_name.title()} varieties"

            query_id = store_query(user_message, user_id)

            if query_id:
                store_query_result(
                query_id,
                reply,
                "local",
                response_type="plants_with_images",
                response_data={"plants": varieties}
            )

                return {
            "status": "success",
            "type": "plants_with_images",
            "reply": reply,
            "plants": varieties
        }


    # --------------------------------------------------
    # ✅ NEW: HANDLE COUNT_PLANTS FIRST
    # --------------------------------------------------
    
    if canonical_key == "COUNT_PLANTS":
        row = db.session.execute(
            text("SELECT COUNT(*) FROM plant")
        ).fetchone()

        count = row[0] if row else 0
        reply = f"🌱 There are {count} plants available in our nursery."
        
        query_id = store_query(user_message, user_id)
        if query_id:
            store_query_result(query_id, reply, "local")

        return {
            "status": "success",
            "type": "text",
            "reply": reply,
            "canonical_key": canonical_key
        }

    # --------------------------------------------------
    # 4️⃣ CHECK IF CATEGORY BEFORE SINGLE IMAGE
    # --------------------------------------------------
    # ================= VARIETY PRIORITY =================
    if canonical_key.startswith("LIST_") and canonical_key.endswith("_VARIETIES"):
        pass  # handled by general variety handler (already added)

    if canonical_key == "SHOW_PLANT_IMAGE":        # ✅ FIRST: Check if it's asking for a CATEGORY of plants with images
        intent_result = map_intent_to_category(user_message)
        
        if intent_result and intent_result["type"] == "category":
            category = intent_result["value"]
            
            print(f"🌿 Category detected: {category}")
            
            cid_row = db.session.execute(text("""
                SELECT category_id
                FROM category
                WHERE LOWER(category_name) = :c
            """), {"c": category}).fetchone()

            if cid_row:
                plants = db.session.execute(text("""
                    SELECT p.plant_id, p.plant_name, p.description, p.price, p.image_path
                    FROM plant p
                    JOIN plant_category pc ON pc.plant_id = p.plant_id
                    WHERE pc.category_id = :cid
                    ORDER BY p.plant_name ASC
                """), {"cid": cid_row[0]}).fetchall()

                plants_with_images = []
                for p in plants:
                    plant_obj = Plant.query.get(p[0])
                    image_url = get_plant_image_url(plant_obj)
                    
                    plants_with_images.append({
                        "plant_id": p[0],
                        "plant_name": p[1],
                        "description": p[2],
                        "price": float(p[3]) if p[3] else 0,
                        "image_url": image_url
                    })

                reply = f"🌱 Found {len(plants_with_images)} {category} plants"
                query_id = store_query(user_message, user_id)
                
                if query_id:
                    store_query_result(
                        query_id, 
                        reply, 
                        "local",
                        response_type="plants_with_images",
                        response_data={
                            "plants": plants_with_images,
                            "category": category
                        }
                    )

                return {
                    "status": "success",
                    "type": "plants_with_images",
                    "reply": reply,
                    "plants": plants_with_images,
                    "count": len(plants_with_images),
                    "category": category
                }
        
        # ✅ FALLBACK: Single plant image from Pexels
        query = re.sub(
            r'\b(show|see|picture|image|photo|display|of|me|a|an|the)\b',
            '',
            text_lower
        ).strip()

        img = search_plant_image(query)

        if img.get("status") == "success":
            response = {
                "status": "success",
                "type": "image",
                "reply": f"Here is an image of {query.title()} 🌸",
                "image_url": img["image_url"],
                "image_credit": img["image_credit"]
            }

            query_id = store_query(user_message, user_id)
            if query_id:
                store_query_result(query_id, response["reply"], "local", 
                                 response_type="image",
                                 response_data={
                                     "image_url": img["image_url"],
                                     "image_credit": img["image_credit"]
                                 })
            
            return response

        return {
            "status": "error",
            "type": "text",
            "reply": "Couldn't find image 🌧️"
        }

    # --------------------------------------------------
    # 6️⃣ CATEGORY / INTENT HANDLING (for non-image queries)
    # --------------------------------------------------
    intent_result = map_intent_to_category(user_message)

    # ========== CASE 1: CATEGORY ==========
    if intent_result and intent_result["type"] == "category":
        category = intent_result["value"]

        cid_row = db.session.execute(text("""
            SELECT category_id
            FROM category
            WHERE LOWER(category_name) = :c
        """), {"c": category}).fetchone()

        if cid_row:

            log_category_interest(user_id, cid_row[0], "search")

            plants = db.session.execute(text("""
                SELECT p.plant_id, p.plant_name, p.description, p.price, p.image_path
                FROM plant p
                JOIN plant_category pc ON pc.plant_id = p.plant_id
                WHERE pc.category_id = :cid
                ORDER BY p.plant_name ASC
            """), {"cid": cid_row[0]}).fetchall()

            plants_with_images = []
            for p in plants:
                plant_obj = Plant.query.get(p[0])
                image_url = get_plant_image_url(plant_obj)
                
                plants_with_images.append({
                    "plant_id": p[0],
                    "plant_name": p[1],
                    "description": p[2],
                    "price": float(p[3]) if p[3] else 0,
                    "image_url": image_url
                })

            reply = f"🌱 Found {len(plants_with_images)} {category} plants"
            query_id = store_query(user_message, user_id)
            
            if query_id:
                store_query_result(
                    query_id, 
                    reply, 
                    "local",
                    response_type="plants_with_images",
                    response_data={
                        "plants": plants_with_images,
                        "category": category
                    }
                )

            return {
                "status": "success",
                "type": "plants_with_images",
                "reply": reply,
                "plants": plants_with_images,
                "count": len(plants_with_images),
                "category": category
            }
    
    # ========== CASE 2: PLANTS FROM LLM ==========
    if intent_result and intent_result["type"] == "plants":
        plants_with_images = []

        for plant in intent_result["value"]:
            plant_obj = Plant.query.get(plant["plant_id"])
            plant["image_url"] = get_plant_image_url(plant_obj)
            plants_with_images.append(plant)

        reply = "✨ Found matching plants 🌿"
        query_id = store_query(user_message, user_id)
        
        if query_id:
            store_query_result(query_id, reply, "llm")

        return {
            "status": "success",
            "type": "plants_with_images",
            "reply": reply,
            "plants": plants_with_images
        }

    # --------------------------------------------------
    # 7️⃣ COLOR LOGIC
    # --------------------------------------------------
    COLORS = [
        "red", "yellow", "pink", "white", "blue",
        "green", "purple", "orange", "brown", "violet", "golden"
    ]

    color = next((c for c in COLORS if c in text_lower), None)

    if color:
        rows = db.session.execute(text("""
            SELECT plant_id, plant_name, description, price, image_path
            FROM plant
            WHERE LOWER(description) LIKE :c
               OR LOWER(plant_name) LIKE :c
            ORDER BY plant_name ASC
        """), {"c": f"%{color}%"}).fetchall()

        if rows:
            plants = []
            for r in rows:
                plant_obj = Plant.query.get(r[0])
                image_url = get_plant_image_url(plant_obj)
                
                plants.append({
                    "plant_id": r[0],
                    "plant_name": r[1],
                    "description": r[2],
                    "price": float(r[3]) if r[3] else 0,
                    "image_url": image_url
                })

            reply = f"🌸 Found {len(plants)} {color} plants"
            query_id = store_query(user_message, user_id)
            
            if query_id:
                store_query_result(query_id, reply, "local")

            return {
                "status": "success",
                "type": "plants_with_images",
                "reply": reply,
                "plants": plants,
                "count": len(plants),
                "color": color
            }

    # --------------------------------------------------
    # 8️⃣ FALLBACK
    # --------------------------------------------------
    reply = "I didn't understand that. Try asking for a plant category or color 🌿"
    query_id = store_query(user_message, user_id)
    
    if query_id:
        store_query_result(query_id, reply, "fallback")

    return {
        "status": "error",
        "type": "text",
        "reply": reply
    }


@app.route('/api/validate_query', methods=['POST'])
def check_query_validity():
    """
    Check if a user query is valid for the plant shop using LLM
    """
    try:
        data = request.json
        user_message = data.get("message", "").strip()
        
        if not user_message:
            return jsonify({"status": "error", "message": "No message provided"}), 400
        
        # ✅ Validate the query using LLM
        validation = validate_user_query(user_message)
        
        return jsonify({
            "status": "success",
            "is_valid": validation["status"] == "valid",
            "reason": validation.get("reason"),
            "confidence": validation.get("confidence"),
            "reply": validation.get("reply"),
            "message": validation.get("message")
        })
    
    except Exception as e:
        print(f"Query validation error: {e}")
        return jsonify({"status": "error", "message": "Validation failed"}), 500

def generate_speech(text):
    """Generate speech from text using selected TTS engine"""
    try:
        # Mute mode → no audio
        if TTS_OPTION == "none":
            return {"status": "success", "audio": None}

        # gTTS engine
        elif TTS_OPTION == "gtts":
            try:
                mp3_fp = io.BytesIO()
                tts = gTTS(text=text, lang='en', slow=False)
                tts.write_to_fp(mp3_fp)
                mp3_fp.seek(0)
                audio_base64 = base64.b64encode(mp3_fp.read()).decode()
                return {"status": "success", "audio": audio_base64}
            except Exception as e:
                print(f"❌ gTTS error: {e}")
                return tts_failsafe(text)

        # Azure engine
        elif TTS_OPTION == "azure":
            try:
                # TODO: Add Azure TTS integration
                print("⚠️ Azure TTS not configured yet")
                return tts_failsafe(text)
            except Exception as e:
                print(f"❌ Azure TTS error: {e}")
                return tts_failsafe(text)

        # ElevenLabs engine
        elif TTS_OPTION == "elevenlabs":
            try:
                # TODO: Add ElevenLabs TTS integration
                print("⚠️ ElevenLabs not configured yet")
                return tts_failsafe(text)
            except Exception as e:
                print(f"❌ ElevenLabs error: {e}")
                return tts_failsafe(text)

        else:
            print(f"❌ Unknown TTS option: {TTS_OPTION}")
            return tts_failsafe(text)

    except Exception as e:
        print(f"TTS error: {str(e)}")
        return tts_failsafe(text)


def tts_failsafe(text):
    """Failsafe TTS using gTTS"""
    print("🔄 Using failsafe TTS (gTTS)...")
    try:
        mp3_fp = io.BytesIO()
        tts = gTTS(text=text, lang='en', slow=False)
        tts.write_to_fp(mp3_fp)
        mp3_fp.seek(0)
        audio_base64 = base64.b64encode(mp3_fp.read()).decode()
        return {"status": "success", "audio": audio_base64}
    except Exception as e:
        print(f"❌ gTTS failsafe error: {e}")
        return {"status": "error", "message": "Could not generate speech"}
    
def get_plant_image_url(plant):

    if not plant:
        return None

    if plant.image_path:
        path = plant.image_path.lstrip("/")
        return f"{request.host_url}static/{path}"

    return None


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

        # ✅ Check if plant exists FIRST (case-insensitive)
        existing_plant = db.session.execute(
            text("SELECT plant_id, plant_name, stock, price FROM plant WHERE LOWER(plant_name) = LOWER(:name)"),
            {"name": plant_name}
        ).fetchone()

        if existing_plant:
            # ✅ Plant already exists - just return it
            plant_id = existing_plant[0]
            existed = True
            current_stock = existing_plant[2]
            current_price = existing_plant[3]
            
            print(f"✅ Plant '{plant_name}' already exists with ID: {plant_id}")

        else:
            # ✅ Create new plant
            try:
                result = db.session.execute(
                    text("""
                        INSERT INTO plant (plant_name, description, price, stock)
                        VALUES (:name, :desc, :price, :stock)
                        RETURNING plant_id, stock, price
                    """),
                    {"name": plant_name, "desc": description, "price": price, "stock": stock_count}
                )
                row = result.fetchone()
                plant_id = row[0]
                current_stock = row[1]
                current_price = row[2]
                db.session.commit()
                existed = False
                print(f"✅ Created new plant '{plant_name}' with ID: {plant_id}")

            except Exception as insert_error:
                db.session.rollback()
                print(f"⚠️ Insert error (retrying): {insert_error}")
                
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
                    print(f"✅ Plant exists (concurrent request): ID {plant_id}")
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
            "message": "✅ Plant already available" if existed else "✨ New plant created"
        })

    except Exception as e:
        print(f"❌ Error in suggest_varieties: {e}")
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
            # Reset sequence to max_id + 1
            next_val = max_id + 1
            db.session.execute(text(f"ALTER SEQUENCE plant_plant_id_seq RESTART WITH {next_val}"))
            db.session.commit()
            print(f"✅ Reset plant sequence to start at {next_val} (max was {max_id})")
        else:
            print("⚠️ No plants found in database")
        
        return True
    except Exception as e:
        print(f"⚠️ Sequence reset warning: {e}")
        # Don't fail completely - the retry logic will handle it
        return False
# Call this once on startup
with app.app_context():
    reset_plant_sequence()
    
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
def get_inventory():
    """Get all plants with inventory info"""
    try:
        plants = db.session.execute(text("""
            SELECT plant_id, plant_name, description, price, stock
            FROM plant
            ORDER BY plant_name ASC
        """)).fetchall()

        plant_list = []
        total_stock = 0

        for p in plants:
            plant_list.append({
                "plant_id": p[0],
                "plant_name": p[1],
                "description": p[2],
                "price": float(p[3]) if p[3] else 0,
                "stock": p[4] or 0
            })
            total_stock += (p[4] or 0)

        return jsonify({
            "status": "success",
            "plants": plant_list,
            "total_plants": len(plant_list),
            "total_stock": total_stock
        })

    except Exception as e:
        print(f"Error in get_inventory: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route('/api/admin/restock/<int:plant_id>', methods=['POST'])
def restock_plant(plant_id):
    """Restock a plant by adding quantity"""
    try:
        data = request.json
        quantity = data.get('quantity', 0)

        if quantity <= 0:
            return jsonify({"status": "error", "message": "Quantity must be positive"}), 400

        plant = db.session.get(Plant, plant_id)
        if not plant:
            return jsonify({"status": "error", "message": "Plant not found"}), 404

        old_stock = plant.stock or 0
        plant.stock = old_stock + quantity

        db.session.commit()

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
def get_plant_details(plant_id):
    """Get detailed info about a plant"""
    try:
        plant = db.session.get(Plant, plant_id)
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
                "image_path": plant.image_path
            }
        })

    except Exception as e:
        print(f"Error in get_plant_details: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route('/api/admin/update-plant/<int:plant_id>', methods=['PUT'])
def update_plant_details(plant_id):
    """Update plant details"""
    try:
        data = request.json
        plant = db.session.get(Plant, plant_id)
        
        if not plant:
            return jsonify({"status": "error", "message": "Plant not found"}), 404

        if 'plant_name' in data:
            plant.plant_name = data['plant_name']
        if 'description' in data:
            plant.description = data['description']
        if 'price' in data:
            plant.price = float(data['price'])
        if 'stock' in data:
            plant.stock = int(data['stock'])

        db.session.commit()

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
def delete_plant_endpoint(plant_id):
    """Delete a plant"""
    try:
        plant = db.session.get(Plant, plant_id)
        
        if not plant:
            return jsonify({"status": "error", "message": "Plant not found"}), 404

        plant_name = plant.plant_name

        # Delete related records first
        db.session.execute(text("DELETE FROM plant_category WHERE plant_id = :id"), {"id": plant_id})
        db.session.execute(text("DELETE FROM variety WHERE plant_id = :id"), {"id": plant_id})
        db.session.execute(text("DELETE FROM cart_new WHERE plant_id = :id"), {"id": plant_id})

        # Delete the plant
        db.session.delete(plant)
        db.session.commit()

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
def search_plants_admin():
    """Search for plants by name"""
    try:
        query = request.args.get('q', '').strip()
        
        if not query or len(query) < 2:
            return jsonify({"status": "error", "message": "Query too short"}), 400

        plants = db.session.execute(text("""
            SELECT plant_id, plant_name, description, price, stock
            FROM plant
            WHERE LOWER(plant_name) LIKE :q
            ORDER BY plant_name ASC
            LIMIT 20
        """), {"q": f"%{query.lower()}%"}).fetchall()

        plant_list = [{
            "plant_id": p[0],
            "plant_name": p[1],
            "description": p[2],
            "price": float(p[3]) if p[3] else 0,
            "stock": p[4] or 0
        } for p in plants]

        return jsonify({
            "status": "success",
            "plants": plant_list,
            "count": len(plant_list)
        })

    except Exception as e:
        print(f"Error in search_plants_admin: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500

    

@app.route('/api/debug/plants-images', methods=['GET'])
def debug_plants_images():
    """Debug: Check plant image paths"""
    import os
    
    plants = db.session.execute(text("""
        SELECT plant_id, plant_name, image_path
        FROM plant
        LIMIT 10
    """)).fetchall()
    
    static_dir = os.path.join(os.getcwd(), 'static')
    
    debug_data = []
    for p in plants:
        plant_id, plant_name, image_path = p
        
        if image_path:
            full_path = os.path.join(static_dir, image_path.lstrip('/'))
            exists = os.path.exists(full_path)
            url = f"http://localhost:5001/static/{image_path.lstrip('/')}"
        else:
            full_path = None
            exists = False
            url = None
        
        debug_data.append({
            "plant_id": plant_id,
            "plant_name": plant_name,
            "image_path": image_path,
            "full_path": full_path,
            "file_exists": exists,
            "url": url
        })
    
    return jsonify({
        "static_dir": static_dir,
        "plants": debug_data,
        "files_in_static": os.listdir(static_dir) if os.path.exists(static_dir) else []
    })                                   
# ---------- ROUTES ----------
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

        response = process_message(user_message, location, user_id)
        
        if response.get("type") == "text":
            speech_response = generate_speech(response["reply"])
            if speech_response["status"] == "success":
                response["audio"] = speech_response["audio"]

        return jsonify(response)

    except Exception as e:
        print(f"Error in /api/chat: {str(e)}")
        return jsonify({
            "status": "error",
            "type": "text",
            "reply": "Sorry, I encountered an error processing your request."
        }), 500




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
            category = Category.query.get(category_id)
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
            
            syn = CategorySynonym.query.get(synonym_id)
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
                print(f"✅ LLM changed to: {LLM_OPTION}")
            
            if "STT_OPTION" in data:
                STT_OPTION = data["STT_OPTION"]
                print(f"✅ STT changed to: {STT_OPTION}")
            
            if "TTS_OPTION" in data:
                TTS_OPTION = data["TTS_OPTION"]
                print(f"✅ TTS changed to: {TTS_OPTION}")
            
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



@app.route('/api/plant/<int:plant_id>/image', methods=['GET', 'POST', 'DELETE'])
def manage_plant_image(plant_id):
    """Upload, retrieve, or delete plant image"""
    try:
        plant = Plant.query.get(plant_id)
        if not plant:
            return jsonify({"status": "error", "message": "Plant not found"}), 404

        if request.method == "GET":
            # Return image from database
            if plant.image_data:
                image_b64 = base64.b64encode(plant.image_data).decode()
                return jsonify({
                    "status": "success",
                    "image_url": f"data:{plant.image_mime_type};base64,{image_b64}",
                    "source": "database",
                    "uploaded_at": plant.uploaded_at.strftime("%Y-%m-%d %H:%M:%S") if plant.uploaded_at else None
                })
            elif plant.image_url:
                return jsonify({
                    "status": "success",
                    "image_url": plant.image_url,
                    "source": "external"
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

            # Store in database
            plant.image_data = image_data
            plant.image_mime_type = file.content_type or 'image/jpeg'
            plant.uploaded_at = datetime.now(timezone.utc)
            db.session.commit()

            return jsonify({
                "status": "success",
                "message": f"Image uploaded for {plant.plant_name}",
                "plant_id": plant_id,
                "size": len(image_data)
            })

        elif request.method == "DELETE":
            # Delete image
            plant.image_data = None
            plant.image_url = None
            plant.uploaded_at = None
            db.session.commit()
            return jsonify({"status": "success", "message": "Image deleted"})

    except Exception as e:
        print(f"Plant image API error: {str(e)}")
        db.session.rollback()
        return jsonify({"status": "error", "message": str(e)}), 500



def is_plant_shop_query_llm(user_message):
    """
    Use LLM to intelligently determine if query is relevant to plant shop.
    Returns: (is_valid, reason, confidence)
    """
    
    prompt = f"""
Determine if this user message is relevant to a PLANT SHOP business.

User message: "{user_message}"

Answer YES if:
- Asking about plants, flowers, gardening
- Prices, availability, delivery
- Care instructions, plant recommendations
- Ordering, cart, account, history
- Any plant-related topics

Answer NO if:
- Weather, sports, news, politics
- Unrelated topics (math, coding, etc)
- Personal/identity questions ("who are you?")
- Topics outside plant shop scope

Respond in JSON format ONLY:
{{
  "is_valid": true/false,
  "reason": "short reason",
  "confidence": 0.0-1.0
}}
"""
    
    try:
        response = llm_call(prompt)
        
        # Parse JSON response
        import json
        result = json.loads(response)
        
        is_valid = result.get("is_valid", False)
        reason = result.get("reason", "Unknown")
        confidence = result.get("confidence", 0.5)
        
        print(f"🤖 LLM Validation: valid={is_valid}, reason='{reason}', confidence={confidence:.0%}")
        
        return is_valid, reason, confidence
        
    except Exception as e:
        print(f"⚠️ LLM validation error: {e}")
        # Fallback to simple keyword check
        return fallback_query_validation(user_message)


def fallback_query_validation(user_message):
    """
    Fallback validation using keyword matching if LLM fails
    """
    text_lower = user_message.lower()
    
    # Plant-related keywords
    plant_keywords = {
        "plant", "plants", "flower", "flowers", "garden", "gardening",
        "indoor", "outdoor", "succulent", "cactus", "tree", "herb",
        "price", "cost", "buy", "purchase", "order", "cart", "delivery",
        "care", "water", "sunlight", "soil", "pot", "grow", "nursery"
    }
    
    # Invalid keywords
    invalid_keywords = {
        "weather", "sports", "news", "politics", "movie", "game",
        "math", "code", "homework", "recipe", "cook", "who are you",
        "tell me about yourself"
    }
    
    # Check for invalid keywords first
    for keyword in invalid_keywords:
        if keyword in text_lower:
            return False, f"Out of scope: {keyword}", 1.0
    
    # Check for plant keywords
    plant_count = sum(1 for kw in plant_keywords if kw in text_lower)
    
    if plant_count >= 1:
        return True, "Plant-related query", 0.7
    
    # Ambiguous
    return False, "Could not classify query", 0.3


def validate_user_query(user_message):
    """
    Main validation function - uses LLM with fallback
    Returns formatted response for invalid queries
    """
    
    is_valid, reason, confidence = is_plant_shop_query_llm(user_message)
    
    if not is_valid:
        responses = {
            "out of scope": "🌿 I'm a plant shop assistant. I can only help with plant-related questions! Try asking about our plants, prices, or delivery. 🚚",
            "unrelated": "I'm sorry, I can only assist with plant shop queries. Is there anything about our plants or orders I can help with? 🌱",
            "personal": "I'm an AI assistant for our plant shop! 🤖 Let me help you find the perfect plant instead. What are you looking for? 🌿",
            "could not": "Sorry, I didn't quite understand. Could you rephrase your question? 🌿\n\nI can help with:\n✅ Plant browsing & prices\n✅ Care instructions\n✅ Orders & delivery\n✅ Inventory"
        }
        
        # Match reason to response
        reply = responses.get("could not")
        for key, msg in responses.items():
            if key.lower() in reason.lower():
                reply = msg
                break
        
        return {
            "status": "invalid",
            "type": "text",
            "reply": reply,
            "reason": reason,
            "confidence": confidence
        }
    
    return {
        "status": "valid",
        "reason": reason,
        "confidence": confidence
    }



@app.route('/api/plants/batch-upload-images', methods=['POST'])
def batch_upload_images():
    """Bulk upload images from Pexels to database"""
    try:
        plants = Plant.query.filter(Plant.image_data.is_(None)).all()
        
        uploaded_count = 0
        failed_count = 0
        
        for plant in plants:
            try:
                img = search_plant_image(plant.plant_name)
                if img.get("status") == "success":
                    # Download and store image
                    response = requests.get(img["image_url"], timeout=10)
                    if response.status_code == 200:
                        plant.image_data = response.content
                        plant.image_url = img["image_url"]
                        plant.image_mime_type = response.headers.get('content-type', 'image/jpeg')
                        plant.uploaded_at = datetime.now(timezone.utc)
                        db.session.add(plant)
                        uploaded_count += 1
                else:
                    failed_count += 1
            except Exception as e:
                print(f"Error uploading image for {plant.plant_name}: {e}")
                failed_count += 1
                continue
        
        db.session.commit()
        return jsonify({
            "status": "success",
            "message": f"Uploaded {uploaded_count} images, {failed_count} failed",
            "uploaded": uploaded_count,
            "failed": failed_count
        })

    except Exception as e:
        print(f"Batch upload error: {str(e)}")
        db.session.rollback()
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/api/voice_vosk', methods=['POST', 'OPTIONS'])
def voice_vosk():
    """Handle voice input using VOSK"""
    
    # ✅ Handle CORS preflight
    if request.method == 'OPTIONS':
        return '', 204
    
    temp_dir = os.path.join(os.getcwd(), 'temp')
    os.makedirs(temp_dir, exist_ok=True)
    temp_webm = None
    temp_wav = None

    try:
        if 'audio' not in request.files:
            return jsonify({'status': 'error', 'message': 'No audio file provided'}), 400

        audio_file = request.files['audio']
        user_id = request.form.get('user_id', 'anonymous')
        timestamp = int(time.time() * 1000)
        temp_webm = os.path.join(temp_dir, f'voice_{timestamp}.webm')
        temp_wav = os.path.join(temp_dir, f'voice_{timestamp}.wav')

        print(f"📝 Received audio file: {audio_file.filename}")
        print(f"👤 User ID: {user_id}")

        # ✅ Save WebM
        audio_file.save(temp_webm)
        print(f"✅ Saved WebM: {temp_webm}")
        
        if not os.path.exists(temp_webm):
            return make_response(jsonify({'status': 'error', 'message': 'Failed to save audio file'}), 400)

        # ✅ Convert WebM to WAV using ffmpeg
        print(f"🔄 Converting to WAV...")
        result = subprocess.run([
            'ffmpeg',
            '-y',
            '-i', temp_webm,
            '-ar', '16000',
            '-ac', '1',
            temp_wav
        ], capture_output=True, text=True, timeout=30)
        
        if result.returncode != 0:
            print(f"❌ FFmpeg error: {result.stderr}")
            return make_response(jsonify({
                'status': 'error', 
                'message': 'FFmpeg conversion failed',
                'details': result.stderr
            }), 400)
        
        print(f"✅ Converted to WAV: {temp_wav}")

        if not os.path.exists(temp_wav):
            return make_response(jsonify({
                'status': 'error', 
                'message': 'WAV file not created'
            }), 400)

        # ✅ Process with VOSK
        if not VOSK_AVAILABLE:
            return jsonify({'status': 'error', 'message': 'Voice recognition is not available on this server'}), 503

        print(f"🎤 Processing with VOSK...")
        rec = KaldiRecognizer(vosk_model, 16000)
        rec.SetWords(["plant", "plants", "indoor", "outdoor", "hello", "show", "image", "picture"])
        
        with wave.open(temp_wav, 'rb') as wf:
            # Verify WAV format
            channels = wf.getnchannels()
            sample_width = wf.getsampwidth()
            framerate = wf.getframerate()
            
            if channels != 1 or sample_width != 2 or framerate != 16000:
                print(f"⚠️ WAV format: channels={channels}, width={sample_width}, rate={framerate}")
                return make_response(jsonify({
                    'status': 'error',
                    'message': f'Invalid WAV format. Expected: 1 channel, 16-bit, 16kHz'
                }), 400)

            print(f"✅ WAV format verified: {channels} channel, {sample_width*8}-bit, {framerate}Hz")
            
            # Process audio in chunks
            while True:
                data = wf.readframes(4000)
                if len(data) == 0:
                    break
                if rec.AcceptWaveform(data):
                    partial = json.loads(rec.PartialResult())
                    print(f"📊 Partial: {partial.get('partial', '')}")
        
        # Get final result
        final_result = rec.FinalResult()
        print(f"📊 VOSK Final Result: {final_result}")
        
        try:
            result_json = json.loads(final_result)
        except json.JSONDecodeError as e:
            print(f"❌ JSON decode error: {e}")
            return make_response(jsonify({
                'status': 'error',
                'message': 'Failed to parse VOSK result'
            }), 500)
        
        # ✅ FIXED: Extract words from result dict array
        transcribed_text = result_json.get('result', [])
        if isinstance(transcribed_text, list) and len(transcribed_text) > 0:
            # result is list of dicts: [{"word": "hello", ...}, ...]
            if isinstance(transcribed_text[0], dict):
                transcribed_text = ' '.join([item.get('word', '') for item in transcribed_text if item.get('word')])
            else:
                # result is list of strings (fallback)
                transcribed_text = ' '.join(str(item) for item in transcribed_text)
        else:
            # Fallback to 'text' field
            transcribed_text = result_json.get('text', '').strip()

        print(f"✅ VOSK TRANSCRIBED: '{transcribed_text}'")

        if not transcribed_text or transcribed_text.strip() == '':
            print("⚠️ No text transcribed")
            return make_response(jsonify({
                'status': 'error',
                'message': "Could not transcribe audio - no speech detected",
                'transcribed_text': ''
            }), 400)

        # ✅ Pass to process_message
        print(f"💬 Processing message: '{transcribed_text}'")
        chat_response = process_message(transcribed_text, user_id=user_id)
        print(f"✅ Chat response status: {chat_response.get('status')}")
        print(f"✅ Chat response type: {chat_response.get('type')}")
        print(f"✅ Chat response reply: {chat_response.get('reply')}")

        # Add audio to response
        if chat_response.get("type") == "text" and chat_response.get("reply"):
            print(f"🔊 Generating speech for: {chat_response['reply']}")
            speech_response = generate_speech(chat_response["reply"])
            if speech_response.get("status") == "success":
                chat_response["audio"] = speech_response["audio"]
                print(f"✅ Speech generated")
            else:
                print(f"⚠️ Speech generation failed: {speech_response}")

        final_response = {
            'status': 'success',
            'transcribed_text': transcribed_text,
            'chat_response': chat_response
        }
        
        print(f"📤 Sending response: status={final_response['status']}, text={final_response['transcribed_text']}")
        
        # ✅ Use make_response to ensure CORS headers
        response = make_response(jsonify(final_response))
        response.headers['Content-Type'] = 'application/json'
        response.headers['Access-Control-Allow-Origin'] = '*'
        response.headers['Access-Control-Allow-Methods'] = 'POST, OPTIONS'
        response.headers['Access-Control-Allow-Headers'] = 'Content-Type'
        
        return response, 200

    except subprocess.TimeoutExpired:
        print(f"❌ FFmpeg timeout")
        return make_response(jsonify({
            'status': 'error',
            'message': 'FFmpeg conversion timeout'
        }), 500)
        
    except Exception as e:
        print(f"❌ Voice VOSK error: {str(e)}")
        import traceback
        traceback.print_exc()
        response = make_response(jsonify({
            'status': 'error',
            'message': str(e)
        }))
        response.headers['Access-Control-Allow-Origin'] = '*'
        return response, 500

    finally:
        # Cleanup
        for temp_file in [temp_webm, temp_wav]:
            if temp_file and os.path.exists(temp_file):
                try:
                    os.unlink(temp_file)
                    print(f"🗑️ Cleaned up: {temp_file}")
                except Exception as e:
                    print(f"⚠️ Failed to delete {temp_file}: {e}")


@app.route('/api/query_results', methods=['GET', 'POST'])
def manage_query_results():
    """Get or store query results"""
    try:
        if request.method == "GET":
            query_id = request.args.get("query_id")
            if query_id:
                results = QueryResult.query.filter_by(query_id=query_id).all()
            else:
                results = QueryResult.query.all()
            
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
            data = request.json
            query_id = data.get("query_id")
            result_description = data.get("result_description")
            result_source = data.get("result_source")  # 'llm' or 'local'
            
            if not query_id or not result_description or not result_source:
                return jsonify({"status": "error", "message": "query_id, result_description, and result_source required"}), 400
            
            if result_source not in ['llm', 'local']:
                return jsonify({"status": "error", "message": "result_source must be 'llm' or 'local'"}), 400
            
            result_id = store_query_result(query_id, result_description, result_source)
            return jsonify({
                "status": "success",
                "message": "Query result stored",
                "result_id": result_id
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
            SET last_logout = NOW(),
                is_online = FALSE
            WHERE id = :uid
        """), {"uid": user_id})

        db.session.commit()

    return jsonify({'status': 'success'})


def get_query_synonym(user_message):
    row = db.session.execute(text("""
        SELECT canonical_key
        FROM query_synonym
        WHERE :msg ILIKE '%' || synonym || '%'
        ORDER BY LENGTH(synonym) DESC
        LIMIT 1
    """), {"msg": user_message.lower()}).fetchone()

    return row[0] if row else None


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
        db.session.execute(text("""
            INSERT INTO cart_new (user_id, plant_id)
            VALUES (:user_id, :plant_id)
            ON CONFLICT (user_id, plant_id) DO NOTHING
        """), {"user_id": user_id, "plant_id": plant_id})

        db.session.commit()

        # ✅ AUTO LOG PURCHASE
        log_user_interest(user_id, plant_id, "purchase", "Added to cart")

        return jsonify({"status": "success", "message": "Added to cart ✅"})

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

    rows = db.session.execute(text("""
        SELECT plant_id
        FROM cart_new
        WHERE user_id = :u
    """), {"u": user_id}).fetchall()

    if not rows:
        return jsonify({"error":"Cart empty"}),400

    for r in rows:

        plant = db.session.get(Plant, r[0])

        if plant.stock <= 0:
            return jsonify({"error":f"{plant.plant_name} out of stock"}),400

        plant.stock -= 1   # ✅ REDUCE HERE

    db.session.execute(text("""
        DELETE FROM cart_new WHERE user_id=:u
    """), {"u": user_id})

    db.session.commit()

    return jsonify({"status":"success","message":"Order placed"})



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
        return jsonify({"status": "success", "message": "Removed from cart ✅"})

    except Exception as e:
        db.session.rollback()
        return jsonify({"status": "error", "message": str(e)}), 500



@app.route("/api/history/<user_id>", methods=["GET"])
def get_history(user_id):
    rows = db.session.execute(text("""
        SELECT query_id, description, created_at
        FROM query
        WHERE user_id = :uid
        ORDER BY created_at DESC
        LIMIT 50
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




if __name__ == "__main__":
    try:
        with app.app_context():
            db.create_all()
    except Exception as e:
        print("⚠️ DB not reachable, skipping create_all:", e)

    port = int(os.getenv("PORT", 5001))
    app.run(host='0.0.0.0', port=port)

