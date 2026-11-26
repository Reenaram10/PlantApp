import os
import sys
import re
import io
import json
import time
import random
import difflib
import string
import requests
from contextlib import nullcontext
from datetime import datetime, timezone
from sqlalchemy import text
from flask import has_app_context, current_app

from server.config import APP_ROOT, LLM_OPTION, SYSTEM_PROMPT, GREETINGS
from server.database import db, utc_now
from server.models import (
    Plant, Variety, User, Category, CategorySynonym,
    UserInterest, UserQuery, Query, QueryResult, LLMResult,
    PlantKnowledge, DiseaseKnowledge
)
from server.telemetry import log_langfuse_trace
from server.services.llm_service import llm_call, slm_call, groq_call_failsafe
from server.services.rag_service import faiss_rag, history_faiss

def app_context_guard():
    if has_app_context():
        return nullcontext()
    try:
        import app as _root_app
        return _root_app.app.app_context()
    except Exception:
        return nullcontext()

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
        with app_context_guard():
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
                    with app_context_guard():
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
            with app_context_guard():
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
