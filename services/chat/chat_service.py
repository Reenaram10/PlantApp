"""
Chat Service
Orchestrates plant consultation, dialog management, inventory queries, multi-turn state machine,
semantic FAISS retrieval, and LLM botanical synthesis.
"""
import os
import csv
import json
import time
import re
from typing import Any
from sqlalchemy import text
from flask import has_app_context, current_app

from config import APP_ROOT
from extensions import db
from models.catalog import Plant, PlantKnowledge
from models.commerce import Cart
from services.observability import log_langfuse_trace
from services.rag_service import faiss_rag, history_faiss
from services.chat.dialog_state import USER_DIALOG_STATE
from services.chat.intent import (
    clean_text, GREETINGS, extract_quantity_nlp,
    classify_affirmation, classify_query_intent, classify_category
)
from services.chat.plant_lookup import (
    fuzzy_resolve_plant, fetch_db_plant_details, check_db_plant_availability,
    format_plant_profile, make_plant_unavailable_guardrail
)

def _handle_dialog_state(user_key, current_dialog_state, user_message, user_id=None):
    """
    Evaluates multi-turn sequential conversation dialogue state machine:
    State 1: AWAITING_PLANT_CONFIRMATION
    State 2: AWAITING_QUANTITY
    Returns response dict if handled, or None if falling through.
    """
    state_type = current_dialog_state.get("state")

    # STATE 1: Awaiting Confirmation ("Do you mean Rose?")
    if state_type == "AWAITING_PLANT_CONFIRMATION":
        aff_result = classify_affirmation(user_message)

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
        # Fall-through when "OTHER": state is preserved, continue to next checks

    # STATE 2: Awaiting Quantity ("How many Rose plants?")
    elif state_type == "AWAITING_QUANTITY":
        plant_info = current_dialog_state.get("plant_info")
        quantity = extract_quantity_nlp(user_message)
        
        plant_id = plant_info["plant_id"]
        plant_name = plant_info["plant_name"]

        plant_db = db.session.get(Plant, plant_id)
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

            if not has_app_context():
                with current_app.app_context():
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

    return None

def _handle_cart_display(normalized_query, user_id, user_message, get_meta):
    """Checks for cart view intent and returns cart overview."""
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
        
        if not has_app_context():
            with current_app.app_context():
                cart_reply = _fetch_cart_reply()
        else:
            cart_reply = _fetch_cart_reply()
            
        log_langfuse_trace("cart_display_query", user_message, cart_reply, get_meta("SUCCESS", {"intent": "VIEW_CART"}))
        return {"status": "success", "type": "text", "reply": cart_reply}
    return None

def _handle_db_inventory_query(user_message, normalized_query, user_key, get_meta):
    """Checks for stock, price, database inventory inquiries."""
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
    return None

def _handle_buy_intent(user_message, normalized_query, user_key):
    """Handles plant purchase/order intent or fuzzy plant match."""
    buy_keywords = ["buy", "order", "purchase", "add to cart", "place order"]
    is_buy_intent = any(k in normalized_query for k in buy_keywords)

    fuzzy_match = fuzzy_resolve_plant(user_message)
    if fuzzy_match and is_buy_intent:
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
    return None

def process_message(user_message, location="", user_id=None, history=None):
    """
    PlantCare AI retrieve-and-format logic using plant_profiles.csv and RAG.
    Tracks sub-component execution timings (latencies) and failure telemetry via Langfuse.
    """
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

    normalized_query = clean_text(user_message)
    query_words = set(normalized_query.split())

    user_key = str(user_id) if user_id else "anonymous"
    current_dialog_state = USER_DIALOG_STATE.get(user_key)

    # 1. MULTI-TURN SEQUENTIAL CONVERSATION DIALOGUE STATE MACHINE
    if current_dialog_state:
        dialog_res = _handle_dialog_state(user_key, current_dialog_state, user_message, user_id=user_id)
        if dialog_res is not None:
            return dialog_res

    # 2. DISPLAY CART INTENT CHECK
    cart_res = _handle_cart_display(normalized_query, user_id, user_message, get_meta)
    if cart_res is not None:
        return cart_res

    # 3. DATABASE & INVENTORY QUERY INTENT CHECK
    inv_res = _handle_db_inventory_query(user_message, normalized_query, user_key, get_meta)
    if inv_res is not None:
        return inv_res

    # 4. BUY INTENT CHECK
    buy_res = _handle_buy_intent(user_message, normalized_query, user_key)
    if buy_res is not None:
        return buy_res

    # 5. HISTORY AND CART CONTEXT
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

    # 6. GREETING CHECK
    if normalized_msg := clean_text(user_message):
        if normalized_msg in GREETINGS:
            reply = "Hello! I am your PlantCare AI assistant. How can I help you today?"
            log_langfuse_trace("greeting_query", user_message, reply, get_meta("SUCCESS", {"intent": "GREETING"}))
            return {
                "status": "success",
                "type": "text",
                "reply": reply
            }

    # 7. RAG RETRIEVAL
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

    # 8. INTENT CLASSIFICATION
    intent = "PROFILE"
    t_intent_start = time.perf_counter()
    try:
        intent = classify_query_intent(user_message, history_context)
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

    # 9. CATEGORY CLASSIFICATION
    rag_category = None
    if intent == "GENERAL":
        t_cat_start = time.perf_counter()
        try:
            rag_category = classify_category(user_message, history_context)
            timings["category_classification_ms"] = round((time.perf_counter() - t_cat_start) * 1000, 2)
        except Exception as e:
            timings["category_classification_ms"] = round((time.perf_counter() - t_cat_start) * 1000, 2)
            failures["category_classification_error"] = str(e)
            rag_category = None

    # 10. CSV SEARCH
    best_plant = None
    best_score = 0
    t_csv_start = time.perf_counter()
    csv_path = os.path.join(APP_ROOT, "plant_profiles.csv")
    
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

    # 11. DB FIRST PLANT LOOKUP CHECK
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

    # 12. CSV PROFILE
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

    # 13. GENERAL RAG ANSWER
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

    # 14. LLM PROFILE GENERATION
    if not best_plant or best_score <= 0 or intent == "PROFILE":
        matched_db_plant = check_db_plant_availability(user_message, failures=failures, timings=timings)
        
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

        # 15. GUARDRAIL
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
