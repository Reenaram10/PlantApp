"""
LLM and SLM Inference Service
Unified provider router (Groq, DeepSeek, Gemini, Local Ollama SLM) with failsafe fallback
and database caching for botanical AI queries.
"""
import os
import re
import requests
from datetime import datetime, timezone
from sqlalchemy import text
from groq import Groq
from openai import OpenAI
import google.generativeai as genai

from config import runtime_settings
from extensions import db
from models.chat import LLMResult

def normalize_query(query_text):
    """Normalize query text, keeping alphanumeric characters, spaces, and emojis."""
    if not query_text:
        return ""
    query_text = query_text.lower().strip()
    query_text = re.sub(r'[^\w\s\u2600-\u27BF\U0001f300-\U0001faff]', ' ', query_text)
    query_text = re.sub(r'\s+', ' ', query_text).strip()
    return query_text

def groq_call_failsafe(prompt):
    """
    Failsafe method using simple botanical keyword matching.
    Works without external API calls or network connectivity.
    """
    print(" Using failsafe LLM (keyword matching)...")
    prompt_lower = prompt.lower()
    
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

def llm_call(prompt, provider=None):
    """
    Unified LLM caller routing to Groq, DeepSeek, Gemini, SLM, or local failsafe.
    Accepts explicit provider parameter or defaults to runtime_settings.llm_option.
    """
    opt = provider or runtime_settings.llm_option

    # GROQ
    if opt == "groq":
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

    # DEEPSEEK (OpenAI-Compatible API)
    elif opt == "deepseek":
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

    # GEMINI
    elif opt == "gemini":
        try:
            genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
            model = genai.GenerativeModel("gemini-1.5-flash")
            response = model.generate_content(prompt)
            return response.text.strip()
        except Exception as e:
            print(f" Gemini error: {e}")
            return groq_call_failsafe(prompt)

    # SLM (Small Language Model via Local Ollama)
    elif opt == "slm":
        return slm_call(prompt)

    # LOCAL fallback
    elif opt == "local":
        return groq_call_failsafe(prompt)

    else:
        print(f" Unknown LLM option: {opt}")
        return groq_call_failsafe(prompt)

def slm_call(prompt, fallback_provider="groq"):
    """
    Calls a specialized Small Language Model (Phi-3) via Local Ollama API.
    Falls back gracefully without mutating runtime settings or global variables.
    """
    api_url = "http://127.0.0.1:11434/api/generate"
    model_id = "phi3"
    
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
        
    # FALLBACK: Try fallback provider (e.g. Groq) without modifying globals()
    print(" [SLM] Falling back to Groq for better quality...")
    try:
        expert_prompt = f"Expert Advice Needed: {prompt}. Please answer as a botanical expert."
        return llm_call(expert_prompt, provider=fallback_provider)
    except Exception:
        return groq_call_failsafe(prompt)

def get_cached_llm_result(normalized_query):
    """Retrieve unexpired cached LLM response for a normalized query."""
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

def store_canonical_answer(canonical_key, reply, user_message=None):
    """Persist canonical LLM response in database for caching."""
    try:
        row = LLMResult(
            user_query=normalize_query(user_message) if user_message else canonical_key,
            llm_response=reply,
            category_suggested=canonical_key,
            plants_found=0,
            llm_name=runtime_settings.llm_option
        )
        db.session.add(row)
        db.session.commit()
    except Exception as e:
        db.session.rollback()
        print(" Canonical store failed:", e)
