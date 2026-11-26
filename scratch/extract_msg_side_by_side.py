import os
import sys
import json
from unittest.mock import patch, MagicMock

sys.path.insert(0, r'd:\PlantApp')

os.environ['KERAS_HOME'] = r'd:\PlantApp\temp\.keras'
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'

from create_app import create_app
flask_app = create_app()

from services.chat.chat_service import process_message
from services.chat import USER_DIALOG_STATE
from models.commerce import Cart
from extensions import db
import services.observability as obs_mod
import services.llm_service as llm_mod
from langchain_core.runnables import RunnableSequence
from langchain_community.llms import Ollama
import groq

with open('scratch/baseline_chat_outputs.json', 'r', encoding='utf-8') as f:
    baseline_data = json.load(f)

def mock_llm_invoke(*args, **kwargs):
    prompt_str = ""
    for a in args:
        if isinstance(a, str):
            prompt_str = a
            break
        elif isinstance(a, dict):
            prompt_str = str(a)
            break
    if not prompt_str and "input" in kwargs:
        prompt_str = str(kwargs["input"])
    
    prompt_l = prompt_str.lower()
    if "classify" in prompt_l or "intent" in prompt_l:
        if "france" in prompt_l or "capital" in prompt_l:
            return "OFF_TOPIC"
        elif "care" in prompt_l or "profile" in prompt_l:
            return "PROFILE"
        return "GENERAL"
    return "Mocked botanical LLM response for: " + prompt_str[:50]

def mock_db_execute(statement, params=None):
    sql = str(statement).upper()
    mock_res = MagicMock()
    if "COUNT(*)" in sql:
        mock_res.fetchone.return_value = (15,)
    elif "FROM PLANT" in sql and "TOP 1" in sql:
        mock_res.fetchone.return_value = (1, "Tomato", "Fresh juicy tomato plant", 50.0, "/static/uploads/tomato.jpg")
    elif "FROM PLANT" in sql:
        mock_res.fetchall.return_value = [
            (1, "Rose", 250.0, "/static/uploads/rose.jpg", "gulab"),
            (2, "Tomato", 50.0, "/static/uploads/tomato.jpg", "tamatar"),
            (3, "Hibiscus", 150.0, "/static/uploads/hibiscus.jpg", "jaswand")
        ]
        mock_res.fetchone.return_value = (1, "Rose", 250.0, "/static/uploads/rose.jpg", "gulab")
    elif "CART" in sql:
        mock_res.fetchall.return_value = [
            (1, "Rose", 250.0, 2, 500.0)
        ]
        mock_res.fetchone.return_value = (1, "Rose", 250.0, 2, 500.0)
    else:
        mock_res.fetchall.return_value = []
        mock_res.fetchone.return_value = None
    return mock_res

mock_groq_resp = MagicMock()
mock_groq_resp.choices = [MagicMock(message=MagicMock(content="Mocked Groq expert response"))]
mock_groq_client = MagicMock()
mock_groq_client.chat.completions.create.return_value = mock_groq_resp

mock_cart_item = MagicMock()
mock_cart_item.plant_id = 1
mock_cart_item.quantity = 2

mock_plant_obj = MagicMock()
mock_plant_obj.plant_name = "Tomato"
mock_plant_obj.price = 50.0
mock_plant_obj.stock = 10

def mock_affirmation(user_input):
    clean = str(user_input).strip().lower()
    if any(w in clean for w in ["yes", "sure", "add it", "ok", "yep"]):
        return "POSITIVE"
    elif any(w in clean for w in ["no", "nah", "cancel", "dont", "nope"]):
        return "NEGATIVE"
    return "OTHER"

with flask_app.app_context():
    with patch.object(Cart, 'query') as mock_cart_query, \
         patch.object(db.session, 'get', return_value=mock_plant_obj), \
         patch.object(db.session, 'execute', side_effect=mock_db_execute), \
         patch.object(obs_mod, 'log_langfuse_trace', side_effect=lambda *a, **k: None), \
         patch('services.chat.chat_service.log_langfuse_trace', side_effect=lambda *a, **k: None), \
         patch('services.chat.chat_service.classify_affirmation', side_effect=mock_affirmation), \
         patch.object(RunnableSequence, 'invoke', side_effect=mock_llm_invoke), \
         patch.object(Ollama, 'invoke', side_effect=mock_llm_invoke), \
         patch.object(groq, 'Groq', return_value=mock_groq_client), \
         patch.object(llm_mod, 'llm_call', side_effect=mock_llm_invoke), \
         patch.object(llm_mod, 'slm_call', side_effect=mock_llm_invoke), \
         patch.object(llm_mod, 'groq_call_failsafe', return_value="general"):

        mock_cart_query.filter_by.return_value.all.return_value = [mock_cart_item]

        for idx in [0, 7, 13]:
            USER_DIALOG_STATE.clear()
            entry = baseline_data[idx]
            inp = entry['input']
            base_out = entry['output']
            new_out = process_message(inp['user_message'], location=inp['location'], user_id=inp['user_id'])
            print("=" * 80)
            print(f"MESSAGE {idx+1}: Input: \"{inp['user_message']}\"")
            print("-" * 80)
            print("BASELINE REPLY:")
            print(base_out['reply'])
            print("-" * 40)
            print("NEW REPLY:")
            print(new_out['reply'])
            print(f"EXACT MATCH: {base_out['reply'] == new_out['reply']}")
            print("=" * 80 + "\n")

        # Multi-turn flow
        print("=" * 80)
        print("MULTI-TURN FLOW COMPARISON (Turn by Turn)")
        print("=" * 80)
        USER_DIALOG_STATE.clear()
        uid = 42
        r1 = process_message("i want to buy rase nuh", user_id=uid)
        print("Turn 1 ('i want to buy rase nuh'):")
        print(f"  Reply: {r1['reply']}")
        r2 = process_message("yes", user_id=uid)
        print("Turn 2 ('yes'):")
        print(f"  Reply: {r2['reply']}")
        mock_cart_query.filter_by.return_value.first.return_value = None
        r3 = process_message("2", user_id=uid)
        print("Turn 3 ('2'):")
        print(f"  Reply: {r3['reply']}")
        print("=" * 80)
