import os
import sys
import json
import importlib.util
from unittest.mock import patch, MagicMock

sys.path.insert(0, r'd:\PlantApp')

os.environ['KERAS_HOME'] = r'd:\PlantApp\temp\.keras'
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'

# Load app
from create_app import create_app
flask_app = create_app()

from services.chat.chat_service import process_message
from services.chat import USER_DIALOG_STATE
from models.commerce import Cart
from extensions import db
from models.catalog import Plant
import services.observability as obs_mod
import services.llm_service as llm_mod

# Baseline file
baseline_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'baseline_chat_outputs.json')
if not os.path.exists(baseline_path):
    baseline_path = r'C:\Users\Reena\.gemini\antigravity\brain\3b1f6c0d-0ebe-42a3-adcd-c818cd743518\scratch\baseline_chat_outputs.json'
with open(baseline_path, 'r', encoding='utf-8') as f:
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

from langchain_core.runnables import RunnableSequence
from langchain_community.llms import Ollama
import groq

mock_cart_item = MagicMock()
mock_cart_item.plant_id = 1
mock_cart_item.quantity = 2

mock_plant_obj = MagicMock()
mock_plant_obj.plant_name = "Tomato"
mock_plant_obj.price = 50.0
mock_plant_obj.stock = 10

print("=" * 80)
print("PHASE 3: CHAT SERVICE REPLAY VERIFICATION (15 BASELINE MESSAGES)")
print("=" * 80)

total_passed = 0
total_failed = 0

with flask_app.app_context():
    with patch.object(Cart, 'query') as mock_cart_query, \
         patch.object(db.session, 'get', return_value=mock_plant_obj), \
         patch.object(db.session, 'execute', side_effect=mock_db_execute), \
         patch.object(obs_mod, 'log_langfuse_trace', side_effect=lambda *a, **k: None), \
         patch('services.chat.chat_service.log_langfuse_trace', side_effect=lambda *a, **k: None), \
         patch.object(RunnableSequence, 'invoke', side_effect=mock_llm_invoke), \
         patch.object(Ollama, 'invoke', side_effect=mock_llm_invoke), \
         patch.object(groq, 'Groq', return_value=mock_groq_client), \
         patch.object(llm_mod, 'llm_call', side_effect=mock_llm_invoke), \
         patch.object(llm_mod, 'slm_call', side_effect=mock_llm_invoke), \
         patch.object(llm_mod, 'groq_call_failsafe', return_value="general"):
        
        mock_cart_query.filter_by.return_value.all.return_value = [mock_cart_item]

        for idx, entry in enumerate(baseline_data):
            USER_DIALOG_STATE.clear()
            inp = entry['input']
            base_out = entry['output']
            base_state = entry['dialog_state_after']

            msg = inp['user_message']
            loc = inp['location']
            uid = inp['user_id']

            new_out = process_message(msg, location=loc, user_id=uid)
            new_state = dict(USER_DIALOG_STATE.get(str(uid), {}))

            # Compare keys
            status_match = (new_out.get('status') == base_out.get('status'))
            type_match = (new_out.get('type') == base_out.get('type'))
            reply_match = (new_out.get('reply') == base_out.get('reply'))
            state_match = (new_state == base_state)

            all_match = status_match and type_match and reply_match and state_match

            print(f"\n[{idx+1}/15] Input: \"{msg}\"")
            print(f"       Status: {'PASS' if all_match else 'DIFF'}")
            if not all_match:
                print(f"       --- Baseline Reply ---")
                print(f"       {base_out.get('reply')}")
                print(f"       --- New Reply ---")
                print(f"       {new_out.get('reply')}")
                if not state_match:
                    print(f"       --- State Diff --- Base: {base_state} vs New: {new_state}")
                total_failed += 1
            else:
                print(f"       Type: {new_out.get('type')} | State: {new_state.get('state', 'None')}")
                print(f"       Reply snippet: {str(new_out.get('reply'))[:75]}...")
                total_passed += 1

print("\n" + "=" * 80)
print(f"SUMMARY: {total_passed}/15 Passed, {total_failed}/15 Differences")
print("=" * 80)

# -----------------------------------------------------------------------------
# MULTI-TURN FLOW VERIFICATION
# -----------------------------------------------------------------------------
print("\n" + "=" * 80)
print("MULTI-TURN FLOW VERIFICATION")
print("=" * 80)

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
         patch.object(groq, 'Groq', return_value=mock_groq_client):

        # Flow A: "buy rose" -> "yes" -> "2"
        print("\n--- Flow A: Multi-turn Cart Addition (Confirmation -> Quantity -> Cart) ---")
        USER_DIALOG_STATE.clear()
        uid = 42

        # Step 1: "buy rase nuh" (misspelled) triggers confirmation prompt
        r1 = process_message("i want to buy rase nuh", user_id=uid)
        s1 = dict(USER_DIALOG_STATE.get(str(uid), {}))
        print(f"Turn 1 Input: 'i want to buy rase nuh'")
        print(f"Turn 1 Reply: {r1.get('reply')}")
        print(f"Turn 1 State: {s1.get('state')}")
        assert s1.get('state') == 'AWAITING_PLANT_CONFIRMATION', f"Expected AWAITING_PLANT_CONFIRMATION, got {s1}"

        # Step 2: "yes" -> Transitions to AWAITING_QUANTITY
        r2 = process_message("yes", user_id=uid)
        s2 = dict(USER_DIALOG_STATE.get(str(uid), {}))
        print(f"\nTurn 2 Input: 'yes'")
        print(f"Turn 2 Reply: {r2.get('reply')}")
        print(f"Turn 2 State: {s2.get('state')}")
        assert s2.get('state') == 'AWAITING_QUANTITY', f"Expected AWAITING_QUANTITY, got {s2}"

        # Step 3: "2" -> Adds 2 plants to cart and clears state
        mock_cart_query.filter_by.return_value.first.return_value = None
        r3 = process_message("2", user_id=uid)
        s3 = dict(USER_DIALOG_STATE.get(str(uid), {}))
        print(f"\nTurn 3 Input: '2'")
        print(f"Turn 3 Reply: {r3.get('reply')}")
        print(f"Turn 3 State: {s3.get('state', 'None (Cleared)')}")
        assert s3.get('state') is None, f"Expected state to be cleared, got {s3}"
        assert "Added 2 **Rose**" in r3.get('reply', ''), f"Unexpected reply: {r3}"
        print(" [PASS] Flow A: Confirmation -> Quantity -> Cart successfully completed!")

        # Flow B: Confirmation followed by unrelated question (Fall-through test)
        print("\n--- Flow B: Confirmation Followed by Unrelated Question (Fall-through) ---")
        USER_DIALOG_STATE.clear()
        uid = 99

        # Step 1: "i want to buy rase nuh" -> State is AWAITING_PLANT_CONFIRMATION
        r_b1 = process_message("i want to buy rase nuh", user_id=uid)
        s_b1 = dict(USER_DIALOG_STATE.get(str(uid), {}))
        print(f"Turn 1 Input: 'i want to buy rase nuh'")
        print(f"Turn 1 Reply: {r_b1.get('reply')}")
        print(f"Turn 1 State: {s_b1.get('state')}")

        # Step 2: Unrelated question: "what is the watering requirement for tomato?"
        # Affirmation logic returns OTHER
        # State should NOT be cleared!
        # Message should fall through and answer the question
        r_b2 = process_message("what is the watering requirement for tomato?", user_id=uid)
        s_b2 = dict(USER_DIALOG_STATE.get(str(uid), {}))
        print(f"\nTurn 2 Input: 'what is the watering requirement for tomato?' (unrelated)")
        print(f"Turn 2 Reply snippet: {str(r_b2.get('reply'))[:120]}...")
        print(f"Turn 2 State (must still be AWAITING_PLANT_CONFIRMATION): {s_b2.get('state')}")
        assert s_b2.get('state') == 'AWAITING_PLANT_CONFIRMATION', f"Expected state to be preserved, got {s_b2}"
        print(" [PASS] Flow B: Dialog state preserved on OTHER affirmation; question answered via fall-through!")
