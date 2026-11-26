import os
import sys
import json

sys.path.insert(0, os.path.abspath('.'))

os.environ['KERAS_HOME'] = r'd:\PlantApp\temp\.keras'
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'

import app_legacy
legacy_app = app_legacy.app

from create_app import create_app
new_app = create_app()

from auth.jwt_utils import generate_jwt
valid_token = generate_jwt({
    "user_id": 1,
    "email": "admin@gmail.com",
    "role": "SUPER_ADMIN",
    "business_id": "SYSTEM"
})

auth_header = {"Authorization": f"Bearer {valid_token}"}

legacy_client = legacy_app.test_client()
new_client = new_app.test_client()

test_cases = [
    {
        "name": "Health check (/health)",
        "method": "GET",
        "path": "/health",
        "kwargs": {}
    },
    {
        "name": "Config GET (/api/config)",
        "method": "GET",
        "path": "/api/config",
        "kwargs": {}
    },
    {
        "name": "Config POST (/api/config)",
        "method": "POST",
        "path": "/api/config",
        "kwargs": {"json": {"LLM_OPTION": "groq"}}
    },
    {
        "name": "Shop plants (/api/shop/plants)",
        "method": "GET",
        "path": "/api/shop/plants",
        "kwargs": {}
    },
    {
        "name": "Cart GET (/api/cart?user_id=1)",
        "method": "GET",
        "path": "/api/cart?user_id=1",
        "kwargs": {}
    },
    {
        "name": "Chat greeting (/api/chat)",
        "method": "POST",
        "path": "/api/chat",
        "kwargs": {"json": {"message": "hello", "user_id": 1}}
    },
    {
        "name": "Chat buy flow (/api/chat)",
        "method": "POST",
        "path": "/api/chat",
        "kwargs": {"json": {"message": "buy rose", "user_id": 1}}
    },
    {
        "name": "Auth login alias (/login)",
        "method": "POST",
        "path": "/login",
        "kwargs": {"json": {"email": "admin@gmail.com", "password": "wrongpassword"}}
    },
    {
        "name": "Auth login API (/api/auth/login)",
        "method": "POST",
        "path": "/api/auth/login",
        "kwargs": {"json": {"email": "admin@gmail.com", "password": "wrongpassword"}}
    },
    {
        "name": "Admin inventory (No Token)",
        "method": "GET",
        "path": "/api/admin/inventory",
        "kwargs": {}
    },
    {
        "name": "Admin inventory (With Token)",
        "method": "GET",
        "path": "/api/admin/inventory",
        "kwargs": {"headers": auth_header}
    },
    {
        "name": "Orders by user (/api/orders/1)",
        "method": "GET",
        "path": "/api/orders/1",
        "kwargs": {}
    },
    {
        "name": "WhatsApp cart (/api/whatsapp/cart?phone=9876543210)",
        "method": "GET",
        "path": "/api/whatsapp/cart?phone=9876543210",
        "kwargs": {}
    },
    {
        "name": "Plant diagnosis empty (/api/plant-diagnosis)",
        "method": "POST",
        "path": "/api/plant-diagnosis",
        "kwargs": {"json": {}}
    },
    {
        "name": "Admin dashboard (/admin)",
        "method": "GET",
        "path": "/admin",
        "kwargs": {}
    }
]

print("=" * 90)
print(f"{'TEST NAME':<35} | {'LEGACY':<18} | {'NEW':<18} | {'MATCH':<6}")
print("=" * 90)

all_passed = True

for tc in test_cases:
    name: str = str(tc["name"])
    method: str = str(tc["method"]).lower()
    path: str = str(tc["path"])
    kwargs_val = tc["kwargs"]
    kwargs: dict = kwargs_val if isinstance(kwargs_val, dict) else {}

    # Send request to legacy
    leg_func = getattr(legacy_client, method)
    leg_res = leg_func(path, **kwargs)
    
    # Send request to new
    new_func = getattr(new_client, method)
    new_res = new_func(path, **kwargs)

    leg_status = leg_res.status_code
    new_status = new_res.status_code

    try:
        leg_json = leg_res.get_json()
        leg_keys = sorted(list(leg_json.keys())) if isinstance(leg_json, dict) else f"list({len(leg_json)})"
    except Exception:
        leg_keys = f"HTML({len(leg_res.data)}b)"

    try:
        new_json = new_res.get_json()
        new_keys = sorted(list(new_json.keys())) if isinstance(new_json, dict) else f"list({len(new_json)})"
    except Exception:
        new_keys = f"HTML({len(new_res.data)}b)"

    status_match = (leg_status == new_status)
    keys_match = (leg_keys == new_keys)
    match = status_match and keys_match

    if not match:
        all_passed = False

    leg_summary = f"{leg_status} {str(leg_keys)[:12]}"
    new_summary = f"{new_status} {str(new_keys)[:12]}"
    status_str = "YES" if match else "DIFF"

    print(f"{name:<35} | {leg_summary:<18} | {new_summary:<18} | {status_str:<6}")
    if not match:
        print(f"   --> Legacy: {leg_status}, keys: {leg_keys}")
        print(f"   --> New:    {new_status}, keys: {new_keys}")

print("=" * 90)
print(f"ALL TESTS MATCHED: {all_passed}")
print("=" * 90)
