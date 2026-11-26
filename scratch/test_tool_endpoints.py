import os
import sys
import json

sys.path.insert(0, os.path.abspath('.'))

os.environ['KERAS_HOME'] = r'd:\PlantApp\temp\.keras'
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'

import app_legacy
legacy_app = app_legacy.app
legacy_client = legacy_app.test_client()

from create_app import create_app
new_app = create_app()
new_client = new_app.test_client()

img_abs_path = os.path.abspath("guava.jpg")

print("=" * 70)
print("TEST 1: POST /api/tools/plant-identify")
print("=" * 70)
payload_plant_id = {"image_path": img_abs_path}

res_leg_id = legacy_client.post('/api/tools/plant-identify', json=payload_plant_id)
print(f"Legacy status: {res_leg_id.status_code}")
print(f"Legacy JSON:   {res_leg_id.get_json()}")

res_new_id = new_client.post('/api/tools/plant-identify', json=payload_plant_id)
print(f"New status:    {res_new_id.status_code}")
print(f"New JSON:      {res_new_id.get_json()}")

print("\n" + "=" * 70)
print("TEST 2: POST /api/history/query")
print("=" * 70)
payload_hist_query = {"query": "How often should I water?", "plant_name": "Tomato"}

res_leg_hist = legacy_client.post('/api/history/query', json=payload_hist_query)
print(f"Legacy status: {res_leg_hist.status_code}")
print(f"Legacy JSON:   {res_leg_hist.get_json()}")

res_new_hist = new_client.post('/api/history/query', json=payload_hist_query)
print(f"New status:    {res_new_hist.status_code}")
print(f"New JSON:      {res_new_hist.get_json()}")

print("\n" + "=" * 70)
print("CHECKING RE-IMPORT / MULTIPLE FLASK INSTANCES:")
print("=" * 70)
print(f"sys.modules['app']: {sys.modules.get('app')}")
flask_apps = [v for k, v in list(sys.modules.items()) if hasattr(v, 'app') and hasattr(getattr(v, 'app'), 'url_map')]
print(f"Modules with .app (Flask): {[m.__name__ for m in flask_apps]}")
print("=" * 70)
