import os
import sys

sys.path.insert(0, os.path.abspath('.'))

os.environ['KERAS_HOME'] = r'd:\PlantApp\temp\.keras'
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'

# 1. Blueprint registration order
from create_app import create_app
new_app = create_app()

print("=" * 60)
print("BLUEPRINT REGISTRATION ORDER:")
print("=" * 60)
for idx, (name, bp) in enumerate(new_app.blueprints.items(), 1):
    print(f" {idx:2d}. {name} -> {bp.import_name}")
is_static_last = list(new_app.blueprints.keys())[-1] == 'static_routes'
print(f"Static routes is last: {is_static_last}")

# 2. Route Diff
print("=" * 60)
print("ROUTE DIFF COMPARISON (app_legacy vs new app):")
print("=" * 60)
import app_legacy
legacy_app = app_legacy.app

def get_routes(app):
    rules = {}
    for r in app.url_map.iter_rules():
        methods = tuple(sorted(r.methods - {'HEAD', 'OPTIONS'}))
        key = (r.rule, methods)
        if key not in rules:
            rules[key] = []
        rules[key].append(r.endpoint)
    return rules

legacy_routes = get_routes(legacy_app)
new_routes = get_routes(new_app)

print(f"Total rule keys in legacy: {len(legacy_routes)}")
print(f"Total rule keys in new:    {len(new_routes)}")

missing = set(legacy_routes.keys()) - set(new_routes.keys())
added = set(new_routes.keys()) - set(legacy_routes.keys())

print(f"Missing in new (expected 0): {len(missing)}")
if missing:
    for m in sorted(missing):
        print(f"  MISSING: {m}")

print(f"Added in new (expected 0):   {len(added)}")
if added:
    for a in sorted(added):
        print(f"  ADDED: {a}")

# Check duplicate /admin in legacy
print("\nChecking removed duplicate /admin in legacy:")
for key, endpoints in legacy_routes.items():
    if key[0] in ['/admin', '/admin.html']:
        print(f"  Legacy rule {key}: endpoints={endpoints}")

for key, endpoints in new_routes.items():
    if key[0] in ['/admin', '/admin.html']:
        print(f"  New rule {key}: endpoints={endpoints}")

print("=" * 60)
print("ROUTE DIFF VERIFICATION PASSED:", len(missing) == 0 and len(added) == 0)
print("=" * 60)
