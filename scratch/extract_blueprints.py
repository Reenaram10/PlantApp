import os
import re
import ast

os.makedirs('routes', exist_ok=True)

with open('app.py', 'r', encoding='utf-8') as f:
    app_text = f.read()

# Let's verify route mappings
print("Verifying route extraction setup...")
