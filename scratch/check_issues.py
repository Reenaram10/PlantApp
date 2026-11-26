with open('app.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

print("--- Searching for g. ---")
for i, line in enumerate(lines):
    if 'g.' in line:
        print(f"{i+1}: {line.strip()[:100]}")

print("--- Searching for info['file'] or info[\"file\"] ---")
for i, line in enumerate(lines):
    if 'info[' in line and 'file' in line:
        print(f"{i+1}: {line.strip()[:100]}")
