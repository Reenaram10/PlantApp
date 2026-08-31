import os

files_to_update = [
    r"d:\PlantApp\index.html",
    r"d:\PlantApp\shop.html",
    r"d:\PlantApp\auth.html",
    r"d:\PlantApp\plants.js",
    r"d:\PlantApp\chatbot.js",
    r"d:\PlantApp\verify_system.py",
    r"d:\PlantApp\test_fruit_discovery.py",
    r"d:\PlantApp\test_disease_detection.py",
    r"d:\PlantApp\test_diagnostics.py",
    r"d:\PlantApp\test_api_chat.py"
]

for filepath in files_to_update:
    if os.path.exists(filepath):
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Replace occurrences of 8080 with 8082
        new_content = content.replace("8080", "8082")
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(new_content)
        print(f"Updated {filepath}")
    else:
        print(f"File not found: {filepath}")
