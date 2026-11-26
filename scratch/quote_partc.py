import json

data = json.load(open("scratch/logs/partC_20261125_103901.json", encoding="utf-8"))
print("Total requests in partC JSON:", len(data))
for item in data:
    if any(k in item["title"] for k in ["6. Add Plant", "8. Restock", "13. Checkout", "23. Super-Admin Login"]):
        print("===", item["title"], f"[{item['method']} {item['path']}]", "===")
        print("LEGACY:", item["legacy_status"], json.dumps(item["legacy_body"], ensure_ascii=False))
        print("NEW:   ", item["new_status"], json.dumps(item["new_body"], ensure_ascii=False))
