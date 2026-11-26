import json

d = json.load(open("scratch/logs/ast_compare_20261125_101232.json", encoding="utf-8"))
for item in d["differing"]:
    lines = [
        l for l in item["diff"].splitlines()
        if (l.startswith("+") or l.startswith("-"))
        and not l.startswith("+++")
        and not l.startswith("---")
    ]
    print(f"{item['name']} ({item['new_path']}:{item['new_line']}) [{len(lines)} diff lines]")
    for l in lines[:8]:
        print("  ", l[:120])
    if len(lines) > 8:
        print(f"   ... (+{len(lines) - 8} more)")
