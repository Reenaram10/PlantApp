import json

data = json.load(open("scratch/logs/partC_20261125_112624.json", encoding="utf-8"))
IGNORE_KEYS = {"user_id", "plant_id", "id", "order_id", "token", "order_group_id", "tracking_number", "timestamp", "created_at", "updated_at", "order_date", "audio"}

def strip_ignored(obj):
    if isinstance(obj, dict):
        return {k: strip_ignored(v) for k, v in obj.items() if k not in IGNORE_KEYS}
    if isinstance(obj, list):
        return [strip_ignored(x) for x in obj]
    return obj

diffs = []
print("| # | Request | Legacy | New | Keys Equal? | Values Equal? |")
print("| :-: | :--- | :---: | :---: | :---: | :---: |")
for idx, r in enumerate(data, 1):
    lb = r["legacy_body"]
    nb = r["new_body"]
    if isinstance(lb, dict) and isinstance(nb, dict):
        k_eq = sorted(lb.keys()) == sorted(nb.keys())
        v_eq = strip_ignored(lb) == strip_ignored(nb)
    elif isinstance(lb, list) and isinstance(nb, list):
        k_eq = len(lb) == len(nb)
        v_eq = strip_ignored(lb) == strip_ignored(nb)
    else:
        k_eq = True
        v_eq = (lb == nb)
    print(f"| {idx} | {r['title']} (`{r['method']} {r['path']}`) | {r['legacy_status']} | {r['new_status']} | {k_eq} | {v_eq} |")
    if r["legacy_status"] != r["new_status"] or not k_eq or not v_eq:
        diffs.append((idx, r))

print("\nDIFF_COUNT:", len(diffs))
for idx, r in diffs:
    print(f"\n--- DIFF #{idx}: {r['title']} ---")
    print("LEGACY:", r["legacy_status"], json.dumps(r["legacy_body"], ensure_ascii=False)[:600])
    print("NEW:   ", r["new_status"], json.dumps(r["new_body"], ensure_ascii=False)[:600])
