import json

data = json.load(open('scratch/sweep_results.json', encoding='utf-8'))
for s in data['rule_summaries']:
    m_str = "/".join(s['methods'])
    print(f"| {s['num']:2d} | `{s['rule']}` | {m_str} | {s['no_auth_status']} | {s['admin_status']} | **{s['verdict']}** |")
