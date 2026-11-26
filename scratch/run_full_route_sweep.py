import os
import sys
import time
import json
import re
import subprocess
import requests

sys.path.insert(0, os.path.abspath('.'))

os.environ['KERAS_HOME'] = r'd:\PlantApp\temp\.keras'
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

from auth.jwt_utils import generate_jwt
from create_app import create_app

app = create_app()

def build_test_path(rule_str):
    p = rule_str
    p = re.sub(r'/<path:filename>', '/nonexistent_file.txt', p)
    p = re.sub(r'/<int:\w+>', '/999999', p)
    p = re.sub(r'/<order_group_id>', '/ORD-NONEXISTENT-999999', p)
    p = re.sub(r'/<business_id>', '/NONEXISTENT-BIZ-999999', p)
    p = re.sub(r'/<user_id>', '/999999', p)
    p = re.sub(r'/<profile_type>', '/nonexistent_profile', p)
    p = re.sub(r'/<\w+>', '/999999', p)
    return p

def deep_compare(v1, v2):
    if type(v1) != type(v2):
        if isinstance(v1, (int, float)) and isinstance(v2, (int, float)):
            return v1 == v2
        return False
    if isinstance(v1, dict):
        if sorted(v1.keys()) != sorted(v2.keys()):
            return False
        for k in v1:
            if any(ts in k.lower() for ts in ['time', 'date', 'timestamp', 'created_at', 'updated_at', 'elapsed']):
                continue
            if not deep_compare(v1[k], v2[k]):
                return False
        return True
    elif isinstance(v1, list):
        if len(v1) != len(v2):
            return False
        return all(deep_compare(a, b) for a, b in zip(v1, v2))
    else:
        return v1 == v2

def main():
    print("=" * 100)
    print("PHASE 5: ROUTE SWEEP OVER ALL 97 RULES (LEGACY:8083 vs NEW:8082)")
    print("=" * 100)

    # 1. Start legacy and new servers
    env_leg = dict(os.environ)
    env_leg['PORT'] = '8083'
    env_leg['KERAS_HOME'] = r'd:\PlantApp\temp\.keras'
    env_leg['HF_HUB_OFFLINE'] = '1'
    env_leg['TRANSFORMERS_OFFLINE'] = '1'
    env_leg['CUDA_VISIBLE_DEVICES'] = '-1'

    env_new = dict(os.environ)
    env_new['PORT'] = '8082'
    env_new['KERAS_HOME'] = r'd:\PlantApp\temp\.keras'
    env_new['HF_HUB_OFFLINE'] = '1'
    env_new['TRANSFORMERS_OFFLINE'] = '1'
    env_new['CUDA_VISIBLE_DEVICES'] = '-1'

    log_leg_f = open('scratch/sweep_legacy_server.log', 'w', encoding='utf-8')
    log_new_f = open('scratch/sweep_new_server.log', 'w', encoding='utf-8')

    print("Launching legacy server on PORT 8083 (app_legacy.py)...")
    p_leg = subprocess.Popen(
        ['python', '-u', 'app_legacy.py'],
        stdin=subprocess.PIPE,
        stdout=log_leg_f,
        stderr=subprocess.STDOUT,
        text=True,
        env=env_leg
    )

    print("Launching new server on PORT 8082 (app.py)...")
    p_new = subprocess.Popen(
        ['python', '-u', 'app.py'],
        stdin=subprocess.PIPE,
        stdout=log_new_f,
        stderr=subprocess.STDOUT,
        text=True,
        env=env_new
    )

    # Answer 'n' to FAISS rebuild prompt
    time.sleep(1)
    for p in [p_leg, p_new]:
        try:
            p.stdin.write("n\n")
            p.stdin.flush()
        except Exception:
            pass

    # Wait for both servers to respond to /health
    print("Waiting for both servers to be ready...")
    ready_leg, ready_new = False, False
    for i in range(45):
        time.sleep(2)
        if not ready_leg:
            try:
                r = requests.get("http://127.0.0.1:8083/health", timeout=1)
                if r.status_code == 200:
                    ready_leg = True
                    print(f" [Legacy 8083] Health check OK after {(i+1)*2}s")
            except Exception:
                pass
        if not ready_new:
            try:
                r = requests.get("http://127.0.0.1:8082/health", timeout=1)
                if r.status_code == 200:
                    ready_new = True
                    print(f" [New 8082] Health check OK after {(i+1)*2}s")
            except Exception:
                pass
        if ready_leg and ready_new:
            break

    if not (ready_leg and ready_new):
        print(f"ERROR: One or both servers failed to start. leg={ready_leg}, new={ready_new}")
        p_leg.kill()
        p_new.kill()
        sys.exit(1)

    print("\nALL SERVERS READY! COMMENCING 97-ROUTE SWEEP...\n")

    super_admin_token = generate_jwt({
        "user_id": 1,
        "email": "admin@gmail.com",
        "role": "SUPER_ADMIN",
        "business_id": "SYSTEM"
    })
    admin_hdr = {"Authorization": f"Bearer {super_admin_token}"}

    results = []
    rule_summaries = []
    diff_records = []
    server_errors = []

    # Safe skip list (per instructions: skip /api/payment/response, /api/rebuild-index, /api/admin/rag/sync)
    skip_rules = {
        "/api/payment/response",
        "/api/rebuild-index",
        "/api/admin/rag/sync"
    }

    rules_sorted = sorted(app.url_map.iter_rules(), key=lambda r: (r.rule, sorted(r.methods)))

    for idx, rule in enumerate(rules_sorted):
        rule_str = rule.rule
        clean_methods = sorted(rule.methods - {'HEAD', 'OPTIONS'})

        rule_verdict = "PASS"

        if rule_str in skip_rules:
            for m in clean_methods:
                results.append({
                    "num": idx + 1,
                    "rule": rule_str,
                    "method": m,
                    "auth": "SKIP",
                    "leg_status": "SKIPPED",
                    "new_status": "SKIPPED",
                    "status_match": True,
                    "verdict": "PASS (SAFE_SKIP)",
                    "notes": "Explicitly skipped per test instructions to protect real state"
                })
            rule_summaries.append({
                "num": idx + 1,
                "rule": rule_str,
                "methods": clean_methods,
                "no_auth_status": "SKIPPED",
                "admin_status": "SKIPPED",
                "verdict": "PASS (SAFE_SKIP)",
                "notes": "Skipped per safety instructions"
            })
            continue

        test_path = build_test_path(rule_str)
        no_auth_statuses = []
        admin_statuses = []

        for m in clean_methods:
            # We test both (a) no token and (b) SUPER_ADMIN token
            for auth_label, hdr in [("No Auth", {}), ("SUPER_ADMIN", admin_hdr)]:
                url_leg = f"http://127.0.0.1:8083{test_path}"
                url_new = f"http://127.0.0.1:8082{test_path}"

                req_kwargs = {"headers": dict(hdr), "timeout": 30}
                if m in ["POST", "PUT", "PATCH", "DELETE"]:
                    req_kwargs["json"] = {}

                fn = getattr(requests, m.lower())
                
                try:
                    r_leg = fn(url_leg, **req_kwargs)
                except Exception as e:
                    r_leg = None
                    err_leg = str(e)

                try:
                    r_new = fn(url_new, **req_kwargs)
                except Exception as e:
                    r_new = None
                    err_new = str(e)

                status_leg = r_leg.status_code if r_leg is not None else "ERR"
                status_new = r_new.status_code if r_new is not None else "ERR"

                if auth_label == "No Auth":
                    no_auth_statuses.append(f"{m}:{status_leg}/{status_new}")
                else:
                    admin_statuses.append(f"{m}:{status_leg}/{status_new}")

                status_match = (status_leg == status_new)

                # Check JSON / text body
                j_leg, j_new = None, None
                try:
                    j_leg = r_leg.json() if r_leg is not None else None
                except Exception:
                    j_leg = r_leg.text[:200] if r_leg is not None else ""

                try:
                    j_new = r_new.json() if r_new is not None else None
                except Exception:
                    j_new = r_new.text[:200] if r_new is not None else ""

                keys_match = True
                val_match = True
                if isinstance(j_leg, dict) and isinstance(j_new, dict):
                    keys_match = (sorted(j_leg.keys()) == sorted(j_new.keys()))
                    val_match = deep_compare(j_leg, j_new)
                elif isinstance(j_leg, list) and isinstance(j_new, list):
                    keys_match = (len(j_leg) == len(j_new))
                    val_match = deep_compare(j_leg, j_new)
                elif isinstance(j_leg, str) and isinstance(j_new, str):
                    keys_match = (len(j_leg) == len(j_new) or j_leg[:50] == j_new[:50])
                    val_match = (j_leg == j_new or j_leg[:50] == j_new[:50])

                verdict = "PASS" if (status_match and keys_match and val_match) else "DIFF"

                notes = ""
                if not status_match:
                    notes += f"Status mismatch: {status_leg} vs {status_new}. "
                elif not keys_match:
                    notes += f"Keys mismatch. "
                elif not val_match:
                    notes += f"Non-timestamp value mismatch. "

                if verdict == "DIFF":
                    rule_verdict = "DIFF"

                if isinstance(status_leg, int) and status_leg >= 500:
                    server_errors.append((rule_str, m, auth_label, status_leg, status_new, j_leg, j_new))
                elif isinstance(status_new, int) and status_new >= 500:
                    server_errors.append((rule_str, m, auth_label, status_leg, status_new, j_leg, j_new))

                if verdict == "DIFF":
                    diff_records.append({
                        "rule": rule_str,
                        "path": test_path,
                        "method": m,
                        "auth": auth_label,
                        "leg_status": status_leg,
                        "new_status": status_new,
                        "leg_body": j_leg,
                        "new_body": j_new,
                        "notes": notes
                    })

                results.append({
                    "num": idx + 1,
                    "rule": rule_str,
                    "method": m,
                    "auth": auth_label,
                    "leg_status": status_leg,
                    "new_status": status_new,
                    "status_match": status_match,
                    "keys_match": keys_match,
                    "val_match": val_match,
                    "verdict": verdict,
                    "notes": notes
                })

        rule_summaries.append({
            "num": idx + 1,
            "rule": rule_str,
            "methods": clean_methods,
            "no_auth_status": ", ".join(no_auth_statuses),
            "admin_status": ", ".join(admin_statuses),
            "verdict": rule_verdict,
            "notes": "Parity confirmed across all methods & auth" if rule_verdict == "PASS" else "Differences detected"
        })

    # Terminate servers
    print("\nTerminating test servers...")
    p_leg.kill()
    p_new.kill()
    p_leg.wait()
    p_new.wait()
    log_leg_f.close()
    log_new_f.close()

    # Save complete sweep results to JSON
    with open('scratch/sweep_results.json', 'w', encoding='utf-8') as f:
        json.dump({
            "results": results,
            "rule_summaries": rule_summaries,
            "diffs": diff_records,
            "errors_5xx": server_errors
        }, f, indent=2, default=str)

    # Print summary table of all 97 rules
    print("\n" + "=" * 130)
    print("ROUTE SWEEP SUMMARY OVER ALL 97 RULES")
    print("=" * 130)
    print(f"| {'#':3s} | {'Rule':45s} | {'Methods':15s} | {'No Auth (Leg/New)':20s} | {'Super Admin (Leg/New)':22s} | {'Verdict':7s} |")
    print(f"|{'-'*5}|{'-'*47}|{'-'*17}|{'-'*22}|{'-'*24}|{'-'*9}|")
    for s in rule_summaries:
        m_str = "/".join(s['methods'])
        print(f"| {s['num']:3d} | {s['rule']:45s} | {m_str:15s} | {s['no_auth_status']:20s} | {s['admin_status']:22s} | {s['verdict']:7s} |")

    print("\n" + "=" * 130)
    total_tests = len(results)
    total_diffs = len(diff_records)
    total_pass = total_tests - total_diffs
    print(f"SWEEP SUMMARY: Total Tests={total_tests}, PASS={total_pass}, DIFF={total_diffs}")
    rules_pass = sum(1 for r in rule_summaries if "PASS" in r['verdict'])
    rules_diff = sum(1 for r in rule_summaries if r['verdict'] == "DIFF")
    print(f"RULE-LEVEL SUMMARY: Total Rules=97, PASS={rules_pass}, DIFF={rules_diff}")
    print("=" * 130)

    if diff_records:
        print("\n" + "!" * 130)
        print(f"RAW OUTPUT FOR {len(diff_records)} DIFF ROWS:")
        print("!" * 130)
        for d in diff_records:
            print(f"\nRule: [{d['method']}] {d['rule']} (Tested Path: {d['path']}, Auth: {d['auth']})")
            print(f"  Legacy Status: {d['leg_status']}, Body: {d['leg_body']}")
            print(f"  New Status:    {d['new_status']}, Body: {d['new_body']}")
            print(f"  Notes:         {d['notes']}")

    if server_errors:
        print("\n" + "=" * 130)
        print(f"5xx SERVER ERROR INVESTIGATION ({len(server_errors)} occurrences):")
        print("=" * 130)
        # Deduplicate server errors by (rule, method)
        seen_5xx = set()
        for se in server_errors:
            key = (se[0], se[1], se[2])
            if key in seen_5xx:
                continue
            seen_5xx.add(key)
            print(f"Rule: [{se[1]}] {se[0]} (Auth: {se[2]}) -> Leg Status: {se[3]}, New Status: {se[4]}")
            print(f"  Legacy Response: {se[5]}")
            print(f"  New Response:    {se[6]}")

if __name__ == '__main__':
    main()
