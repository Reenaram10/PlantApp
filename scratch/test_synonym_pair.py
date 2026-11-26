import os, sys, time, json, datetime, subprocess, requests
sys.path.insert(0, os.path.abspath("."))

from sqlalchemy import create_engine, text
from config import SQLALCHEMY_DATABASE_URI

def main():
    env_leg = dict(os.environ, PORT="8083", KERAS_HOME=r"d:\PlantApp\temp\.keras", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", CUDA_VISIBLE_DEVICES="-1")
    env_new = dict(os.environ, PORT="8082", KERAS_HOME=r"d:\PlantApp\temp\.keras", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", CUDA_VISIBLE_DEVICES="-1")

    log_l = open("scratch/syn_leg.log", "w", encoding="utf-8")
    log_n = open("scratch/syn_new.log", "w", encoding="utf-8")
    p_leg = subprocess.Popen(["python", "-u", "app_legacy.py"], stdin=subprocess.PIPE, stdout=log_l, stderr=subprocess.STDOUT, text=True, env=env_leg)
    p_new = subprocess.Popen(["python", "-u", "app.py"], stdin=subprocess.PIPE, stdout=log_n, stderr=subprocess.STDOUT, text=True, env=env_new)

    time.sleep(1)
    for p in (p_leg, p_new):
        try:
            if p.stdin:
                p.stdin.write("n\n")
                p.stdin.flush()
        except Exception:
            pass

    ready_l = ready_n = False
    for _ in range(45):
        time.sleep(2)
        if not ready_l:
            try:
                if requests.get("http://127.0.0.1:8083/health", timeout=1).status_code == 200:
                    ready_l = True
            except Exception:
                pass
        if not ready_n:
            try:
                if requests.get("http://127.0.0.1:8082/health", timeout=1).status_code == 200:
                    ready_n = True
            except Exception:
                pass
        if ready_l and ready_n:
            break

    if not (ready_l and ready_n):
        p_leg.kill(); p_new.kill()
        raise RuntimeError(f"Servers failed to boot: leg={ready_l}, new={ready_n}")

    engine = create_engine(SQLALCHEMY_DATABASE_URI)
    results = []
    cat_id_leg = cat_id_new = None

    try:
        # 1. Create throwaway category TEST_cat (separate rows or shared? Use TEST_cat_leg / TEST_cat or sequential)
        # User asked: "with a throwaway category named TEST_cat: POST a synonym (check synonym_id is an int), POST the same synonym again, GET the list, DELETE it, then delete TEST_cat."
        # Both servers share PlantAppDB, so we create TEST_cat once, test on Legacy, then test on New (or create/delete per server with identical category name 'TEST_cat').
        # Running sequential on the shared DB ensures identical state for 'TEST_cat'!
        for label, base_url in [("legacy", "http://127.0.0.1:8083"), ("new", "http://127.0.0.1:8082")]:
            r_cat = requests.post(f"{base_url}/api/admin/categories", json={"category_name": "TEST_cat"}, timeout=15)
            cat_json = r_cat.json()
            # fetch category_id
            with engine.connect() as conn:
                row = conn.execute(text("SELECT TOP 1 category_id FROM category WHERE category_name = 'TEST_cat' ORDER BY category_id DESC")).fetchone()
                cid = row[0]

            r_post1 = requests.post(f"{base_url}/api/category_synonyms", json={"synonym": "test_syn_xyz", "category_id": cid}, timeout=15)
            post1_json = r_post1.json()
            syn_id = post1_json.get("synonym_id")
            is_int = isinstance(syn_id, int) and not isinstance(syn_id, bool)

            r_post2 = requests.post(f"{base_url}/api/category_synonyms", json={"synonym": "test_syn_xyz", "category_id": cid}, timeout=15)
            r_get = requests.get(f"{base_url}/api/category_synonyms", timeout=15)
            r_del_syn = requests.delete(f"{base_url}/api/category_synonyms", json={"synonym_id": syn_id}, timeout=15)
            r_del_cat = requests.delete(f"{base_url}/api/admin/categories/{cid}", timeout=15)

            results.append({
                "server": label,
                "create_cat": {"status": r_cat.status_code, "body": cat_json, "category_id": cid},
                "post_synonym_1": {"status": r_post1.status_code, "body": post1_json, "synonym_id_is_int": is_int},
                "post_synonym_2_dup": {"status": r_post2.status_code, "body": r_post2.json()},
                "get_synonyms_count": len(r_get.json().get("synonyms", [])),
                "get_synonyms_matching_test": [s for s in r_get.json().get("synonyms", []) if s.get("synonym") == "test_syn_xyz"],
                "delete_synonym": {"status": r_del_syn.status_code, "body": r_del_syn.json()},
                "delete_cat": {"status": r_del_cat.status_code, "body": r_del_cat.json()},
            })
    finally:
        with engine.connect() as conn:
            conn.execute(text("DELETE FROM category_synonym WHERE synonym = 'test_syn_xyz'"))
            conn.execute(text("DELETE FROM category WHERE category_name = 'TEST_cat'"))
            conn.commit()
        p_leg.kill(); p_new.kill()
        p_leg.wait(); p_new.wait()
        log_l.close(); log_n.close()

    os.makedirs("scratch/logs", exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = f"scratch/logs/synonym_test_{ts}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Saved synonym test to {out_path}")
    print(json.dumps(results, indent=2))

if __name__ == "__main__":
    main()
