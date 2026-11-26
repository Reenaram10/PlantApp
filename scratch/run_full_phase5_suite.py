import os
import sys
import time
import json
import subprocess
import requests

sys.path.insert(0, os.path.abspath('.'))

from auth.jwt_utils import generate_jwt
from config import SQLALCHEMY_DATABASE_URI
from sqlalchemy import create_engine, text

engine = create_engine(SQLALCHEMY_DATABASE_URI)

def format_json_for_display(data):
    if isinstance(data, dict):
        clean = {}
        for k, v in data.items():
            if k == "audio" and isinstance(v, str) and len(v) > 80:
                clean[k] = f"<base64 audio len={len(v)}>"
            elif isinstance(v, (dict, list)):
                clean[k] = format_json_for_display(v)
            else:
                clean[k] = v
        return clean
    elif isinstance(data, list):
        return [format_json_for_display(item) for item in data]
    return data

def main():
    print("=" * 80)
    print("PHASE 5 PART C: REAL SMOKE TESTS WITH RUNNING SQL SERVER")
    print("=" * 80)

    # 1. Confirm database connection & DB name
    with engine.connect() as conn:
        row = conn.execute(text("SELECT DB_NAME(), @@SERVERNAME")).fetchone()
        print(f"Connected to SQL Server: Server={row[1]}, Database={row[0]}")
        print(f"Connection URI: {SQLALCHEMY_DATABASE_URI}\n")

    # 2. Start legacy and new servers with CPU-only TF to avoid CUDA collisions
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

    log_leg_f = open('scratch/legacy_server.log', 'w', encoding='utf-8')
    log_new_f = open('scratch/new_server.log', 'w', encoding='utf-8')

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

    print("\nALL SERVERS READY! EXECUTING REQUESTS...\n")

    super_admin_token = generate_jwt({
        "user_id": 1,
        "email": "admin@gmail.com",
        "role": "SUPER_ADMIN",
        "business_id": "SYSTEM"
    })
    admin_hdr = {"Authorization": f"Bearer {super_admin_token}"}

    test_uid_leg = None
    test_uid_new = None
    test_pid_leg = None
    test_pid_new = None
    test_ogid_leg = None
    test_ogid_new = None

    raw_log = []

    def execute_test(title, method, path, req_kwargs=None, path_leg=None, path_new=None, kw_leg=None, kw_new=None, check_deterministic=False):
        p_l = path_leg or path
        p_n = path_new or path
        url_leg = f"http://127.0.0.1:8083{p_l}"
        url_new = f"http://127.0.0.1:8082{p_n}"

        k_l = dict(kw_leg if kw_leg is not None else (req_kwargs or {}))
        k_n = dict(kw_new if kw_new is not None else (req_kwargs or {}))
        if 'timeout' not in k_l: k_l['timeout'] = 90
        if 'timeout' not in k_n: k_n['timeout'] = 90

        fn = getattr(requests, method.lower())
        r_leg = fn(url_leg, **k_l)
        r_new = fn(url_new, **k_n)

        try:
            j_leg = r_leg.json()
        except Exception:
            j_leg = r_leg.text[:300]

        try:
            j_new = r_new.json()
        except Exception:
            j_new = r_new.text[:300]

        disp_leg = format_json_for_display(j_leg)
        disp_new = format_json_for_display(j_new)

        print("=" * 80)
        print(f"REQUEST: {title} [{method} {path}]")
        print("-" * 80)
        print(f"LEGACY Status: {r_leg.status_code}")
        print(f"LEGACY JSON:   {json.dumps(disp_leg, default=str) if isinstance(disp_leg, (dict, list)) else disp_leg}")
        print("-" * 40)
        print(f"NEW Status:    {r_new.status_code}")
        print(f"NEW JSON:      {json.dumps(disp_new, default=str) if isinstance(disp_new, (dict, list)) else disp_new}")
        
        status_match = (r_leg.status_code == r_new.status_code)
        print(f"STATUS MATCH:  {status_match}")

        # Keys check
        if isinstance(j_leg, dict) and isinstance(j_new, dict):
            keys_match = (sorted(j_leg.keys()) == sorted(j_new.keys()))
            print(f"KEYS MATCH:    {keys_match} (Keys: {sorted(j_leg.keys())})")
            if check_deterministic:
                # Compare deterministic values (exclude token, user_id, order_group_id)
                diffs = []
                for k in j_leg.keys():
                    if k not in ['token', 'user_id', 'order_group_id', 'order_id', 'plant_id', 'audio']:
                        if j_leg[k] != j_new[k]:
                            diffs.append((k, j_leg[k], j_new[k]))
                print(f"VALUES MATCH:  {len(diffs) == 0}" + (f" (Diffs: {diffs})" if diffs else ""))
        elif isinstance(j_leg, list) and isinstance(j_new, list):
            print(f"LIST LENGTH MATCH: {len(j_leg) == len(j_new)} (Legacy: {len(j_leg)}, New: {len(j_new)})")
        print("=" * 80 + "\n")

        raw_log.append({
            "title": title,
            "method": method,
            "path": path,
            "legacy_status": r_leg.status_code,
            "legacy_body": disp_leg,
            "new_status": r_new.status_code,
            "new_body": disp_new,
        })

        return r_leg, r_new

    try:
        # Pre-cleanup throwaways
        with engine.connect() as conn:
            conn.execute(text("DELETE FROM login WHERE username LIKE 'TEST_%' OR email LIKE 'test_%'"))
            conn.execute(text("DELETE FROM plant WHERE plant_name LIKE 'TEST_%'"))
            conn.execute(text("DELETE FROM cart_new WHERE user_id LIKE 'TEST_%'"))
            conn.execute(text("DELETE FROM orders WHERE plant_name LIKE 'TEST_%' OR shipping_address LIKE '%Evergreen%'"))
            conn.commit()

        # 1. Register valid
        reg_leg = {
            "username": "TEST_user_smoke_leg",
            "email": "test_smoke_leg@example.com",
            "password": "Password123!",
            "business_id": "NURS-0001",
            "role": "CUSTOMER"
        }
        reg_new = {
            "username": "TEST_user_smoke_new",
            "email": "test_smoke_new@example.com",
            "password": "Password123!",
            "business_id": "NURS-0001",
            "role": "CUSTOMER"
        }
        execute_test("1. Register (Valid User)", "POST", "/register", kw_leg={"json": reg_leg}, kw_new={"json": reg_new})

        with engine.connect() as conn:
            row_ul = conn.execute(text("SELECT id FROM login WHERE email = 'test_smoke_leg@example.com'")).fetchone()
            row_un = conn.execute(text("SELECT id FROM login WHERE email = 'test_smoke_new@example.com'")).fetchone()
            test_uid_leg = row_ul[0] if row_ul else 99991
            test_uid_new = row_un[0] if row_un else 99992

        # Customer tokens
        cust_token_leg = generate_jwt({
            "user_id": test_uid_leg,
            "email": "test_smoke_leg@example.com",
            "role": "CUSTOMER",
            "business_id": "NURS-0001"
        })
        cust_hdr_leg = {"Authorization": f"Bearer {cust_token_leg}"}

        cust_token_new = generate_jwt({
            "user_id": test_uid_new,
            "email": "test_smoke_new@example.com",
            "role": "CUSTOMER",
            "business_id": "NURS-0001"
        })
        cust_hdr_new = {"Authorization": f"Bearer {cust_token_new}"}

        # 2. Login valid
        execute_test("2. Login (Valid Credentials)", "POST", "/login",
                     kw_leg={"json": {"email": "test_smoke_leg@example.com", "password": "Password123!"}},
                     kw_new={"json": {"email": "test_smoke_new@example.com", "password": "Password123!"}},
                     check_deterministic=True)

        # 3. Login invalid
        execute_test("3. Login (Invalid Password)", "POST", "/login",
                     kw_leg={"json": {"email": "test_smoke_leg@example.com", "password": "WrongPassword!"}},
                     kw_new={"json": {"email": "test_smoke_new@example.com", "password": "WrongPassword!"}},
                     check_deterministic=True)

        # 4. Add-plant no token
        execute_test("4. Add Plant (No Token)", "POST", "/api/admin/add-plant",
                     req_kwargs={"json": {"plant_name": "TEST_Fern", "price": 120.0}},
                     check_deterministic=True)

        # 5. Add-plant wrong role (CUSTOMER)
        execute_test("5. Add Plant (Wrong Role - CUSTOMER)", "POST", "/api/admin/add-plant",
                     kw_leg={"json": {"plant_name": "TEST_Fern", "price": 120.0}, "headers": cust_hdr_leg},
                     kw_new={"json": {"plant_name": "TEST_Fern", "price": 120.0}, "headers": cust_hdr_new},
                     check_deterministic=True)

        # 6. Add-plant valid (SUPER_ADMIN)
        plant_leg = {
            "plant_name": "TEST_Fern_Leg",
            "price": 120.0,
            "stock": 25,
            "description": "Test Fern for Smoke Testing"
        }
        plant_new = {
            "plant_name": "TEST_Fern_New",
            "price": 120.0,
            "stock": 25,
            "description": "Test Fern for Smoke Testing"
        }
        execute_test("6. Add Plant (Valid SUPER_ADMIN)", "POST", "/api/admin/add-plant",
                     kw_leg={"json": plant_leg, "headers": admin_hdr},
                     kw_new={"json": plant_new, "headers": admin_hdr},
                     check_deterministic=True)

        with engine.connect() as conn:
            row_pl = conn.execute(text("SELECT TOP 1 plant_id FROM plant WHERE plant_name = 'TEST_Fern_Leg' ORDER BY plant_id DESC")).fetchone()
            row_pn = conn.execute(text("SELECT TOP 1 plant_id FROM plant WHERE plant_name = 'TEST_Fern_New' ORDER BY plant_id DESC")).fetchone()
            test_pid_leg = row_pl[0] if row_pl else 1
            test_pid_new = row_pn[0] if row_pn else 1

        # 7. Admin inventory (no token & SUPER_ADMIN)
        execute_test("7a. Admin Inventory (No Token)", "GET", "/api/admin/inventory")
        execute_test("7b. Admin Inventory (SUPER_ADMIN)", "GET", "/api/admin/inventory", req_kwargs={"headers": admin_hdr})

        # 8. Restock
        execute_test("8. Restock Plant", "POST", "/api/admin/restock",
                     path_leg=f"/api/admin/restock/{test_pid_leg}",
                     path_new=f"/api/admin/restock/{test_pid_new}",
                     req_kwargs={"json": {"quantity": 10}, "headers": admin_hdr},
                     check_deterministic=True)

        # 9. Shop plants
        execute_test("9. Shop Plants (/api/shop/plants)", "GET", "/api/shop/plants")

        # 10. Cart add
        execute_test("10. Cart Add Item", "POST", "/api/cart",
                     kw_leg={"json": {"user_id": test_uid_leg, "plant_id": test_pid_leg, "quantity": 2}},
                     kw_new={"json": {"user_id": test_uid_new, "plant_id": test_pid_new, "quantity": 2}},
                     check_deterministic=True)

        # 11. Cart update
        execute_test("11. Cart Update Quantity", "POST", "/api/cart/update",
                     kw_leg={"json": {"user_id": test_uid_leg, "plant_id": test_pid_leg, "action": "increase"}},
                     kw_new={"json": {"user_id": test_uid_new, "plant_id": test_pid_new, "action": "increase"}},
                     check_deterministic=True)

        # 12. Cart get
        execute_test("12. Cart Get (/api/cart)", "GET", "/api/cart",
                     path_leg=f"/api/cart?user_id={test_uid_leg}",
                     path_new=f"/api/cart?user_id={test_uid_new}")

        # 13. Cart remove
        execute_test("13. Cart Remove Item", "POST", "/api/cart/remove",
                     kw_leg={"json": {"user_id": test_uid_leg, "plant_id": test_pid_leg}},
                     kw_new={"json": {"user_id": test_uid_new, "plant_id": test_pid_new}},
                     check_deterministic=True)

        # Re-add for checkout
        requests.post("http://127.0.0.1:8083/api/cart", json={"user_id": test_uid_leg, "plant_id": test_pid_leg, "quantity": 1})
        requests.post("http://127.0.0.1:8082/api/cart", json={"user_id": test_uid_new, "plant_id": test_pid_new, "quantity": 1})

        # 14. Checkout
        co_leg = {
            "user_id": test_uid_leg,
            "address": "789 Evergreen Terrace",
            "payment_method": "Cash on Delivery",
            "phone": "9876543210",
            "customer_name": "TEST Customer Leg"
        }
        co_new = {
            "user_id": test_uid_new,
            "address": "789 Evergreen Terrace",
            "payment_method": "Cash on Delivery",
            "phone": "9876543210",
            "customer_name": "TEST Customer New"
        }
        r_co_l, r_co_n = execute_test("14. Checkout (/api/checkout)", "POST", "/api/checkout",
                                      kw_leg={"json": co_leg},
                                      kw_new={"json": co_new})

        try:
            test_ogid_leg = r_co_l.json().get('order_group_id', 'ORD-0001')
            test_ogid_new = r_co_n.json().get('order_group_id', 'ORD-0002')
        except Exception:
            test_ogid_leg = "ORD-0001"
            test_ogid_new = "ORD-0002"

        # 15. Orders by user
        execute_test("15. Orders by User (/api/orders/<user_id>)", "GET", "/api/orders",
                     path_leg=f"/api/orders/{test_uid_leg}",
                     path_new=f"/api/orders/{test_uid_new}")

        # 16. Order tracking
        execute_test("16. Order Tracking (/api/orders/track/<id>)", "GET", "/api/orders/track",
                     path_leg=f"/api/orders/track/{test_ogid_leg}",
                     path_new=f"/api/orders/track/{test_ogid_new}")

        # 17. Chat greeting
        execute_test("17. Chat Greeting", "POST", "/api/chat",
                     kw_leg={"json": {"message": "hello", "user_id": test_uid_leg}},
                     kw_new={"json": {"message": "hello", "user_id": test_uid_new}},
                     check_deterministic=True)

        # 18. Chat buy flow (a: buy rose -> b: yes -> c: 2)
        execute_test("18a. Chat Buy Flow (buy rose)", "POST", "/api/chat",
                     kw_leg={"json": {"message": "buy rose", "user_id": test_uid_leg}},
                     kw_new={"json": {"message": "buy rose", "user_id": test_uid_new}},
                     check_deterministic=True)
        execute_test("18b. Chat Buy Flow (yes)", "POST", "/api/chat",
                     kw_leg={"json": {"message": "yes", "user_id": test_uid_leg}},
                     kw_new={"json": {"message": "yes", "user_id": test_uid_new}},
                     check_deterministic=True)
        execute_test("18c. Chat Buy Flow (2)", "POST", "/api/chat",
                     kw_leg={"json": {"message": "2", "user_id": test_uid_leg}},
                     kw_new={"json": {"message": "2", "user_id": test_uid_new}},
                     check_deterministic=True)

        # 19. Chat stock query
        execute_test("19. Chat Stock Query", "POST", "/api/chat",
                     kw_leg={"json": {"message": "how many rose do you have", "user_id": test_uid_leg}},
                     kw_new={"json": {"message": "how many rose do you have", "user_id": test_uid_new}},
                     check_deterministic=True)

        # 20. Chat off-topic
        execute_test("20. Chat Off-Topic", "POST", "/api/chat",
                     kw_leg={"json": {"message": "what is the capital of France", "user_id": test_uid_leg}},
                     kw_new={"json": {"message": "what is the capital of France", "user_id": test_uid_new}},
                     check_deterministic=True)

        # 21. WhatsApp cart
        execute_test("21. WhatsApp Cart", "GET", "/api/whatsapp/cart?phone=9876543210")

        # 22. WhatsApp checkout
        wa_co = {
            "phone": "9876543210",
            "address": "456 Palm Grove Road",
            "payment_method": "Cash on Delivery",
            "customer_name": "TEST WA Customer"
        }
        execute_test("22. WhatsApp Checkout", "POST", "/api/whatsapp/checkout", req_kwargs={"json": wa_co}, check_deterministic=True)

        # 23. Super-admin login (invalid & valid)
        execute_test("23a. Super-Admin Login (Invalid)", "POST", "/api/super-admin/login",
                     req_kwargs={"json": {"username": "admin", "password": "admin@123"}},
                     check_deterministic=True)
        execute_test("23b. Super-Admin Login (Valid)", "POST", "/api/super-admin/login",
                     req_kwargs={"json": {"email": "admin@system.com", "password": "admin123"}},
                     check_deterministic=True)

        # 24. Super-admin stats
        execute_test("24. Super-Admin Stats", "GET", "/api/super-admin/stats", req_kwargs={"headers": admin_hdr})

        # 25. Tool plant-identify with real image
        guava_abs = os.path.abspath("guava.jpg")
        execute_test("25. Plant Identify with Image", "POST", "/api/tools/plant-identify",
                     req_kwargs={"json": {"image_path": guava_abs}},
                     check_deterministic=True)

        # 26. Static files
        execute_test("26a. Static Root (/)", "GET", "/")
        execute_test("26b. Static Admin (/admin)", "GET", "/admin")
        execute_test("26c. Static Plant Image (/static/images/plants/plant_19_rose.jpeg)", "GET", "/static/images/plants/plant_19_rose.jpeg")

    finally:
        print("\n" + "=" * 80)
        print("CLEANING UP TEST DATA FROM DATABASE...")
        print("=" * 80)
        with engine.connect() as conn:
            conn.execute(text("DELETE FROM cart_new WHERE user_id LIKE 'TEST_%' OR user_id IN (:ul, :un)"),
                         {"ul": str(test_uid_leg or 0), "un": str(test_uid_new or 0)})
            conn.execute(text("DELETE FROM orders WHERE user_id IN (:ul, :un) OR plant_name LIKE 'TEST_%' OR shipping_address LIKE '%Evergreen%'"),
                         {"ul": test_uid_leg or 0, "un": test_uid_new or 0})
            conn.execute(text("DELETE FROM plant WHERE plant_name LIKE 'TEST_%'"))
            conn.execute(text("DELETE FROM login WHERE username LIKE 'TEST_%' OR email LIKE 'test_%'"))
            conn.commit()

            check_u = conn.execute(text("SELECT count(*) FROM login WHERE username LIKE 'TEST_%' OR email LIKE 'test_%'")).fetchone()[0]
            check_p = conn.execute(text("SELECT count(*) FROM plant WHERE plant_name LIKE 'TEST_%'")).fetchone()[0]
            check_c = conn.execute(text("SELECT count(*) FROM cart_new WHERE user_id LIKE 'TEST_%' OR user_id IN (:ul, :un)"),
                                   {"ul": str(test_uid_leg or 0), "un": str(test_uid_new or 0)}).fetchone()[0]
            check_o = conn.execute(text("SELECT count(*) FROM orders WHERE user_id IN (:ul, :un) OR plant_name LIKE 'TEST_%' OR shipping_address LIKE '%Evergreen%'"),
                                   {"ul": test_uid_leg or 0, "un": test_uid_new or 0}).fetchone()[0]
            print(f" Leftover TEST_ users:  {check_u}")
            print(f" Leftover TEST_ plants: {check_p}")
            print(f" Leftover TEST_ cart:   {check_c}")
            print(f" Leftover TEST_ orders: {check_o}")

        os.makedirs("scratch/logs", exist_ok=True)
        import datetime
        ts_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        raw_log_path = f"scratch/logs/partC_{ts_str}.json"
        with open(raw_log_path, "w", encoding="utf-8") as rf:
            json.dump(raw_log, rf, indent=2, default=str)
        print(f"Saved raw Part C responses to {raw_log_path}")

        print("\nShutting down server processes...")
        try:
            p_leg.kill()
            p_new.kill()
            p_leg.wait()
            p_new.wait()
        except Exception:
            pass
        try:
            log_leg_f.close()
            log_new_f.close()
        except Exception:
            pass
        print("Both servers terminated cleanly.")

if __name__ == '__main__':
    main()
