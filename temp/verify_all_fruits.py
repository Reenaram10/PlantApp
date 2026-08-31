import os
import sys
import json

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

from app import app

tomato_fruit_dir = os.path.join(BASE_DIR, "plant_disease_detection", "plant_disease_dataset", "Tomato", "Fruit")
potato_fruit_dir = os.path.join(BASE_DIR, "plant_disease_detection", "plant_disease_dataset", "Potato", "Fruit")

t_files = [f for f in os.listdir(tomato_fruit_dir) if f.lower().endswith(('.jpg','.jpeg','.png'))]
p_files = [f for f in os.listdir(potato_fruit_dir) if f.lower().endswith(('.jpg','.jpeg','.png'))]

out_file = os.path.join(BASE_DIR, "temp", "all_fruits_verification.txt")
with open(out_file, "w", encoding="utf-8") as out:
    def p(text=""):
        print(text)
        out.write(text + "\n")

    with app.test_client() as client:
        p("=== TOMATO FRUIT /API/IDENTIFY VERIFICATION ===")
        t_correct = 0
        for f in t_files:
            img_path = os.path.join(tomato_fruit_dir, f)
            with open(img_path, 'rb') as img_f:
                resp = client.post('/api/identify', data={'image': (img_f, f)})
                data = resp.get_json()
                ident = data.get("identification", {})
                detected = ident.get("detected_plant")
                plant_name = ident.get("plant_name")
                conf = ident.get("confidence")
                ok = (detected == "Tomato" and plant_name == "Tomato")
                if ok: t_correct += 1
                p(f"File: {f:45s} | Plant: {plant_name:10s} | Detected: {detected:10s} | Status: {'[OK]' if ok else '[FAIL]'}")
        p(f"Tomato Fruit Endpoint Accuracy: {t_correct}/{len(t_files)} ({t_correct/len(t_files)*100:.1f}%)\n")

        p("=== POTATO FRUIT /API/IDENTIFY VERIFICATION ===")
        p_correct = 0
        for f in p_files:
            img_path = os.path.join(potato_fruit_dir, f)
            with open(img_path, 'rb') as img_f:
                resp = client.post('/api/identify', data={'image': (img_f, f)})
                data = resp.get_json()
                ident = data.get("identification", {})
                detected = ident.get("detected_plant")
                plant_name = ident.get("plant_name")
                conf = ident.get("confidence")
                ok = (detected == "Potato" and plant_name == "Potato")
                if ok: p_correct += 1
                p(f"File: {f:25s} | Plant: {plant_name:10s} | Detected: {detected:10s} | Status: {'[OK]' if ok else '[FAIL]'}")
        p(f"Potato Fruit Endpoint Accuracy: {p_correct}/{len(p_files)} ({p_correct/len(p_files)*100:.1f}%)")
