import os
import sys
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "plant_disease_detection", "model")

potato_model = tf.keras.models.load_model(os.path.join(MODEL_DIR, "potato_disease_model.h5"))
tomato_model = tf.keras.models.load_model(os.path.join(MODEL_DIR, "tomato_disease_model.h5"))
flower_model = tf.keras.models.load_model(os.path.join(MODEL_DIR, "flower_crop_model.h5"))
crop_model = tf.keras.models.load_model(os.path.join(MODEL_DIR, "crop_type_model.h5"))

with open(os.path.join(MODEL_DIR, "potato_indices.json")) as f: potato_indices = json.load(f)
with open(os.path.join(MODEL_DIR, "tomato_indices.json")) as f: tomato_indices = json.load(f)
with open(os.path.join(MODEL_DIR, "flower_indices.json")) as f: flower_indices = json.load(f)
with open(os.path.join(MODEL_DIR, "crop_indices.json")) as f: crop_indices = json.load(f)

def simulate_api_identify(img_path):
    img = image.load_img(img_path, target_size=(224,224))
    arr = preprocess_input(image.img_to_array(img))
    img_batch = np.expand_dims(arr, axis=0)

    p_preds = potato_model.predict(img_batch, verbose=0)
    t_preds = tomato_model.predict(img_batch, verbose=0)

    p_idx = np.argmax(p_preds[0])
    t_idx = np.argmax(t_preds[0])

    p_label = potato_indices.get(str(p_idx), "")
    t_label = tomato_indices.get(str(t_idx), "")

    is_flower_detected = (p_label == "Potato___Flower") or (t_label == "Tomato___Flower")

    if is_flower_detected and flower_model is not None:
        f_preds = flower_model.predict(img_batch, verbose=0)
        f_idx = np.argmax(f_preds[0])
        f_conf = float(np.max(f_preds[0]))
        detected_crop = flower_indices.get(str(f_idx), "Potato")
        result_label = f"{detected_crop}___Flower"
        confidence = f_conf
    else:
        crop_preds = crop_model.predict(img_batch, verbose=0)
        crop_idx = np.argmax(crop_preds[0])
        crop_conf = float(np.max(crop_preds[0]))
        detected_crop = crop_indices.get(str(crop_idx), "Potato")

        if detected_crop == "Potato":
            disease_preds = potato_model.predict(img_batch, verbose=0)
            disease_idx = np.argmax(disease_preds[0])
            disease_conf = float(np.max(disease_preds[0]))
            result_label = potato_indices.get(str(disease_idx), "Potato___healthy")
        else:
            disease_preds = tomato_model.predict(img_batch, verbose=0)
            disease_idx = np.argmax(disease_preds[0])
            disease_conf = float(np.max(disease_preds[0]))
            result_label = tomato_indices.get(str(disease_idx), "Tomato___healthy")
        confidence = float(crop_conf * disease_conf)

    if "potato" in result_label.lower():
        detected_plant = "Potato"
    elif "tomato" in result_label.lower():
        detected_plant = "Tomato"
    else:
        detected_plant = "Unknown Plant"

    is_flower_result = "flower" in result_label.lower()
    is_healthy = "healthy" in result_label.lower()

    if is_flower_result:
        inferred_status = "Flower Identification"
        disease_display = f"{detected_plant} Plant"
        plant_display_name = detected_plant
    else:
        inferred_status = "Healthy" if is_healthy else "Diseased"
        clean_disease = result_label.replace("Potato___", "").replace("Tomato___", "").replace("_", " ")
        disease_display = "Healthy (No Disease Detected)" if is_healthy else clean_disease.title()
        plant_display_name = f"{detected_plant} ({inferred_status})"

    return {
        "plant_name": plant_display_name,
        "scientific_name": f"Diagnosis: {disease_display}",
        "detected_plant": detected_plant,
        "health_status": inferred_status,
        "disease_name": disease_display,
        "confidence": round(confidence * 100, 2)
    }

potato_dir = os.path.join(BASE_DIR, "plant_disease_detection", "plant_disease_dataset", "Potato", "Flower")
tomato_dir = os.path.join(BASE_DIR, "plant_disease_detection", "plant_disease_dataset", "Tomato", "Flower")

out_file = os.path.join(BASE_DIR, "temp", "full_identify_verification.txt")
with open(out_file, "w", encoding="utf-8") as out:
    def p(text=""):
        print(text)
        out.write(text + "\n")

    p("=== POTATO FLOWER IDENTIFY PIPELINE VERIFICATION ===")
    p_pass = 0
    files = [f for f in os.listdir(potato_dir) if f.lower().endswith(('.jpg','.jpeg','.png'))]
    for f in files:
        res = simulate_api_identify(os.path.join(potato_dir, f))
        ok = (res["detected_plant"] == "Potato" and res["plant_name"] == "Potato")
        if ok: p_pass += 1
        p(f"File: {f:18s} | Plant: {res['plant_name']:10s} | Diag: {res['disease_name']:15s} | Conf: {res['confidence']}% | Status: {'[OK]' if ok else '[FAIL]'}")
    p(f"Potato Flower Identification Accuracy: {p_pass}/{len(files)} ({p_pass/len(files)*100:.1f}%)\n")

    p("=== TOMATO FLOWER IDENTIFY PIPELINE VERIFICATION ===")
    t_pass = 0
    files = [f for f in os.listdir(tomato_dir) if f.lower().endswith(('.jpg','.jpeg','.png'))]
    for f in files:
        res = simulate_api_identify(os.path.join(tomato_dir, f))
        ok = (res["detected_plant"] == "Tomato" and res["plant_name"] == "Tomato")
        if ok: t_pass += 1
        p(f"File: {f:18s} | Plant: {res['plant_name']:10s} | Diag: {res['disease_name']:15s} | Conf: {res['confidence']}% | Status: {'[OK]' if ok else '[FAIL]'}")
    p(f"Tomato Flower Identification Accuracy: {t_pass}/{len(files)} ({t_pass/len(files)*100:.1f}%)")
