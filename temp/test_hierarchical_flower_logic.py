import os
import sys
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CROP_MODEL_PATH = os.path.join(BASE_DIR, "plant_disease_detection", "model", "crop_type_model.h5")
POTATO_MODEL_PATH = os.path.join(BASE_DIR, "plant_disease_detection", "model", "potato_disease_model.h5")
TOMATO_MODEL_PATH = os.path.join(BASE_DIR, "plant_disease_detection", "model", "tomato_disease_model.h5")

CROP_INDICES_PATH = os.path.join(BASE_DIR, "plant_disease_detection", "model", "crop_indices.json")
POTATO_INDICES_PATH = os.path.join(BASE_DIR, "plant_disease_detection", "model", "potato_indices.json")
TOMATO_INDICES_PATH = os.path.join(BASE_DIR, "plant_disease_detection", "model", "tomato_indices.json")

crop_model = tf.keras.models.load_model(CROP_MODEL_PATH)
potato_model = tf.keras.models.load_model(POTATO_MODEL_PATH)
tomato_model = tf.keras.models.load_model(TOMATO_MODEL_PATH)

with open(CROP_INDICES_PATH) as f: crop_indices = json.load(f)
with open(POTATO_INDICES_PATH) as f: potato_indices = json.load(f)
with open(TOMATO_INDICES_PATH) as f: tomato_indices = json.load(f)

potato_flower_dir = os.path.join(BASE_DIR, "plant_disease_detection", "plant_disease_dataset", "Potato", "Flower")
tomato_flower_dir = os.path.join(BASE_DIR, "plant_disease_detection", "plant_disease_dataset", "Tomato", "Flower")

def classify_plant(img_path):
    img = image.load_img(img_path, target_size=(224,224))
    arr = preprocess_input(image.img_to_array(img))
    img_batch = np.expand_dims(arr, axis=0)

    p_preds = potato_model.predict(img_batch, verbose=0)[0]
    t_preds = tomato_model.predict(img_batch, verbose=0)[0]

    p_label = potato_indices[str(np.argmax(p_preds))]
    p_conf = float(np.max(p_preds))

    t_label = tomato_indices[str(np.argmax(t_preds))]
    t_conf = float(np.max(t_preds))

    # Check for Flower detection
    p_is_flower = (p_label == "Potato___Flower")
    t_is_flower = (t_label == "Tomato___Flower")

    crop_preds = crop_model.predict(img_batch, verbose=0)[0]
    crop_idx = np.argmax(crop_preds)
    detected_crop = crop_indices.get(str(crop_idx), "Potato")

    if p_is_flower and t_is_flower:
        # Both specialist models detected a flower!
        # Compare prediction margins or relative probability of non-flower vs flower
        p_flower_prob = p_preds[int([k for k,v in potato_indices.items() if v=="Potato___Flower"][0])]
        t_flower_prob = t_preds[int([k for k,v in tomato_indices.items() if v=="Tomato___Flower"][0])]

        # If potato specialist is very confident it's Potato Flower (>95%), and tomato model has lower relative flower dominance
        # OR compare potato flower confidence vs tomato flower confidence
        # Let's inspect values!
        return "Potato___Flower" if p_conf >= t_conf else "Tomato___Flower", detected_crop, p_conf, t_conf
    elif p_is_flower:
        return "Potato___Flower", "Potato", p_conf, t_conf
    elif t_is_flower:
        return "Tomato___Flower", "Tomato", p_conf, t_conf
    else:
        if detected_crop == "Potato":
            return p_label, "Potato", p_conf, t_conf
        else:
            return t_label, "Tomato", p_conf, t_conf

out_file = os.path.join(BASE_DIR, "temp", "hierarchical_flower_results.txt")
with open(out_file, "w", encoding="utf-8") as out:
    def p(text=""):
        print(text)
        out.write(text + "\n")

    p("=== POTATO FLOWERS ===")
    p_correct = 0
    files = [f for f in os.listdir(potato_flower_dir) if f.lower().endswith(('.jpg','.jpeg','.png'))]
    for f in files:
        lbl, crop, pc, tc = classify_plant(os.path.join(potato_flower_dir, f))
        if lbl == "Potato___Flower": p_correct += 1
        p(f"File: {f:18s} | Classified: {lbl:20s} | P_Conf: {pc*100:.1f}% | T_Conf: {tc*100:.1f}%")
    p(f"Potato Flower Accuracy: {p_correct}/{len(files)} ({p_correct/len(files)*100:.1f}%)\n")

    p("=== TOMATO FLOWERS ===")
    t_correct = 0
    files = [f for f in os.listdir(tomato_flower_dir) if f.lower().endswith(('.jpg','.jpeg','.png'))]
    for f in files:
        lbl, crop, pc, tc = classify_plant(os.path.join(tomato_flower_dir, f))
        if lbl == "Tomato___Flower": t_correct += 1
        p(f"File: {f:18s} | Classified: {lbl:20s} | P_Conf: {pc*100:.1f}% | T_Conf: {tc*100:.1f}%")
    p(f"Tomato Flower Accuracy: {t_correct}/{len(files)} ({t_correct/len(files)*100:.1f}%)")
