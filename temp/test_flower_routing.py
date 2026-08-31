import os
import cv2
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
POTATO_MODEL_PATH = os.path.join(BASE_DIR, "plant_disease_detection", "model", "potato_disease_model.h5")
TOMATO_MODEL_PATH = os.path.join(BASE_DIR, "plant_disease_detection", "model", "tomato_disease_model.h5")
CROP_MODEL_PATH = os.path.join(BASE_DIR, "plant_disease_detection", "model", "crop_type_model.h5")

POTATO_INDICES_PATH = os.path.join(BASE_DIR, "plant_disease_detection", "model", "potato_indices.json")
TOMATO_INDICES_PATH = os.path.join(BASE_DIR, "plant_disease_detection", "model", "tomato_indices.json")
CROP_INDICES_PATH = os.path.join(BASE_DIR, "plant_disease_detection", "model", "crop_indices.json")

potato_model = tf.keras.models.load_model(POTATO_MODEL_PATH)
tomato_model = tf.keras.models.load_model(TOMATO_MODEL_PATH)
crop_model = tf.keras.models.load_model(CROP_MODEL_PATH)

with open(POTATO_INDICES_PATH) as f: potato_indices = json.load(f)
with open(TOMATO_INDICES_PATH) as f: tomato_indices = json.load(f)
with open(CROP_INDICES_PATH) as f: crop_indices = json.load(f)

def detect_flower_color(img_path):
    img = cv2.imread(img_path)
    if img is None: return "unknown"
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # Bright Yellow range
    yellow_mask = cv2.inRange(hsv, np.array([15, 80, 80]), np.array([35, 255, 255]))
    yellow_pct = (np.count_nonzero(yellow_mask) / (img.shape[0] * img.shape[1])) * 100.0
    # White/Pale range
    white_mask = cv2.inRange(hsv, np.array([0, 0, 160]), np.array([180, 70, 255]))
    white_pct = (np.count_nonzero(white_mask) / (img.shape[0] * img.shape[1])) * 100.0
    return "yellow" if yellow_pct > white_pct else "white_purple"

def run_pipeline(img_path):
    img = image.load_img(img_path, target_size=(224,224))
    arr = preprocess_input(image.img_to_array(img))
    img_batch = np.expand_dims(arr, axis=0)

    p_preds = potato_model.predict(img_batch, verbose=0)[0]
    t_preds = tomato_model.predict(img_batch, verbose=0)[0]

    p_label = potato_indices[str(np.argmax(p_preds))]
    p_conf = float(np.max(p_preds))

    t_label = tomato_indices[str(np.argmax(t_preds))]
    t_conf = float(np.max(t_preds))

    if p_label == "Potato___Flower" or t_label == "Tomato___Flower":
        color = detect_flower_color(img_path)
        if color == "white_purple":
            return "Potato___Flower", "Potato", p_conf
        else:
            return "Tomato___Flower", "Tomato", t_conf
    else:
        c_preds = crop_model.predict(img_batch, verbose=0)[0]
        c_crop = crop_indices[str(np.argmax(c_preds))]
        if c_crop == "Potato":
            return p_label, "Potato", p_conf
        else:
            return t_label, "Tomato", t_conf

potato_dir = os.path.join(BASE_DIR, "plant_disease_detection", "plant_disease_dataset", "Potato", "Flower")
tomato_dir = os.path.join(BASE_DIR, "plant_disease_detection", "plant_disease_dataset", "Tomato", "Flower")

p_correct, t_correct = 0, 0

print("--- TESTING POTATO FLOWERS ---")
p_files = [f for f in os.listdir(potato_dir) if f.lower().endswith(('.jpg','.jpeg','.png'))]
for f in p_files:
    lbl, crop, conf = run_pipeline(os.path.join(potato_dir, f))
    if lbl == "Potato___Flower": p_correct += 1
    print(f"File: {f:18s} -> {crop:8s} ({lbl:16s}) | Conf: {conf*100:.1f}%")
print(f"Potato Flower Accuracy: {p_correct}/{len(p_files)} ({p_correct/len(p_files)*100:.1f}%)\n")

print("--- TESTING TOMATO FLOWERS ---")
t_files = [f for f in os.listdir(tomato_dir) if f.lower().endswith(('.jpg','.jpeg','.png'))]
for f in t_files:
    lbl, crop, conf = run_pipeline(os.path.join(tomato_dir, f))
    if lbl == "Tomato___Flower": t_correct += 1
    print(f"File: {f:18s} -> {crop:8s} ({lbl:16s}) | Conf: {conf*100:.1f}%")
print(f"Tomato Flower Accuracy: {t_correct}/{len(t_files)} ({t_correct/len(t_files)*100:.1f}%)")
