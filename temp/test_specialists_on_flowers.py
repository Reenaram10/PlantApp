import os
import sys
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
POTATO_MODEL_PATH = os.path.join(BASE_DIR, "plant_disease_detection", "model", "potato_disease_model.h5")
TOMATO_MODEL_PATH = os.path.join(BASE_DIR, "plant_disease_detection", "model", "tomato_disease_model.h5")

POTATO_INDICES_PATH = os.path.join(BASE_DIR, "plant_disease_detection", "model", "potato_indices.json")
TOMATO_INDICES_PATH = os.path.join(BASE_DIR, "plant_disease_detection", "model", "tomato_indices.json")

potato_model = tf.keras.models.load_model(POTATO_MODEL_PATH)
tomato_model = tf.keras.models.load_model(TOMATO_MODEL_PATH)

with open(POTATO_INDICES_PATH) as f:
    potato_indices = json.load(f)
with open(TOMATO_INDICES_PATH) as f:
    tomato_indices = json.load(f)

potato_flower_dir = os.path.join(BASE_DIR, "plant_disease_detection", "plant_disease_dataset", "Potato", "Flower")
tomato_flower_dir = os.path.join(BASE_DIR, "plant_disease_detection", "plant_disease_dataset", "Tomato", "Flower")

out_file = os.path.join(BASE_DIR, "temp", "specialists_results.txt")
with open(out_file, "w", encoding="utf-8") as out:
    def p(text=""):
        print(text)
        out.write(text + "\n")

    p("=== POTATO FLOWER IMAGES ===")
    for f in os.listdir(potato_flower_dir):
        if not f.lower().endswith(('.jpg','.jpeg','.png')): continue
        img = image.load_img(os.path.join(potato_flower_dir, f), target_size=(224,224))
        arr = preprocess_input(image.img_to_array(img))
        img_batch = np.expand_dims(arr, axis=0)

        p_preds = potato_model.predict(img_batch, verbose=0)[0]
        t_preds = tomato_model.predict(img_batch, verbose=0)[0]

        p_label = potato_indices[str(np.argmax(p_preds))]
        p_conf = np.max(p_preds)

        t_label = tomato_indices[str(np.argmax(t_preds))]
        t_conf = np.max(t_preds)

        p(f"File: {f:18s} | PotatoModel: {p_label:20s} ({p_conf*100:.1f}%) | TomatoModel: {t_label:25s} ({t_conf*100:.1f}%)")

    p("\n=== TOMATO FLOWER IMAGES ===")
    for f in os.listdir(tomato_flower_dir):
        if not f.lower().endswith(('.jpg','.jpeg','.png')): continue
        img = image.load_img(os.path.join(tomato_flower_dir, f), target_size=(224,224))
        arr = preprocess_input(image.img_to_array(img))
        img_batch = np.expand_dims(arr, axis=0)

        p_preds = potato_model.predict(img_batch, verbose=0)[0]
        t_preds = tomato_model.predict(img_batch, verbose=0)[0]

        p_label = potato_indices[str(np.argmax(p_preds))]
        p_conf = np.max(p_preds)

        t_label = tomato_indices[str(np.argmax(t_preds))]
        t_conf = np.max(t_preds)

        p(f"File: {f:18s} | PotatoModel: {p_label:20s} ({p_conf*100:.1f}%) | TomatoModel: {t_label:25s} ({t_conf*100:.1f}%)")
