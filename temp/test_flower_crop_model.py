import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "plant_disease_detection", "model", "flower_crop_model.h5")
CLASSES_PATH = os.path.join(BASE_DIR, "plant_disease_detection", "model", "flower_indices.json")

model = tf.keras.models.load_model(MODEL_PATH)
with open(CLASSES_PATH) as f:
    indices = json.load(f)

potato_dir = os.path.join(BASE_DIR, "plant_disease_detection", "plant_disease_dataset", "Potato", "Flower")
tomato_dir = os.path.join(BASE_DIR, "plant_disease_detection", "plant_disease_dataset", "Tomato", "Flower")

out_file = os.path.join(BASE_DIR, "temp", "flower_model_test_results.txt")
with open(out_file, "w", encoding="utf-8") as out:
    def p(text=""):
        print(text)
        out.write(text + "\n")

    p("--- POTATO FLOWER TEST ---")
    p_correct = 0
    files = [f for f in os.listdir(potato_dir) if f.lower().endswith(('.jpg','.jpeg','.png'))]
    for f in files:
        img = image.load_img(os.path.join(potato_dir, f), target_size=(224,224))
        arr = preprocess_input(image.img_to_array(img))
        preds = model.predict(np.expand_dims(arr, axis=0), verbose=0)[0]
        pred_label = indices[str(np.argmax(preds))]
        if pred_label == "Potato": p_correct += 1
        p(f"File: {f:18s} | Prediction: {pred_label:10s} | Confidence: {np.max(preds)*100:.2f}%")
    p(f"Potato Flower Accuracy: {p_correct}/{len(files)} ({p_correct/len(files)*100:.1f}%)\n")

    p("--- TOMATO FLOWER TEST ---")
    t_correct = 0
    files = [f for f in os.listdir(tomato_dir) if f.lower().endswith(('.jpg','.jpeg','.png'))]
    for f in files:
        img = image.load_img(os.path.join(tomato_dir, f), target_size=(224,224))
        arr = preprocess_input(image.img_to_array(img))
        preds = model.predict(np.expand_dims(arr, axis=0), verbose=0)[0]
        pred_label = indices[str(np.argmax(preds))]
        if pred_label == "Tomato": t_correct += 1
        p(f"File: {f:18s} | Prediction: {pred_label:10s} | Confidence: {np.max(preds)*100:.2f}%")
    p(f"Tomato Flower Accuracy: {t_correct}/{len(files)} ({t_correct/len(files)*100:.1f}%)")
