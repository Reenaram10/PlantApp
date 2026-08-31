import os
import sys
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "plant_disease_detection", "model")

tomato_model = tf.keras.models.load_model(os.path.join(MODEL_DIR, "tomato_disease_model.h5"))
with open(os.path.join(MODEL_DIR, "tomato_indices.json")) as f:
    tomato_indices = json.load(f)

print("Loaded Tomato Class Mapping:", tomato_indices)

fruit_dir = os.path.join(BASE_DIR, "plant_disease_detection", "plant_disease_dataset", "Tomato", "Fruit")
files = [f for f in os.listdir(fruit_dir) if f.lower().endswith(('.jpg','.jpeg','.png'))]

out_file = os.path.join(BASE_DIR, "temp", "tomato_fruit_verification.txt")
with open(out_file, "w", encoding="utf-8") as out:
    def p(text=""):
        print(text)
        out.write(text + "\n")

    p("=== TOMATO FRUIT MODEL PREDICTION VERIFICATION ===")
    correct = 0
    for f in files:
        img = image.load_img(os.path.join(fruit_dir, f), target_size=(224,224))
        arr = preprocess_input(image.img_to_array(img))
        img_batch = np.expand_dims(arr, axis=0)

        preds = tomato_model.predict(img_batch, verbose=0)[0]
        pred_idx = np.argmax(preds)
        pred_label = tomato_indices.get(str(pred_idx), "")
        conf = float(np.max(preds))

        ok = (pred_label == "Tomato___Fruit")
        if ok: correct += 1
        p(f"File: {f:45s} | Prediction: {pred_label:20s} | Conf: {conf*100:.2f}% | Status: {'[OK]' if ok else '[FAIL]'}")

    p(f"\nTomato Fruit Classification Accuracy: {correct}/{len(files)} ({correct/len(files)*100:.1f}%)")
