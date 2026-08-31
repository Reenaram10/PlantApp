import os
import sys
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)
MODEL_PATH = os.path.join(BASE_DIR, "plant_disease_detection", "model", "tomato_disease_model.h5")
CLASSES_PATH = os.path.join(BASE_DIR, "plant_disease_detection", "model", "tomato_indices.json")
FLOWER_DIR = os.path.join(BASE_DIR, "plant_disease_detection", "plant_disease_dataset", "Tomato", "Flower")

print("--- Testing Tomato Flower Model Predictions ---")
print(f"Model Path: {MODEL_PATH}")
print(f"Classes Path: {CLASSES_PATH}")

if not os.path.exists(MODEL_PATH) or not os.path.exists(CLASSES_PATH):
    print("❌ Error: Model or class indices file not found!")
    sys.exit(1)

model = tf.keras.models.load_model(MODEL_PATH)
with open(CLASSES_PATH, "r") as f:
    class_indices = json.load(f)

print("Loaded class indices:", class_indices)

flower_files = [f for f in os.listdir(FLOWER_DIR) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
print(f"Found {len(flower_files)} flower images in {FLOWER_DIR}\n")

correct_count = 0
for f in flower_files:
    img_path = os.path.join(FLOWER_DIR, f)
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = preprocess_input(img_array)
    img_batch = np.expand_dims(img_array, axis=0)

    preds = model.predict(img_batch, verbose=0)[0]
    predicted_idx = str(np.argmax(preds))
    confidence = float(np.max(preds))
    predicted_label = class_indices.get(predicted_idx, "Unknown")

    is_flower = "Flower" in predicted_label
    if is_flower:
        correct_count += 1
    status = "✅ PASS" if is_flower else "❌ FAIL"
    print(f"{status} | Image: {f:20s} | Prediction: {predicted_label:30s} | Confidence: {confidence*100:.2f}%")

print(f"\n--- Accuracy: {correct_count}/{len(flower_files)} ({correct_count/len(flower_files)*100:.1f}%) ---")

print("\n--- Testing TomatoOpenCVDetector Pipeline ---")
from plant_disease_detection.tomato_disease_opencv import tomato_detector

test_img_path = os.path.join(FLOWER_DIR, flower_files[0])
res = tomato_detector.predict_tomato_disease(test_img_path)
print(f"Status: {res.get('status')}")
print(f"Plant: {res.get('plant')}")
print(f"Disease Key: {res.get('disease_key')}")
print(f"Disease Name: {res.get('disease_name')}")
print(f"Confidence: {res.get('confidence_pct')}%")
print(f"Description: {res.get('description')}")
print(f"Treatment: {res.get('treatment')}")

