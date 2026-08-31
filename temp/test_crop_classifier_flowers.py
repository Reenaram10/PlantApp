import os
import sys
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "plant_disease_detection", "model", "crop_type_model.h5")
CLASSES_PATH = os.path.join(BASE_DIR, "plant_disease_detection", "model", "crop_indices.json")

print("--- Testing Stage 1 Crop Classifier on Flower Images ---")
model = tf.keras.models.load_model(MODEL_PATH)
with open(CLASSES_PATH, "r") as f:
    indices = json.load(f)

print("Crop Indices:", indices)

potato_flower_dir = os.path.join(BASE_DIR, "plant_disease_detection", "plant_disease_dataset", "Potato", "Flower")
tomato_flower_dir = os.path.join(BASE_DIR, "plant_disease_detection", "plant_disease_dataset", "Tomato", "Flower")

out_file = os.path.join(BASE_DIR, "temp", "crop_classifier_results.txt")
with open(out_file, "w", encoding="utf-8") as out:
    def p(text=""):
        print(text)
        out.write(text + "\n")

    p("--- POTATO FLOWER IMAGES ---")
    potato_correct = 0
    files = [f for f in os.listdir(potato_flower_dir) if f.lower().endswith(('.jpg','.jpeg','.png'))]
    for f in files:
        img = image.load_img(os.path.join(potato_flower_dir, f), target_size=(224,224))
        arr = preprocess_input(image.img_to_array(img))
        preds = model.predict(np.expand_dims(arr, axis=0), verbose=0)[0]
        pred_label = indices[str(np.argmax(preds))]
        if pred_label == 'Potato': potato_correct += 1
        p(f"Image: {f:20s} | Prediction: {pred_label:10s} | Confidence: {np.max(preds)*100:.2f}%")
    p(f"Potato Flower Accuracy: {potato_correct}/{len(files)} ({potato_correct/len(files)*100:.1f}%)\n")

    p("--- TOMATO FLOWER IMAGES ---")
    tomato_correct = 0
    files = [f for f in os.listdir(tomato_flower_dir) if f.lower().endswith(('.jpg','.jpeg','.png'))]
    for f in files:
        img = image.load_img(os.path.join(tomato_flower_dir, f), target_size=(224,224))
        arr = preprocess_input(image.img_to_array(img))
        preds = model.predict(np.expand_dims(arr, axis=0), verbose=0)[0]
        pred_label = indices[str(np.argmax(preds))]
        if pred_label == 'Tomato': tomato_correct += 1
        p(f"Image: {f:20s} | Prediction: {pred_label:10s} | Confidence: {np.max(preds)*100:.2f}%")
    p(f"Tomato Flower Accuracy: {tomato_correct}/{len(files)} ({tomato_correct/len(files)*100:.1f}%)")

