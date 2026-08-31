"""
train_flower_classifier.py - Dedicated Binary Flower Classifier (Potato Flower vs Tomato Flower)
=================================================================================================
Trains MobileNetV2 exclusively on distinguishing Potato Flowers from Tomato Flowers.
Output files:
- model/flower_crop_model.h5
- model/flower_indices.json
"""

import os
import json
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator

IMG_SIZE = 224
BATCH_SIZE = 4
EPOCHS = 15

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "model")
MODEL_PATH = os.path.join(MODEL_DIR, "flower_crop_model.h5")
CLASSES_PATH = os.path.join(MODEL_DIR, "flower_indices.json")

DATASET_DIR = os.path.join(BASE_DIR, "plant_disease_dataset")

filepaths, labels = [], []

potato_flower_dir = os.path.join(DATASET_DIR, "Potato", "Flower")
tomato_flower_dir = os.path.join(DATASET_DIR, "Tomato", "Flower")

for f in os.listdir(potato_flower_dir):
    if f.lower().endswith(('.jpg','.jpeg','.png')):
        filepaths.append(os.path.join(potato_flower_dir, f))
        labels.append("Potato")

for f in os.listdir(tomato_flower_dir):
    if f.lower().endswith(('.jpg','.jpeg','.png')):
        filepaths.append(os.path.join(tomato_flower_dir, f))
        labels.append("Tomato")

df = pd.DataFrame({"filepath": filepaths, "label": labels})
print(f"--- Dataset Loaded: {df['label'].value_counts().to_dict()} ---")

train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.3,
    horizontal_flip=True,
    vertical_flip=True,
    fill_mode="nearest"
)

train_gen = train_datagen.flow_from_dataframe(
    df, x_col="filepath", y_col="label",
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=True
)

class_indices = train_gen.class_indices
json_indices = {str(v): k for k, v in class_indices.items()}
with open(CLASSES_PATH, "w") as f:
    json.dump(json_indices, f, indent=2)
print(f"Saved Flower Class Mapping -> {CLASSES_PATH}: {json_indices}")

base = MobileNetV2(weights="imagenet", include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))
base.trainable = False

x = base.output
x = GlobalAveragePooling2D()(x)
x = Dense(64, activation="relu")(x)
x = Dropout(0.3)(x)
out = Dense(2, activation="softmax")(x)

model = Model(inputs=base.input, outputs=out)
model.compile(optimizer=Adam(1e-3), loss="categorical_crossentropy", metrics=["accuracy"])

print("\n--- Training Dedicated Flower Classifier ---")
model.fit(train_gen, epochs=EPOCHS)

print("\n--- Fine-Tuning Base Layers ---")
base.trainable = True
for layer in base.layers[:-30]:
    layer.trainable = False

model.compile(optimizer=Adam(1e-4), loss="categorical_crossentropy", metrics=["accuracy"])
model.fit(train_gen, epochs=10)

model.save(MODEL_PATH)
print(f"\n [OK] Flower Crop Model successfully saved -> {MODEL_PATH}")
