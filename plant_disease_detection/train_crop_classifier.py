"""
train_crop_classifier.py - Stage 1 Binary Crop Classifier (Potato vs Tomato)
=============================================================================
Trains MobileNetV2 exclusively on binary Potato vs Tomato classification.
Output files:
- model/crop_type_model.h5
- model/crop_indices.json
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
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.utils.class_weight import compute_class_weight

IMG_SIZE = 224
BATCH_SIZE = 16
EPOCHS = 10

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "model")
MODEL_PATH = os.path.join(MODEL_DIR, "crop_type_model.h5")
CLASSES_PATH = os.path.join(MODEL_DIR, "crop_indices.json")

DATASET_DIR = os.path.join(BASE_DIR, "plant_disease_dataset")

filepaths, labels = [], []

print(f"--- Scanning Dataset for Crop Classification (Potato vs Tomato) ---")

for crop in ["Potato", "Tomato"]:
    crop_dir = os.path.join(DATASET_DIR, crop)
    if not os.path.exists(crop_dir):
        continue
    for root, _, files in os.walk(crop_dir):
        for f in files:
            if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                filepaths.append(os.path.join(root, f))
                labels.append(crop)

df = pd.DataFrame({"filepath": filepaths, "label": labels})
print(f" Found {len(df)} total images: {df['label'].value_counts().to_dict()}")

classes = sorted(df["label"].unique().tolist())
num_classes = len(classes)

os.makedirs(MODEL_DIR, exist_ok=True)

# Data Generators
train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    validation_split=0.2,
    rotation_range=20,
    zoom_range=0.2,
    width_shift_range=0.15,
    height_shift_range=0.15,
    horizontal_flip=True,
    fill_mode="nearest"
)

val_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    validation_split=0.2
)

train_gen = train_datagen.flow_from_dataframe(
    df, x_col="filepath", y_col="label",
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="training",
    shuffle=True
)

val_gen = val_datagen.flow_from_dataframe(
    df, x_col="filepath", y_col="label",
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="validation",
    shuffle=False
)

class_indices = train_gen.class_indices
json_indices = {str(v): k for k, v in class_indices.items()}
with open(CLASSES_PATH, "w") as f:
    json.dump(json_indices, f, indent=2)
print(f"Saved Crop Class Mapping -> {CLASSES_PATH}: {json_indices}")

# Build Binary MobileNetV2
base = MobileNetV2(weights="imagenet", include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))
base.trainable = False

x = base.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation="relu")(x)
x = Dropout(0.2)(x)
out = Dense(num_classes, activation="softmax")(x)

model = Model(inputs=base.input, outputs=out)
model.compile(optimizer=Adam(1e-3), loss="categorical_crossentropy", metrics=["accuracy"])

callbacks = [
    ModelCheckpoint(MODEL_PATH, save_best_only=True, monitor="val_accuracy", verbose=1),
    EarlyStopping(patience=4, restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, verbose=1)
]

print("\n--- Phase 1: Training Model Head ---")
model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS,
    callbacks=callbacks
)

print("\n--- Phase 2: Fine-Tuning MobileNetV2 Base ---")
base.trainable = True
for layer in base.layers[:-30]:
    layer.trainable = False

model.compile(optimizer=Adam(1e-5), loss="categorical_crossentropy", metrics=["accuracy"])
model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=5,
    callbacks=callbacks
)

print(f"\n [OK] Crop Type Classifier successfully saved -> {MODEL_PATH}")
