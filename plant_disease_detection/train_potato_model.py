"""
train_potato_model.py - Stage 2 Potato Disease Specialist Model
===============================================================
Trains MobileNetV2 exclusively on the 3 Potato classes:
- Potato___Early_blight
- Potato___Late_blight
- Potato___healthy

Output files:
- model/potato_disease_model.h5
- model/potato_indices.json
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

IMG_SIZE = 224
BATCH_SIZE = 16
EPOCHS = 10

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "model")
MODEL_PATH = os.path.join(MODEL_DIR, "potato_disease_model.h5")
CLASSES_PATH = os.path.join(MODEL_DIR, "potato_indices.json")

DATASET_DIR = os.path.join(BASE_DIR, "plant_disease_dataset", "Potato")

filepaths, labels = [], []

print(f"--- Scanning Potato Dataset: {DATASET_DIR} ---")

for folder_name in os.listdir(DATASET_DIR):
    folder_path = os.path.join(DATASET_DIR, folder_name)
    if not os.path.isdir(folder_path):
        continue
    folder_lower = folder_name.lower()
    if "early" in folder_lower:
        label = "Potato___Early_blight"
    elif "late" in folder_lower:
        label = "Potato___Late_blight"
    elif "healthy" in folder_lower:
        label = "Potato___healthy"
    elif "flower" in folder_lower:
        label = "Potato___Flower"
    elif "fruit" in folder_lower:
        label = "Potato___Fruit"
    else:
        label = f"Potato___{folder_name}"

    for f in os.listdir(folder_path):
        if f.lower().endswith(('.jpg', '.jpeg', '.png')):
            filepaths.append(os.path.join(folder_path, f))
            labels.append(label)

df = pd.DataFrame({"filepath": filepaths, "label": labels})
print(f" Found {len(df)} Potato images: {df['label'].value_counts().to_dict()}")

os.makedirs(MODEL_DIR, exist_ok=True)

train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    validation_split=0.2,
    rotation_range=20,
    zoom_range=0.2,
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
print(f"Saved Potato Class Mapping -> {CLASSES_PATH}: {json_indices}")

base = MobileNetV2(weights="imagenet", include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))
base.trainable = False

x = base.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation="relu")(x)
x = Dropout(0.2)(x)
out = Dense(len(class_indices), activation="softmax")(x)

model = Model(inputs=base.input, outputs=out)
model.compile(optimizer=Adam(1e-3), loss="categorical_crossentropy", metrics=["accuracy"])

callbacks = [
    ModelCheckpoint(MODEL_PATH, save_best_only=True, monitor="val_accuracy", verbose=1),
    EarlyStopping(patience=4, restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, verbose=1)
]

print("\n--- Training Potato Specialist Model ---")
model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS,
    callbacks=callbacks
)

print(f"\n [OK] Potato Specialist Model saved -> {MODEL_PATH}")
