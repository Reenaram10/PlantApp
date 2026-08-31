"""
train_plant_model.py - Optimized MobileNetV2 Trainer for Potato & Tomato Diseases
==================================================================================
Uses tf.keras.applications.mobilenet_v2.preprocess_input and compute_class_weight
to balance Potato (100 samples/class) and Tomato (48 samples/class) datasets.
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
EPOCHS = 20

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "model")
MODEL_PATH = os.path.join(MODEL_DIR, "plant_disease_model.h5")
CLASSES_PATH = os.path.join(MODEL_DIR, "class_indices.json")

DATASET_DIR = os.path.join(BASE_DIR, "plant_disease_dataset")

filepaths, labels = [], []

print(f"--- Scanning Dataset Directory: {DATASET_DIR} ---")

if not os.path.exists(DATASET_DIR):
    print(f" [ERROR] Directory not found: {DATASET_DIR}")
    exit(1)

for crop in ["Potato", "Tomato"]:
    crop_dir = os.path.join(DATASET_DIR, crop)
    if not os.path.exists(crop_dir):
        continue

    for folder_name in os.listdir(crop_dir):
        folder_path = os.path.join(crop_dir, folder_name)
        if not os.path.isdir(folder_path):
            continue

        folder_lower = folder_name.lower()
        
        if "___" in folder_name:
            canonical_label = folder_name
        else:
            canonical_label = None
            if crop.lower() == "potato":
                if "early" in folder_lower:
                    canonical_label = "Potato___Early_blight"
                elif "late" in folder_lower:
                    canonical_label = "Potato___Late_blight"
                elif "healthy" in folder_lower:
                    canonical_label = "Potato___healthy"
                elif "flower" in folder_lower:
                    canonical_label = "Potato___Flower"
                elif "fruit" in folder_lower:
                    canonical_label = "Potato___Fruit"
            elif crop.lower() == "tomato":
                if "bacterial" in folder_lower:
                    canonical_label = "Tomato___Bacterial_spot"
                elif "early" in folder_lower:
                    canonical_label = "Tomato___Early_blight"
                elif "late" in folder_lower:
                    canonical_label = "Tomato___Late_blight"
                elif "mold" in folder_lower:
                    canonical_label = "Tomato___Leaf_Mold"
                elif "septoria" in folder_lower:
                    canonical_label = "Tomato___Septoria_leaf_spot"
                elif "healthy" in folder_lower:
                    canonical_label = "Tomato___healthy"
                elif "flower" in folder_lower:
                    canonical_label = "Tomato___Flower"
                elif "fruit" in folder_lower:
                    canonical_label = "Tomato___Fruit"

            if not canonical_label:
                canonical_label = f"{crop}___{folder_name}"

        for f in os.listdir(folder_path):
            if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                filepaths.append(os.path.join(folder_path, f))
                labels.append(canonical_label)

df = pd.DataFrame({"filepath": filepaths, "label": labels})

if df.empty:
    print(f" [ERROR] No images found inside {DATASET_DIR}!")
    exit(1)

classes = sorted(df["label"].unique().tolist())
num_classes = len(classes)

print(f" [OK] Found {len(df)} images across {num_classes} categories:")
for c in classes:
    count = len(df[df["label"] == c])
    print(f"  - {c}: {count} images")

os.makedirs(MODEL_DIR, exist_ok=True)
idx_to_class = {str(i): c for i, c in enumerate(classes)}
with open(CLASSES_PATH, "w") as f:
    json.dump(idx_to_class, f, indent=2)
print(f"\nSaved class mapping -> {CLASSES_PATH}")

# Data Generators using MobileNetV2 preprocess_input
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

# Align Keras train_gen class mapping to JSON
class_indices = train_gen.class_indices
json_indices = {str(v): k for k, v in class_indices.items()}
with open(CLASSES_PATH, "w") as f:
    json.dump(json_indices, f, indent=2)
print(f"Verified Keras class mapping saved to -> {CLASSES_PATH}")

# Calculate Class Weights to balance Potato & Tomato samples
y_train_indices = train_gen.classes
class_weights_vals = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(y_train_indices),
    y=y_train_indices
)
class_weights_dict = {i: weight for i, weight in enumerate(class_weights_vals)}
print(f"\nCalculated Class Weights: {class_weights_dict}")

# Build Model
print("\nBuilding MobileNetV2 model...")
base = MobileNetV2(weights="imagenet", include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))
base.trainable = False

x = base.output
x = GlobalAveragePooling2D()(x)
x = Dense(256, activation="relu")(x)
x = Dropout(0.3)(x)
out = Dense(num_classes, activation="softmax")(x)

model = Model(inputs=base.input, outputs=out)
model.compile(optimizer=Adam(1e-3), loss="categorical_crossentropy", metrics=["accuracy"])

callbacks = [
    ModelCheckpoint(MODEL_PATH, save_best_only=True, monitor="val_accuracy", verbose=1),
    EarlyStopping(patience=6, restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, verbose=1)
]

print("\n--- Training Model Head ---")
model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=12,
    class_weight=class_weights_dict,
    callbacks=callbacks
)

print("\n--- Fine-tuning MobileNetV2 Base ---")
base.trainable = True
for layer in base.layers[:-40]:
    layer.trainable = False

model.compile(optimizer=Adam(1e-4), loss="categorical_crossentropy", metrics=["accuracy"])
model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=10,
    class_weight=class_weights_dict,
    callbacks=callbacks
)

print(f"\n [OK] Plant Disease Model successfully trained and saved!")
print(f"   Model file  -> {MODEL_PATH}")
print(f"   Class Index -> {CLASSES_PATH}")
