"""
train.py - Plant Disease Detection with MobileNetV2
====================================================
Downloads PlantVillage from Kaggle automatically (if kaggle API key configured),
OR falls back to your local plant_disease_dataset/ folder.

Kaggle setup (one time):
  pip install kaggle
  Put kaggle.json (from kaggle.com/account) in C:/Users/<you>/.kaggle/kaggle.json
  Then run: python train.py

Without Kaggle: just run python train.py — it uses your local images
with aggressive augmentation.
"""

import os
import json
import shutil
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# --------------------------------------------------
# Config
# --------------------------------------------------
IMG_SIZE      = 224
BATCH_SIZE    = 16
EPOCHS        = 30
MODEL_DIR     = "model"
DATASET_PATH  = "plant_disease_dataset"
MODEL_PATH    = os.path.join(MODEL_DIR, "plant_disease_model.h5")
CLASSES_PATH  = os.path.join(MODEL_DIR, "class_indices.json")

# Use local plant_disease_dataset exclusively
use_kaggle = False
search_root = DATASET_PATH
print(f"Using local dataset at: {search_root}")

for item in os.listdir(search_root):
    item_path = os.path.join(search_root, item)
    if not os.path.isdir(item_path):
        continue

    inner = os.listdir(item_path)
    # Nested layout: Plant/Disease/img.jpg
    has_subdirs = any(os.path.isdir(os.path.join(item_path, x)) for x in inner)
    if has_subdirs:
        for disease in os.listdir(item_path):
            disease_path = os.path.join(item_path, disease)
            if not os.path.isdir(disease_path):
                continue
            label = f"{item}_{disease}"
            for img in os.listdir(disease_path):
                if img.lower().endswith(('.jpg', '.jpeg', '.png')):
                    filepaths.append(os.path.join(disease_path, img))
                    labels.append(label)
    else:
        # Flat layout: Disease/img.jpg (Kaggle color folder)
        label = item
        for img in inner:
            if img.lower().endswith(('.jpg', '.jpeg', '.png')):
                filepaths.append(os.path.join(item_path, img))
                labels.append(label)

df = pd.DataFrame({"filepath": filepaths, "label": labels})

classes = sorted(df["label"].unique().tolist())
num_classes = len(classes)

print(f"\n✅ Found {len(df)} images across {num_classes} classes.")
print("Classes:", classes)

# --------------------------------------------------
# 3. Save class_indices.json
# --------------------------------------------------
os.makedirs(MODEL_DIR, exist_ok=True)
idx_to_class = {str(i): c for i, c in enumerate(classes)}
with open(CLASSES_PATH, "w") as f:
    json.dump(idx_to_class, f, indent=2)
print(f"Saved: {CLASSES_PATH}")

# --------------------------------------------------
# 4. Data Generators
# --------------------------------------------------
# When dataset is small (<500 imgs) use heavier augmentation
heavy_aug = len(df) < 500
print(f"Using {'HEAVY' if heavy_aug else 'standard'} augmentation (dataset has {len(df)} images).")

train_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    rotation_range=40         if heavy_aug else 20,
    zoom_range=0.35           if heavy_aug else 0.2,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.25          if heavy_aug else 0.1,
    brightness_range=[0.7, 1.3] if heavy_aug else [0.85, 1.15],
    horizontal_flip=True,
    fill_mode="nearest"
)

val_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

train_gen = train_datagen.flow_from_dataframe(
    df, x_col="filepath", y_col="label",
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="training"
)

val_gen = val_datagen.flow_from_dataframe(
    df, x_col="filepath", y_col="label",
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="validation"
)

# --------------------------------------------------
# 5. Model (MobileNetV2 + custom head)
# --------------------------------------------------
print("Building model...")
base = MobileNetV2(weights="imagenet", include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))
base.trainable = False

x = base.output
x = GlobalAveragePooling2D()(x)
x = Dense(256, activation="relu")(x)
x = Dropout(0.4)(x)
out = Dense(num_classes, activation="softmax")(x)

model = Model(inputs=base.input, outputs=out)
model.compile(optimizer=Adam(1e-4), loss="categorical_crossentropy", metrics=["accuracy"])
model.summary()

# --------------------------------------------------
# 6. Phase 1 training (head only)
# --------------------------------------------------
callbacks = [
    ModelCheckpoint(MODEL_PATH, save_best_only=True, monitor="val_accuracy", verbose=1),
    EarlyStopping(patience=6, restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, verbose=1)
]

print("\n--- Phase 1: Training head only ---")
model.fit(train_gen, validation_data=val_gen, epochs=EPOCHS, callbacks=callbacks)

# --------------------------------------------------
# 7. Phase 2: Fine-tune last 30 base layers
# --------------------------------------------------
print("\n--- Phase 2: Fine-tuning top layers ---")
base.trainable = True
for layer in base.layers[:-30]:
    layer.trainable = False

model.compile(optimizer=Adam(1e-5), loss="categorical_crossentropy", metrics=["accuracy"])
model.fit(train_gen, validation_data=val_gen, epochs=10, callbacks=callbacks)

print(f"\n✅ Training done! Model saved → {MODEL_PATH}")
print(f"   Class mapping  → {CLASSES_PATH}")