
import os
import json
import numpy as np
import tensorflow as tf

from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)

MODEL_DIR = os.path.join(
    BASE_DIR, "plant_disease_detection", "model"
)

MODEL_PATH = os.path.join(MODEL_DIR, "crop_type_model.h5")
LABEL_PATH = os.path.join(MODEL_DIR, "crop_indices.json")

SCIENTIFIC_NAMES = {
    "Tomato": "Solanum lycopersicum",
    "Potato": "Solanum tuberosum"
}

# Load trained crop model once
crop_model = tf.keras.models.load_model(MODEL_PATH)

with open(LABEL_PATH, "r", encoding="utf-8") as f:
    crop_indices = json.load(f)


def identify_plant(image_path: str) -> dict:

    if not os.path.isfile(image_path):
        raise FileNotFoundError(
            f"Image not found: {image_path}"
        )

    img = image.load_img(
        image_path,
        target_size=(224, 224)
    )

    arr = preprocess_input(
        image.img_to_array(img)
    )

    img_batch = np.expand_dims(arr, axis=0)

    predictions = crop_model.predict(
        img_batch,
        verbose=0
    )[0]

    predicted_index = int(np.argmax(predictions))

    plant_name = crop_indices.get(
        str(predicted_index)
    )

    confidence = float(np.max(predictions))

    if not plant_name:
        raise ValueError("Unknown crop prediction")

    if confidence < 0.90:
        return {
            "name": "Unknown Plant",
            "scientific_name": "Unknown",
            "confidence": round(confidence, 4),
            "image_type": "unknown",
            "status": "uncertain"
        }

    return {
        "name": plant_name,
        "scientific_name": SCIENTIFIC_NAMES.get(
            plant_name, "Unknown"
        ),
        "confidence": round(confidence, 4),
        "image_type": "unknown",
        "status": "identified"
    }
