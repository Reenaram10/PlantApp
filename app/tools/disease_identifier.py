
import os
import json
import numpy as np
import tensorflow as tf

from keras.preprocessing import image
from keras.applications.mobilenet_v2 import preprocess_input

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)

MODEL_DIR = os.path.join(
    BASE_DIR, "plant_disease_detection", "model"
)

CONFIDENCE_THRESHOLD = 0.90

# Load both trained disease models once
potato_model = tf.keras.models.load_model(
    os.path.join(MODEL_DIR, "potato_disease_model.h5")
)

tomato_model = tf.keras.models.load_model(
    os.path.join(MODEL_DIR, "tomato_disease_model.h5")
)

with open(os.path.join(MODEL_DIR, "potato_indices.json"), encoding="utf-8") as f:
    potato_indices = json.load(f)

with open(os.path.join(MODEL_DIR, "tomato_indices.json"), encoding="utf-8") as f:
    tomato_indices = json.load(f)


def format_disease_card(disease_info: dict) -> str:

    plant_name = disease_info.get("plant_name", "Unknown")
    disease_name = disease_info.get("disease_name", "Unknown Disease")
    confidence_pct = f"{int(round(disease_info.get('confidence', 0.0) * 100))}%"
    status = disease_info.get("status", "uncertain")

    if status == "healthy":
        status_display = "Healthy"
    elif status == "uncertain":
        status_display = "Uncertain"
    elif status == "not_applicable":
        status_display = "Not Applicable"
    else:
        status_display = "Disease Detected"

    return (
        "\n========== DISEASE IDENTIFICATION ==========\n"
        f"Plant Name  : {plant_name}\n"
        f"Disease     : {disease_name}\n"
        f"Confidence  : {confidence_pct}\n"
        f"Status      : {status_display}\n"
        "============================================"
    )


def identify_disease(image_path: str, plant_name: str) -> dict:
    """
    Identifies disease using the trained Potato or Tomato model.

    The plant name is received from the plant identification tool.
    """

    if not os.path.isfile(image_path):
        raise FileNotFoundError(
            f"Image not found: {image_path}"
        )

    plant_name = plant_name.strip().title()

    if plant_name == "Potato":
        model = potato_model
        labels = potato_indices

    elif plant_name == "Tomato":
        model = tomato_model
        labels = tomato_indices

    else:
        raise ValueError(
            f"Disease model is not available for {plant_name}"
        )

    # Image preprocessing
    img = image.load_img(
        image_path,
        target_size=(224, 224)
    )

    arr = preprocess_input(
        image.img_to_array(img)
    )

    img_batch = np.expand_dims(arr, axis=0)

    # Actual trained model prediction
    predictions = model.predict(
        img_batch,
        verbose=0
    )[0]

    predicted_index = int(np.argmax(predictions))

    result_label = labels.get(str(predicted_index))

    if not result_label:
        raise ValueError("Unable to map disease prediction")

    confidence = float(np.max(predictions))

    # Extract the predicted class
    clean_label = result_label.split("___")[-1]

    clean_label = clean_label.replace("_", " ").title()

    # Identify image type
    if clean_label.lower() == "flower":
        image_type = "flower"
        disease_name = "Flower"
        status = "not_applicable"

    elif clean_label.lower() == "fruit":
        image_type = "fruit"
        disease_name = "Fruit"
        status = "not_applicable"

    elif clean_label.lower() == "healthy":
        image_type = "leaf"
        disease_name = "None Detected"
        status = "healthy"

    elif confidence < CONFIDENCE_THRESHOLD:
        image_type = "leaf"
        disease_name = "Unknown Disease"
        status = "uncertain"

    else:
        image_type = "leaf"
        disease_name = clean_label
        status = "detected"

    disease_result = {
        "plant_name": plant_name,
        "disease_name": disease_name,
        "confidence": round(confidence, 4),
        "status": status,
        "image_type": image_type,
        "predicted_class": result_label
    }

    print(format_disease_card(disease_result))

    return disease_result
