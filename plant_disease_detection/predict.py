import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image


# Load model
model = tf.keras.models.load_model(
    "model/plant_disease_model.h5"
)


import json
import sys

# Class names (dynamically loaded from train output)
try:
    with open("model/class_indices.json", "r") as f:
        class_mapping = json.load(f)
        classes = [class_mapping[str(i)] for i in range(len(class_mapping))]
except FileNotFoundError:
    print("❌ Error: model/class_indices.json not found. Run train.py first!")
    sys.exit(1)

# Test image
img_path =  "test_leaf.jpg.jpg"


# Image preprocessing
img = image.load_img(
    img_path,
    target_size=(224,224)
)

img_array = image.img_to_array(img)

img_array = img_array / 255.0

img_array = np.expand_dims(
    img_array,
    axis=0
)


# Prediction
prediction = model.predict(img_array)


predicted_index = np.argmax(prediction)


disease = classes[predicted_index]


confidence = np.max(prediction)



print("-----------------------------")
print("Disease Prediction :", disease)
print("Confidence :", round(confidence * 100, 2), "%")
print("-----------------------------")