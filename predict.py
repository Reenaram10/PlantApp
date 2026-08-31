import os
import sys
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image # type: ignore

MODEL_PATH = "plant_disease_model.h5"
CLASSES_PATH = "class_indices.json"
IMG_SIZE = (224, 224)

def load_prediction_model():
    if not os.path.exists(MODEL_PATH) or not os.path.exists(CLASSES_PATH):
        print("❌ Error: Model or class indices not found. Run train.py first!")
        sys.exit(1)
        
    model = tf.keras.models.load_model(MODEL_PATH)
    
    with open(CLASSES_PATH, 'r') as f:
        class_indices = json.load(f)
        
    return model, class_indices

def predict(img_path):
    model, class_indices = load_prediction_model()
    
    if not os.path.exists(img_path):
        print(f"❌ Error: Image '{img_path}' not found.")
        return
        
    # Load and preprocess the image exactly as we did in training
    img = image.load_img(img_path, target_size=IMG_SIZE)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) # Convert single image to a batch
    img_array /= 255.0 # Normalize 0-1
    
    # Predict
    predictions = model.predict(img_array)
    predicted_class_idx = np.argmax(predictions[0])
    confidence = predictions[0][predicted_class_idx]
    
    # Needs string key for JSON dictionary mapping
    predicted_class_name = class_indices[str(predicted_class_idx)]
    
    print("-" * 40)
    print(f"🌿 Image: {os.path.basename(img_path)}")
    print(f"🔍 Predicted Disease: {predicted_class_name}")
    print(f"📊 Confidence: {confidence*100:.2f}%")
    print("-" * 40)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python predict.py <path_to_leaf_image.jpg>")
    else:
        predict(sys.argv[1])
