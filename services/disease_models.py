"""
Plant Disease Detection Models Service
Lazy-loaded Keras / TensorFlow neural models and classification indices.
"""
import os
import json
from config import (
    APP_ROOT, CROP_MODEL_PATH, CROP_CLASSES_PATH,
    POTATO_MODEL_PATH, POTATO_CLASSES_PATH,
    TOMATO_MODEL_PATH, TOMATO_CLASSES_PATH,
    FLOWER_MODEL_PATH, FLOWER_CLASSES_PATH
)

_cached_models = {}
_cached_indices = {}

def load_json_mapping(path):
    """Load JSON class index mapping safely."""
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            print(f" [Model] Error loading {path}: {e}")
    return {}

def _load_keras_model(model_name, model_path, indices):
    """Internal lazy loader for a Keras model with exact legacy logging."""
    if model_name in _cached_models:
        return _cached_models[model_name]
    
    try:
        # Import TensorFlow/Keras only inside disease_models service
        try:
            from keras.models import load_model
        except ImportError:
            from tensorflow.keras.models import load_model  # type: ignore[import-untyped,import-not-found]
        
        display_names = {
            "crop_type_model": "Stage 1 Crop Type model",
            "potato_disease_model": "Stage 2 Potato Disease model",
            "tomato_disease_model": "Stage 2 Tomato Disease model",
            "flower_crop_model": "Dedicated Flower Crop model"
        }
        disp = display_names.get(model_name, model_name)
        print(f" [Model] Loading {disp} from {model_path}...")
        model = load_model(model_path)
        print(f" [Model] {model_name.replace('_model', '').replace('_', ' ').title()} model loaded: {indices}")
        _cached_models[model_name] = model
        return model
    except Exception as e:
        print(f" [Model] Error loading {model_name}: {e}")
        _cached_models[model_name] = None
        return None

def get_crop_indices():
    if "crop_indices" not in _cached_indices:
        _cached_indices["crop_indices"] = load_json_mapping(CROP_CLASSES_PATH)
    return _cached_indices["crop_indices"]

def get_potato_indices():
    if "potato_indices" not in _cached_indices:
        _cached_indices["potato_indices"] = load_json_mapping(POTATO_CLASSES_PATH)
    return _cached_indices["potato_indices"]

def get_tomato_indices():
    if "tomato_indices" not in _cached_indices:
        _cached_indices["tomato_indices"] = load_json_mapping(TOMATO_CLASSES_PATH)
    return _cached_indices["tomato_indices"]

def get_flower_indices():
    if "flower_indices" not in _cached_indices:
        _cached_indices["flower_indices"] = load_json_mapping(FLOWER_CLASSES_PATH)
    return _cached_indices["flower_indices"]

def get_crop_type_model():
    return _load_keras_model("crop_type_model", CROP_MODEL_PATH, get_crop_indices())

def get_potato_disease_model():
    return _load_keras_model("potato_disease_model", POTATO_MODEL_PATH, get_potato_indices())

def get_tomato_disease_model():
    return _load_keras_model("tomato_disease_model", TOMATO_MODEL_PATH, get_tomato_indices())

def get_flower_crop_model():
    return _load_keras_model("flower_crop_model", FLOWER_MODEL_PATH, get_flower_indices())

# Module-level __getattr__ enables seamless attribute access (e.g. disease_models.crop_type_model)
def __getattr__(name):
    if name == "crop_indices":
        val = get_crop_indices()
    elif name == "potato_indices":
        val = get_potato_indices()
    elif name == "tomato_indices":
        val = get_tomato_indices()
    elif name == "flower_indices":
        val = get_flower_indices()
    elif name == "crop_type_model":
        val = get_crop_type_model()
    elif name == "potato_disease_model":
        val = get_potato_disease_model()
    elif name == "tomato_disease_model":
        val = get_tomato_disease_model()
    elif name == "flower_crop_model":
        val = get_flower_crop_model()
    else:
        raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
    globals()[name] = val
    return val
