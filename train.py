import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator # type: ignore
from tensorflow.keras.applications import MobileNetV2 # type: ignore
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout # type: ignore
from tensorflow.keras.models import Model # type: ignore
import json

# Setup parameters
# The user needs to put their images in subfolders inside the DATASET_DIR.
# Example: 
# dataset/
# ├── Potato_Early_blight/
# │   ├── image1.jpg...
# ├── Potato_healthy/
# ├── Tomato_Early_blight/
# └── Tomato_healthy/
DATASET_DIR = "dataset" 
MODEL_SAVE_PATH = "plant_disease_model.h5"
CLASSES_SAVE_PATH = "class_indices.json"

IMG_SIZE = (224, 224)
BATCH_SIZE = 8 # Small batch size because dataset is very small
EPOCHS = 15    # Number of times to train on the data

def build_model(num_classes):
    # 1. Load Pretrained MobileNetV2 (Transfer Learning)
    # This model already understands edges, shapes, and textures from millions of images
    base_model = MobileNetV2(
        weights='imagenet', 
        include_top=False, 
        input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)
    )
    
    # Freeze the base model so we don't destroy the pre-trained weights
    base_model.trainable = False

    # 2. Add custom classification head on top
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x) # Helps prevent overfitting since dataset is small
    predictions = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=predictions)
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), 
        loss='categorical_crossentropy', 
        metrics=['accuracy']
    )
    return model

def train():
    if not os.path.exists(DATASET_DIR):
        print(f"❌ Error: Please create a folder named '{DATASET_DIR}' and put your class subfolders inside it.")
        return

    # Data Augmentation (CRITICAL for very small datasets of 6-7 images)
    # This creates fake variations of your 6 images (rotated, zoomed, flipped)
    train_datagen = ImageDataGenerator(
        rescale=1./255,           # Normalize pixel values
        rotation_range=40,        # Rotate by up to 40 degrees
        width_shift_range=0.2,    # Shift width
        height_shift_range=0.2,   # Shift height
        shear_range=0.2,          # Distort along an axis
        zoom_range=0.2,           # Zoom in/out
        horizontal_flip=True,     # Flip left/right
        fill_mode='nearest'     
    )

    print("Load dataset...")
    # Read the dataset
    train_generator = train_datagen.flow_from_directory(
        DATASET_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical'
    )

    num_classes = len(train_generator.class_indices)
    
    if num_classes == 0:
        print("❌ Error: No images found. Ensure you have subdirectories for each disease class.")
        return

    print(f"Classes detected: {train_generator.class_indices}")

    # Save the class dictionary to a JSON file so predict.py knows the labels
    with open(CLASSES_SAVE_PATH, 'w') as f:
        # Swap key-value so we map index -> class name
        idx_to_class = {v: k for k, v in train_generator.class_indices.items()}
        json.dump(idx_to_class, f)

    # Build and Train model
    print("Building model...")
    model = build_model(num_classes)

    print("Starting training...")
    model.fit(
        train_generator,
        epochs=EPOCHS
    )

    # Save model
    model.save(MODEL_SAVE_PATH)
    print(f"✅ Training complete! Model saved to '{MODEL_SAVE_PATH}'")
    print(f"✅ Class mappings saved to '{CLASSES_SAVE_PATH}'")

if __name__ == "__main__":
    train()
