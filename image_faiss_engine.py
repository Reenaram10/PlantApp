import os
import json
import numpy as np
import faiss
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
from tensorflow.keras.preprocessing import image

class ImageFAISSRAG:
    """
    A FAISS-backed RAG engine specifically for Images.
    Uses MobileNetV2 as a feature extractor to convert images into 1280-dimensional vectors.
    """
    def __init__(self, index_path="local_image_faiss_index"):
        print(" [Image FAISS] Initializing MobileNetV2 feature extractor...")
        # Load MobileNetV2 without the top classification layer, use average pooling
        self.model = MobileNetV2(weights='imagenet', include_top=False, pooling='avg')
        self.index_path = index_path
        self.metadata_path = os.path.join(index_path, "metadata.json")
        self.faiss_index_path = os.path.join(index_path, "index.faiss")
        
        # MobileNetV2 'avg' pooling outputs a vector of size 1280
        self.dimension = 1280
        self.vector_db = None
        self.metadata = []

    def get_embedding(self, img_path):
        """Extract feature vector from a single image."""
        try:
            # MobileNetV2 expects 224x224 images
            img = image.load_img(img_path, target_size=(224, 224))
            x = image.img_to_array(img)
            x = np.expand_dims(x, axis=0)
            x = preprocess_input(x)
            
            # Predict returns a (1, 1280) array
            features = self.model.predict(x, verbose=0)
            return features[0]
        except Exception as e:
            print(f" [Image FAISS] Error processing {img_path}: {e}")
            return None

    def build_index(self, directories):
        """Build FAISS index from all common image files in the given directories."""
        valid_extensions = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
        
        if not os.path.exists(self.index_path):
            os.makedirs(self.index_path)

        embeddings_list = []
        metadata_list = []

        for directory in directories:
            if not os.path.exists(directory):
                print(f" [Image FAISS] Directory not found: {directory}. Skipping...")
                continue
                
            for root, _, files in os.walk(directory):
                for file in files:
                    ext = os.path.splitext(file)[1].lower()
                    if ext in valid_extensions:
                        full_path = os.path.abspath(os.path.join(root, file))
                        print(f" [Image FAISS] Extracting vector for: {file}")
                        vector = self.get_embedding(full_path)
                        if vector is not None:
                            embeddings_list.append(vector)
                            metadata_list.append({
                                "filename": file,
                                "path": full_path,
                                "directory": os.path.basename(root)
                            })

        if not embeddings_list:
            print(" [Image FAISS] No valid images found to index.")
            return

        # Convert to numpy array of type float32 (required by faiss)
        embeddings_matrix = np.array(embeddings_list).astype('float32')

        # faiss.normalize_L2(embeddings_matrix) # Uncomment if using inner product
        # Initialize L2 distance index
        self.vector_db = faiss.IndexFlatL2(self.dimension)
        self.vector_db.add(embeddings_matrix)
        self.metadata = metadata_list

        # Save index
        faiss.write_index(self.vector_db, self.faiss_index_path)
        print(f" [Image FAISS] Saved FAISS index with {len(self.metadata)} items to {self.faiss_index_path}")

        # Save metadata
        with open(self.metadata_path, 'w', encoding='utf-8') as f:
            json.dump(self.metadata, f, indent=4)
        print(f" [Image FAISS] Saved metadata to {self.metadata_path}")

    def load_index(self):
        """Load the FAISS index and metadata from disk."""
        if os.path.exists(self.faiss_index_path) and os.path.exists(self.metadata_path):
            try:
                self.vector_db = faiss.read_index(self.faiss_index_path)
                with open(self.metadata_path, 'r', encoding='utf-8') as f:
                    self.metadata = json.load(f)
                print(f" [Image FAISS] Loaded index containing {self.vector_db.ntotal} images.")
                return True
            except Exception as e:
                print(f" [Image FAISS] Error loading index: {e}")
                return False
        return False

    def retrieve_similar(self, query_img_path, k=5):
        """Find the top-k most similar images to a query image."""
        if not self.vector_db:
            if not self.load_index():
                print(" [Image FAISS] Index missing. Cannot retrieve.")
                return []

        query_vector = self.get_embedding(query_img_path)
        if query_vector is None:
            return []

        query_matrix = np.array([query_vector]).astype('float32')
        # Perform distance search
        distances, indices = self.vector_db.search(query_matrix, k)

        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx == -1: # Invalid index
                continue
            
            # Convert raw L2 distance to similarity score 0-1
            similarity = 1.0 / (1.0 + float(dist))
            meta = self.metadata[idx]
            results.append({
                "similarity_score": similarity,
                "distance": float(dist),
                "metadata": meta
            })

        return results
