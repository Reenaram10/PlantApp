import json
import os
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class PlantRAG:
    def __init__(self, json_path):
        self.json_path = json_path
        self.knowledge_base = []
        self.vectorizer = TfidfVectorizer(stop_words='english')
        self.tfidf_matrix = None
        self.load_knowledge()

    def load_knowledge(self):
        if os.path.exists(self.json_path):
            with open(self.json_path, 'r') as f:
                self.knowledge_base = json.load(f)
            
            # Prepare texts for vectorization
            texts = [item['content'] for item in self.knowledge_base]
            if texts:
                self.tfidf_matrix = self.vectorizer.fit_transform(texts)
                print(f" [RAG] Indexed {len(self.knowledge_base)} knowledge chunks.")
        else:
            print(f" [RAG] ERROR: Knowledge base not found at {self.json_path}")

    def retrieve(self, query, top_k=3):
        if not self.knowledge_base or self.tfidf_matrix is None:
            return []
        
        query_vec = self.vectorizer.transform([query])
        similarities = cosine_similarity(query_vec, self.tfidf_matrix).flatten()
        
        # Get top-k indices
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            if similarities[idx] > 0.1: # Threshold to ensure relevance
                item = self.knowledge_base[idx]
                results.append({
                    "content": item['content'],
                    "category": item['category'],
                    "score": float(similarities[idx]),
                    "type": item.get("type", "general"),
                    "target_id": item.get("target_id"),
                    "plant_name": item.get("plant_name"),
                    "category_name": item.get("category_name")
                })
        
        return results

    def get_profile_summary(self):
        # Concatenate all content for a full profile
        return "\n\n".join([f"### {item['category']}\n{item['content']}" for item in self.knowledge_base])

# Global instance
DEFAULT_KB_PATH = os.path.join(os.path.dirname(__file__), "unified_plants_knowledge.json")
tomato_rag = PlantRAG(DEFAULT_KB_PATH)
# alias for general use
plant_rag = tomato_rag
