import os
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

class PlantFAISSRAG:
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        print(f" [FAISS] Initializing embeddings with {model_name}...")
        self.embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        self.vector_db = None
        self.index_path = "local_faiss_index"

    def sync_from_db(self, PlantKnowledgeModel, DiseaseKnowledgeModel=None):
        """Load data from database and CSV files to create FAISS index"""
        print(" [FAISS] Syncing knowledge from database and CSV files...")
        import csv
        try:
            documents = []
            
            # --- 1. Index Plants from DB ---
            plant_records = []
            try:
                plant_records = PlantKnowledgeModel.query.all()
            except Exception as dberr:
                print(f" [FAISS] DB query failed (maybe no context): {dberr}")

            for r in plant_records:
                content = f"Plant Name: {r.plant_name}\n" \
                          f"Scientific Name: {r.scientific_name}\n" \
                          f"Description: {r.description}\n" \
                          f"Sunlight Requirement: {r.sunlight}\n" \
                          f"Water Requirement: {r.water_requirement}\n" \
                          f"Soil Type: {r.soil_type}\n" \
                          f"Medicinal Uses: {r.medicinal_uses}\n" \
                          f"Common Diseases: {r.common_diseases}\n" \
                          f"Care Tips: {r.care_tips}"
                
                doc = Document(
                    page_content=content,
                    metadata={"db_id": r.id, "name": r.plant_name, "type": "plant"}
                )
                documents.append(doc)
            
            # --- 2. Index Diseases from DB ---
            if DiseaseKnowledgeModel:
                disease_records = []
                try:
                    disease_records = DiseaseKnowledgeModel.query.all()
                except Exception as dberr:
                    print(f" [FAISS] Disease DB query failed: {dberr}")

                for r in disease_records:
                    content = f"Disease Name: {r.disease_name}\n" \
                              f"Symptoms: {r.symptoms}\n" \
                              f"Causes: {r.causes}\n" \
                              f"Organic Treatment: {r.organic_treatment}\n" \
                              f"Chemical Treatment: {r.chemical_treatment}\n" \
                              f"Prevention: {r.prevention}"
                    
                    doc = Document(
                        page_content=content,
                        metadata={"db_id": r.id, "name": r.disease_name, "type": "disease"}
                    )
                    documents.append(doc)

            # --- 3. Index GEOGRAPHY_PROFILEE.csv ---
            csv_geography = os.path.join(os.path.dirname(os.path.abspath(__file__)), "GEOGRAPHY_PROFILEE.csv")
            if os.path.exists(csv_geography):
                print(f" [FAISS] Loading geography profiles from {csv_geography}")
                try:
                    with open(csv_geography, 'r', encoding='latin1') as f:
                        reader = csv.DictReader(f)
                        for row in reader:
                            plant_name = row.get('plant_name', '').strip()
                            if not plant_name:
                                continue
                            content = (
                                f"Plant Name: {plant_name}\n"
                                f"Geography & Climate Profile:\n"
                                f"- Country/Region: {row.get('country', '')} / {row.get('state_or_region', '')} ({row.get('native_status', '')})\n"
                                f"- Climate Zone: {row.get('climate_zone', '')}\n"
                                f"- Best Growing Season & Months: {row.get('best_growing_season', '')} ({row.get('best_growing_months', '')})\n"
                                f"- Temperature Range: {row.get('temperature_min', '')}°C to {row.get('temperature_max', '')}°C\n"
                                f"- Annual Rainfall: {row.get('annual_rainfall_min', '')} mm to {row.get('annual_rainfall_max', '')} mm\n"
                                f"- Humidity Range: {row.get('humidity_min', '')}% to {row.get('humidity_max', '')}%\n"
                                f"- Soil Suitability: {row.get('soil_suitability', '')}\n"
                                f"- Sunlight: {row.get('sunlight_condition', '')}\n"
                                f"- Tolerances: Frost: {row.get('frost_tolerance', '')}, Heat: {row.get('heat_tolerance', '')}, Drought: {row.get('drought_tolerance', '')}\n"
                                f"- Remarks: {row.get('remarks', '')}"
                            )
                            doc = Document(
                                page_content=content,
                                metadata={"name": plant_name, "type": "geography"}
                            )
                            documents.append(doc)
                except Exception as csv_err:
                    print(f" [FAISS] Error reading geography CSV: {csv_err}")

            # --- 4. Index seasonal_profile.csv ---
            csv_season = os.path.join(os.path.dirname(os.path.abspath(__file__)), "seasonal_profile.csv")
            if os.path.exists(csv_season):
                print(f" [FAISS] Loading seasonal profiles from {csv_season}")
                try:
                    with open(csv_season, 'r', encoding='latin1') as f:
                        reader = csv.DictReader(f)
                        for row in reader:
                            plant_name = row.get('plant_name', '').strip()
                            if not plant_name:
                                continue
                            content = (
                                f"Plant Name: {plant_name}\n"
                                f"Seasonal Care Profile:\n"
                                f"- Season: {row.get('season', '')}\n"
                                f"- Watering: {row.get('watering_frequency', '')} ({row.get('water_amount_liters', '')} Liters/frequency)\n"
                                f"- Fertilizer: {row.get('fertilizer_type', '')} ({row.get('fertilizer_amount', '')})\n"
                                f"- NPK Nutrients: Nitrogen: {row.get('nitrogen_requirement', '')}, Phosphorus: {row.get('phosphorus_requirement', '')}, Potassium: {row.get('potassium_requirement', '')}\n"
                                f"- Sunlight & Temp: {row.get('sunlight_requirement', '')}, {row.get('temperature_range_celsius', '')}°C\n"
                                f"- Wind Tolerance & Care: {row.get('wind_tolerance', '')}. Care Instructions: {row.get('care_instructions', '')}"
                            )
                            doc = Document(
                                page_content=content,
                                metadata={"name": plant_name, "type": "season"}
                            )
                            documents.append(doc)
                except Exception as csv_err:
                    print(f" [FAISS] Error reading seasonal CSV: {csv_err}")

            # --- 5. Index plant_disease_profiles (1).csv ---
            csv_disease = os.path.join(os.path.dirname(os.path.abspath(__file__)), "plant_disease_profiles (1).csv")
            if os.path.exists(csv_disease):
                print(f" [FAISS] Loading disease profiles from {csv_disease}")
                try:
                    with open(csv_disease, 'r', encoding='latin1') as f:
                        reader = csv.DictReader(f)
                        for row in reader:
                            plant_name = row.get('plant_name', '').strip()
                            if not plant_name:
                                continue
                            content = (
                                f"Plant Name: {plant_name}\n"
                                f"Disease Profile:\n"
                                f"- Disease Name: {row.get('disease_name', '')} ({row.get('disease_type', '')})\n"
                                f"- Affected Parts & Severity: {row.get('affected_parts', '')} ({row.get('severity', '')})\n"
                                f"- Symptoms: {row.get('symptoms', '')}\n"
                                f"- Causes: {row.get('causes', '')}\n"
                                f"- Prevention: {row.get('prevention', '')}\n"
                                f"- Treatment: {row.get('treatment', '')}"
                            )
                            doc = Document(
                                page_content=content,
                                metadata={"name": plant_name, "type": "disease"}
                            )
                            documents.append(doc)
                except Exception as csv_err:
                    print(f" [FAISS] Error reading disease CSV: {csv_err}")

            # --- 6. Index plant_profiles.csv ---
            csv_profiles = os.path.join(os.path.dirname(os.path.abspath(__file__)), "plant_profiles.csv")
            if os.path.exists(csv_profiles):
                print(f" [FAISS] Loading general profiles from {csv_profiles}")
                try:
                    with open(csv_profiles, 'r', encoding='latin1') as f:
                        reader = csv.DictReader(f)
                        for row in reader:
                            plant_name = row.get('plant_name', '').strip()
                            if not plant_name:
                                continue
                            content = (
                                f"Plant Name: {plant_name}\n"
                                f"General Profile:\n"
                                f"- Scientific Name: {row.get('scientific_name', '')}\n"
                                f"- Growing Conditions: Temp: {row.get('temperature', '')}, Humidity: {row.get('humidity', '')}, Moisture: {row.get('moisture', '')}, Sunlight: {row.get('sunlight', '')}\n"
                                f"- Soil & Water: Soil: {row.get('soil_type', '')}, Watering: {row.get('watering_frequency', '')}\n"
                                f"- Nutrients: Nitrogen: {row.get('nitrogen', '')}, Phosphorus: {row.get('phosphorus', '')}, Potassium: {row.get('potassium', '')}\n"
                                f"- Growth Duration: {row.get('growth_duration', '')}"
                            )
                            doc = Document(
                                page_content=content,
                                metadata={"name": plant_name, "type": "plant"}
                            )
                            documents.append(doc)
                except Exception as csv_err:
                    print(f" [FAISS] Error reading general profiles CSV: {csv_err}")
            
            if documents:
                self.vector_db = FAISS.from_documents(documents, self.embeddings)
                self.vector_db.save_local(self.index_path)
                print(f" [FAISS] Successfully indexed {len(documents)} total knowledge items.")
            else:
                print(" [FAISS] No records found to index.")
        except Exception as e:
            print(f" [FAISS] Error during sync: {e}")

    def load_index(self):
        """Load index from disk if it exists"""
        if os.path.exists(self.index_path):
            try:
                self.vector_db = FAISS.load_local(
                    self.index_path, 
                    self.embeddings, 
                    allow_dangerous_deserialization=True
                )
                print(" [FAISS] Loaded index from disk.")
                return True
            except Exception as e:
                print(f" [FAISS] Error loading index: {e}")
                return False
        return False

    def retrieve(self, query, k=3, category_filter=None):
        """Retrieve most similar plant knowledge, optionally filtered by category filter"""
        if not self.vector_db:
            if not self.load_index():
                print(" [FAISS] Index not found and not loaded.")
                return []
        
        try:
            search_filter = None
            if category_filter:
                search_filter = {"type": category_filter}
                
            results = self.vector_db.similarity_search_with_score(query, k=k, filter=search_filter)
            
            formatted = []
            for doc, distance in results:
                # Convert distance to a similarity score (0 to 1)
                similarity = 1.0 / (1.0 + distance)
                
                formatted.append({
                    "content": doc.page_content,
                    "metadata": doc.metadata,
                    "score": float(similarity),
                    "raw_distance": float(distance)
                })
            return formatted
        except Exception as e:
            print(f" [FAISS] Error during retrieval: {e}")
            return []
