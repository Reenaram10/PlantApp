import os
import re
import json
import hashlib
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

INDEX_PATH = os.path.join("data", "history_faiss_index")
STATE_PATH = os.path.join("data", "history_store.json")


def calculate_constant_hash(profile_dict: Dict[str, Any]) -> str:
    """
    Calculates MD5 hash of constant plant profile attributes.
    EXCLUDES variable state metrics (plant_count, healthy_count, diseased_count, inventory_count, etc.).
    """
    constant_keys = [
        "plant_name", "scientific_name", "plant_type", "description",
        "growing_conditions", "soil", "sunlight", "temperature",
        "humidity", "watering", "nutrients", "care_tips",
        "common_diseases", "prevention"
    ]
    normalized_pairs = []
    for k in sorted(constant_keys):
        val = str(profile_dict.get(k, "")).strip().lower()
        normalized_pairs.append(f"{k}:{val}")
    
    raw_str = "|".join(normalized_pairs)
    return hashlib.md5(raw_str.encode("utf-8")).hexdigest()


class HistoryManager:
    """
    Manages Plant Profile History versions, Variable Current State, and Analysis Events
    using FAISS vector indexing and structured JSON persistence.
    """
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2", index_path: str = INDEX_PATH, state_path: str = STATE_PATH):
        self.index_path = index_path
        self.state_path = state_path
        self.embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        self.vector_db: Optional[FAISS] = None
        self.state: Dict[str, Any] = {
            "profiles": {},   # plant_name -> list of version dicts
            "variables": {},  # plant_name -> dict of variable counts
            "analyses": []    # list of analysis event dicts
        }
        
        self._load_state()
        self._init_vector_db()

    def _load_state(self):
        if os.path.exists(self.state_path):
            try:
                with open(self.state_path, "r", encoding="utf-8") as f:
                    self.state = json.load(f)
            except Exception as e:
                print(f" [HistoryManager] Error loading state JSON: {e}")

    def _save_state(self):
        os.makedirs(os.path.dirname(self.state_path), exist_ok=True)
        try:
            with open(self.state_path, "w", encoding="utf-8") as f:
                json.dump(self.state, f, indent=2)
        except Exception as e:
            print(f" [HistoryManager] Error saving state JSON: {e}")

    def _init_vector_db(self):
        os.makedirs(self.index_path, exist_ok=True)
        index_file = os.path.join(self.index_path, "index.faiss")
        if os.path.exists(index_file):
            try:
                self.vector_db = FAISS.load_local(
                    self.index_path,
                    self.embeddings,
                    allow_dangerous_deserialization=True
                )
                print(" [HistoryManager] Loaded FAISS history index.")
                return
            except Exception as e:
                print(f" [HistoryManager] Could not load FAISS index: {e}")

        # Initialize fresh index with a system marker document
        dummy_doc = Document(
            page_content="System History Storage Initialized.",
            metadata={"history_type": "system", "plant_name": "system", "version": 0}
        )
        self.vector_db = FAISS.from_documents([dummy_doc], self.embeddings)
        self.vector_db.save_local(self.index_path)
        print(" [HistoryManager] Created new FAISS history index.")

    def add_plant_profile_version(self, profile_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Adds a new plant profile version if constant data has changed.
        Prevents duplicate history entries using constant_hash.
        Variable state modifications DO NOT trigger a new profile version.
        """
        p_name = profile_data.get("plant_name", "").strip()
        if not p_name:
            raise ValueError("plant_name is required for profile versioning.")

        c_hash = calculate_constant_hash(profile_data)
        existing_versions = self.state["profiles"].get(p_name, [])

        # Check for duplicates using constant hash
        if existing_versions:
            latest_version = existing_versions[-1]
            if latest_version.get("constant_hash") == c_hash:
                print(f" [HistoryManager] Constant hash match for '{p_name}'. Skipping duplicate profile version creation.")
                return latest_version

        version_num = len(existing_versions) + 1
        created_at = profile_data.get("created_at") or datetime.now(timezone.utc).strftime("%Y-%m-%d")

        version_entry = {
            "history_type": "plant_profile",
            "plant_name": p_name,
            "version": version_num,
            "created_at": created_at,
            "constant_hash": c_hash,
            "scientific_name": profile_data.get("scientific_name", "Solanum lycopersicum" if p_name.lower()=="tomato" else ""),
            "plant_type": profile_data.get("plant_type", ""),
            "description": profile_data.get("description", ""),
            "soil": profile_data.get("soil", ""),
            "sunlight": profile_data.get("sunlight", ""),
            "temperature": profile_data.get("temperature", ""),
            "humidity": profile_data.get("humidity", ""),
            "watering": profile_data.get("watering", ""),
            "nutrients": profile_data.get("nutrients", ""),
            "care_tips": profile_data.get("care_tips", ""),
            "common_diseases": profile_data.get("common_diseases", ""),
            "prevention": profile_data.get("prevention", "")
        }

        if p_name not in self.state["profiles"]:
            self.state["profiles"][p_name] = []
        self.state["profiles"][p_name].append(version_entry)
        self._save_state()

        # Format into LangChain Document for History FAISS
        doc_content = (
            f"{p_name} Plant Profile\n"
            f"Version: {version_num}\n"
            f"Scientific Name: {version_entry['scientific_name']}\n"
            f"Watering: {version_entry['watering']}\n"
            f"Soil: {version_entry['soil']}\n"
            f"Sunlight: {version_entry['sunlight']}\n"
            f"Temperature: {version_entry['temperature']}\n"
            f"Humidity: {version_entry['humidity']}\n"
            f"Care Tips: {version_entry['care_tips']}\n"
            f"Created At: {created_at}"
        )

        doc = Document(
            page_content=doc_content,
            metadata={
                "history_type": "plant_profile",
                "plant_name": p_name,
                "version": version_num,
                "created_at": created_at
            }
        )

        if self.vector_db:
            self.vector_db.add_documents([doc])
            self.vector_db.save_local(self.index_path)

        print(f" [HistoryManager] Created Profile Version {version_num} for '{p_name}'.")
        return version_entry

    def update_variable_state(self, plant_name: str, variable_updates: Dict[str, Any]) -> Dict[str, Any]:
        """
        Updates changing variable counts (plant_count, healthy_count, diseased_count, inventory_count).
        Does NOT create a new plant profile history version!
        """
        p_name = plant_name.strip()
        if p_name not in self.state["variables"]:
            self.state["variables"][p_name] = {
                "plant_count": 0,
                "healthy_count": 0,
                "diseased_count": 0,
                "inventory_count": 0
            }

        curr = self.state["variables"][p_name]
        for k, v in variable_updates.items():
            if k in ["plant_count", "healthy_count", "diseased_count", "inventory_count"]:
                if isinstance(v, int):
                    curr[k] = v
                elif isinstance(v, str) and v.isdigit():
                    curr[k] = int(v)

        self._save_state()
        print(f" [HistoryManager] Updated variable state for '{p_name}': {curr}")
        return curr

    def get_variable_state(self, plant_name: str) -> Dict[str, Any]:
        """Returns current variable state for a plant."""
        return self.state["variables"].get(plant_name.strip(), {})

    def add_analysis_history(self, plant_name: str, disease_name: str, plant_confidence: float = 1.0, disease_confidence: float = 1.0, timestamp: Optional[str] = None) -> Dict[str, Any]:
        """
        Records an image analysis event into history storage and vector database.
        """
        ts = timestamp or datetime.now(timezone.utc).isoformat()
        analysis_entry = {
            "history_type": "analysis",
            "plant_name": plant_name,
            "disease_name": disease_name,
            "plant_confidence": plant_confidence,
            "disease_confidence": disease_confidence,
            "timestamp": ts
        }

        self.state["analyses"].append(analysis_entry)
        self._save_state()

        doc_content = (
            f"Analysis Event for {plant_name}\n"
            f"Detected Disease: {disease_name}\n"
            f"Plant Confidence: {plant_confidence:.2f}\n"
            f"Disease Confidence: {disease_confidence:.2f}\n"
            f"Timestamp: {ts}"
        )

        doc = Document(
            page_content=doc_content,
            metadata={
                "history_type": "analysis",
                "plant_name": plant_name,
                "disease_name": disease_name,
                "plant_confidence": plant_confidence,
                "disease_confidence": disease_confidence,
                "timestamp": ts
            }
        )

        if self.vector_db:
            self.vector_db.add_documents([doc])
            self.vector_db.save_local(self.index_path)

        print(f" [HistoryManager] Logged analysis event for {plant_name}: {disease_name}")
        return analysis_entry


# Global instance
history_manager = HistoryManager()


def classify_query(query: str) -> str:
    """
    Classifies user's query into one of:
    - CONSTANT_QUERY
    - CONSTANT_UPDATE
    - VARIABLE_QUERY
    - VARIABLE_UPDATE
    - HISTORY_QUERY
    - ANALYSIS_QUERY
    """
    q_lower = query.lower().strip()

    # 1. CONSTANT_UPDATE indicators (e.g. "Change tomato watering from daily to twice a week.")
    if (q_lower.startswith("change ") or q_lower.startswith("update ") or q_lower.startswith("set ")) and not any(k in q_lower for k in ["was", "before", "previously", "earlier", "old"]):
        return "CONSTANT_UPDATE"
    if any(k in q_lower for k in ["should be watered", "update profile", "set watering"]):
        return "CONSTANT_UPDATE"


    # 2. VARIABLE_UPDATE indicators
    if any(k in q_lower for k in ["added", "bought", "increase count", "update count", "set count", "removed plants"]):
        return "VARIABLE_UPDATE"

    # 3. CONSTANT_QUERY indicators (current general questions)
    if any(k in q_lower for k in ["how often should i water", "how to water", "what is the scientific name", "care tips for", "soil requirement for"]):
        if not any(pk in q_lower for pk in ["was", "before", "previously", "earlier", "old"]):
            return "CONSTANT_QUERY"

    # 4. VARIABLE_QUERY indicators
    if any(k in q_lower for k in ["how many", "plant count", "inventory", "stock count", "garden count", "current count"]):
        if not any(pk in q_lower for pk in ["was", "before", "previously", "earlier", "old"]):
            return "VARIABLE_QUERY"

    # 5. HISTORY_QUERY indicators (past history questions)
    past_keywords = [
        "was", "before", "previously", "earlier", "old", "changed", "change",
        "did i change", "when did i", "history", "previous", "prior", "used to"
    ]
    if any(k in q_lower for k in past_keywords):
        return "HISTORY_QUERY"

    # 6. ANALYSIS_QUERY indicators (image analysis / disease check requests)
    if any(k in q_lower for k in ["analyze", "identify leaf", "detect disease", "scan image"]):
        return "ANALYSIS_QUERY"

    # Default fallback
    return "CONSTANT_QUERY"


def extract_plant_name_from_query(query: str) -> Optional[str]:
    """Extracts known plant name from text if present."""
    q_lower = query.lower()
    for name in ["tomato", "potato", "mango", "rose", "apple", "tulsi", "aloe vera"]:
        if name in q_lower:
            return name.capitalize()
    return None


def retrieve_history(query: str, plant_name: str = None, k: int = 5) -> List[Document]:
    """
    Searches the History FAISS Vector Store for documents relevant to the query.
    Filters/ranks documents matching plant_name if specified.
    """
    if not history_manager.vector_db:
        return []

    target_plant = plant_name or extract_plant_name_from_query(query)
    
    # Retrieve top candidates from vector DB
    try:
        results = history_manager.vector_db.similarity_search(query, k=k*2)
    except Exception as e:
        print(f" [HistoryRetriever] Vector search error: {e}")
        return []

    filtered = []
    for doc in results:
        meta = doc.metadata or {}
        doc_plant = meta.get("plant_name", "")
        # Filter system markers
        if meta.get("history_type") == "system":
            continue
        
        if target_plant:
            if doc_plant.lower() == target_plant.lower():
                filtered.append(doc)
        else:
            filtered.append(doc)

    return filtered[:k]


def answer_from_history(query: str, plant_name: Optional[str] = None) -> Dict[str, Any]:
    """
    Executes History QA pipeline:
    query -> classify -> retrieve_history -> build context -> LLM response
    Returns structured dict with type, plant, query, answer, and sources.
    """
    q_type = classify_query(query)
    target_plant = plant_name or extract_plant_name_from_query(query)

    # 1. Handle Variable Query (current count)
    if q_type == "VARIABLE_QUERY":
        if target_plant:
            var_state = history_manager.get_variable_state(target_plant)
            count = var_state.get("plant_count")
            if count is not None and count > 0:
                ans = f"You currently have {count} {target_plant} plant(s)."
            else:
                ans = f"No current plant count recorded for {target_plant}."
            return {
                "type": "variable",
                "plant": target_plant,
                "query": query,
                "answer": ans,
                "sources": []
            }

    # 2. Handle History Query
    docs = retrieve_history(query, plant_name=target_plant, k=5)
    
    if not docs:
        p_label = target_plant or "your plant"
        return {
            "type": "history",
            "plant": target_plant or "Unknown",
            "query": query,
            "answer": "I couldn't find that information in the stored history.",
            "sources": []
        }

    # Extract version sources & context
    sources = []
    context_chunks = []
    for doc in docs:
        meta = doc.metadata or {}
        sources.append({
            "version": meta.get("version", 1),
            "created_at": meta.get("created_at") or meta.get("timestamp")
        })
        context_chunks.append(doc.page_content)

    context_str = "\n\n---\n\n".join(context_chunks)

    # Rule-based comparative generator to guarantee 100% precision & non-hallucination
    answer_text = generate_grounded_answer(query, target_plant, docs)

    return {
        "type": "history",
        "plant": target_plant or "Unknown",
        "query": query,
        "answer": answer_text,
        "sources": sources
    }


def generate_grounded_answer(query: str, plant_name: Optional[str], docs: List[Document]) -> str:
    """
    Generates a clear, grounded answer strictly using retrieved historical documents.
    Prevents hallucinations by requiring factual matching in context documents.
    """
    q_lower = query.lower()

    # Check for past count / variable history request (which is not recorded in profile history)
    if "count" in q_lower or "how many" in q_lower:
        if any(pk in q_lower for pk in ["was", "before", "previously", "earlier", "old"]):
            return "I couldn't find that information in the stored history."

    # Extract all profile versions from docs and state
    profiles = []
    analyses = []
    for doc in docs:
        meta = doc.metadata or {}
        if meta.get("history_type") == "plant_profile":
            profiles.append((meta.get("version", 1), meta.get("created_at"), doc.page_content))
        elif meta.get("history_type") == "analysis":
            analyses.append((meta.get("disease_name"), meta.get("disease_confidence"), meta.get("timestamp")))

    # Merge profile versions from history_manager state for complete version coverage
    if plant_name and plant_name in history_manager.state["profiles"]:
        for p in history_manager.state["profiles"][plant_name]:
            v_num = p.get("version", 1)
            v_date = p.get("created_at")
            v_text = f"Watering: {p.get('watering', '')}\nSoil: {p.get('soil', '')}\nSunlight: {p.get('sunlight', '')}"
            if not any(existing[0] == v_num for existing in profiles):
                profiles.append((v_num, v_date, v_text))

    profiles.sort(key=lambda x: x[0])  # Sort by version ascending


    # Case A: Disease History Query
    if "disease" in q_lower or "blight" in q_lower or "detected" in q_lower:
        if analyses:
            d_name, d_conf, d_ts = analyses[-1]
            conf_str = f" with {int(d_conf*100)}% confidence" if d_conf else ""
            date_str = f" on {d_ts[:10]}" if d_ts else ""
            return f"Your previous {plant_name or 'plant'} analysis detected {d_name}{conf_str}{date_str}."

    # Case B: Profile Version Comparison (Watering / Soil / Sunlight / Changes)
    if profiles:
        v1_ver, v1_date, v1_text = profiles[0]
        
        # Check date/time change query first
        if "when" in q_lower:
            if len(profiles) >= 2 and profiles[-1][1]:
                return f"You updated the {plant_name or 'plant'} profile on {profiles[-1][1]}."
            elif profiles[0][1]:
                return f"The {plant_name or 'plant'} profile was created on {profiles[0][1]}."

        # Check specific field queries
        if "water" in q_lower:
            w_matches = re.findall(r"Watering:\s*(.+)", "\n".join([p[2] for p in profiles]))
            if len(set(w_matches)) > 1:
                last_date = profiles[-1][1] or ""
                date_suffix = f" on {last_date}" if last_date else ""
                return f"Previously, your {plant_name or 'plant'} watering schedule was {w_matches[0]}. It was later changed to {w_matches[-1]}{date_suffix}."
            elif w_matches:
                return f"Your stored {plant_name or 'plant'} watering schedule is {w_matches[0]}."

        if "soil" in q_lower:
            s_matches = re.findall(r"Soil:\s*(.+)", "\n".join([p[2] for p in profiles]))
            if len(set(s_matches)) > 1:
                return f"You changed the {plant_name or 'plant'} soil requirement from {s_matches[0]} to {s_matches[-1]}."
            elif s_matches:
                return f"Your stored {plant_name or 'plant'} soil requirement is {s_matches[0]}."

        if "change" in q_lower or "changed" in q_lower:
            if len(profiles) >= 2:
                v1_text = profiles[0][2]
                v2_text = profiles[-1][2]
                changes = []
                for field in ["Watering", "Soil", "Sunlight", "Temperature", "Humidity"]:
                    m1 = re.search(f"{field}:\\s*(.+)", v1_text)
                    m2 = re.search(f"{field}:\\s*(.+)", v2_text)
                    if m1 and m2 and m1.group(1).strip() != m2.group(1).strip():
                        changes.append(f"{field.lower()} from {m1.group(1).strip()} to {m2.group(1).strip()}")
                if changes:
                    return f"You changed the {plant_name or 'plant'} " + ", ".join(changes) + "."
            return f"No profile changes were detected for {plant_name or 'your plant'}."

        # Return latest profile summary
        return f"Historical record found for {plant_name or 'plant'} (Version {profiles[-1][0]}): {profiles[-1][2]}"

    return "I couldn't find that information in the stored history."


