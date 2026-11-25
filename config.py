import os
import sys
import warnings
from dotenv import load_dotenv
from sqlalchemy.exc import SAWarning

# Define APP_ROOT explicitly from this config file at the root of the project
APP_ROOT = os.path.dirname(os.path.abspath(__file__))
if APP_ROOT not in sys.path:
    sys.path.insert(0, APP_ROOT)

load_dotenv()

# Suppress unrecognized SQL Server version warning (SQL Server 2025 compatibility)
warnings.filterwarnings("ignore", category=SAWarning, message=r".*Unrecognized server version info.*")

# Core Directory & File Paths (All anchored to APP_ROOT)
STATIC_DIR = os.path.join(APP_ROOT, "static")
UPLOADS_DIR = os.path.join(APP_ROOT, "uploads")
TEMP_DIR = os.path.join(APP_ROOT, "temp")
LOCAL_FAISS_INDEX = os.path.join(APP_ROOT, "local_faiss_index")
LOCAL_HISTORY_FAISS_INDEX = os.path.join(APP_ROOT, "local_history_faiss_index")
DEFAULT_KB_PATH = os.path.join(APP_ROOT, "unified_plants_knowledge.json")

# Dataset CSV Paths (Anchored to APP_ROOT)
GEOGRAPHY_CSV_PATH = os.path.join(APP_ROOT, "GEOGRAPHY_PROFILEE.csv")
SEASONAL_CSV_PATH = os.path.join(APP_ROOT, "seasonal_profile.csv")
DISEASE_CSV_PATH = os.path.join(APP_ROOT, "plant_disease_profiles (1).csv")
PLANT_PROFILES_CSV_PATH = os.path.join(APP_ROOT, "plant_profiles.csv")

# Disease Detection Model Paths (Anchored to APP_ROOT)
CROP_MODEL_PATH = os.path.join(APP_ROOT, "plant_disease_detection", "model", "crop_type_model.h5")
CROP_CLASSES_PATH = os.path.join(APP_ROOT, "plant_disease_detection", "model", "crop_indices.json")

POTATO_MODEL_PATH = os.path.join(APP_ROOT, "plant_disease_detection", "model", "potato_disease_model.h5")
POTATO_CLASSES_PATH = os.path.join(APP_ROOT, "plant_disease_detection", "model", "potato_indices.json")

TOMATO_MODEL_PATH = os.path.join(APP_ROOT, "plant_disease_detection", "model", "tomato_disease_model.h5")
TOMATO_CLASSES_PATH = os.path.join(APP_ROOT, "plant_disease_detection", "model", "tomato_indices.json")

FLOWER_MODEL_PATH = os.path.join(APP_ROOT, "plant_disease_detection", "model", "flower_crop_model.h5")
FLOWER_CLASSES_PATH = os.path.join(APP_ROOT, "plant_disease_detection", "model", "flower_indices.json")

# ---- SQL Server Database Configuration ----
_db_server   = os.getenv("DB_SERVER", "localhost")
_db_name     = os.getenv("DB_NAME", "PlantAppDB")
_db_user     = os.getenv("DB_USER", "")
_db_password = os.getenv("DB_PASSWORD", "")
_db_trusted  = os.getenv("DB_TRUSTED_CONNECTION", "yes")

# Auto-detect newest installed Microsoft ODBC Driver
_odbc_driver = "ODBC+Driver+18+for+SQL+Server"
try:
    import pyodbc
    _available = pyodbc.drivers()
    if "ODBC Driver 18 for SQL Server" in _available:
        _odbc_driver = "ODBC+Driver+18+for+SQL+Server"
    elif "ODBC Driver 17 for SQL Server" in _available:
        _odbc_driver = "ODBC+Driver+17+for+SQL+Server"
    elif "SQL Server" in _available:
        _odbc_driver = "SQL+Server"
except Exception:
    pass

if _db_trusted.lower() == "yes":
    SQLALCHEMY_DATABASE_URI = (
        f"mssql+pyodbc://@{_db_server}/{_db_name}"
        f"?driver={_odbc_driver}"
        f"&TrustServerCertificate=yes"
    )
else:
    from urllib.parse import quote_plus
    SQLALCHEMY_DATABASE_URI = (
        f"mssql+pyodbc://{quote_plus(_db_user)}:{quote_plus(_db_password)}"
        f"@{_db_server}/{_db_name}"
        f"?driver={_odbc_driver}"
        f"&TrustServerCertificate=yes"
    )

SQLALCHEMY_TRACK_MODIFICATIONS = False
SQLALCHEMY_ENGINE_OPTIONS = {
    "pool_pre_ping": True,
    "pool_recycle": 300,
    "pool_timeout": 30,
}

# JWT Secret
JWT_SECRET = os.getenv("JWT_SECRET", "super-secret-saas-key-2026")

# Runtime Mutable Settings
class RuntimeSettings:
    """
    Mutable configuration object allowing runtime modification of options via /api/config.
    Eliminates globals() mutation across modules while ensuring state consistency.
    """
    def __init__(self):
        self.llm_option = "slm"         # Options: "groq", "deepseek", "gemini", "slm", "local"
        self.stt_option = "vosk"        # Options: "vosk", "whisper", "google", "azure"
        self.tts_option = "gtts"        # Options: "gtts", "azure", "elevenlabs", "none"
        self.llm_failsafe = "local"     # Falls back to keyword matching
        self.stt_failsafe = "vosk"      # Falls back to VOSK
        self.tts_failsafe = "gtts"      # Falls back to gTTS

runtime_settings = RuntimeSettings()

# File Upload Settings
ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png', 'gif', 'webp'}

# Scientific Names Mapping
SCIENTIFIC_NAMES_MAP = {
    "Tomato": "Solanum lycopersicum",
    "Potato": "Solanum tuberosum"
}

# System Prompt for Chat Assistant
SYSTEM_PROMPT = """
You are Green-Buy's helpful plant assistant. When helping customers:

1. SPELLING CHECK RULES:
   - For any misspelled plant name, ONLY respond with: "Did you mean <correct_name> (ID:<id>)?"
   - Do not proceed until user confirms

2. AFTER SPELLING CONFIRMATION:
   - Only proceed if user confirms with "yes" or similar affirmative
   - Then ask about living space, light conditions, experience level, maintenance preferences, location/climate

3. FOR PURCHASE REQUESTS:
   - Only process after spelling is confirmed
   - Include care instructions after order confirmation

4. FOR DELETION REQUESTS:
   - When user asks to remove/delete/cancel an order
   - Respond with: "DELETE_ORDER:<plant_id>"

5. FOR IMAGE REQUESTS:
   - When user asks "show me/show picture of/show image of <plant>", display the image
   - Include basic plant information with the image

Always use plant IDs from the database in your responses.
"""

# Profile Datasets Metadata & Template Configurations (All paths anchored to APP_ROOT)
PROFILE_DATASETS = {
    "geography": {
        "file": "GEOGRAPHY_PROFILEE.csv",
        "path": GEOGRAPHY_CSV_PATH,
        "title": "Geography & Climate Profiles",
        "key_col": "plant_name",
        "required_cols": ["plant_name"],
        "columns": [
            "plant_name", "country", "state_or_region", "continent", "native_status",
            "cultivation_status", "climate_zone", "best_growing_season", "best_growing_months",
            "temperature_min", "temperature_max", "annual_rainfall_min", "annual_rainfall_max",
            "humidity_min", "humidity_max", "altitude_min", "altitude_max", "soil_suitability",
            "water_availability", "sunlight_condition", "frost_tolerance", "heat_tolerance",
            "drought_tolerance", "salinity_tolerance", "wind_tolerance", "common_growing_method",
            "commercial_production", "remarks"
        ],
        "example": {
            "plant_name": "Mango", "country": "India", "state_or_region": "Maharashtra",
            "continent": "Asia", "native_status": "Native", "cultivation_status": "Widely Cultivated",
            "climate_zone": "Tropical", "best_growing_season": "Summer", "best_growing_months": "March-June",
            "temperature_min": "24", "temperature_max": "35", "annual_rainfall_min": "750",
            "annual_rainfall_max": "2500", "humidity_min": "50", "humidity_max": "70",
            "altitude_min": "0", "altitude_max": "1200", "soil_suitability": "Well-drained alluvial loam",
            "water_availability": "Moderate", "sunlight_condition": "Full Sun", "frost_tolerance": "Low",
            "heat_tolerance": "High", "drought_tolerance": "Moderate", "salinity_tolerance": "Low",
            "wind_tolerance": "Moderate", "common_growing_method": "Grafting", "commercial_production": "High",
            "remarks": "Requires dry weather during flowering."
        }
    },
    "plant_profile": {
        "file": "plant_profiles.csv",
        "path": PLANT_PROFILES_CSV_PATH,
        "title": "Plant Profiles Catalog",
        "key_col": "plant_name",
        "required_cols": ["plant_name"],
        "columns": [
            "plant_name", "scientific_name", "moisture", "temperature", "humidity",
            "sunlight", "soil_type", "nitrogen", "phosphorus", "potassium",
            "watering_frequency", "growth_duration"
        ],
        "example": {
            "plant_name": "Mango", "scientific_name": "Mangifera indica", "moisture": "50-70%",
            "temperature": "24-35°C", "humidity": "50-60%", "sunlight": "Full Sun",
            "soil_type": "Well-drained loamy soil", "nitrogen": "High", "phosphorus": "Medium",
            "potassium": "High", "watering_frequency": "2-3 times per week", "growth_duration": "5-8 years"
        }
    },
    "disease_profile": {
        "file": "plant_disease_profiles (1).csv",
        "path": DISEASE_CSV_PATH,
        "title": "Disease Profiles & Identification",
        "key_col": "disease_name",
        "required_cols": ["plant_name", "disease_name"],
        "columns": [
            "plant_name", "disease_name", "disease_type", "symptoms", "causes",
            "prevention", "treatment", "affected_parts", "severity"
        ],
        "example": {
            "plant_name": "Tomato", "disease_name": "Early Blight", "disease_type": "Fungal",
            "symptoms": "Concentric dark spots on leaves", "causes": "Alternaria solani fungus",
            "prevention": "Crop rotation, clean seeds", "treatment": "Copper-based fungicide spray",
            "affected_parts": "Leaves, stems", "severity": "Moderate to High"
        }
    },
    "seasonal_profile": {
        "file": "seasonal_profile.csv",
        "path": SEASONAL_CSV_PATH,
        "title": "Seasonal Care & Nutrient Profiles",
        "key_col": "plant_name",
        "required_cols": ["plant_name", "season"],
        "columns": [
            "plant_name", "season", "watering_frequency", "water_amount_liters",
            "fertilizer_type", "fertilizer_amount", "nitrogen_requirement",
            "phosphorus_requirement", "potassium_requirement", "sunlight_requirement",
            "temperature_range_celsius", "wind_tolerance", "care_instructions"
        ],
        "example": {
            "plant_name": "Tomato", "season": "Summer", "watering_frequency": "Daily",
            "water_amount_liters": "1.5", "fertilizer_type": "NPK 10-10-10", "fertilizer_amount": "50g",
            "nitrogen_requirement": "Medium", "phosphorus_requirement": "High",
            "potassium_requirement": "High", "sunlight_requirement": "6-8 hours full sun",
            "temperature_range_celsius": "20-30", "wind_tolerance": "Moderate",
            "care_instructions": "Mulch soil surface to preserve moisture."
        }
    }
}
