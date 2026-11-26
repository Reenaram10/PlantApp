"""
Services Package
Aggregates application services for telemetry, payments, image utilities, speech,
disease detection models, RAG search engines, and LLM inference.
"""
from services.observability import log_langfuse_trace
from services.payment import encrypt, decrypt
from services.image_utils import allowed_file, get_plant_image_url, search_plant_image
from services.speech_service import tts_failsafe, generate_speech
from services.disease_models import (
    get_crop_type_model, get_potato_disease_model, get_tomato_disease_model, get_flower_crop_model,
    get_crop_indices, get_potato_indices, get_tomato_indices, get_flower_indices
)
from services.rag_service import faiss_rag, history_faiss, ask_rebuild_index, tomato_rag, plant_rag
from services.llm_service import (
    normalize_query, groq_call_failsafe, llm_call, slm_call,
    get_cached_llm_result, store_canonical_answer
)
from services.chat import (
    USER_DIALOG_STATE, process_message, clean_text, GREETINGS,
    extract_quantity_nlp, fuzzy_resolve_plant, fuzzy_find_plant,
    get_category_from_synonym, handle_count_plants, log_category_interest,
    log_user_interest, store_query, fetch_db_plant_details
)

__all__ = [
    "log_langfuse_trace",
    "encrypt", "decrypt",
    "allowed_file", "get_plant_image_url", "search_plant_image",
    "tts_failsafe", "generate_speech",
    "get_crop_type_model", "get_potato_disease_model", "get_tomato_disease_model", "get_flower_crop_model",
    "get_crop_indices", "get_potato_indices", "get_tomato_indices", "get_flower_indices",
    "faiss_rag", "history_faiss", "ask_rebuild_index", "tomato_rag", "plant_rag",
    "normalize_query", "groq_call_failsafe", "llm_call", "slm_call",
    "get_cached_llm_result", "store_canonical_answer",
    "USER_DIALOG_STATE", "process_message", "clean_text", "GREETINGS",
    "extract_quantity_nlp", "fuzzy_resolve_plant", "fuzzy_find_plant",
    "get_category_from_synonym", "handle_count_plants", "log_category_interest",
    "log_user_interest", "store_query", "fetch_db_plant_details"
]
