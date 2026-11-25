"""
Chat Services Package
Includes dialog state singleton, intent and affirmation classifiers, plant lookups,
and the modular process_message engine.
"""
from services.chat.dialog_state import USER_DIALOG_STATE
from services.chat.intent import (
    clean_text, GREETINGS, extract_quantity_nlp,
    classify_affirmation, classify_query_intent, classify_category
)
from services.chat.plant_lookup import (
    resolve_plant_synonym, fuzzy_resolve_plant, fuzzy_find_plant,
    get_category_from_synonym, add_category_synonym, handle_count_plants,
    log_category_interest, log_user_interest, store_query,
    fetch_db_plant_details, check_db_plant_availability,
    format_plant_profile, make_plant_unavailable_guardrail
)
from services.chat.chat_service import process_message

__all__ = [
    "USER_DIALOG_STATE",
    "clean_text", "GREETINGS", "extract_quantity_nlp",
    "classify_affirmation", "classify_query_intent", "classify_category",
    "resolve_plant_synonym", "fuzzy_resolve_plant", "fuzzy_find_plant",
    "get_category_from_synonym", "add_category_synonym", "handle_count_plants",
    "log_category_interest", "log_user_interest", "store_query",
    "fetch_db_plant_details", "check_db_plant_availability",
    "format_plant_profile", "make_plant_unavailable_guardrail",
    "process_message"
]
