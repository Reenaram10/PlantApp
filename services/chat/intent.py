"""
Chat Intent & Classification Service
Provides text normalization, greeting detection, quantity parsing, and NLP intent routing.
"""
import re

GREETINGS = {
    "hi", "hello", "hey", "hii", "heyy",
    "thanks", "thank you", "thx",
    "good morning", "good evening", "good night"
}

def clean_text(text):
    """Normalize text: lowercase, remove non-alphanumeric symbols, collapse whitespace."""
    if not text:
        return ""
    text = text.lower()
    text = re.sub(r'[^\w\s]', ' ', text)
    return ' '.join(text.split())

def extract_quantity_nlp(text):
    """
    Extracts numerical quantity from user string using an LCEL AI Chain.
    e.g. 'two', '5 plants', 'add double' -> int
    """
    try:
        from langchain_core.prompts import PromptTemplate
        from langchain_community.llms import Ollama
        from langchain_core.output_parsers import StrOutputParser

        quantity_prompt = PromptTemplate.from_template(
            "The user was asked how many plants they want. User input: '{user_input}'. "
            "Extract the exact numerical integer. If none is given, assume 1. Respond ONLY with the integer digits."
        )
        quantity_chain = quantity_prompt | Ollama(model="phi3", temperature=0) | StrOutputParser()
        
        qty_str = quantity_chain.invoke({"user_input": text}).strip()
        match = re.search(r'\d+', qty_str)
        if match:
            return max(1, int(match.group()))
        return 1
    except Exception as e:
        print(f"Error in LangChain quantity extraction: {e}")
        return 1

def classify_affirmation(user_input):
    """
    Categorize confirmation reply into POSITIVE, NEGATIVE, or OTHER.
    """
    clean_input = clean_text(user_input).strip()
    try:
        from langchain_core.prompts import PromptTemplate
        from langchain_community.llms import Ollama
        from langchain_core.output_parsers import StrOutputParser
        
        affirmation_prompt = PromptTemplate.from_template(
            "The user was asked if they want to add a plant to their cart.\n"
            "Their response was: '{user_input}'\n\n"
            "Task: Categorize this response into EXACTLY one of three words: POSITIVE, NEGATIVE, or OTHER.\n"
            "- Output POSITIVE ONLY if they are explicitly agreeing to add the plant (e.g. 'yes', 'sure', 'add it').\n"
            "- Output NEGATIVE ONLY if they are declining (e.g. 'no', 'nah').\n"
            "- Output OTHER if they are asking a new question, changing the topic, or making a statement unrelated to answering 'yes/no' (e.g. 'do you sell X', 'what is Y').\n\n"
            "Classification:"
        )
        affirmation_chain = affirmation_prompt | Ollama(model="phi3", temperature=0) | StrOutputParser()
        aff_result = affirmation_chain.invoke({"user_input": clean_input}).strip().upper()
        return aff_result
    except Exception as e:
        print(f"Affirmation logic failed: {e}")
        return "OTHER"

def classify_query_intent(user_message, history_context=""):
    """
    Classify query intent into PROFILE, GENERAL, or OFF_TOPIC.
    """
    class_prompt = (
        f"{history_context}"
        f"Analyze the user's message: '{user_message}'.\n"
        "Classify the intent into exactly one of three categories:\n"
        "1. PROFILE: If the user is requesting a profile, care-sheet, or specific growing/care parameters of a single specific plant name (e.g., 'tell me about rose', 'Mango', 'Acanthocereus', 'unsupported plant xyz', 'Tomato').\n"
        "2. GENERAL: If the user is asking a general botanical question, list of plants in our nursery, providing their geographic location/climate for recommendations, care tips, growing guides, etc.\n"
        "3. OFF_TOPIC: If the user is asking about topics completely unrelated to plants, gardening, farming, or our catalog. Note: Geographic locations, cities, or climates (e.g., 'tamilnadu', 'florida', 'hot climate') are NOT OFF_TOPIC; they are GENERAL intent used for plant matching.\n\n"
        "Respond with exactly one word: PROFILE, GENERAL, or OFF_TOPIC. Do not write any other text or markdown code wraps."
    )
    from langchain_community.llms import Ollama
    llm = Ollama(model="phi3", temperature=0.0)
    sys_msg = "You are a routing classification system for a Plant Care Assistant. You must classify user inputs into PROFILE, GENERAL, or OFF_TOPIC. Every question that is not about plants, botany, diseases, gardening, or agriculture is OFF_TOPIC.\n\n"
    return llm.invoke(sys_msg + class_prompt).strip().upper()

def classify_category(user_message, history_context=""):
    """
    Determine knowledge category for GENERAL botanical queries: GEOGRAPHY, SEASON, DISEASE, or PLANT.
    Returns lowercase category name or None.
    """
    cat_prompt = (
        f"{history_context}"
        f"Determine the knowledge category for the user query: '{user_message}'.\n"
        "Pick exactly one of these categories:\n"
        "- GEOGRAPHY: Questions about native regions, countries, climate zones, altitude, frost/drought tolerance, where a plant grows.\n"
        "- SEASON: Questions about seasonal care, winter/summer watering, seasonal fertilization, seasonal temperature or sunlight requirements.\n"
        "- DISEASE: Questions about plant diseases, symptoms, pests, yellowing leaves, fungal/bacterial infections, treatment, prevention.\n"
        "- PLANT: Any other plant care, profile, soil, nutrients, watering, growth duration, medicinal uses.\n"
        "Respond with exactly one word: GEOGRAPHY, SEASON, DISEASE, or PLANT."
    )
    from langchain_community.llms import Ollama
    llm = Ollama(model="phi3", temperature=0.0)
    sys_msg = "You are a data source router. Respond with GEOGRAPHY, SEASON, DISEASE, or PLANT only.\n\n"
    cat_label = llm.invoke(sys_msg + cat_prompt).strip().upper()
    
    category_map = {"GEOGRAPHY": "geography", "SEASON": "season", "DISEASE": "disease", "PLANT": "plant"}
    return category_map.get(cat_label, None)
