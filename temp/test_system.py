import os
import sys
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

plants_to_test = ["unsupported plant xyz", "Rosemary", "hello", "Acanthocereus"]

for plant in plants_to_test:
    sys_prompt = "You are a strict database classification system. Reply ONLY in the requested JSON format (or UNKNOWN). Do not use conversational filler or markdown blocks."
    
    prompt = f"""
    Check if '{plant}' is a real, recognized plant name/species.
    If NOT, reply with exactly the word "UNKNOWN".
    If YES, reply with exactly a JSON object having keys: "plant_name", "scientific_name", "moisture", "temperature", "humidity", "sunlight", "soil_type", "nitrogen", "phosphorus", "potassium", "watering_frequency", "growth_duration".
    """
    res = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": prompt}
        ],
        temperature=0.0
    ).choices[0].message.content.strip()
    print(f"Plant: {plant} \nResult:\n{res}\n" + "="*30)
