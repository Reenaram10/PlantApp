import os
import sys
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

plants_to_test = ["unsupported plant xyz", "Rosemary", "hello", "Acanthocereus"]

for plant in plants_to_test:
    prompt = f"""
    State whether the plant '{plant}' is a real, recognized plant.
    If '{plant}' is a real plant, output a JSON profile containing keys: "plant_name", "scientific_name", "moisture", "temperature", "humidity", "sunlight", "soil_type", "nitrogen", "phosphorus", "potassium", "watering_frequency", "growth_duration".
    If it is not a recognized plant (for example, greetings like 'hello', or fake/invalid/unsupported plants like 'unsupported plant xyz', etc.), output ONLY "UNKNOWN".
    Do not output any markdown formatting, backticks, or other text.
    """
    res = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0
    ).choices[0].message.content.strip()
    print(f"Plant: {plant} \nResult:\n{res}\n" + "="*30)
