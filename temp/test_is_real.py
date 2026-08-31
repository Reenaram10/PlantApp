import os
import sys
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

plants_to_test = ["unsupported plant xyz", "Rosemary", "hello", "Acanthocereus", "xyz plant"]

for plant in plants_to_test:
    prompt = f"""
    You are a botanical database checker.
    Is "{plant}" a recognized plant species (either common name or scientific name)?
    Reply ONLY with 'YES' or 'NO'. No other words, no explanation, no punctuation.
    """
    res = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0
    ).choices[0].message.content.strip().upper()
    print(f"Plant: {plant} -> Real: {res}")
