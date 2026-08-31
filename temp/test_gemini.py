import os
import sys
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
model = genai.GenerativeModel("gemini-1.5-flash")

prompt = """
Analyze this plant profile request: "unsupported plant xyz".
Is "unsupported plant xyz" a real plant?
- If it is not a real plant, or if it is made up/gibberish, reply ONLY with the word "UNKNOWN".
- If it is a real plant, generate a valid JSON profile with the keys.
Output ONLY the JSON or "UNKNOWN".
"""

res = model.generate_content(prompt).text.strip()
print("Result:", repr(res))
