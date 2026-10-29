import os
from dotenv import load_dotenv
from groq import Groq

load_dotenv()
client = Groq()

models = client.models.list()
for i, m in enumerate(models.data):
    print(f"{i}. {m.id}")
