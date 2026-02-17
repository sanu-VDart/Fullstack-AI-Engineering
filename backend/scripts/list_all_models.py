import os
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

if not api_key:
    print("Error: No API key found in .env")
    exit(1)

print(f"Using API key: {api_key[:10]}...")

genai.configure(api_key=api_key)

print("\nAvailable models and their capabilities:")
try:
    for m in genai.list_models():
        methods = ", ".join(m.supported_generation_methods)
        print(f"Model: {m.name}")
        print(f"  Methods: {methods}")
        print(f"  Description: {m.description}")
        print("-" * 20)
except Exception as e:
    print(f"Error listing models: {e}")
