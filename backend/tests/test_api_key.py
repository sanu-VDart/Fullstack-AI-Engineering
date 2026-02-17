"""Test Gemini API key and model access"""
import os
from dotenv import load_dotenv
load_dotenv()

import google.generativeai as genai

api_key = os.getenv("GOOGLE_API_KEY")
print(f"API Key: {api_key[:20]}...")

genai.configure(api_key=api_key)

# List available models
print("\nAvailable models:")
for m in genai.list_models():
    if 'generateContent' in m.supported_generation_methods:
        print(f"  - {m.name}")

# Test gemini-1.5-flash
print("\nTesting gemini-1.5-flash...")
try:
    model = genai.GenerativeModel('gemini-1.5-flash')
    response = model.generate_content("Say hello")
    print(f"SUCCESS: {response.text[:50]}")
except Exception as e:
    print(f"ERROR: {e}")
