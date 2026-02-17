import google.generativeai as genai
import os
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

genai.configure(api_key=api_key)

print("Listing ALL models with generateContent support:\n")
models = genai.list_models()

working_model = None
for m in models:
    if 'generateContent' in m.supported_generation_methods:
        print(f"✓ {m.name}")
        
        # Try to use the first one we find
        if working_model is None:
            try:
                print(f"  Testing {m.name}...")
                test_model = genai.GenerativeModel(m.name)
                response = test_model.generate_content("Say OK")
                print(f"  ✅ WORKS! Response: {response.text[:50]}")
                working_model = m.name
                break
            except Exception as e:
                print(f"  ❌ Failed: {str(e)[:80]}")

if working_model:
    print(f"\n🎉 WORKING MODEL FOUND: {working_model}")
else:
    print("\n❌ No working models found!")
