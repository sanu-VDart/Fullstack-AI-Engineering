import os
from dotenv import load_dotenv
load_dotenv()

from langchain_google_genai import ChatGoogleGenerativeAI

api_key = os.getenv("GOOGLE_API_KEY")
print(f"Testing API key: {api_key[:15]}...")

# Test different models
models_to_test = [
    "gemini-2.0-flash-exp",
    "gemini-1.5-flash", 
    "gemini-1.5-pro",
    "gemini-pro"
]

for model_name in models_to_test:
    print(f"\nTesting {model_name}...")
    try:
        llm = ChatGoogleGenerativeAI(model=model_name, temperature=0.1)
        response = llm.invoke("Say 'OK'")
        print(f"  ✅ SUCCESS: {response.content[:30]}")
        break  # Stop at first working model
    except Exception as e:
        error_msg = str(e)[:100]
        print(f"  ❌ FAILED: {error_msg}")
