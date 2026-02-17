"""Direct test of the API key with google-generativeai library"""
import os
from dotenv import load_dotenv
load_dotenv()

import google.generativeai as genai

api_key = os.getenv("GOOGLE_API_KEY")
print(f"Testing API key: {api_key}")
print(f"Key length: {len(api_key)}")

genai.configure(api_key=api_key)

# Try the simplest possible call
print("\nTesting gemini-pro with google-generativeai library...")
try:
    model = genai.GenerativeModel('gemini-pro')
    response = model.generate_content("Say hello")
    print(f"✅ SUCCESS with google-generativeai!")
    print(f"Response: {response.text}")
except Exception as e:
    print(f"❌ FAILED: {e}")

# Now test with LangChain
print("\nTesting gemini-pro with LangChain...")
try:
    from langchain_google_genai import ChatGoogleGenerativeAI
    llm = ChatGoogleGenerativeAI(model="gemini-pro", google_api_key=api_key)
    response = llm.invoke("Say hello")
    print(f"✅ SUCCESS with LangChain!")
    print(f"Response: {response.content}")
except Exception as e:
    print(f"❌ FAILED: {e}")
    import traceback
    traceback.print_exc()
