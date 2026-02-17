"""Test API key with direct HTTP request to see exact error"""
import os
import requests
from dotenv import load_dotenv
load_dotenv()

api_key = os.getenv("GOOGLE_API_KEY")
print(f"Testing API key: {api_key}")

# Direct API call to Gemini
url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent?key={api_key}"

payload = {
    "contents": [{
        "parts": [{"text": "Say hello"}]
    }]
}

print("\nMaking direct API call to Google...")
response = requests.post(url, json=payload)

print(f"Status Code: {response.status_code}")
print(f"Response: {response.text[:500]}")

if response.status_code == 200:
    print("\n✅ API KEY WORKS!")
    data = response.json()
    print(f"Response text: {data['candidates'][0]['content']['parts'][0]['text']}")
else:
    print("\n❌ API KEY FAILED")
    print("Full error:", response.json())
