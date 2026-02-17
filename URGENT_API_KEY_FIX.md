# 🚨 URGENT: API KEY ISSUE

## Problem
Your current API key does NOT work with ANY Gemini model:
- ❌ gemini-pro → 404 NOT_FOUND
- ❌ gemini-1.5-flash → 404 NOT_FOUND  
- ❌ gemini-2.0-flash-exp → 404 NOT_FOUND
- ❌ gemini-2.5-flash → 404 NOT_FOUND

Current key: `AIzaSyCuOyj8GgrEhSpJNFgxOCL8ppyytDvf3g4`

## IMMEDIATE FIX (2 minutes)

### Step 1: Get NEW API Key
1. Go to: https://aistudio.google.com/app/apikey
2. Click "Create API Key"
3. Copy the new key

### Step 2: Update .env file
```bash
# Open .env file
# Replace line 2 with your NEW key:
GOOGLE_API_KEY="YOUR_NEW_KEY_HERE"
```

### Step 3: Restart Backend
```bash
# Stop current server (Ctrl+C)
python -m uvicorn src.main:app --reload --port 8000
```

## For Your Manager Demo RIGHT NOW

**Good news:** All prediction features work WITHOUT chat:
1. ✅ Select engine from dropdown
2. ✅ Click "PREDICT RUL" → Shows predictions
3. ✅ View maintenance recommendations
4. ✅ Analyze engine health

**Chat will work** once you get the new API key (takes 2 minutes).

## Why This Happened
The API key you're using either:
- Was revoked/expired
- Doesn't have Gemini API access enabled
- Is from a different Google Cloud project

Get a fresh key from AI Studio and it will work immediately.
