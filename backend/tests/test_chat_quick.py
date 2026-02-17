"""Quick test to see what error the chat is getting"""
import os
import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent))

from dotenv import load_dotenv
load_dotenv()

from src.agents.graph import create_assistant

print("Creating assistant...")
assistant = create_assistant()
print("Assistant created successfully!")

print("\nTesting chat...")
try:
    response = assistant.chat("hello")
    print(f"SUCCESS: {response[:100]}")
except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()
