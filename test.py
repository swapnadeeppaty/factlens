import os
from dotenv import load_dotenv
load_dotenv()
import google.generativeai as genai

key = os.environ.get('GEMINI_API_KEY', '')
genai.configure(api_key=key)

# Test Gemini is working
print("Testing Gemini...")
try:
    model = genai.GenerativeModel('models/gemini-2.0-flash')
    r = model.generate_content('Say hello')
    print(f"Gemini OK: {r.text.strip()}")
except Exception as e:
    print(f"Gemini FAILED: {e}")
    exit()