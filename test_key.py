"""Check that the Gemini API key loads and Gemini can answer a test prompt."""

import os

from dotenv import load_dotenv
from google import genai

load_dotenv()
key = os.getenv("GEMINI_API_KEY")
print("Key मिली?", bool(key), "| लंबाई:", len(key) if key else 0)

if not key:
	raise SystemExit("GEMINI_API_KEY .env file mein set nahi hai.")

client = genai.Client(api_key=key)
response = client.models.generate_content(
	model="gemini-3.8-flash",
	contents="Say hi",
)
print(response.text)
