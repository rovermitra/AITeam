# test_matches.py
from utils import llm_find_matches
import openai
from dotenv import load_dotenv
#from utils import parse_llm_json, safe_write_json, prefilter_candidates
import os
# Load env
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

new_user = {
    "name": "Test User",
    "age": 30,
    "gender": "female",
    "location": "Paris",
    "country": "France",
    "occupation": "Designer",
    "interests": ["art", "travel", "hiking"],
    "languages": ["English", "French"],
    "bio": "I love exploring new cultures and nature trails."
}

db_users = [
    {
        "name": "Alex",
        "age": 32,
        "gender": "male",
        "location": "Berlin",
        "country": "Germany",
        "occupation": "Engineer",
        "interests": ["travel", "photography"],
        "languages": ["English", "German"],
        "bio": "Adventurous and easygoing."
    },
    {
        "name": "Sophie",
        "age": 29,
        "gender": "female",
        "location": "Rome",
        "country": "Italy",
        "occupation": "Chef",
        "interests": ["cooking", "hiking"],
        "languages": ["Italian", "English"],
        "bio": "Food lover and nature enthusiast."
    }
]

matches = llm_find_matches(new_user, db_users, top_k=5)
print(matches)
