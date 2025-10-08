# main.py
import sys
import os
# Ensure project root is importable when running from anywhere
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

import json
import uuid
import openai
from dotenv import load_dotenv
from utils import parse_llm_json, safe_write_json, prefilter_candidates

# Load env
load_dotenv()
openai.api_key = os.getenv("API")

DB_FILE_PATH = "data/travel_ready_user_profiles.json"

def profile_to_natural_text(user):
    lines = []
    lines.append(f"Name: {user.get('name', 'N/A')}")
    lines.append(f"Age: {user.get('age', 'N/A')}")
    lines.append(f"Gender: {user.get('gender', 'N/A')}")
    location_parts = [p for p in (user.get('location'), user.get('country')) if p]
    lines.append(f"Location: {', '.join(location_parts) if location_parts else 'N/A'}")
    lines.append(f"Occupation: {user.get('occupation', 'N/A')}")
    lines.append(f"Interests: {', '.join(user.get('interests', [])) or 'N/A'}")
    lines.append(f"Languages: {', '.join(user.get('languages', [])) or 'N/A'}")

    if 'bio' in user and user['bio']:
        lines.append(f"Bio: {user['bio']}")

    if 'personality' in user and isinstance(user['personality'], dict):
        sorted_traits = sorted(user['personality'].items(), key=lambda x: -x[1])
        # print the normalized (0.0-1.0) values as 1-10 for readability:
        best_traits = ', '.join([f"{k}: {round(v*10, 1)}" for k, v in sorted_traits[:5]])
        lines.append(f"Top personality traits (1-10): {best_traits}")

    return "\n".join(lines)

def build_llm_prompt(new_user, db_users, top_k=5):
    text = (
        "You are a psychologist and travel-match expert. Given a query user and candidate profiles, "
        "recommend only those candidates with a compatibility of at least 75% for being an excellent travel companion. "
        "Think holistically: personality dynamics, emotional style, interests/activities, values and life priorities, "
        "learning/communication styles, travel style and pace, budget and accommodation fit, destination types, planning style, "
        "money attitude, sleep/chronotype, cleanliness/organization, diet/substances, safety/risk tolerance, work mode, "
        "cultural/religious needs, languages, and explicit dealbreakers.\n\n"
        "Age policy: prefer matches within each person's stated age_preference or within their age_gap_tolerance. "
        "If missing, infer cautiously from openness/adaptability/maturity; avoid large gaps unless both have high age_openness.\n\n"
        "Trip context: compatibility should hold for trips from a day to many months and across multiple countries. "
        "Favor pairs who can realistically enjoy shared activities, handle stress and logistics together, and balance each other's styles "
        "without persistent friction.\n\n"
        "OUTPUT STYLE: Return ONLY a JSON list of the top {k} matches with compatibility >= 75%. For each, include:\n"
        "- name\n"
        "- explanation: ONE short sentence, second-person, beginning with 'For you,' briefly stating the strongest reason(s) "
        "(e.g., shared key interests + complementary travel style/pace/budget or aligned values/conflict style). Keep it concise and specific.\n"
        "- compatibility_score: integer percentage with a % symbol (e.g., \"78%\")\n\n"
        "Do not include extra fields.\n\n"
        "Query User Profile:\n"
    ).format(k=top_k)

    text += profile_to_natural_text(new_user) + "\n\n"

    text += "Candidate Profiles:\n"
    for i, u in enumerate(db_users):
        text += f"\n[{i+1}] " + profile_to_natural_text(u)

    text += (
        "\n\nInstructions:\n"
        "1) Silently filter out candidates who violate hard dealbreakers (e.g., incompatible diet/substances, "
        "non-overlapping trip_duration_pref, irreconcilable budget or accommodation_style, or age outside stated preferences with low age_openness).\n"
        "2) Among remaining candidates, reason about multi-dimensional fit (psychological + travel logistics).\n"
        "3) Only include candidates where compatibility ≥ 75%.\n"
        "4) Output the top {k} as JSON with the exact keys: name, explanation, compatibility_score (percentage string, e.g., \"82%\").\n"
    ).format(k=top_k)

    return text


def llm_find_matches(new_user, db_users):
    if not openai.api_key:
        print("Error: OPENAI_API_KEY not found. Please set in your .env file.")
        return []

    prompt = build_llm_prompt(new_user, db_users)

    functions = [
        {
            "name": "match_response",
            "description": "Provide travel match results",
            "parameters": {
                "type": "object",
                "properties": {
                    "matches": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "name": {"type": "string"},
                                "explanation": {"type": "string"},
                                "compatibility_score": {"type": "number", "minimum": 0.0, "maximum": 1.0}
                            },
                            "required": ["name", "explanation", "compatibility_score"]
                        }
                    }
                },
                "required": ["matches"]
            }
        }
    ]

    try:
        response = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful matchmaking assistant."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.3,
            max_tokens=1500,
            functions=functions,
            function_call={"name": "match_response"}
        )

        msg = response.choices[0].message
        func_call = getattr(msg, "function_call", None)
        func_args = func_call.arguments if func_call and getattr(func_call, "arguments", None) is not None else "{}"

        parsed = parse_llm_json(func_args)

        matches = []
        if isinstance(parsed, list):
            matches = parsed
        elif isinstance(parsed, dict):
            if isinstance(parsed.get("matches"), list):
                matches = parsed["matches"]
            else:
                for v in parsed.values():
                    if isinstance(v, list):
                        matches = v
                        break

        print("Parsed LLM function arguments:", parsed)
        print("Extracted matches:", matches)

        return matches

    except Exception as e:
        print(f"Error during API call or parsing: {e}")
        return []


    except Exception as e:
        # Helpful debug to inspect the raw response shape (safe to remove later)
        try:
            import traceback
            traceback.print_exc()
        except Exception:
            pass
        print(f"Error during API call or parsing: {e}")
        return []

def read_int_1_10(prompt, default=None):
    while True:
        val = input(prompt).strip()
        if not val and default is not None:
            return default
        try:
            i = int(val)
            if 1 <= i <= 10:
                return i
            print("Please enter an integer between 1 and 10.")
        except ValueError:
            print("Please enter a valid integer between 1 and 10.")

<<<<<<< HEAD
def read_int(prompt, default=None, min_val=None, max_val=None):
=======
def read_int(prompt, default=None):
>>>>>>> 0e2f315be2dc8ded97ab57f9ae3cf19467983165
    while True:
        val = input(prompt).strip()
        if not val and default is not None:
            return default
        try:
<<<<<<< HEAD
            # Clean input - remove non-numeric characters except minus sign
            cleaned = ''.join(c for c in val if c.isdigit() or c == '-')
            if not cleaned:
                print("Please enter a valid number.")
                continue
            
            num = int(cleaned)
            
            # Validate range
            if min_val is not None and num < min_val:
                print(f"Please enter a number >= {min_val}.")
                continue
            if max_val is not None and num > max_val:
                print(f"Please enter a number <= {max_val}.")
                continue
                
            return num
=======
            return int(val)
>>>>>>> 0e2f315be2dc8ded97ab57f9ae3cf19467983165
        except ValueError:
            print("Please enter a valid integer.")

def create_manual_profile():
    user = {"personality": {}, "id": str(uuid.uuid4())}
    print("--- Create Your Travel Profile ---")
    user['name'] = input("Name: ").strip()
<<<<<<< HEAD
    user['age'] = read_int("Age: ", min_val=18, max_val=80)
    user['gender'] = input("Gender: ").strip()
    # Import validation function
    try:
        from Updated_main import validate_city_country
        while True:
            try:
                city = input("Location (e.g., Berlin): ").strip()
                country = input("Country (e.g., Germany): ").strip()
                validated_city, validated_country = validate_city_country(city, country)
                user['location'] = validated_city
                user['country'] = validated_country
                print(f"✅ Validated: {validated_city}, {validated_country}")
                break
            except ValueError as e:
                print(f"{e}")
                print("Please try again with a valid city-country combination.\n")
    except ImportError:
        # Fallback if validation not available
        user['location'] = input("Location (e.g., Berlin): ").strip()
        user['country'] = input("Country (e.g., Germany): ").strip()
=======
    user['age'] = read_int("Age: ")
    user['gender'] = input("Gender: ").strip()
    user['location'] = input("Location (e.g., Berlin): ").strip()
    user['country'] = input("Country (e.g., Germany): ").strip()
>>>>>>> 0e2f315be2dc8ded97ab57f9ae3cf19467983165
    user['occupation'] = input("Occupation: ").strip()
    user['interests'] = [i.strip() for i in input("Enter interests (comma-separated): ").split(',') if i.strip()]
    user['languages'] = [l.strip() for l in input("Enter languages (comma-separated, e.g., en, de): ").split(',') if l.strip()]
    user['bio'] = input("Bio: ").strip()

    print("\n--- Personality (score 1 to 10) ---")
    # We accept 1-10 from user, but normalize to 0.0-1.0 for internal use & LLM
    user['personality']['openness'] = read_int_1_10("Openness to new experiences: ") / 10.0
    user['personality']['extraversion'] = read_int_1_10("Extraversion (outgoingness): ") / 10.0
    user['personality']['agreeableness'] = read_int_1_10("Agreeableness: ") / 10.0
    user['personality']['conscientiousness'] = read_int_1_10("Conscientiousness (organization): ") / 10.0

    print("\nProfile created successfully!")
    return user

def add_user_to_db(user, db_path):
    try:
        with open(db_path, "r", encoding="utf-8") as f:
            db_data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        db_data = []

    db_data.append(user)
    safe_write_json(db_data, db_path)
    print(f"User '{user['name']}' has been added to {db_path}.")

if __name__ == "__main__":
    new_user_profile = create_manual_profile()
    add_user_to_db(new_user_profile, DB_FILE_PATH)

    try:
        with open(DB_FILE_PATH, "r", encoding="utf-8") as f:
            db_profiles = json.load(f)
    except FileNotFoundError:
        print(f"Database file not found at {DB_FILE_PATH}")
        db_profiles = []

    if db_profiles:
        # Filter out the current user
        candidates = [p for p in db_profiles if p.get('id') != new_user_profile['id']]
        # Prefilter candidates by shared interests and languages
        filtered_candidates = prefilter_candidates(new_user_profile, candidates, max_results=5)

        print(f"\nFinding matches for {new_user_profile['name']} among {len(filtered_candidates)} candidates...")
        matches = llm_find_matches(new_user_profile, filtered_candidates)

        print("\n--- Top Matches Found ---")
        if matches:
            print(json.dumps(matches, indent=2))
        else:
            print("No matches found or an error occurred.")