# utils.py
import json
import tempfile
import os
import re

def parse_llm_json(raw_content):
    """
    Safely parse JSON from a string or dict. Returns the parsed object.
    If input is a dict with a single top-level key, it unwraps that key and returns its value.
    """
    if isinstance(raw_content, dict):
        data = raw_content
    else:
        try:
            data = json.loads(raw_content)
        except (json.JSONDecodeError, TypeError):
            # Attempt to find JSON substring
            match = re.search(r"(\{.*\}|\[.*\])", str(raw_content), re.S)
            if match:
                data = json.loads(match.group())
            else:
                raise ValueError("No valid JSON found in input.")
    # Unwrap single-key dicts
    if isinstance(data, dict) and len(data) == 1:
        # return the single value (could be list)
        key = next(iter(data))
        return data[key]
    return data

def safe_write_json(data, path):
    """
    Write JSON atomically to avoid file corruption.
    """
    dirpath = os.path.dirname(path)
    os.makedirs(dirpath, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", delete=False, dir=dirpath, encoding="utf-8") as tf:
        json.dump(data, tf, indent=4, ensure_ascii=False)
        tempname = tf.name
    os.replace(tempname, path)

def prefilter_candidates(new_user, candidates, max_results=5):
    """
    Prefilter candidates by shared interests and languages.
    Returns top max_results candidates with highest shared counts.
    """
    def score(candidate):
        shared_interests = len(set(new_user.get("interests", [])) & set(candidate.get("interests", [])))
        shared_languages = len(set(new_user.get("languages", [])) & set(candidate.get("languages", [])))
        return shared_interests * 2 + shared_languages  # weigh interests more

    scored = [(score(c), c) for c in candidates]
    scored.sort(key=lambda x: x[0], reverse=True)
    filtered = [c for s, c in scored if s > 0]  # only keep if shares something
    return filtered[:max_results]


def build_llm_prompt(new_user, db_users):
    """
    Constructs a detailed prompt describing the new user and candidate users
    for the LLM to generate matches.
    """
    prompt = f"Find the best travel matches for this user:\n\nUser Profile:\n"
    prompt += f"Name: {new_user.get('name')}\n"
    prompt += f"Age: {new_user.get('age')}\n"
    prompt += f"Gender: {new_user.get('gender')}\n"
    prompt += f"Location: {new_user.get('location')}, {new_user.get('country')}\n"
    prompt += f"Occupation: {new_user.get('occupation')}\n"
    prompt += f"Interests: {', '.join(new_user.get('interests', []))}\n"
    prompt += f"Languages: {', '.join(new_user.get('languages', []))}\n"
    prompt += f"Bio: {new_user.get('bio')}\n"
    prompt += f"Personality:\n"
    for trait, val in new_user.get('personality', {}).items():
        prompt += f"  - {trait}: {val:.2f}\n"

    prompt += "\nCandidate Profiles:\n"
    for candidate in db_users:
        prompt += f"- Name: {candidate.get('name')}, Age: {candidate.get('age')}, Location: {candidate.get('location')}, Interests: {', '.join(candidate.get('interests', []))}\n"

    prompt += "\nProvide a JSON output with matches, including name, explanation, and compatibility_score (0.0 to 1.0)."
    return prompt

import openai

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
        return matches
    except Exception as e:
        print(f"Error during API call or parsing: {e}")
        return []
