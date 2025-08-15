# utils.py (updated LLM matching)

import json
import re
import random
import openai

import os
import json
import tempfile

def safe_write_json(data, path):
    """
    Atomic JSON write to avoid file corruption.
    """
    dirpath = os.path.dirname(path)
    os.makedirs(dirpath, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", delete=False, dir=dirpath, encoding="utf-8") as tf:
        json.dump(data, tf, indent=4, ensure_ascii=False)
        tempname = tf.name
    os.replace(tempname, path)


# ----------------------------
# JSON helpers
# ----------------------------
def parse_llm_json(raw_content):
    if isinstance(raw_content, (dict, list)):
        data = raw_content
    else:
        try:
            data = json.loads(raw_content)
        except (json.JSONDecodeError, TypeError):
            match = re.search(r"(\{[\s\S]*\}|\[[\s\S]*\])", str(raw_content))
            if match:
                data = json.loads(match.group(0))
            else:
                raise ValueError("No valid JSON found in input.")
    if isinstance(data, dict) and len(data) == 1:
        key = next(iter(data))
        v = data[key]
        if isinstance(v, list):
            return v
    return data

# ----------------------------
# Candidate prefiltering
# ----------------------------
def prefilter_candidates(candidates, max_results=20):
    """
    Randomly sample candidates to reduce token usage.
    """
    if max_results is None or max_results >= len(candidates):
        return candidates
    return random.sample(candidates, k=max_results)

def summarize_candidates(candidates, max_interests=5, max_languages=3):
    """
    Keep only essential info per candidate to reduce tokens.
    """
    summarized = []
    for u in candidates:
        s = {
            "id": u.get("id"),
            "name": u.get("name"),
            "age": u.get("age"),
            "gender": u.get("gender"),
            "location": u.get("location"),
            "country": u.get("country"),
            "occupation": u.get("occupation"),
            "interests": (u.get("interests") or [])[:max_interests],
            "languages": (u.get("languages") or [])[:max_languages],
            "bio": (u.get("bio") or "")[:200],  # truncate long bios
            "personality": {k: round(v,2) for k,v in (u.get("personality") or {}).items()},
        }
        summarized.append(s)
    return summarized

# ----------------------------
# Profile -> text
# ----------------------------
def profile_to_natural_text(u):
    lines = []
    lines.append(f"Name: {u.get('name','')}")
    lines.append(f"Age: {u.get('age','')}")
    lines.append(f"Gender: {u.get('gender','')}")
    loc = u.get('location','')
    ctry = u.get('country','')
    lines.append(f"Location: {loc}{', ' + ctry if ctry else ''}")
    lines.append(f"Occupation: {u.get('occupation','')}")
    ints = ", ".join(u.get('interests', []) or [])
    langs = ", ".join(u.get('languages', []) or [])
    if ints: lines.append(f"Interests: {ints}")
    if langs: lines.append(f"Languages: {langs}")
    bio = u.get('bio')
    if bio: lines.append(f"Bio: {bio}")
    pers = u.get('personality', {}) or {}
    if pers:
        traits = ", ".join([f"{k}:{v}" for k,v in pers.items() if v is not None])
        if traits: lines.append(f"Personality: {traits}")
    return "\n".join(lines)

# ----------------------------
# Build prompt
# ----------------------------
def build_llm_prompt(new_user, db_users, top_k=5):
    text = (
        "You are a psychologist and travel-match expert. Assess compatibility for being an excellent travel companion.\n\n"
        "OUTPUT RULES:\n"
        "1. Output a JSON array ONLY.\n"
        "2. Each object must have keys: name, explanation, compatibility_score (0.0-1.0).\n"
        "Query User Profile:\n"
    )
    text += profile_to_natural_text(new_user) + "\n\n"
    text += "Candidate Profiles:\n"
    for i, u in enumerate(db_users, 1):
        text += f"\n[{i}]\n" + profile_to_natural_text(u)
    text += f"\n\nInstructions:\nOutput top {top_k} matches only."
    return text

# ----------------------------
# LLM call
# ----------------------------
def llm_find_matches(new_user, db_users, top_k=5, max_candidates=20):
    """
    Prefilter and summarize candidates to avoid token overflow.
    """
    if not openai.api_key:
        print("Error: OPENAI_API_KEY not provided")
        return []

    # 1. Prefilter
    candidates = prefilter_candidates(db_users, max_results=max_candidates)

    # 2. Summarize
    candidates_summarized = summarize_candidates(candidates)

    # 3. Build prompt
    prompt = build_llm_prompt(new_user, candidates_summarized, top_k=top_k)

    try:
        resp = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a precise, trustworthy travel matchmaker. Output valid JSON only."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.3,
            max_tokens=1800,
        )
        content = resp.choices[0].message.content
        parsed = parse_llm_json(content)

        matches = []
        if isinstance(parsed, list):
            for m in parsed:
                if not isinstance(m, dict):
                    continue
                name = m.get("name")
                explanation = m.get("explanation")
                score_val = m.get("compatibility_score")
                if not (name and explanation and isinstance(score_val, (int, float))):
                    continue
                if 0.0 <= score_val <= 1.0:
                    matches.append({
                        "name": name,
                        "explanation": explanation,
                        "compatibility_score": float(score_val)
                    })
        return matches
    except Exception as e:
        print(f"Error during API call or parsing: {e}")
        return []
