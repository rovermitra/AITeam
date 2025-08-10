# utils.py
import json
import tempfile
import os
import re
import random
import openai

# ----------------------------
# JSON helpers & safe write
# ----------------------------
def parse_llm_json(raw_content):
    """
    Safely parse JSON from a string or dict. Returns the parsed object.
    Tries to extract the first JSON object/array if raw text is noisy.
    If top-level is a single-key dict whose value is a list, unwrap it.
    """
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
# NO HARDCODED PREFILTERING
# ----------------------------
def prefilter_candidates(new_user, candidates, max_results=None):
    """
    No hard rules. Return candidates as-is (optionally random-sampled to control token size).
    This avoids brittle heuristics (e.g., exact-match interests/languages).
    """
    if max_results is None or max_results >= len(candidates):
        return candidates
    # random sample to keep prompt size manageable
    return random.sample(candidates, k=max_results)

# ----------------------------
# Prompt builder (LLM does all)
# ----------------------------
def profile_to_natural_text(u):
    # Compact, but includes enough signal for LLM
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
        # only show what exists; values could be 0..1 or 1..10 normalized
        traits = ", ".join([f"{k}:{round(float(v),2)}" for k,v in pers.items() if v is not None])
        if traits:
            lines.append(f"Personality: {traits}")
    # travel-specific fields (if your dataset includes them later)
    tp = u.get('travel_preferences', {}) or {}
    if tp:
        for k, v in tp.items():
            if isinstance(v, list):
                v = ", ".join(v)
            lines.append(f"{k}: {v}")
    return "\n".join(lines)

def build_llm_prompt(new_user, db_users, top_k=5):
    """
    Holistic, psychologist-style, travel context; short, second-person explanations;
    only >= 75%; percentage score string.
    """
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
        "OUTPUT RULES:\n"
        "1. Output a JSON array ONLY.\n"
        "2. Each object must have keys: name, explanation, compatibility_score.\n"
        "3. compatibility_score must be an integer percentage string ending with '%', e.g., \"82%\".\n"
        "4. Do not invent people not in the candidate list.\n"
        "Query User Profile:\n"
    ).format(k=top_k)

    text += profile_to_natural_text(new_user) + "\n\n"

    text += "Candidate Profiles:\n"
    for i, u in enumerate(db_users, 1):
        text += f"\n[{i}]\n" + profile_to_natural_text(u)

    text += (
        "\n\nInstructions:\n"
        "1) Drop candidates that violate hard dealbreakers or yield < 75% compatibility.\n"
        "2) Choose realistic, sustainable pairings (not just superficial similarity).\n"
        "3) Output the top {k} as a JSON array with EXACT keys: name, explanation, compatibility_score (e.g., \"79%\").\n"
    ).format(k=top_k)

    return text

# ----------------------------
# LLM call
# ----------------------------
def llm_find_matches(new_user, db_users, top_k=5):
    """
    Calls OpenAI to get matches; expects JSON array as per the prompt.
    """
    if not openai.api_key:
        print("Error: OPENAI_API_KEY not provided (env var 'API').")
        return []

    prompt = build_llm_prompt(new_user, db_users, top_k=top_k)

    try:
        # Using Chat Completions with function-like schema is optional; here we just ask for JSON directly.
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

        # Ensure it's a list of dicts with the exact keys and >= 75%
        matches = []
        if isinstance(parsed, list):
            for m in parsed:
                if not isinstance(m, dict):
                    continue
                name = m.get("name")
                explanation = m.get("explanation")
                score_str = m.get("compatibility_score", "")
                if not (name and explanation and isinstance(score_str, str) and score_str.endswith("%")):
                    continue
                try:
                    pct = int(score_str.strip().replace("%", ""))
                except ValueError:
                    continue
                if pct >= 75:
                    matches.append({"name": name, "explanation": explanation, "compatibility_score": f"{pct}%"})
        return matches
    except Exception as e:
        print(f"Error during API call or parsing: {e}")
        return []
