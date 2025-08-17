#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RoverMitra Matchmaker – interactive, integrated, error-free.

- Prompts user for a guided profile.
- Appends ONLY the new user to data/travel_ready_user_profiles.json.
- Loads candidate pool from:
    users/data/users_core.json
    MatchMaker/data/matchmaker_profiles.json
- Pipeline:
    Hard Prefilters  →  Free AI Prefilter (S-BERT / TF-IDF / safe fallback)  →  Final Ranking
- Final Ranking:
    If OPENAI_API_KEY is present, uses OpenAI (gpt-4o) automatically.
    Else uses strong local explainer (specific one-liners, no generic text).
- Prints results to console; does not save match results.
"""

import os, re, json, uuid, math
from pathlib import Path
from typing import Any, Dict, List, Optional

# ----------------------------
# Project paths
# ----------------------------
BASE_DIR  = Path(__file__).resolve().parent
USERS_CORE_PATH  = BASE_DIR / "users/data/users_core.json"
MM_PATH          = BASE_DIR / "MatchMaker/data/matchmaker_profiles.json"
LOCAL_DB_PATH    = BASE_DIR / "data/travel_ready_user_profiles.json"  # append new users here

# ----------------------------
# Optional: OpenAI final ranking
# ----------------------------
from dotenv import load_dotenv
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
try:
    import openai
    _OPENAI_OK = True
except Exception:
    _OPENAI_OK = False

# ----------------------------
# Free AI prefilter options
# ----------------------------
_USE_ST_EMB = True
try:
    import numpy as np
    from sentence_transformers import SentenceTransformer
    _EMB = None
except Exception:
    _USE_ST_EMB = False
    _EMB = None
    np = None

_TFIDF_OK = True
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
except Exception:
    _TFIDF_OK = False

# ----------------------------
# Utils
# ----------------------------
def load_json(path: Path) -> list:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else []

def safe_write_json(obj: Any, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")

def append_to_json(obj: dict, path: Path):
    data = load_json(path)
    data.append(obj)
    safe_write_json(data, path)

def index_by(lst: list, key: str) -> Dict[str, Any]:
    return {x.get(key): x for x in lst if isinstance(x, dict) and x.get(key)}

def budget_band(amount: Optional[float]) -> str:
    if amount is None: return "mid"
    if amount <= 90:   return "budget"
    if amount >= 180:  return "lux"
    return "mid"

def shared_language_count(a: List[str], b: List[str]) -> int:
    return len(set(a or []) & set(b or []))

def jaccard(a: List[str], b: List[str]) -> float:
    A, B = set(a or []), set(b or [])
    if not A and not B: return 0.0
    return len(A & B) / max(1, len(A | B))

def tokenize(text: str) -> set:
    return set(re.findall(r"[A-Za-zÀ-ÿ0-9']+", text.lower()))

# ----------------------------
# Vocab / parsing aids (expanded)
# ----------------------------
LANG_ALIASES = {
    # Europe + English
    "english":"en","en":"en",
    "german":"de","deutsch":"de","de":"de",
    "french":"fr","français":"fr","francais":"fr","fr":"fr",
    "italian":"it","italiano":"it","it":"it",
    "spanish":"es","español":"es","es":"es",
    "portuguese":"pt","português":"pt","pt":"pt",
    "dutch":"nl","netherlands":"nl","nl":"nl",
    "polish":"pl","pl":"pl","russian":"ru","ru":"ru",
    "czech":"cs","cz":"cs","cs":"cs","danish":"da","da":"da",
    "swedish":"sv","sv":"sv","norwegian":"no","no":"no",
    "finnish":"fi","fi":"fi",
    # South Asia / MEA
    "hindi":"hi","hi":"hi","urdu":"ur","ur":"ur","punjabi":"pa","pa":"pa",
    "bengali":"bn","bangla":"bn","bn":"bn",
    "tamil":"ta","ta":"ta","telugu":"te","te":"te","marathi":"mr","mr":"mr",
    "gujarati":"gu","gu":"gu","malayalam":"ml","ml":"ml","sinhala":"si","si":"si","nepali":"ne","ne":"ne",
    "arabic":"ar","ar":"ar","farsi":"fa","persian":"fa","fa":"fa","turkish":"tr","tr":"tr","hebrew":"he","he":"he",
    # East / SE Asia
    "japanese":"ja","jp":"ja","ja":"ja",
    "korean":"ko","kr":"ko","ko":"ko",
    "chinese":"zh","mandarin":"zh","zh":"zh",
    "thai":"th","th":"th","vietnamese":"vi","vi":"vi",
    "malay":"ms","ms":"ms","indonesian":"id","id":"id",
    "filipino":"tl","tagalog":"tl","tl":"tl",
}

DIETS = {"none","vegetarian","vegan","halal","kosher","gluten-free","no pork","lactose-free","pescatarian"}
PACE  = {"relaxed","balanced","packed"}
ALCOHOL = {"none","moderate","social"}
SMOKING = {"never","occasionally","regular"}

def clean_list_csv(s: str) -> List[str]:
    return [x.strip() for x in (s or "").split(",") if x.strip()]

# ----------------------------
# Interactive profile builder
# ----------------------------
def ask(prompt: str, default: Optional[str]=None) -> str:
    v = input(f"{prompt}{' ['+default+']' if default else ''}: ").strip()
    return v if v else (default or "")

def ask_int(prompt: str, default: Optional[int]=None) -> int:
    while True:
        v = input(f"{prompt}{' ['+str(default)+']' if default is not None else ''}: ").strip()
        if not v and default is not None:
            return default
        try:
            return int(v)
        except:
            print("Please enter a number.")

def ask_choice(prompt: str, choices: List[str], default: Optional[str]=None) -> str:
    cs = "/".join(choices)
    while True:
        v = input(f"{prompt} ({cs}){(' ['+default+']' if default else '')}: ").strip().lower()
        if not v and default: return default
        if v in choices: return v
        print(f"Please choose one of: {cs}")

def interactive_new_user() -> Dict[str, Any]:
    print("\n--- Create Your Travel Profile ---")
    name = ask("Name")
    age  = ask_int("Age", 27)
    gender = ask_choice("Gender", ["male","female","non-binary","other"], "other").title()
    city = ask("Home city (e.g., Berlin)", "Berlin")
    country = ask("Country (e.g., Germany)", "Germany")

    langs_in = clean_list_csv(ask("Languages (comma, e.g., en, de, ur)", "en"))
    langs = []
    for w in langs_in:
        key = w.lower()
        langs.append(LANG_ALIASES.get(key, key))
    langs = sorted(set(langs))

    interests = clean_list_csv(ask("Top interests (comma, e.g., museums, scenic trains, mountain hiking)", "museums, scenic trains, coffee crawls"))
    pace = ask_choice("Travel pace", ["relaxed","balanced","packed"], "balanced")
    budget = ask_int("Budget per day (number)", 150)
    diet = ask_choice("Diet", list(DIETS), "none")
    smoking = ask_choice("Smoking", list(SMOKING), "never")
    alcohol = ask_choice("Alcohol", list(ALCOHOL), "moderate")
    bio = ask("Short bio (optional)", "")

    user = {
        "id": f"u_local_{uuid.uuid4().hex[:8]}",
        "name": name,
        "age": age,
        "gender": gender,
        "home_base": {"city": city, "country": country, "nearby_nodes": [], "willing_radius_km": 40},
        "languages": langs,
        "interests": interests,
        "travel_prefs": {"pace": pace},
        "budget": {"type":"per_day","amount": budget, "currency": "EUR", "split_rule": "each_own"},
        "diet_health": {"diet": diet, "allergies": ["none"], "accessibility": "none"},
        "comfort": {"smoking": smoking, "alcohol": alcohol, "risk_tolerance":"medium","noise_tolerance":"medium","cleanliness_preference":"medium"},
        "bio": bio
    }
    return user

# ----------------------------
# Candidate pool
# ----------------------------
def load_pool():
    users = load_json(USERS_CORE_PATH)
    mm    = load_json(MM_PATH)
    mm_by_uid = index_by(mm, "user_id")
    pool = []
    for u in users:
        uid = u.get("user_id")
        if not uid: continue
        pool.append({"user": u, "mm": mm_by_uid.get(uid)})
    return pool

# ----------------------------
# Summaries → text for embeddings
# ----------------------------
def summarize_user(u: Dict[str,Any], mm: Optional[Dict[str,Any]]) -> str:
    parts = []
    parts.append(f"{u.get('name','')}, age {u.get('age','')}")
    hb = u.get("home_base") or {}
    parts.append(f"city={hb.get('city','')}")
    parts.append(f"langs={'/'.join(u.get('languages',[]))}")
    bud = (u.get('budget') or {}).get('amount')
    parts.append(f"budget={bud}")
    dh = u.get("diet_health") or {}
    parts.append(f"diet={dh.get('diet','none')}")
    cmf = u.get("comfort") or {}
    parts.append(f"smoking={cmf.get('smoking','never')} alcohol={cmf.get('alcohol','moderate')}")
    pace = (u.get("travel_prefs") or {}).get("pace") or "balanced"
    parts.append(f"pace={pace}")
    intr = u.get("interests") or []
    parts.append("interests=" + ",".join(intr[:12]))
    vals = u.get("values") or []
    if vals: parts.append("values=" + ",".join(vals[:8]))
    if mm:
        intents = mm.get("match_intent") or []
        parts.append("intent=" + ",".join(intents))
        lp = mm.get("language_policy") or {}
        parts.append(f"min_shared_lang={lp.get('min_shared_languages',1)}")
        sdeals = mm.get("hard_dealbreakers") or []
        if sdeals: parts.append("dealbreakers=" + ",".join(sdeals[:6]))
    return " | ".join(parts)

def query_text(q: Dict[str,Any]) -> str:
    hb = q.get("home_base") or {}
    return (
        f"Name: {q.get('name')}\n"
        f"Age: {q.get('age')}\n"
        f"Gender: {q.get('gender')}\n"
        f"Location: {hb.get('city','')} {hb.get('country','')}\n"
        f"Languages: {', '.join(q.get('languages',[]))}\n"
        f"Pace: {q.get('travel_prefs',{}).get('pace','balanced')}\n"
        f"BudgetPerDay: {(q.get('budget') or {}).get('amount')}\n"
        f"Diet: {(q.get('diet_health') or {}).get('diet','none')}\n"
        f"Smoking: {(q.get('comfort') or {}).get('smoking','never')}\n"
        f"Alcohol: {(q.get('comfort') or {}).get('alcohol','moderate')}\n"
        f"Interests: {', '.join(q.get('interests',[]))}"
    )

# ----------------------------
# Hard prefilters
# ----------------------------
def age_ok(query_age: int, cand_user: Dict[str,Any], cand_mm: Dict[str,Any]) -> bool:
    pc = (cand_mm or {}).get("preferred_companion") or {}
    r  = pc.get("age_range")
    if r and isinstance(r, list) and len(r) == 2:
        return r[0] <= query_age <= r[1]
    return True

def gender_ok(query_gender: str, cand_mm: Dict[str,Any]) -> bool:
    pc = (cand_mm or {}).get("preferred_companion") or {}
    allowed  = [g.lower() for g in (pc.get("genders") or ["any"])]
    qg = (query_gender or "").lower()
    return ("any" in allowed) or (qg in allowed)

def langs_ok(query_langs: List[str], cand_mm: Dict[str,Any], cand_user: Dict[str,Any]) -> bool:
    lp = (cand_mm or {}).get("language_policy") or {}
    need = lp.get("min_shared_languages", 1)
    cand_pref = lp.get("preferred_chat_languages") or cand_user.get("languages",[])
    if need <= 0: 
        return True
    return shared_language_count(query_langs, cand_pref) >= need

def budget_ok(query_amount: int, cand_user: Dict[str,Any]) -> bool:
    cand_amount = (cand_user.get("budget") or {}).get("amount")
    qa = budget_band(query_amount)
    ca = budget_band(cand_amount)
    order = ["budget","mid","lux"]
    return abs(order.index(qa) - order.index(ca)) <= 1

def pace_ok(query_pace: str, cand_user: Dict[str,Any], cand_mm: Dict[str,Any]) -> bool:
    pref = (cand_mm or {}).get("soft_preferences", {})
    want_same = pref.get("prefer_same_pace") == "prefer_same"
    cand_pace = (cand_user.get("travel_prefs") or {}).get("pace")
    if not want_same or cand_pace is None:
        return True
    return cand_pace == query_pace

def hard_prefilter(q: Dict[str,Any], pool: List[Dict[str,Any]]) -> List[Dict[str,Any]]:
    out = []
    q_age = int(q.get("age", 30))
    q_gender = q.get("gender","Other")
    q_langs = q.get("languages",[])
    q_budget = (q.get("budget") or {}).get("amount", 150)
    q_pace = (q.get("travel_prefs") or {}).get("pace","balanced")
    for rec in pool:
        cu, cm = rec["user"], rec["mm"]
        if not langs_ok(q_langs, cm, cu):           continue
        if not age_ok(q_age, cu, cm):               continue
        if not gender_ok(q_gender, cm):             continue
        if not budget_ok(q_budget, cu):             continue
        if not pace_ok(q_pace, cu, cm):             continue
        out.append(rec)
    return out

# ----------------------------
# Free AI prefilter
# ----------------------------
def ensure_emb():
    global _EMB
    if _USE_ST_EMB and _EMB is None:
        _EMB = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    return _EMB

def ai_prefilter(q_user: Dict[str, Any],
                 cands: List[Dict[str, Any]],
                 percent: float = 0.02,
                 min_k: int = 80) -> List[Dict[str, Any]]:
    """
    Shortlist candidates using:
      1) Semantic similarity between query text and candidate summaries
         - Sentence-Transformers (all-MiniLM-L6-v2) if available
         - Fallback: TF-IDF + cosine similarity
         - Safe fallback: token Jaccard if neither is available
      2) Symbolic bonuses (interests, values, pace, languages, budget proximity)
    """
    if not cands:
        return []

    # 1) Texts
    qtext = query_text(q_user)
    summaries = [summarize_user(rec["user"], rec.get("mm")) for rec in cands]
    texts = [qtext] + summaries

    # 2) Similarities
    sims: List[float] = []
    used_backend = None
    if _USE_ST_EMB:
        used_backend = "sbert"
        m = ensure_emb()
        vecs = m.encode(texts, normalize_embeddings=True)
        qv = vecs[0:1]
        cv = vecs[1:]
        sims = (cv @ qv.T).ravel().astype(float).tolist()
    elif _TFIDF_OK:
        used_backend = "tfidf"
        tfidf = TfidfVectorizer(min_df=1, max_df=0.95)
        mat = tfidf.fit_transform(texts)
        qv = mat[0]
        cv = mat[1:]
        sims = cosine_similarity(cv, qv).ravel().tolist()
    else:
        used_backend = "naive"
        tq = tokenize(qtext)
        for s in summaries:
            ts = tokenize(s)
            sims.append(jaccard(list(tq), list(ts)))

    # 3) Symbolic bonuses
    def _band(v):
        if v is None: return "mid"
        if v <=  90:  return "budget"
        if v >= 180:  return "lux"
        return "mid"
    def _band_idx(b):
        order = ["budget","mid","lux"]
        return order.index(b) if b in order else 1

    q_interests = set(q_user.get("interests", []) or [])
    q_values    = set(q_user.get("values", []) or [])
    q_langs     = set(q_user.get("languages", []) or [])
    q_pace      = (q_user.get("travel_prefs") or {}).get("pace", "balanced")
    q_budget_b  = _band((q_user.get("budget") or {}).get("amount"))

    bonuses = []
    for rec in cands:
        u = rec["user"]
        mm = rec.get("mm") or {}
        ui  = set(u.get("interests", []) or [])
        uv  = set(u.get("values", []) or [])
        b   = 0.0
        # interests / values
        b  += 0.60 * (len(q_interests & ui) / max(1, len(q_interests | ui)))
        b  += 0.30 * (len(q_values & uv) / max(1, len(q_values | uv)))
        # pace
        c_pace = (u.get("travel_prefs") or {}).get("pace") or "balanced"
        want_same = ((mm.get("soft_preferences") or {}).get("prefer_same_pace") == "prefer_same")
        if want_same and c_pace == q_pace:
            b += 0.20
        # languages (cap at one language bonus)
        langs = set(u.get("languages", []) or [])
        b += 0.20 * min(1, len(q_langs & langs))
        # budget band proximity
        cand_amt = (u.get("budget") or {}).get("amount")
        c_band = _band(cand_amt)
        gap = abs(_band_idx(q_budget_b) - _band_idx(c_band))
        if gap == 0: b += 0.20
        elif gap == 1: b += 0.10
        bonuses.append(b)

    # Normalize sims to [0,1] for stability
    s_min, s_max = min(sims), max(sims)
    if s_max > s_min:
        sims_norm = [(s - s_min) / (s_max - s_min) for s in sims]
    else:
        sims_norm = [0.5] * len(sims)

    combined = [(rec, sims_norm[i] + bonuses[i]) for i, rec in enumerate(cands)]
    combined.sort(key=lambda x: x[1], reverse=True)

    k = max(min_k, int(math.ceil(len(combined) * percent)))
    k = min(k, len(combined))
    shortlist = [rec for rec, _ in combined[:k]]
    return shortlist

# ----------------------------
# Local specific reason generator (no generic lines)
# ----------------------------
def craft_specific_reason(q: Dict[str,Any], u: Dict[str,Any], mm: Optional[Dict[str,Any]]) -> str:
    qi = set(q.get("interests",[]))
    ui = set(u.get("interests",[]))
    shared_i = sorted(qi & ui)
    pace_q = (q.get("travel_prefs") or {}).get("pace","balanced")
    pace_u = (u.get("travel_prefs") or {}).get("pace","balanced")
    langs = sorted(set(q.get("languages",[])) & set(u.get("languages",[])))
    budget_gap = abs((q.get("budget") or {}).get("amount",150) - (u.get("budget") or {}).get("amount",150))
    diet_q = (q.get("diet_health") or {}).get("diet","none")
    diet_u = (u.get("diet_health") or {}).get("diet","none")
    city_u = (u.get("home_base") or {}).get("city","")

    hooks = []
    if shared_i: hooks.append(f"shared love for {', '.join(shared_i[:2])}")
    if langs:    hooks.append(f"you both speak {', '.join(langs[:2])}")
    if pace_q == pace_u: hooks.append(f"matching {pace_q} pace")
    if budget_gap <= 30: hooks.append("similar daily budgets")
    if diet_u != "none" and diet_u == diet_q: hooks.append(f"both {diet_u}")
    if city_u:   hooks.append(f"and they’re based in {city_u}")
    if not hooks:
        hooks.append("complementary interests and compatible travel habits")
    return "For you, this match fits because of " + ", ".join(hooks) + "."

# ----------------------------
# Final ranking
# ----------------------------
def build_llm_prompt(q_user: Dict[str,Any], shortlist: List[Dict[str,Any]], top_k:int=5) -> str:
    head = (
        "You are a psychologist + travel-match expert. Rank candidates for holistic trip compatibility.\n"
        "Consider personality fit, conflict style, shared & complementary interests, languages/communication, pace, budget, diet/substances,\n"
        "safety/risk tolerance, work needs, and values. Trip context: weekend to multi-week.\n"
        "Return JSON with: user_id, name, explanation (ONE sentence, must start with 'For you,' and be specific), compatibility_score (0.0–1.0).\n\n"
        "Query User:\n"
        f"{query_text(q_user)}\n\n"
        "Candidates:\n"
    )
    body = []
    for i, rec in enumerate(shortlist):
        u, m = rec["user"], rec["mm"]
        body.append(f"[{i+1}] user_id={u.get('user_id')} | {summarize_user(u, m)}")
    return head + "\n".join(body)

def llm_rank(q_user: Dict[str,Any], shortlist: List[Dict[str,Any]], out_top:int=5) -> List[Dict[str,Any]]:
    if not (OPENAI_API_KEY and _OPENAI_OK):
        # Local, specific one-liners
        results = []
        for rec in shortlist[:out_top]:
            u = rec["user"]
            reason = craft_specific_reason(q_user, u, rec.get("mm"))
            score = 0.70 + 0.25 * jaccard(q_user.get("interests",[]), u.get("interests",[]))
            results.append({
                "user_id": u.get("user_id"),
                "name": u.get("name"),
                "explanation": reason,
                "compatibility_score": round(min(max(score, 0.0), 0.99), 2)
            })
        return results

    openai.api_key = OPENAI_API_KEY
    prompt = build_llm_prompt(q_user, shortlist, top_k=out_top)

    functions = [{
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
                            "user_id": {"type": "string"},
                            "name": {"type": "string"},
                            "explanation": {"type": "string"},
                            "compatibility_score": {"type": "number","minimum":0.0,"maximum":1.0}
                        },
                        "required": ["user_id","name","explanation","compatibility_score"]
                    }
                }
            },
            "required": ["matches"]
        }
    }]

    try:
        resp = openai.chat.completions.create(
            model="gpt-4o",
            temperature=0.25,
            max_tokens=1200,
            functions=functions,
            function_call={"name":"match_response"},
            messages=[
                {"role":"system","content":"You are a precise, concise travel matchmaker."},
                {"role":"user","content": prompt}
            ]
        )
        msg = resp.choices[0].message
        fc  = getattr(msg, "function_call", None)
        raw = fc.arguments if (fc and getattr(fc,"arguments",None) is not None) else "{}"
        try:
            parsed = json.loads(raw)
        except Exception:
            parsed = {}
        matches = parsed.get("matches") or []
        # If model returns empty or malformed, fall back to local reasons
        if not matches:
            return llm_rank.__wrapped__(q_user, shortlist, out_top)  # type: ignore
        return matches[:out_top]
    except Exception:
        # Fall back to local
        return llm_rank.__wrapped__(q_user, shortlist, out_top)  # type: ignore

# Preserve a reference for fallback recursion
llm_rank.__wrapped__ = lambda q, sl, k: [
    {
        "user_id": rec["user"].get("user_id"),
        "name": rec["user"].get("name"),
        "explanation": craft_specific_reason(q, rec["user"], rec.get("mm")),
        "compatibility_score": round(0.70 + 0.25 * jaccard(q.get("interests",[]), rec["user"].get("interests",[])), 2)
    } for rec in sl[:k]
]

# ----------------------------
# Run
# ----------------------------
def main():
    # 1) Build query user interactively
    q_user = interactive_new_user()

    # 2) Append to local user db (ONLY here; not to matchmaker data)
    append_to_json(q_user, LOCAL_DB_PATH)
    print(f"\n✅ Saved your profile to {LOCAL_DB_PATH}")

    # 3) Load candidate pool
    pool = load_pool()
    if not pool:
        print("No candidates found; please generate users_core.json + matchmaker_profiles.json.")
        return

    # 4) Hard prefilters
    hard = hard_prefilter(q_user, pool)
    if not hard:
        print("No candidates remained after hard prefilters; try broader languages/pace/budget.")
        return

    # 5) AI prefilter → keep ~2% (min 80)
    shortlist = ai_prefilter(q_user, hard, percent=0.02, min_k=80)

    # 6) Final ranking (automatic; uses OpenAI if key present, else local)
    final = llm_rank(q_user, shortlist, out_top=5)

    # 7) Print results (no file write)
    print("\n--- Top Recommendations ---")
    if not final:
        print("No matches found.")
        return
    for i, m in enumerate(final, 1):
        try:
            pct = int(round(float(m.get("compatibility_score",0))*100))
        except Exception:
            pct = 0
        print(f"{i}. {m.get('name')} (user_id: {m.get('user_id')})  —  {pct}%")
        print(f"   {m.get('explanation')}\n")

if __name__ == "__main__":
    main()
