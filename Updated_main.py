#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RoverMitra Matchmaker – interactive, integrated, error-free (Playful UX v2)

- Engaging, tap-style questions with examples & emojis (travel-focused, not dating).
- Appends ONLY the new user to data/travel_ready_user_profiles.json.
- Loads candidate pool from:
    users/data/users_core.json
    MatchMaker/data/matchmaker_profiles.json
- Pipeline:
    Hard Prefilters  →  Free AI Prefilter (S-BERT / TF-IDF / safe fallback)  →  Final Ranking
- Final Ranking:
    If OPENAI_API_KEY is present, uses OpenAI (gpt-4o) automatically.
    Else uses specific one-liners (no generic fluff).
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
# Vocab / parsing aids
# ----------------------------
LANG_ALIASES = {
    # Europe + English
<<<<<<< HEAD
    "english":"en","en":"en","english":"eng",    
=======
    "english":"en","en":"en",
>>>>>>> 0e2f315be2dc8ded97ab57f9ae3cf19467983165
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

# ----------------------------
# CLI helpers (playful, fast)
# ----------------------------
def _show_choices(choices: List[str]) -> str:
    return " / ".join(f"{i+1}:{c}" for i, c in enumerate(choices))

def ask(prompt: str, default: Optional[str]=None) -> str:
    v = input(f"{prompt}{' ['+default+']' if default else ''}: ").strip()
    return v if v else (default or "")

def ask_int(prompt: str, default: Optional[int]=None) -> int:
    while True:
        v = input(f"{prompt}{' ['+str(default)+']' if default is not None else ''}: ").strip()
        if not v and default is not None:
            return default
        try:
<<<<<<< HEAD
            num = int(v)
            if num < 0:
                print("Please enter a positive number.")
                continue
            return num
        except:
            print("Please enter a valid number.")
=======
            return int(v)
        except:
            print("Please enter a number.")
>>>>>>> 0e2f315be2dc8ded97ab57f9ae3cf19467983165

def ask_choice(prompt: str, choices: List[str], default: Optional[str]=None) -> str:
    index_hint = _show_choices(choices)
    while True:
        v = input(f"{prompt} ({index_hint}){(' ['+default+']' if default else '')}: ").strip()
        if not v and default: return default
        if v.isdigit():
            i = int(v)-1
            if 0 <= i < len(choices): return choices[i]
        if v in choices: return v
        print(f"Please answer with number 1..{len(choices)} or exact value.")

def ask_yesno(prompt: str, default: bool=True) -> bool:
    d = "Y/n" if default else "y/N"
    while True:
        v = input(f"{prompt} ({d}): ").strip().lower()
        if not v: return default
        if v in {"y","yes"}: return True
        if v in {"n","no"}: return False
        print("Please answer y or n.")

def ask_multi(prompt: str, choices: List[str], min_k:int=1, max_k:int=3, default_idxs: Optional[List[int]]=None) -> List[str]:
    idx_hint = _show_choices(choices)
    default_str = ""
    if default_idxs:
        default_str = " [" + ",".join(str(i+1) for i in default_idxs) + "]"
    while True:
        v = input(f"{prompt} (pick {min_k}-{max_k}) {idx_hint}{default_str}: ").strip()
        if not v and default_idxs:
            picks = [choices[i] for i in default_idxs]
            return picks[:max_k]
        parts = [p.strip() for p in re.split(r"[,\s]+", v) if p.strip()]
        out = []
        ok = True
        for p in parts:
            if p.isdigit():
                i = int(p)-1
                if 0 <= i < len(choices): out.append(choices[i])
                else: ok = False
            else:
                if p in choices: out.append(p)
                else: ok = False
        out = list(dict.fromkeys(out))
        if ok and min_k <= len(out) <= max_k:
            return out
        print("Oops—please choose valid numbers/values within the requested range.")

def clean_list_csv(s: str) -> List[str]:
    return [x.strip() for x in (s or "").split(",") if x.strip()]

# ----------------------------
# Interactive profile builder (travel-first, not-dating)
# ----------------------------
def interactive_new_user() -> Dict[str, Any]:
    print("\n🌍  Welcome! Let’s make trips easy to say yes to.\n(Quick taps, tiny questions, zero boredom.)\n")

    # — Identity & home base —
    print("— Basics —")
    name = ask("Your name (e.g., 'Aisha Khan' or 'Tom Müller')", "Alex Rivera")
    age  = ask_int("Age", 27)
    gender = ask_choice("How do you identify?", ["Male","Female","Non-binary","Other"], "Other")
    city = ask("Home city (e.g., Berlin)", "Berlin")
    country = ask("Country (e.g., Germany)", "Germany")

    # — Languages —
    print("\n— Languages —")
    print("Tip: use names or codes (e.g., english/en, deutsch/de).")
    langs_in = clean_list_csv(ask("Languages you can chat in (comma)", "en, de"))
    langs = []
    for w in langs_in:
        key = w.lower()
        langs.append(LANG_ALIASES.get(key, key))
    langs = sorted(set([x for x in langs if x]))

    # — Interests (snappy chips) —
    print("\n— Pick your vibe —")
    INTEREST_CHIPS = [
        "museum hopping","architecture walks","history sites","city photography","food tours","street food",
        "coffee crawls","vineyards","scenic trains","short hikes","long hikes","mountain hiking","lake swims",
        "beach days","markets","old towns","rooftop views","local crafts","live music","festivals"
    ]
    interests = ask_multi("What sounds fun right now?", INTEREST_CHIPS, 3, 6, default_idxs=[0,1,3])

    # — Pace & rhythm —
    print("\n— Tempo —")
    pace = ask_choice("Trip pace", ["relaxed","balanced","packed"], "balanced")
    chronotype = ask_choice("You’re most alive…", ["early bird","flexible","night owl"], "flexible")

    # — Budget —
    print("\n— Money stuff (kept private) —")
    budget = ask_int("Comfortable budget per day (€)", 150)
    split_rule = ask_choice("Fair way to split shared costs?", ["each_own","50/50","custom"], "each_own")

    # — Diet & allergies —
    print("\n— Food —")
    diet = ask_choice("Diet", sorted(list(DIETS)), "none")
    allergies_raw = clean_list_csv(ask("Allergies? (comma, e.g., nuts, gluten) or 'none'", "none"))
    allergies = ["none"] if (not allergies_raw or allergies_raw == ["none"]) else allergies_raw

    # — Comfort & lifestyle —
    print("\n— Comfort —")
    smoking = ask_choice("Smoking", ["never","occasionally","regular"], "never")
    alcohol = ask_choice("Alcohol", ["none","moderate","social"], "moderate")
    noise = ask_choice("Noise tolerance for stays", ["low","medium","high"], "medium")
    clean = ask_choice("Cleanliness preference", ["low","medium","high"], "medium")
    risk  = ask_choice("Risk tolerance for activities", ["low","medium","high"], "medium")

    # — Work & Wi-Fi —
    print("\n— Work bits —")
    remote_ok = ask_yesno("Need time to work while traveling?")
    hours_online = ask_choice("Hours you must be online per day", ["0","1","2"], "0")
    wifi_need = ask_choice("Wi-Fi needs", ["normal","good","excellent"], "good")

    # — Must-haves at stays —
    print("\n— Stay must-haves —")
    MUSTS = ["wifi","private_bath","kitchen","workspace","near_station","quiet_room","breakfast"]
    must_haves = ask_multi("Pick 1–4 must-haves", MUSTS, 1, 4, default_idxs=[0,3])

    # — Room setup & transport —
    print("\n— Room & transport —")
    room_setup = ask_choice("Room setup you prefer when sharing", ["twin","double","2 rooms","dorm"], "twin")
    TRANSPORT = ["train","plane","bus","car"]
    transport_allowed = ask_multi("Allowed transport modes", TRANSPORT, 1, 3, default_idxs=[0,1])

    # — Trip styles (keeps it travel-focused, not dating) —
    print("\n— Trip styles —")
    STYLES = ["weekend getaway","co-work week","hiking basecamp","city sampler","island hop","festival trip","road trip"]
    trip_styles = ask_multi("Pick a couple", STYLES, 1, 3, default_idxs=[0,3])

    # — Values (for soft matching) —
    print("\n— Values —")
    VALUES = ["adventure","stability","learning","family","budget-minded","luxury-taste","nature","culture","community","fitness","spirituality"]
    values = ask_multi("Choose 2–3", VALUES, 2, 3, default_idxs=[0,6])

    # — Safety/visibility —
    print("\n— Safety —")
    share_home_city = ask_yesno("OK to show your home city on profile?", True)
    pre_meet_video  = ask_yesno("Comfortable with a short pre-trip video call?", True)

    # — Bio (keep it light) —
    print("\n— Last bit —")
    bio = ask("Write a one-liner about you (e.g., 'Berlin-based planner who loves scenic trains + espresso.')", "")

    # Build user object (aligned to your schema fields used by filters/ranker)
    user = {
        "id": f"u_local_{uuid.uuid4().hex[:8]}",
        "name": name,
        "age": age,
        "gender": gender,
        "home_base": {"city": city, "country": country, "nearby_nodes": [], "willing_radius_km": 40},
        "languages": langs,
        "interests": interests,
        "values": values,
        "bio": bio,

        "travel_prefs": {
            "pace": pace,
            "accommodation_types": ["hotel","apartment"],  # sensible default
            "room_setup": room_setup,
            "transport_allowed": transport_allowed,
            "must_haves": must_haves
        },

        "budget": {"type":"per_day","amount": budget, "currency": "EUR", "split_rule": split_rule},

        "diet_health": {"diet": diet, "allergies": allergies if allergies else ["none"], "accessibility": "none"},

        "comfort": {
            "risk_tolerance": risk,
            "noise_tolerance": noise,
            "cleanliness_preference": clean,
            "chronotype": chronotype,
            "alcohol": alcohol,
            "smoking": smoking
        },

        "work": {
            "remote_work_ok": remote_ok,
            "hours_online_needed": int(hours_online),
            "wifi_quality_needed": wifi_need
        },

        "privacy": {
            "share_profile_with_matches": True,
            "share_home_city": share_home_city,
            "pre_meet_video_call_ok": pre_meet_video
        }
    }
    print("\n✅ Thanks! Building your matches…\n")
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
    parts.append(f"smoking={cmf.get('smoking','never')} alcohol={cmf.get('alcohol','moderate')} risk={cmf.get('risk_tolerance','medium')}")
    pace = (u.get("travel_prefs") or {}).get("pace") or "balanced"
    parts.append(f"pace={pace}")
    intr = u.get("interests") or []
    parts.append("interests=" + ",".join(intr[:12]))
    vals = u.get("values") or []
    if vals: parts.append("values=" + ",".join(vals[:6]))
    if cmf.get("chronotype"): parts.append(f"chronotype={cmf.get('chronotype')}")
    if (u.get("work") or {}).get("remote_work_ok"): parts.append("remote-work-ok")
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
        f"Interests: {', '.join(q.get('interests',[]))}\n"
        f"Chronotype: {(q.get('comfort') or {}).get('chronotype','flexible')}\n"
        f"Values: {', '.join(q.get('values',[]))}\n"
        f"MustHaves: {', '.join((q.get('travel_prefs') or {}).get('must_haves',[]))}\n"
        f"Transport: {', '.join((q.get('travel_prefs') or {}).get('transport_allowed',[]))}\n"
        f"RemoteWork: {(q.get('work') or {}).get('remote_work_ok', False)}"
    )

# ----------------------------
# Hard prefilters (unchanged logic + stable)
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
<<<<<<< HEAD
        _EMB = SentenceTransformer("BAAI/bge-m3")
=======
        _EMB = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
>>>>>>> 0e2f315be2dc8ded97ab57f9ae3cf19467983165
    return _EMB

def ai_prefilter(q_user: Dict[str, Any],
                 cands: List[Dict[str, Any]],
                 percent: float = 0.02,
                 min_k: int = 80) -> List[Dict[str, Any]]:
    if not cands:
        return []

    qtext = query_text(q_user)
    summaries = [summarize_user(rec["user"], rec.get("mm")) for rec in cands]
    texts = [qtext] + summaries

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

    # symbolic bonuses
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
        b  += 0.60 * (len(q_interests & ui) / max(1, len(q_interests | ui)))
        b  += 0.30 * (len(q_values & uv) / max(1, len(q_values | uv)))
        c_pace = (u.get("travel_prefs") or {}).get("pace") or "balanced"
        want_same = ((mm.get("soft_preferences") or {}).get("prefer_same_pace") == "prefer_same")
        if want_same and c_pace == q_pace:
            b += 0.20
        langs = set(u.get("languages", []) or [])
        b += 0.20 * min(1, len(q_langs & langs))
        cand_amt = (u.get("budget") or {}).get("amount")
        c_band = _band(cand_amt)
        gap = abs(_band_idx(q_budget_b) - _band_idx(c_band))
        if gap == 0: b += 0.20
        elif gap == 1: b += 0.10
        bonuses.append(b)

    s_min, s_max = min(sims), max(sims)
    sims_norm = [(s - s_min) / (s_max - s_min) if s_max > s_min else 0.5 for s in sims]
    combined = [(rec, sims_norm[i] + bonuses[i]) for i, rec in enumerate(cands)]
    combined.sort(key=lambda x: x[1], reverse=True)

    k = max(min_k, int(math.ceil(len(combined) * percent)))
    k = min(k, len(combined))
    return [rec for rec, _ in combined[:k]]

# ----------------------------
# Local specific reason generator
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
        if not matches:
            return llm_rank.__wrapped__(q_user, shortlist, out_top)  # type: ignore
        return matches[:out_top]
    except Exception:
        return llm_rank.__wrapped__(q_user, shortlist, out_top)  # type: ignore

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
    # 1) Build query user interactively (fun, fast)
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
    shortlist = ai_prefilter(q_user, hard, percent=0.10, min_k=80)

    # 6) Final ranking (OpenAI if key present, else local one-liners)
    final = llm_rank(q_user, shortlist, out_top=5)

    # 7) Print results (no file write)
    print("\n— Top Recommendations —")
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
