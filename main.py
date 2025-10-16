#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RoverMitra Matchmaker – Railway Data + Local Models Only (UI chips: interests/values)

Pipeline:
  Railway Postgres -> Hard Prefilters -> AI Prefilter (Local BGE-M3 + cached embeddings) -> Final Ranking (Local Llama-4bit)

Data Source:
  Fetches data from Railway Postgres database
  Falls back to local JSON files if Railway unavailable

Models (Local Only):
  Uses local BGE-M3 at models/bge-m3 for embeddings
  Uses cached embeddings from models_cache/bge_user_emails.npy and models_cache/bge_embeds_fp16.npy
  Uses local fine-tuned Llama at models/llama-travel-matcher (4-bit on GPU)
  If fine-tuned fails, try base models/llama-3.2-3b-instruct (4-bit on GPU)
  If both fail, fall back to heuristic scorer

First run once:  python build_bge_cache.py
Then run this script normally.

Requires: bitsandbytes, sentence-transformers, transformers, torch (CUDA),
          psycopg2, requests, python-dotenv
"""

from __future__ import annotations
import os, re, json, uuid, math, sys, time
from pathlib import Path
from typing import Any, Dict, List, Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Suppress unimportant warnings
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", message=".*TRANSFORMERS_CACHE.*")
warnings.filterwarnings("ignore", message=".*do_sample.*")
warnings.filterwarnings("ignore", message=".*temperature.*")
warnings.filterwarnings("ignore", message=".*top_p.*")

# Transformer warning control
os.environ["TOKENIZERS_PARALLELISM"] = os.getenv("TOKENIZERS_PARALLELISM", "false")
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = os.getenv("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:512")

# Base dir
BASE_DIR = Path(__file__).resolve().parent

# Environment defaults
os.environ.setdefault("CUDA_VISIBLE_DEVICES", os.getenv("CUDA_VISIBLE_DEVICES", "0"))
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", os.getenv("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1"))
os.environ.setdefault("HF_HOME", os.getenv("HF_HOME", str(BASE_DIR / "models_cache")))
os.environ.setdefault("TRANSFORMERS_CACHE", os.getenv("TRANSFORMERS_CACHE", str(BASE_DIR / "models_cache" / "transformers")))
os.environ.setdefault("TORCH_HOME", os.getenv("TORCH_HOME", str(BASE_DIR / "models_cache" / "torch")))
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", os.getenv("HF_HUB_ENABLE_HF_TRANSFER", "0"))

HF_TOKEN = os.getenv("HF_TOKEN")
if HF_TOKEN:
    os.environ["HF_TOKEN"] = HF_TOKEN

# psutil guard
try:
    import psutil  # noqa
except Exception:
    import types
    dummy = types.SimpleNamespace(
        virtual_memory=lambda: types.SimpleNamespace(total=0),
        cpu_count=lambda: 1,
        Process=lambda *a, **k: types.SimpleNamespace(memory_info=lambda: types.SimpleNamespace(rss=0))
    )
    sys.modules['psutil'] = dummy

# Project paths
USERS_CORE_PATH  = BASE_DIR / "users" / "data" / "users_core.json"
MM_PATH          = BASE_DIR / "MatchMaker" / "data" / "matchmaker_profiles.json"
LOCAL_DB_PATH    = BASE_DIR / "data" / "travel_ready_user_profiles.json"

# Model paths - local
BGE_M3_PATH          = BASE_DIR / "models" / "bge-m3"
LLAMA_FINETUNED_PATH = BASE_DIR / "models" / "llama-travel-matcher"
LLAMA_BASE_PATH      = BASE_DIR / "models" / "llama-3.2-3b-instruct"
CACHE_DIR            = BASE_DIR / "models_cache"
UIDS_PATH            = CACHE_DIR / "bge_user_emails.npy"
EMB_PATH             = CACHE_DIR / "bge_embeds_fp16.npy"

# ML imports
try:
    import numpy as np
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
    from sentence_transformers import SentenceTransformer
    _ML_OK = True
except Exception as e:
    print(f"[warn] ML stack import failed: {e}", file=sys.stderr)
    _ML_OK = False

# Utilities
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

# Vocab aids
LANG_ALIASES = {
    "english":"en","en":"en","german":"de","deutsch":"de","de":"de","french":"fr","français":"fr","francais":"fr","fr":"fr",
    "italian":"it","italiano":"it","it":"it","spanish":"es","español":"es","es":"es","portuguese":"pt","português":"pt","pt":"pt",
    "dutch":"nl","netherlands":"nl","nl":"nl","polish":"pl","pl":"pl","russian":"ru","ru":"ru","czech":"cs","cz":"cs","cs":"cs",
    "danish":"da","da":"da","swedish":"sv","sv":"sv","norwegian":"no","no":"no","finnish":"fi","fi":"fi",
    "hindi":"hi","hi":"hi","urdu":"ur","ur":"ur","punjabi":"pa","pa":"pa","bengali":"bn","bangla":"bn","bn":"bn",
    "tamil":"ta","ta":"ta","telugu":"te","te":"te","marathi":"mr","mr":"mr","gujarati":"gu","gu":"gu","malayalam":"ml","ml":"ml",
    "sinhala":"si","si":"si","nepali":"ne","ne":"ne","arabic":"ar","ar":"ar","farsi":"fa","persian":"fa","fa":"fa","turkish":"tr","tr":"tr",
    "hebrew":"he","he":"he","japanese":"ja","jp":"ja","ja":"ja","korean":"ko","kr":"ko","ko":"ko","chinese":"zh","mandarin":"zh","zh":"zh",
    "thai":"th","th":"th","vietnamese":"vi","vi":"vi","malay":"ms","ms":"ms","indonesian":"id","id":"id","filipino":"tl","tagalog":"tl","tl":"tl",
}
DIETS = {"none","vegetarian","vegan","halal","kosher","gluten-free","no pork","lactose-free","pescatarian"}

FAITH_LABELS = ["Islam","Hindu","Christian","Jewish","Buddhist","Sikh","Other","Prefer not to say"]

def _faith_slug(s: str) -> str:
    return (s or "").strip().lower().replace(" ", "_")

# CLI helpers
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
            return int(v)
        except:
            print("Please enter a number.")

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
        out, ok = [], True
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
        print("Please choose valid numbers or values within range.")

def clean_list_csv(s: str) -> List[str]:
    return [x.strip() for x in (s or "").split(",") if x.strip()]

# Interactive profile builder
def interactive_new_user() -> Dict[str, Any]:
    print("\nWelcome. Quick onboarding.\n")

    print("Basics")
    name = ask("Your name", "Alex Rivera")
    age  = ask_int("Age", 27)
    gender = ask_choice("How do you identify?", ["Male","Female","Non-binary","Other"], "Other")
    city = ask("Home city", "Berlin")
    country = ask("Country", "Germany")

    print("\nLanguages")
    langs_in = clean_list_csv(ask("Languages you can chat in (comma)", "en, de"))
    langs = []
    for w in langs_in:
        key = w.lower()
        langs.append(LANG_ALIASES.get(key, key))
    langs = sorted(set([x for x in langs if x]))

    print("\nInterests")
    INTERESTS = [
        "museum hopping","architecture walks","history sites","city photography","food tours","street food",
        "coffee crawls","vineyards","scenic trains","short hikes","long hikes","mountain hiking","lake swims",
        "beach days","markets","old towns","rooftop views","local crafts","live music","festivals"
    ]
    interests = ask_multi("What sounds fun right now?", INTERESTS, 3, 6, default_idxs=[0,1,3])

    print("\nTempo")
    pace = ask_choice("Trip pace", ["relaxed","balanced","packed"], "balanced")
    chronotype = ask_choice("Most alive", ["early bird","flexible","night owl"], "flexible")

    print("\nBudget")
    budget = ask_int("Comfortable budget per day (€)", 150)
    split_rule = ask_choice("Split shared costs", ["each_own","50/50","custom"], "each_own")

    print("\nFood")
    diet = ask_choice("Diet", sorted(list(DIETS)), "none")
    allergies_raw = clean_list_csv(ask("Allergies? or 'none'", "none"))
    allergies = ["none"] if (not allergies_raw or allergies_raw == ["none"]) else allergies_raw

    print("\nComfort")
    smoking = ask_choice("Smoking", ["never","occasionally","regular"], "never")
    alcohol = ask_choice("Alcohol", ["none","moderate","social"], "moderate")
    noise = ask_choice("Noise tolerance", ["low","medium","high"], "medium")
    clean = ask_choice("Cleanliness preference", ["low","medium","high"], "medium")
    risk  = ask_choice("Risk tolerance", ["low","medium","high"], "medium")

    print("\nWork")
    remote_ok = ask_yesno("Need time to work while traveling?")
    hours_online = ask_choice("Hours online per day", ["0","1","2"], "0")
    wifi_need = ask_choice("Wi-Fi needs", ["normal","good","excellent"], "good")

    print("\nStay must-haves")
    MUSTS = ["wifi","private_bath","kitchen","workspace","near_station","quiet_room","breakfast"]
    must_haves = ask_multi("Pick 1–4 must-haves", MUSTS, 1, 4, default_idxs=[0,3])

    print("\nRoom and transport")
    room_setup = ask_choice("Room setup when sharing", ["twin","double","2 rooms","dorm"], "twin")
    TRANSPORT = ["train","plane","bus","car"]
    transport_allowed = ask_multi("Allowed transport modes", TRANSPORT, 1, 3, default_idxs=[0,1])

    print("\nTrip styles")
    STYLES = ["weekend getaway","co-work week","hiking basecamp","city sampler","island hop","festival trip","road trip"]
    trip_styles = ask_multi("Pick a couple", STYLES, 1, 3, default_idxs=[0,3])

    print("\nValues")
    VALUES = ["adventure","stability","learning","family","budget-minded","luxury-taste","nature","culture","community","fitness","spirituality"]
    values = ask_multi("Choose 2–3", VALUES, 2, 3, default_idxs=[0,6])

    print("\nCompanions")
    companion_pref = ask_choice("Who to travel with?", ["I'm open to travel with anyone", "Men", "Women", "Nonbinary travelers"], "I'm open to travel with anyone")

    print("\nSafety")
    share_home_city = ask_yesno("OK to show your home city on profile?", True)
    pre_meet_video  = ask_yesno("Comfortable with a short pre-trip video call?", True)

    print("\nOptional faith")
    consider_faith = ask_yesno("Consider faith when matching? Optional and private", False)
    faith_block = {"consider_in_matching": False, "religion": "", "policy": "open", "visibility": "private"}
    if consider_faith:
        faith_block["consider_in_matching"] = True
        faith_block["policy"] = ask_choice("How to consider it?", ["open", "prefer_same", "same_only"], "prefer_same")
        faith_pick = ask_choice("If comfortable, which faith?", FAITH_LABELS, "Prefer not to say")
        faith_block["religion"] = "" if faith_pick == "Prefer not to say" else faith_pick

    bio = ask("One-liner about you", "")

    user = {
        "email": f"local_{uuid.uuid4().hex[:8]}@example.com",
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
            "accommodation_types": ["hotel","apartment"],
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
        "companion_preferences": {"genders_ok": [companion_pref]},
        "faith": faith_block,
        "privacy": {
            "share_profile_with_matches": True,
            "share_home_city": share_home_city,
            "pre_meet_video_call_ok": pre_meet_video
        }
    }
    print("\nProfile captured.\n")
    return user

# Data load from Railway or local
def load_pool():
    try:
        import psycopg2
        from datetime import datetime

        DATABASE_URL = os.getenv("DATABASE_URL") or "postgresql://postgres:YOUR_PASSWORD@YOUR_HOST:YOUR_PORT/YOUR_DATABASE"
        print("Connecting to Railway Postgres...")
        conn = psycopg2.connect(DATABASE_URL)
        cursor = conn.cursor()

        cursor.execute("""
            SELECT "Id", "Email", "FirstName", "LastName", "MiddleName", 
                   "DateOfBirth", "Address", "City", "State", "PostalCode", 
                   "Country", "CreatedAt", "UpdatedAt", "IsActive", 
                   "IsEmailVerified", "IsPhoneVerified", "ProfilePictureUrl",
                   "UserName", "PhoneNumber", "IsProfileComplete", 
                   "IsIdVerified", "IdVerifiedAt", "Gender"
            FROM "Users"
        """)
        columns = [desc[0] for desc in cursor.description]
        user_rows = cursor.fetchall()

        def as_dict(row):
            out = {}
            for i, v in enumerate(row):
                if hasattr(v, "isoformat"):
                    out[columns[i]] = v.isoformat()
                else:
                    out[columns[i]] = v
            return out

        users_raw = [as_dict(r) for r in user_rows]

        cursor.execute("""
            SELECT id, match_profile_id, email, status, created_at, updated_at,
                   visibility, preferences, compatibility_scores, raw_data
            FROM matchmaker_profiles
        """)
        mm_columns = [desc[0] for desc in cursor.description]
        mm_rows = cursor.fetchall()

        def as_mm_dict(row):
            d = {}
            for i, v in enumerate(row):
                col = mm_columns[i]
                if col in ["visibility","preferences","compatibility_scores","raw_data"]:
                    if v:
                        try:
                            d[col] = json.loads(v) if isinstance(v, str) else v
                        except json.JSONDecodeError:
                            d[col] = str(v)
                    else:
                        d[col] = None
                elif hasattr(v, "isoformat"):
                    d[col] = v.isoformat()
                else:
                    d[col] = v
            return d

        mm_profiles = [as_mm_dict(r) for r in mm_rows]

        cursor.close()
        conn.close()

        pool = []
        mm_by_email = index_by(mm_profiles, "email")
        for ur in users_raw:
            email = ur.get("Email", "")
            if not email:
                continue
            mm = mm_by_email.get(email)
            # Extract rich data from Railway matchmaker profile
            raw_data = mm.get("raw_data", {}) if mm else {}
            user_profile = raw_data.get("user_profile", {})
            matchmaker_prefs = raw_data.get("matchmaker_preferences", {})
            
            # Extract languages from Railway data
            languages = matchmaker_prefs.get("language_policy", {}).get("preferred_chat_languages", ["en"])
            
            # Extract interests and values from Railway data
            interests = user_profile.get("interests", [])
            values = user_profile.get("values", [])
            
            # Extract budget information
            budget_amount = user_profile.get("budget", {}).get("amount", 100)
            budget_currency = user_profile.get("budget", {}).get("currency", "EUR")
            
            # Extract travel preferences
            pace = user_profile.get("travel_prefs", {}).get("pace", "balanced")
            chronotype = user_profile.get("comfort", {}).get("chronotype", "flexible")
            
            # Extract diet and health preferences
            diet = user_profile.get("diet_health", {}).get("diet", "none")
            allergies = user_profile.get("diet_health", {}).get("allergies", ["none"])
            
            # Extract comfort preferences
            smoking = user_profile.get("comfort", {}).get("smoking", "never")
            alcohol = user_profile.get("comfort", {}).get("alcohol", "moderate")
            risk_tolerance = user_profile.get("comfort", {}).get("risk_tolerance", "medium")
            noise_tolerance = user_profile.get("comfort", {}).get("noise_tolerance", "medium")
            cleanliness = user_profile.get("comfort", {}).get("cleanliness_preference", "medium")
            
            # Extract work preferences
            remote_work_ok = user_profile.get("work", {}).get("remote_work_ok", True)
            hours_online = user_profile.get("work", {}).get("hours_online_needed", 0)
            wifi_quality = user_profile.get("work", {}).get("wifi_quality_needed", "good")
            
            # Extract companion preferences
            companion_genders = matchmaker_prefs.get("preferred_companion", {}).get("genders", ["any"])
            
            # Extract faith preferences
            faith_data = user_profile.get("faith", {})
            faith_consider = faith_data.get("consider_in_matching", False)
            faith_religion = faith_data.get("religion", "")
            faith_policy = faith_data.get("policy", "open")
            
            # Extract match intent and trip styles
            match_intent = matchmaker_prefs.get("match_intent", [])
            
            # Extract accommodation and transport preferences
            accommodation_types = user_profile.get("travel_prefs", {}).get("accommodation_types", ["hotel", "apartment"])
            transport_allowed = user_profile.get("travel_prefs", {}).get("transport_allowed", ["train", "plane"])
            must_haves = user_profile.get("travel_prefs", {}).get("must_haves", [])
            room_setup = user_profile.get("travel_prefs", {}).get("room_setup", "twin")
            
            # Extract bio or create dynamic one
            bio = user_profile.get("bio", f"Travel enthusiast from {ur.get('City','')}, {ur.get('Country','')}")
            
            converted_user = {
                "email": email,
                "name": f"{ur.get('FirstName','')} {ur.get('LastName','')}".strip(),
                "age": calculate_age_from_dob(ur.get("DateOfBirth")),
                "gender": ur.get("Gender", "Other"),
                "home_base": {"city": ur.get("City",""), "country": ur.get("Country","")},
                "languages": languages,
                "interests": interests,
                "values": values,
                "budget": {"amount": budget_amount, "currency": budget_currency},
                "bio": bio,
                "travel_prefs": {
                    "pace": pace,
                    "accommodation_types": accommodation_types,
                    "room_setup": room_setup,
                    "transport_allowed": transport_allowed,
                    "must_haves": must_haves
                },
                "diet_health": {"diet": diet, "allergies": allergies},
                "comfort": {
                    "smoking": smoking,
                    "alcohol": alcohol,
                    "risk_tolerance": risk_tolerance,
                    "noise_tolerance": noise_tolerance,
                    "cleanliness_preference": cleanliness,
                    "chronotype": chronotype
                },
                "work": {
                    "remote_work_ok": remote_work_ok,
                    "hours_online_needed": hours_online,
                    "wifi_quality_needed": wifi_quality
                },
                "companion_preferences": {"genders_ok": companion_genders},
                "faith": {
                    "consider_in_matching": faith_consider,
                    "religion": faith_religion,
                    "policy": faith_policy
                },
                "match_intent": match_intent,
                "railway_raw_data": raw_data  # Keep original data for reference
            }
            pool.append({"user": converted_user, "mm": mm})
        print(f"Pulled {len(pool)} candidates from Railway.")
        return pool

    except Exception as e:
        print(f"Railway load failed: {e}")
        print("Falling back to local files.")
        users = load_json(USERS_CORE_PATH)
        mm    = load_json(MM_PATH)
        mm_by_email = index_by(mm, "email")
        pool = []
        for u in users:
            email = u.get("email")
            if not email:
                continue
            pool.append({"user": u, "mm": mm_by_email.get(email)})
        return pool

def calculate_age_from_dob(dob_str: str) -> int:
    try:
        from datetime import datetime
        if isinstance(dob_str, str) and dob_str:
            dob = datetime.fromisoformat(dob_str.replace('Z', '+00:00'))
        else:
            return 28
        today = datetime.now()
        age = today.year - dob.year - ((today.month, today.day) < (dob.month, dob.day))
        return max(18, min(age, 80))
    except:
        return 28

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
    parts.append(f"alcohol={cmf.get('alcohol','moderate')}")
    pace = (u.get("travel_prefs") or {}).get("pace") or "balanced"
    parts.append(f"pace={pace}")
    intr = u.get("interests") or []
    parts.append("interests=" + ",".join(intr[:10]))
    vals = u.get("values") or []
    if vals: parts.append("values=" + ",".join(vals[:5]))
    faith_info = u.get("faith") or {}
    if faith_info.get("consider_in_matching"):
        faith_policy = faith_info.get("policy", "open")
        faith_religion = faith_info.get("religion", "")
        if faith_religion:
            parts.append(f"faith={faith_religion}({faith_policy})")
    return " | ".join(parts)

def query_text(q: Dict[str,Any]) -> str:
    hb = q.get("home_base") or {}
    fq = q.get("faith") or {}
    faith_str = ""
    if fq.get("consider_in_matching"):
        fp = fq.get("policy","open")
        fr = fq.get("religion","")
        faith_str = f"\nFaithPolicy: {fp}\nFaith: {fr}"
    return (
        f"Name: {q.get('name')}\n"
        f"Age: {q.get('age')}\n"
        f"Gender: {q.get('gender')}\n"
        f"Location: {hb.get('city','')} {hb.get('country','')}\n"
        f"Languages: {', '.join(q.get('languages',[]))}\n"
        f"Pace: {q.get('travel_prefs',{}).get('pace','balanced')}\n"
        f"BudgetPerDay: {(q.get('budget') or {}).get('amount')}\n"
        f"Diet: {(q.get('diet_health') or {}).get('diet','none')}\n"
        f"Alcohol: {(q.get('comfort') or {}).get('alcohol','moderate')}\n"
        f"Interests: {', '.join(q.get('interests',[]))}\n"
        f"Values: {', '.join(q.get('values',[]))}\n"
        f"MustHaves: {', '.join((q.get('travel_prefs') or {}).get('must_haves',[]))}\n"
        f"Transport: {', '.join((q.get('travel_prefs') or {}).get('transport_allowed',[]))}\n"
        f"RemoteWork: {(q.get('work') or {}).get('remote_work_ok', False)}\n"
        f"Bio: {q.get('bio', '')}{faith_str}"
    )

# --------- UI chips helper (interests/values) ----------
def ui_interests_values(q_user: Dict[str, Any], cand_user: Dict[str, Any], n: int = 4) -> List[str]:
    """
    Return up to n short chips drawn from candidate 'interests', preferring overlap with the query user's interests.
    If not enough, fill from candidate 'values'. Nicely titled for UI.
    """
    def tidy(s: str) -> str:
        s = re.sub(r"\s+", " ", s).strip().lower()
        # consistent short chips like "Hiking", "Photography", "Food tours"
        return " ".join(w if w in {"and","of","to","in"} else w.capitalize() for w in s.split())

    qi = [s for s in (q_user.get("interests") or []) if isinstance(s, str)]
    ui = [s for s in (cand_user.get("interests") or []) if isinstance(s, str)]
    uv = [s for s in (cand_user.get("values") or []) if isinstance(s, str)]

    # preserve candidate order but prioritize overlaps
    overlap = [i for i in ui if i in qi]
    non_overlap = [i for i in ui if i not in overlap]
    chips = overlap + non_overlap
    if len(chips) < n:
        # fill from values, avoiding duplicates
        for v in uv:
            if v not in chips:
                chips.append(v)
            if len(chips) >= n:
                break

    chips = chips[:n]
    return [tidy(c) for c in chips]

# Hard prefilters
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

def companion_gender_ok(query_companion_pref: str, cand_user: Dict[str,Any]) -> bool:
    if not query_companion_pref:
        return True
    cand_gender = (cand_user.get("gender") or "").lower()
    if query_companion_pref.lower() == "i'm open to travel with anyone":
        return True
    elif query_companion_pref.lower() == "men":
        return cand_gender in ["male", "man"]
    elif query_companion_pref.lower() == "women":
        return cand_gender in ["female", "woman"]
    elif query_companion_pref.lower() == "nonbinary travelers":
        return cand_gender in ["non-binary", "nonbinary", "other"]
    return True

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

def faith_ok(q: Dict[str,Any], cand_user: Dict[str,Any], cand_mm: Dict[str,Any]) -> bool:
    qf = (q.get("faith") or {})
    consider = bool(qf.get("consider_in_matching"))
    q_token = _faith_slug(qf.get("religion") or "")

    mm = cand_mm or {}
    cand_fp = (mm.get("faith_preference") or {})
    c_token = cand_fp.get("religion_token", "")

    cand_requires_same = "same_faith_required" in (mm.get("hard_dealbreakers") or [])

    if cand_requires_same:
        return bool(q_token and c_token and q_token == c_token)

    if consider and qf.get("policy") == "same_only":
        return bool(q_token and c_token and q_token == c_token)

    return True

def hard_prefilter(q: Dict[str,Any], pool: List[Dict[str,Any]]) -> List[Dict[str,Any]]:
    out = []
    q_age = int(q.get("age", 30))
    q_gender = q.get("gender","Other")
    q_langs = q.get("languages",[])
    q_budget = (q.get("budget") or {}).get("amount", 150)
    q_pace = (q.get("travel_prefs") or {}).get("pace","balanced")
    q_companion_prefs = q.get("companion_preferences", {})
    q_companion_gender = q_companion_prefs.get("genders_ok", ["I'm open to travel with anyone"])[0] if q_companion_prefs.get("genders_ok") else "I'm open to travel with anyone"

    for rec in pool:
        cu, cm = rec["user"], rec["mm"]
        if not langs_ok(q_langs, cm, cu):           continue
        if not age_ok(q_age, cu, cm):               continue
        if not gender_ok(q_gender, cm):             continue
        if not companion_gender_ok(q_companion_gender, cu): continue
        if not budget_ok(q_budget, cu):             continue
        if not pace_ok(q_pace, cu, cm):             continue
        if not faith_ok(q, cu, cm):                 continue
        out.append(rec)
    return out

# AI prefilter with BGE cache or heuristic fallback
_EMB: Optional[SentenceTransformer] = None
_cached_ids = None
_cached_embs = None

def ensure_emb(device: str = "cuda"):
    global _EMB
    if not _ML_OK:
        return None
    if _EMB is None:
        dev = device if (hasattr(torch, "cuda") and torch.cuda.is_available()) else "cpu"
        print(f"Loading BGE-M3 embedding model from local path: {BGE_M3_PATH}...")
        _EMB = SentenceTransformer(str(BGE_M3_PATH), device=dev)
        print("BGE-M3 embedding model loaded.")
    return _EMB

def _load_bge_cache():
    global _cached_ids, _cached_embs
    if _cached_ids is None or _cached_embs is None:
        if not (UIDS_PATH.exists() and EMB_PATH.exists()):
            raise RuntimeError("BGE cache missing. Run: python build_bge_cache.py")
        _cached_ids  = np.load(UIDS_PATH, allow_pickle=True)
        _cached_embs = np.load(EMB_PATH,  allow_pickle=False)   # float16 [N,D]
    return _cached_ids, _cached_embs

def _heuristic_shortlist(q_user: Dict[str, Any], cands: List[Dict[str, Any]], percent: float, min_k: int) -> List[Dict[str, Any]]:
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

    qf = (q_user.get("faith") or {})
    q_token = _faith_slug(qf.get("religion") or "")
    q_policy = qf.get("policy","open") if qf.get("consider_in_matching") else "open"

    scored = []
    for rec in cands:
        u = rec["user"]
        mm = rec.get("mm") or {}
        ui  = set(u.get("interests", []) or [])
        uv  = set(u.get("values", []) or [])
        score = 0.0
        score += 0.60 * (len(q_interests & ui) / max(1, len(q_interests | ui)))
        score += 0.30 * (len(q_values & uv) / max(1, len(q_values | uv)))
        c_pace = (u.get("travel_prefs") or {}).get("pace") or "balanced"
        want_same = ((mm.get("soft_preferences") or {}).get("prefer_same_pace") == "prefer_same")
        if want_same and c_pace == q_pace:
            score += 0.20
        langs = set(u.get("languages", []) or [])
        score += 0.20 * min(1, len(q_langs & langs))
        cand_amt = (u.get("budget") or {}).get("amount")
        c_band = _band(cand_amt)
        gap = abs(_band_idx(q_budget_b) - _band_idx(c_band))
        if gap == 0: score += 0.20
        elif gap == 1: score += 0.10

        c_token = ((mm.get("faith_preference") or {}).get("religion_token") or "")
        if q_policy in {"prefer_same","same_only"} and q_token and c_token and q_token == c_token:
            score += 0.15

        scored.append((rec, score))

    scored.sort(key=lambda x: x[1], reverse=True)
    k = max(min_k, int(math.ceil(len(scored) * percent)))
    k = min(k, len(scored))
    return [rec for rec, _ in scored[:k]]

def ai_prefilter(q_user: Dict[str, Any], cands: List[Dict[str, Any]], percent: float = 0.02, min_k: int = 80) -> List[Dict[str, Any]]:
    if not cands:
        return []
    if not _ML_OK:
        print("[info] ML stack unavailable; using heuristic shortlist.")
        return _heuristic_shortlist(q_user, cands, percent, min_k)

    try:
        ids, embs = _load_bge_cache()
    except Exception as e:
        print(f"[warn] {e}  Using heuristic shortlist instead.")
        return _heuristic_shortlist(q_user, cands, percent, min_k)

    email2idx = {email: i for i, email in enumerate(ids)}
    cand_emails = [rec["user"].get("email") for rec in cands]
    cand_row_idx = [email2idx.get(e) for e in cand_emails]

    model = ensure_emb()
    if model is None:
        print("[info] No embedding model; using heuristic shortlist.")
        return _heuristic_shortlist(q_user, cands, percent, min_k)

    qv = model.encode(
        [query_text(q_user)],
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False
    )[0].astype("float16")

    valid_pairs = [(i, idx) for i, idx in enumerate(cand_row_idx) if idx is not None]
    sims_full = np.full(len(cands), 0.30, dtype="float32")
    if valid_pairs:
        sub = embs[[idx for (_, idx) in valid_pairs]]
        sims = (sub @ qv).astype("float32")
        s_min, s_max = float(sims.min()), float(sims.max())
        sims_norm = (sims - s_min) / (s_max - s_min) if s_max > s_min else np.full_like(sims, 0.5, dtype="float32")
        for (pos, _), val in zip(valid_pairs, sims_norm):
            sims_full[pos] = float(val)

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

    qf = (q_user.get("faith") or {})
    q_token = _faith_slug(qf.get("religion") or "")
    q_policy = qf.get("policy","open") if qf.get("consider_in_matching") else "open"

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
        c_token = ((mm.get("faith_preference") or {}).get("religion_token") or "")
        if q_policy in {"prefer_same","same_only"} and q_token and c_token and q_token == c_token:
            b += 0.15
        bonuses.append(b)

    combined = [(cands[i], float(sims_full[i]) + float(bonuses[i])) for i in range(len(cands))]
    combined.sort(key=lambda x: x[1], reverse=True)

    k = max(min_k, int(math.ceil(len(combined) * percent)))
    k = min(k, len(combined))
    return [rec for rec, _ in combined[:k]]

# Llama ranking
import requests
_LLM_FINETUNED = None
_LLM_BASE = None
_SERVER_AVAILABLE = None

def check_server_availability():
    global _SERVER_AVAILABLE
    if _SERVER_AVAILABLE is not None:
        return _SERVER_AVAILABLE
    try:
        response = requests.get("http://localhost:8002/health", timeout=5)
        if response.status_code == 200:
            data = response.json()
            _SERVER_AVAILABLE = data.get("status") == "ok" and data.get("models_loaded", False)
        else:
            _SERVER_AVAILABLE = False
    except Exception:
        _SERVER_AVAILABLE = False
    return _SERVER_AVAILABLE

def get_llama_ranking_server(prompt, max_new_tokens=600, temperature=0.2, top_p=0.9):
    try:
        response = requests.post(
            "http://localhost:8002/rank",
            json={"prompt": prompt, "max_new_tokens": max_new_tokens, "temperature": temperature, "top_p": top_p},
            timeout=30
        )
        return response.json()["text"]
    except Exception as e:
        print(f"Llama server error: {e}")
        return None

def _load_llama_4bit(model_path: Path):
    bnb_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    print(f"Loading Llama model from local path: {model_path}...")
    tok = AutoTokenizer.from_pretrained(str(model_path), use_fast=True)
    mdl = AutoModelForCausalLM.from_pretrained(
        str(model_path),
        quantization_config=bnb_cfg,
        device_map="auto",
        torch_dtype=torch.float16,
        trust_remote_code=True,
    )
    mdl.config.use_cache = True
    if mdl.config.pad_token_id is None and tok.eos_token_id is not None:
        mdl.config.pad_token_id = tok.eos_token_id
    return {"tokenizer": tok, "model": mdl}

def ensure_llm_finetuned():
    global _LLM_FINETUNED
    if _ML_OK and _LLM_FINETUNED is None:
        try:
            print("Loading fine-tuned Llama (4-bit) from local path...")
            _LLM_FINETUNED = _load_llama_4bit(LLAMA_FINETUNED_PATH)
            print("Fine-tuned Llama loaded from local path")
        except Exception as e:
            print(f"[warn] Fine-tuned Llama load failed: {e}")
            _LLM_FINETUNED = None
    return _LLM_FINETUNED

def ensure_llm_base():
    global _LLM_BASE
    if _ML_OK and _LLM_BASE is None:
        try:
            print("Loading base Llama (4-bit) from local path...")
            _LLM_BASE = _load_llama_4bit(LLAMA_BASE_PATH)
            print("Base Llama loaded from local path")
        except Exception as e:
            print(f"[warn] Base Llama load failed: {e}")
            _LLM_BASE = None
    return _LLM_BASE

def get_llama_ranking_local(prompt, max_new_tokens=120, temperature=0.0, top_p=0.9):
    model_data = ensure_llm_finetuned()
    if model_data is None:
        model_data = ensure_llm_base()
    if model_data is None:
        return None
    tok, mdl = model_data["tokenizer"], model_data["model"]
    inputs = tok(prompt, return_tensors="pt", truncation=True, max_length=4096)
    inputs = {k: v.to(mdl.device) for k, v in inputs.items()}
    try:
        with torch.inference_mode():
            out = mdl.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=(temperature > 0.0),
                temperature=temperature,
                top_p=top_p,
                use_cache=True,
                pad_token_id=tok.eos_token_id,
                eos_token_id=tok.eos_token_id,
            )
        gen = out[0][inputs["input_ids"].shape[1]:]
        text = tok.decode(gen, skip_special_tokens=True)
        return text
    except Exception as e:
        print(f"[warn] Local LLM generation failed: {e}")
        return None

def get_llama_ranking(prompt, max_new_tokens=600, temperature=0.2, top_p=0.9):
    if check_server_availability():
        print("Using Llama API server...")
        result = get_llama_ranking_server(prompt, max_new_tokens, temperature, top_p)
        if result is not None:
            return result
        else:
            print("API server failed, falling back to local model...")
    print("Using local Llama model...")
    return get_llama_ranking_local(prompt, max_new_tokens=120, temperature=0.0, top_p=0.9)

def build_llm_prompt(q_user: Dict[str,Any], shortlist: List[Dict[str,Any]], top_k:int=5) -> str:
    head = (
    "ROLE: You are a discerning travel-matchmaker. Rank candidates for holistic trip compatibility.\n"
    "\n"
    "EVALUATE ON (in order):\n"
    "• Shared & complementary interests/activities (what they'd actually do together)\n"
    "• Pace fit (relaxed/balanced/packed) and daily rhythm (chronotype)\n"
    "• Budget band compatibility (budget/mid/lux) and split expectations\n"
    "• Language overlap (count of shared chat languages)\n"
    "• Diet & alcohol/smoking compatibility; allergies (hard constraints)\n"
    "• Transport allowed, accommodation/room setup, and must-have amenities (wifi, workspace, near_station, etc.)\n"
    "• Remote-work needs (hours online, wifi quality)\n"
    "• Comfort & psychology signals: risk tolerance, noise/cleanliness preferences, values (e.g., culture, nature, community)\n"
    "• Faith policy: if query is 'same_only', enforce same faith; if 'prefer_same', treat as a bonus. Keep faith private—do not disclose it in text.\n"
    "\n"
    "OUTPUT: Return ONLY valid JSON with key 'matches' as an array of objects:\n"
    "  { email, name, explanation, compatibility_score }\n"
    "\n"
    "EXPLANATION STYLE (STRICT):\n"
    "• ONE sentence, MUST start with 'For you,' and be 18–28 words.\n"
    "• Include: (1) 2 concrete shared interests, (2) the pace word, (3) budget band term (budget/mid/lux),\n"
    "  and at least one of {language overlap, diet match, transport/amenity fit, remote-work fit}.\n"
    "• Be specific and activity-oriented; avoid vague traits like 'friendly', 'nice', 'open-minded', or generic buzzwords.\n"
    "• Do NOT mention demographics (age/gender) or private details; do NOT output lists or extra text—only the JSON.\n"
    "\n"
    "SCORING:\n"
    "• compatibility_score is a float 0.00–1.00 (two decimals), calibrated by the above criteria; higher means better trip fit.\n"
    "\n"
    "Query User:\n"
    f"{query_text(q_user)}\n\n"
    "Candidates:\n"
    )

    body = []
    for i, rec in enumerate(shortlist):
        u, m = rec["user"], rec["mm"]
        body.append(f"[{i+1}] email={u.get('email')} | {summarize_user(u, m)}")
    return head + "\n".join(body)

def _extract_json(text: str) -> Optional[dict]:
    start = text.find("{")
    end   = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    chunk = text[start:end+1]
    try:
        return json.loads(chunk)
    except Exception:
        for j in range(end, start, -1):
            try:
                return json.loads(text[start:j])
            except Exception:
                continue
    return None

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
    
    # Extract additional dynamic data from Railway
    raw_data = u.get("railway_raw_data", {})
    match_intent = raw_data.get("match_intent", [])
    chronotype_u = (u.get("comfort") or {}).get("chronotype", "flexible")
    chronotype_q = (q.get("comfort") or {}).get("chronotype", "flexible")
    transport_u = (u.get("travel_prefs") or {}).get("transport_allowed", [])
    transport_q = (q.get("travel_prefs") or {}).get("transport_allowed", [])
    shared_transport = set(transport_u) & set(transport_q)
    must_haves_u = (u.get("travel_prefs") or {}).get("must_haves", [])
    must_haves_q = (q.get("travel_prefs") or {}).get("must_haves", [])
    shared_must_haves = set(must_haves_u) & set(must_haves_q)
    
    # Extract work compatibility
    work_remote_u = (u.get("work") or {}).get("remote_work_ok", True)
    work_remote_q = (q.get("work") or {}).get("remote_work_ok", True)
    work_hours_u = (u.get("work") or {}).get("hours_online_needed", 0)
    work_hours_q = (q.get("work") or {}).get("hours_online_needed", 0)
    
    # Extract values compatibility
    qv = set(q.get("values",[]))
    uv = set(u.get("values",[]))
    shared_values = sorted(qv & uv)

    hooks = []
    
    # Match intent compatibility
    if match_intent:
        hooks.append(f"shared interest in {', '.join(match_intent[:2])}")
    
    # Shared interests
    if shared_i: 
        hooks.append(f"shared love for {', '.join(shared_i[:2])}")
    
    # Language compatibility
    if langs:    
        hooks.append(f"you both speak {', '.join(langs[:2])}")
    
    # Pace compatibility
    if pace_q == pace_u: 
        hooks.append(f"matching {pace_q} pace")
    
    # Chronotype compatibility
    if chronotype_q == chronotype_u and chronotype_u != "flexible":
        hooks.append(f"both {chronotype_u}")
    
    # Budget compatibility
    if budget_gap <= 30: 
        hooks.append("similar daily budgets")
    
    # Diet compatibility
    if diet_u != "none" and diet_u == diet_q: 
        hooks.append(f"both {diet_u}")
    
    # Transport compatibility
    if shared_transport:
        hooks.append(f"prefer {', '.join(list(shared_transport)[:2])} travel")
    
    # Must-haves compatibility
    if shared_must_haves:
        hooks.append(f"both need {', '.join(list(shared_must_haves)[:2])}")
    
    # Work compatibility
    if work_remote_u == work_remote_q and work_remote_q:
        if abs(work_hours_u - work_hours_q) <= 1:
            hooks.append("compatible work schedules")
    
    # Values compatibility
    if shared_values:
        hooks.append(f"shared values like {', '.join(shared_values[:2])}")
    
    # Location
    if city_u:   
        hooks.append(f"and they are based in {city_u}")
    
    # Fallback
    if not hooks:
        hooks.append("complementary interests and compatible travel habits")
    
    return "For you, this match fits because of " + ", ".join(hooks) + "."

def llm_rank_fallback(q_user: Dict[str,Any], shortlist: List[Dict[str,Any]], out_top:int=5) -> List[Dict[str,Any]]:
    results = []
    for rec in shortlist[:out_top]:
        u = rec["user"]
        reason = craft_specific_reason(q_user, u, rec.get("mm"))
        base = 0.60
        bonus = 0.25 * jaccard(q_user.get("interests",[]), u.get("interests",[]))
        lang_bonus = 0.05 * min(1, len(set(q_user.get("languages",[])) & set(u.get("languages",[]))))
        pace_bonus = 0.05 if (q_user.get("travel_prefs",{}).get("pace","balanced") ==
                              (u.get("travel_prefs") or {}).get("pace","balanced")) else 0.0
        score = min(max(base + bonus + lang_bonus + pace_bonus, 0.0), 0.95)
        results.append({
            "email": u.get("email"),
            "name": u.get("name"),
            "explanation": reason,
            "compatibility_score": round(score, 2)
        })
    return results

def llm_rank(q_user: Dict[str,Any], shortlist: List[Dict[str,Any]], out_top:int=5) -> List[Dict[str,Any]]:
    shortlist_for_llm = shortlist[:10]
    system_prompt = "Return ONLY valid JSON with key 'matches'. No explanations. No extra text."
    prompt = build_llm_prompt(q_user, shortlist_for_llm, top_k=out_top)
    full_prompt = f"System: {system_prompt}\nUser: {prompt}\nAssistant:"
    text = get_llama_ranking(full_prompt, max_new_tokens=120, temperature=0.0, top_p=0.9)

    if text is None:
        print("[warn] Llama server unavailable, using fallback")
        return llm_rank_fallback(q_user, shortlist_for_llm, out_top)

    try:
        parsed = _extract_json(text)
        if parsed and isinstance(parsed.get("matches", None), list):
            cleaned = []
            for m in parsed["matches"][:out_top]:
                try:
                    cleaned.append({
                        "email": str(m.get("email","")),
                        "name": str(m.get("name","")),
                        "explanation": str(m.get("explanation","")),
                        "compatibility_score": float(m.get("compatibility_score", 0.0))
                    })
                except Exception:
                    continue
            if cleaned:
                return cleaned
    except Exception as e:
        print(f"[warn] LLM response parsing failed: {e}")

    return llm_rank_fallback(q_user, shortlist_for_llm, out_top)

# Run
def main():
    q_user = interactive_new_user()
    append_to_json(q_user, LOCAL_DB_PATH)
    print(f"Saved your profile to {LOCAL_DB_PATH}")

    pool = load_pool()
    if not pool:
        print("No candidates found. Provide users_core.json and matchmaker_profiles.json.")
        return

    t0 = time.time()
    hard = hard_prefilter(q_user, pool)
    print(f"Hard prefilter: {len(hard)} candidates ({time.time()-t0:.2f}s)")
    if not hard:
        print("No candidates remained after hard prefilters. Loosen languages, pace, or budget.")
        return

    t1 = time.time()
    shortlist = ai_prefilter(q_user, hard, percent=0.02, min_k=80)
    print(f"AI prefilter: {len(shortlist)} candidates ({time.time()-t1:.2f}s)")
    if not shortlist:
        print("No candidates after AI prefilter.")
        return

    t2 = time.time()
    final = llm_rank(q_user, shortlist, out_top=5)
    print(f"Llama ranking produced {len(final)} matches ({time.time()-t2:.2f}s)")

    # Clear and informative presentation similar to a card:
    email_to_user = {rec["user"].get("email"): rec["user"] for rec in shortlist}

    def _band(amount: Optional[float]) -> str:
        if amount is None: return "mid"
        if amount <= 90:   return "budget"
        if amount >= 180:  return "lux"
        return "mid"

    print("\nTop Recommendations")
    print("-------------------")
    for i, m in enumerate(final, 1):
        u = email_to_user.get(m.get("email"), {}) or {}
        hb = u.get("home_base") or {}
        city = hb.get("city", "")
        country = hb.get("country", "")
        langs = "/".join(u.get("languages", [])) or "—"
        pace_u = (u.get("travel_prefs") or {}).get("pace", "balanced")
        budget_u = (u.get("budget") or {}).get("amount", None)
        band = _band(budget_u)

        chips = ui_interests_values(q_user, u, n=4)  # <= 4 short chips from interests/values

        try:
            pct = int(round(float(m.get("compatibility_score", 0)) * 100))
        except Exception:
            pct = 0

        print(f"{i}. {m.get('name','Unknown')} — {pct}% match")
        print(f"   {city or '—'} | {', '.join(chips) if chips else '—'}")
        print(f"   Pace: {pace_u} | Budget: {('€' + str(budget_u)) if budget_u else '—'} [{band}] | Languages: {langs}")
        print(f"   Why: {m.get('explanation','')}\n")

if __name__ == "__main__":
    main()
