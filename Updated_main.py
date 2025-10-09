#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RoverMitra Matchmaker – fast path (cached BGE + 4-bit Llama)

Pipeline:
  Hard Prefilters  ->  AI Prefilter (BGE-M3 embeddings; cached)  ->  Final Ranking (Llama-4bit)

Final Ranking:
  Try local fine-tuned Llama at models/llama-travel-matcher (4-bit on GPU).
  If it fails, try base models/llama-3.2-3b-instruct (4-bit on GPU).
  If both fail, fall back to heuristic scorer.

First run once:  python build_bge_cache.py
Then run this script normally.

Requires: bitsandbytes (for 4-bit), sentence-transformers, transformers, torch (CUDA)
"""

from __future__ import annotations
import os, re, json, uuid, math, sys, time
from pathlib import Path
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------
# Environment (set before any ML import)
# ---------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent

# Let you choose GPU outside; default to GPU0 if not set
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
os.environ.setdefault("HF_HOME", str(BASE_DIR / "models_cache"))
os.environ.setdefault("TRANSFORMERS_CACHE", str(BASE_DIR / "models_cache" / "transformers"))
os.environ.setdefault("TORCH_HOME", str(BASE_DIR / "models_cache" / "torch"))
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")

# --- psutil guard (prevents circular-import crash if system psutil is broken) ---
try:
    import psutil  # noqa
except Exception:
    import types, sys
    dummy = types.SimpleNamespace(
        virtual_memory=lambda: types.SimpleNamespace(total=0),
        cpu_count=lambda: 1,
        Process=lambda *a, **k: types.SimpleNamespace(memory_info=lambda: types.SimpleNamespace(rss=0))
    )
    sys.modules['psutil'] = dummy
# -------------------------------------------------------------------------------

# ---------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------
USERS_CORE_PATH  = BASE_DIR / "users/data/users_core.json"
MM_PATH          = BASE_DIR / "MatchMaker/data/matchmaker_profiles.json"
LOCAL_DB_PATH    = BASE_DIR / "data/travel_ready_user_profiles.json"  # append new users here

MODELS_DIR               = BASE_DIR / "models"
BGE_M3_PATH              = MODELS_DIR / "bge-m3"
LLAMA_FINETUNED_PATH     = MODELS_DIR / "llama-travel-matcher"
LLAMA_BASE_PATH          = MODELS_DIR / "llama-3.2-3b-instruct"

# Embedding cache
CACHE_DIR = BASE_DIR / "models_cache"
UIDS_PATH = CACHE_DIR / "bge_user_ids.npy"
EMB_PATH  = CACHE_DIR / "bge_embeds_fp16.npy"

# ---------------------------------------------------------------------
# Imports (after env)
# ---------------------------------------------------------------------
try:
    import numpy as np
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
    from sentence_transformers import SentenceTransformer
    _ML_OK = True
except Exception as e:
    print(f"[warn] ML stack import failed: {e}", file=sys.stderr)
    _ML_OK = False

# ---------------------------------------------------------------------
# Utils
# ---------------------------------------------------------------------
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

# ---------------------------------------------------------------------
# Vocab aids
# ---------------------------------------------------------------------
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

# ---------------------------------------------------------------------
# CLI helpers (interactive profile builder)
# ---------------------------------------------------------------------
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

# ---------------------------------------------------------------------
# Interactive profile builder
# ---------------------------------------------------------------------
def interactive_new_user() -> Dict[str, Any]:
    print("\n🌍  Welcome! Let’s make trips easy to say yes to.\n(Quick taps, tiny questions, zero boredom.)\n")

    print("— Basics —")
    name = ask("Your name (e.g., 'Aisha Khan' or 'Tom Müller')", "Alex Rivera")
    age  = ask_int("Age", 27)
    gender = ask_choice("How do you identify?", ["Male","Female","Non-binary","Other"], "Other")
    city = ask("Home city (e.g., Berlin)", "Berlin")
    country = ask("Country (e.g., Germany)", "Germany")

    print("\n— Languages —")
    print("Tip: use names or codes (e.g., english/en, deutsch/de).")
    langs_in = clean_list_csv(ask("Languages you can chat in (comma)", "en, de"))
    langs = []
    for w in langs_in:
        key = w.lower()
        langs.append(LANG_ALIASES.get(key, key))
    langs = sorted(set([x for x in langs if x]))

    print("\n— Pick your vibe —")
    INTEREST_CHIPS = [
        "museum hopping","architecture walks","history sites","city photography","food tours","street food",
        "coffee crawls","vineyards","scenic trains","short hikes","long hikes","mountain hiking","lake swims",
        "beach days","markets","old towns","rooftop views","local crafts","live music","festivals"
    ]
    interests = ask_multi("What sounds fun right now?", INTEREST_CHIPS, 3, 6, default_idxs=[0,1,3])

    print("\n— Tempo —")
    pace = ask_choice("Trip pace", ["relaxed","balanced","packed"], "balanced")
    chronotype = ask_choice("You’re most alive…", ["early bird","flexible","night owl"], "flexible")

    print("\n— Money stuff (kept private) —")
    budget = ask_int("Comfortable budget per day (€)", 150)
    split_rule = ask_choice("Fair way to split shared costs?", ["each_own","50/50","custom"], "each_own")

    print("\n— Food —")
    diet = ask_choice("Diet", sorted(list(DIETS)), "none")
    allergies_raw = clean_list_csv(ask("Allergies? (comma, e.g., nuts, gluten) or 'none'", "none"))
    allergies = ["none"] if (not allergies_raw or allergies_raw == ["none"]) else allergies_raw

    print("\n— Comfort —")
    smoking = ask_choice("Smoking", ["never","occasionally","regular"], "never")
    alcohol = ask_choice("Alcohol", ["none","moderate","social"], "moderate")
    noise = ask_choice("Noise tolerance for stays", ["low","medium","high"], "medium")
    clean = ask_choice("Cleanliness preference", ["low","medium","high"], "medium")
    risk  = ask_choice("Risk tolerance for activities", ["low","medium","high"], "medium")

    print("\n— Work bits —")
    remote_ok = ask_yesno("Need time to work while traveling?")
    hours_online = ask_choice("Hours you must be online per day", ["0","1","2"], "0")
    wifi_need = ask_choice("Wi-Fi needs", ["normal","good","excellent"], "good")

    print("\n— Stay must-haves —")
    MUSTS = ["wifi","private_bath","kitchen","workspace","near_station","quiet_room","breakfast"]
    must_haves = ask_multi("Pick 1–4 must-haves", MUSTS, 1, 4, default_idxs=[0,3])

    print("\n— Room & transport —")
    room_setup = ask_choice("Room setup you prefer when sharing", ["twin","double","2 rooms","dorm"], "twin")
    TRANSPORT = ["train","plane","bus","car"]
    transport_allowed = ask_multi("Allowed transport modes", TRANSPORT, 1, 3, default_idxs=[0,1])

    print("\n— Trip styles —")
    STYLES = ["weekend getaway","co-work week","hiking basecamp","city sampler","island hop","festival trip","road trip"]
    trip_styles = ask_multi("Pick a couple", STYLES, 1, 3, default_idxs=[0,3])

    print("\n— Values —")
    VALUES = ["adventure","stability","learning","family","budget-minded","luxury-taste","nature","culture","community","fitness","spirituality"]
    values = ask_multi("Choose 2–3", VALUES, 2, 3, default_idxs=[0,6])

    print("\n— Safety —")
    share_home_city = ask_yesno("OK to show your home city on profile?", True)
    pre_meet_video  = ask_yesno("Comfortable with a short pre-trip video call?", True)

    print("\n— Last bit —")
    bio = ask("Write a one-liner about you (e.g., 'Berlin-based planner who loves scenic trains + espresso.')", "")

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

        "privacy": {
            "share_profile_with_matches": True,
            "share_home_city": share_home_city,
            "pre_meet_video_call_ok": pre_meet_video
        }
    }
    print("\n✅ Thanks! Building your matches…\n")
    return user

# ---------------------------------------------------------------------
# Candidate pool and summarization
# ---------------------------------------------------------------------
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

def summarize_user(u: Dict[str,Any], mm: Optional[Dict[str,Any]]) -> str:
    # Shorter summary (keeps prompt small / fast)
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
        f"Alcohol: {(q.get('comfort') or {}).get('alcohol','moderate')}\n"
        f"Interests: {', '.join(q.get('interests',[]))}\n"
        f"Values: {', '.join(q.get('values',[]))}\n"
        f"MustHaves: {', '.join((q.get('travel_prefs') or {}).get('must_haves',[]))}\n"
        f"Transport: {', '.join((q.get('travel_prefs') or {}).get('transport_allowed',[]))}\n"
        f"RemoteWork: {(q.get('work') or {}).get('remote_work_ok', False)}\n"
        f"Bio: {q.get('bio', '')}"
    )

# ---------------------------------------------------------------------
# Hard prefilters (no bio rule; LLM will reason about bio)
# ---------------------------------------------------------------------
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

# ---------------------------------------------------------------------
# AI prefilter (BGE-M3 with cache; query-only encode)
# ---------------------------------------------------------------------
_EMB: Optional[SentenceTransformer] = None
_cached_ids = None
_cached_embs = None

def ensure_emb(device: str = "cuda"):
    global _EMB
    if _ML_OK and _EMB is None:
        dev = device if torch.cuda.is_available() else "cpu"
        print("Loading BGE-M3 embedding model...")
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

def ai_prefilter(q_user: Dict[str, Any],
                 cands: List[Dict[str, Any]],
                 percent: float = 0.02,
                 min_k: int = 80) -> List[Dict[str, Any]]:
    if not cands:
        return []

    # Map candidates to cached embedding rows
    cand_uids = [rec["user"]["user_id"] for rec in cands]
    ids, embs = _load_bge_cache()              # embs: [N,D] float16
    uid2idx = {uid: i for i, uid in enumerate(ids)}
    idxs = [uid2idx[u] for u in cand_uids if u in uid2idx]

    # Encode query only (GPU if available)
    model = ensure_emb()
    qv = model.encode(
        [query_text(q_user)],
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False
    )[0].astype("float16")

    sub = embs[idxs]                            # [K,D] float16
    sims = (sub @ qv).astype("float32")         # cosine approx

    # symbolic bonuses (same as before)
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

    # normalize sims + add bonuses
    s_min, s_max = float(sims.min()), float(sims.max())
    sims_norm = (sims - s_min) / (s_max - s_min) if s_max > s_min else np.full_like(sims, 0.5, dtype="float32")
    combined = [(cands[i], float(sims_norm[i]) + float(bonuses[i])) for i in range(len(cands))]
    combined.sort(key=lambda x: x[1], reverse=True)

    # keep ~2% (min 80) for general run; you can lower for test
    k = max(min_k, int(math.ceil(len(combined) * percent)))
    k = min(k, len(combined))
    return [rec for rec, _ in combined[:k]]

# ---------------------------------------------------------------------
# Llama ranking (4-bit quant, GPU)
# ---------------------------------------------------------------------
_LLM_FINETUNED = None
_LLM_BASE = None

def _load_llama_4bit(path: Path):
    bnb_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    tok = AutoTokenizer.from_pretrained(str(path), use_fast=True)
    mdl = AutoModelForCausalLM.from_pretrained(
        str(path),
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
            print("Loading fine-tuned Llama (4-bit)...")
            _LLM_FINETUNED = _load_llama_4bit(LLAMA_FINETUNED_PATH)
            print("Fine-tuned Llama loaded.")
        except Exception as e:
            print(f"[warn] Fine-tuned Llama load failed: {e}")
            _LLM_FINETUNED = None
    return _LLM_FINETUNED

def ensure_llm_base():
    global _LLM_BASE
    if _ML_OK and _LLM_BASE is None:
        try:
            print("Loading base Llama (4-bit)...")
            _LLM_BASE = _load_llama_4bit(LLAMA_BASE_PATH)
            print("Base Llama loaded.")
        except Exception as e:
            print(f"[warn] Base Llama load failed: {e}")
            _LLM_BASE = None
    return _LLM_BASE

def build_llm_prompt(q_user: Dict[str,Any], shortlist: List[Dict[str,Any]], top_k:int=5) -> str:
    # Keep prompt compact
    head = (
        "You are a precise travel-match expert. Rank candidates for holistic trip compatibility.\n"
        "Consider: personality fit, conflict style, shared/complementary interests, languages, pace, budget, diet/substances, "
        "risk tolerance, work needs, values, cultural/religious needs, and any constraints from the user's bio.\n"
        "Trip context: weekend to multi-week.\n"
        "Return ONLY valid JSON with key 'matches' as an array of objects with fields: user_id, name, explanation "
        "(ONE sentence, must start with 'For you,' and be specific), compatibility_score (0.0–1.0).\n\n"
        "Query User:\n"
        f"{query_text(q_user)}\n\n"
        "Candidates:\n"
    )
    body = []
    for i, rec in enumerate(shortlist):
        u, m = rec["user"], rec["mm"]
        body.append(f"[{i+1}] user_id={u.get('user_id')} | {summarize_user(u, m)}")
    return head + "\n".join(body)

def _extract_json(text: str) -> Optional[dict]:
    # robust extractor: grab first {...} region
    start = text.find("{")
    end   = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    chunk = text[start:end+1]
    try:
        return json.loads(chunk)
    except Exception:
        # try to trim trailing commas or stray chars
        # simple fallback: find last "}" that yields parse
        for j in range(end, start, -1):
            try:
                return json.loads(text[start:j])
            except Exception:
                continue
    return None

def llm_rank(q_user: Dict[str,Any], shortlist: List[Dict[str,Any]], out_top:int=5) -> List[Dict[str,Any]]:
    if not _ML_OK:
        return llm_rank_fallback(q_user, shortlist, out_top)

    # Keep the LLM work small: only top-10 from shortlist
    shortlist_for_llm = shortlist[:10]

    system_prompt = "Return ONLY valid JSON with key 'matches'. No explanations. No extra text."
    prompt = build_llm_prompt(q_user, shortlist_for_llm, top_k=out_top)

    model_data = ensure_llm_finetuned()
    if model_data is None:
        model_data = ensure_llm_base()
    if model_data is None:
        return llm_rank_fallback(q_user, shortlist_for_llm, out_top)

    tok, mdl = model_data["tokenizer"], model_data["model"]

    # Chat template if available; else simple prefix
    if hasattr(tok, "apply_chat_template") and getattr(tok, "chat_template", None):
        inputs = tok.apply_chat_template(
            [{"role": "system", "content": system_prompt},
             {"role": "user",   "content": prompt}],
            add_generation_prompt=True,
            return_tensors="pt",
            truncation=True,
            max_length=4096
        )
    else:
        text = f"System: {system_prompt}\nUser: {prompt}\nAssistant:"
        inputs = tok(text, return_tensors="pt", truncation=True, max_length=4096)

    inputs = {k: v.to(mdl.device) for k, v in inputs.items()}

    try:
        with torch.inference_mode():
            out = mdl.generate(
                **inputs,
                max_new_tokens=120,     # tight cap = faster & enough
                do_sample=False,        # deterministic for consistency
                use_cache=True,
                pad_token_id=tok.eos_token_id,
                eos_token_id=tok.eos_token_id,
            )
        gen = out[0][inputs["input_ids"].shape[1]:]
        text = tok.decode(gen, skip_special_tokens=True)

        parsed = _extract_json(text)
        if parsed and isinstance(parsed.get("matches", None), list):
            cleaned = []
            for m in parsed["matches"][:out_top]:
                try:
                    cleaned.append({
                        "user_id": str(m.get("user_id","")),
                        "name": str(m.get("name","")),
                        "explanation": str(m.get("explanation","")),
                        "compatibility_score": float(m.get("compatibility_score", 0.0))
                    })
                except Exception:
                    continue
            if cleaned:
                return cleaned
    except Exception as e:
        print(f"[warn] LLM generation failed: {e}")

    return llm_rank_fallback(q_user, shortlist_for_llm, out_top)

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

def llm_rank_fallback(q_user: Dict[str,Any], shortlist: List[Dict[str,Any]], out_top:int=5) -> List[Dict[str,Any]]:
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

# ---------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------
def main():
    q_user = interactive_new_user()
    append_to_json(q_user, LOCAL_DB_PATH)
    print(f"\n✅ Saved your profile to {LOCAL_DB_PATH}")

    pool = load_pool()
    if not pool:
        print("No candidates found. Provide users_core.json and matchmaker_profiles.json.")
        return

    # 1) Hard prefilters
    t0 = time.time()
    hard = hard_prefilter(q_user, pool)
    print(f"✅ Hard prefilter: {len(hard)} candidates (in {time.time()-t0:.2f}s)")
    if not hard:
        print("No candidates remained after hard prefilters. Loosen languages, pace, or budget.")
        return

    # 2) AI prefilter (BGE cache → fast)
    t1 = time.time()
    shortlist = ai_prefilter(q_user, hard, percent=0.02, min_k=80)
    print(f"✅ AI prefilter: {len(shortlist)} candidates (in {time.time()-t1:.2f}s)")
    if not shortlist:
        print("No candidates after AI prefilter.")
        return

    # 3) Llama ranking (uses fine-tuned first)
    t2 = time.time()
    final = llm_rank(q_user, shortlist, out_top=5)
    print(f"✅ Llama ranking produced {len(final)} matches (in {time.time()-t2:.2f}s)")

    # 4) Threshold >= 0.75
    high_quality = [m for m in final if float(m.get("compatibility_score", 0)) >= 0.75]

    print("\n— Top Recommendations —")
    if not high_quality:
        print("No high-quality matches found (score >= 75%). Here are your top results:")
        for i, m in enumerate(final, 1):
            try:
                pct = int(round(float(m.get("compatibility_score",0))*100))
            except Exception:
                pct = 0
            print(f"{i}. {m.get('name')} (user_id: {m.get('user_id')})  —  {pct}%")
            print(f"   {m.get('explanation')}\n")
        return

    for i, m in enumerate(high_quality, 1):
        try:
            pct = int(round(float(m.get("compatibility_score",0))*100))
        except Exception:
            pct = 0
        print(f"{i}. {m.get('name')} (user_id: {m.get('user_id')})  —  {pct}%")
        print(f"   {m.get('explanation')}\n")

if __name__ == "__main__":
    main()
