#!/usr/bin/env python3
"""
Llama API Server for RoverMitra
Pre-loads all models and data for fast inference
"""

import os
import json
import time
import psycopg2
from pathlib import Path
from typing import Dict, List, Any, Optional
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

# Load environment variables
load_dotenv()

# Suppress warnings
import warnings
warnings.filterwarnings("ignore")

# ML imports
try:
    import numpy as np
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
    from sentence_transformers import SentenceTransformer
    _ML_OK = True
except Exception as e:
    print(f"[warn] ML stack import failed: {e}")
    _ML_OK = False

# Base dir
BASE_DIR = Path(__file__).resolve().parent

# Model paths
BGE_M3_PATH = BASE_DIR / "models" / "bge-m3"
LLAMA_FINETUNED_PATH = BASE_DIR / "models" / "llama-travel-matcher"
LLAMA_BASE_PATH = BASE_DIR / "models" / "llama-3.2-3b-instruct"
CACHE_DIR = BASE_DIR / "models_cache"
UIDS_PATH = CACHE_DIR / "bge_user_emails.npy"
EMB_PATH = CACHE_DIR / "bge_embeds_fp16.npy"

# Global variables for loaded models/data
_EMB = None
_LLM_FINETUNED = None
_LLM_BASE = None
_cached_ids = None
_cached_embs = None
_pool_data = None

app = FastAPI(title="RoverMitra Llama API", version="1.0.0")

class RankRequest(BaseModel):
    prompt: str
    max_new_tokens: int = 120
    temperature: float = 0.0
    top_p: float = 0.9

class HealthResponse(BaseModel):
    status: str
    models_loaded: bool
    data_loaded: bool
    pool_size: int

def load_embeddings():
    """Load BGE-M3 embeddings"""
    global _EMB
    if not _ML_OK:
        return False
    
    try:
        dev = "cuda" if (hasattr(torch, "cuda") and torch.cuda.is_available()) else "cpu"
        print(f"Loading BGE-M3 embedding model from local path: {BGE_M3_PATH}...")
        _EMB = SentenceTransformer(str(BGE_M3_PATH), device=dev)
        print("✅ BGE-M3 embedding model loaded.")
        return True
    except Exception as e:
        print(f"❌ Failed to load BGE-M3: {e}")
        return False

def load_bge_cache():
    """Load BGE cache"""
    global _cached_ids, _cached_embs
    try:
        if not (UIDS_PATH.exists() and EMB_PATH.exists()):
            print("❌ BGE cache missing. Run: python build_bge_cache.py")
            return False
        
        _cached_ids = np.load(UIDS_PATH, allow_pickle=True)
        _cached_embs = np.load(EMB_PATH, allow_pickle=False)
        print(f"✅ BGE cache loaded: {len(_cached_ids)} embeddings")
        return True
    except Exception as e:
        print(f"❌ Failed to load BGE cache: {e}")
        return False

def load_llama_4bit(model_path: Path):
    """Load Llama model with 4-bit quantization"""
    if not _ML_OK:
        return None
    
    try:
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
    except Exception as e:
        print(f"❌ Failed to load Llama model: {e}")
        return None

def load_llama_models():
    """Load Llama models"""
    global _LLM_FINETUNED, _LLM_BASE
    
    # Try fine-tuned model first
    try:
        print("Loading fine-tuned Llama (4-bit) from local path...")
        _LLM_FINETUNED = load_llama_4bit(LLAMA_FINETUNED_PATH)
        if _LLM_FINETUNED:
            print("✅ Fine-tuned Llama loaded from local path")
            return True
    except Exception as e:
        print(f"⚠️ Fine-tuned Llama load failed: {e}")
    
    # Try base model as fallback
    try:
        print("Loading base Llama (4-bit) from local path...")
        _LLM_BASE = load_llama_4bit(LLAMA_BASE_PATH)
        if _LLM_BASE:
            print("✅ Base Llama loaded from local path")
            return True
    except Exception as e:
        print(f"⚠️ Base Llama load failed: {e}")
    
    print("❌ No Llama models could be loaded")
    return False

def load_railway_data():
    """Load data from Railway Postgres"""
    global _pool_data
    
    try:
        DATABASE_URL = os.getenv("DATABASE_URL") or "postgresql://postgres:RzBikKnKvwEeEUMDmGYFskiVJStCeOOH@hopper.proxy.rlwy.net:11809/railway"
        print("Connecting to Railway Postgres...")
        conn = psycopg2.connect(DATABASE_URL)
        cursor = conn.cursor()

        # Load users
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

        # Load matchmaker profiles
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

        # Create pool
        pool = []
        mm_by_email = {mm.get("email"): mm for mm in mm_profiles if mm.get("email")}
        
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
        
        _pool_data = pool
        print(f"✅ Loaded {len(pool)} candidates from Railway.")
        return True

    except Exception as e:
        print(f"❌ Failed to load Railway data: {e}")
        return False

def build_llm_prompt(q_user: Dict[str,Any], shortlist: List[Dict[str,Any]], top_k:int=5) -> str:
    """Build LLM prompt"""
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

    head = (
        "You are a precise travel-match expert. Rank candidates for holistic trip compatibility.\n"
        "Focus on: shared interests, complementary activities, budget compatibility, language overlap, "
        "travel pace, dietary needs, transport preferences, accommodation types, and cultural compatibility.\n"
        "Trip context: weekend to multi-week.\n"
        "Return ONLY valid JSON with key 'matches' as an array of objects with fields: email, name, explanation "
        "(ONE sentence, must start with 'For you,' and be specific), compatibility_score (0.0–1.0).\n\n"
        "Query User:\n"
        f"{query_text(q_user)}\n\n"
        "Candidates:\n"
    )
    body = []
    for i, rec in enumerate(shortlist):
        u, m = rec["user"], rec["mm"]
        body.append(f"[{i+1}] email={u.get('email')} | {summarize_user(u, m)}")
    return head + "\n".join(body)

def get_llama_ranking(prompt: str, max_new_tokens: int = 120, temperature: float = 0.0, top_p: float = 0.9) -> str:
    """Get ranking from loaded Llama model"""
    model_data = _LLM_FINETUNED or _LLM_BASE
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
        print(f"[warn] LLM generation failed: {e}")
        return None

@app.on_event("startup")
async def startup_event():
    """Load all models and data on startup"""
    print("🚀 Starting RoverMitra Llama API Server...")
    
    # Load models
    print("📦 Loading models...")
    embeddings_ok = load_embeddings()
    cache_ok = load_bge_cache()
    llama_ok = load_llama_models()
    
    # Load data
    print("📊 Loading data...")
    data_ok = load_railway_data()
    
    print("✅ Server startup complete!")
    print(f"   Embeddings: {'✅' if embeddings_ok else '❌'}")
    print(f"   BGE Cache: {'✅' if cache_ok else '❌'}")
    print(f"   Llama Model: {'✅' if llama_ok else '❌'}")
    print(f"   Railway Data: {'✅' if data_ok else '❌'}")

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="ok",
        models_loaded=_EMB is not None and (_LLM_FINETUNED is not None or _LLM_BASE is not None),
        data_loaded=_pool_data is not None,
        pool_size=len(_pool_data) if _pool_data else 0
    )

@app.post("/rank")
async def rank_candidates(request: RankRequest):
    """Rank candidates using loaded Llama model"""
    try:
        # Get ranking directly from the prompt
        text = get_llama_ranking(request.prompt, request.max_new_tokens, request.temperature, request.top_p)
        
        if text is None:
            raise HTTPException(status_code=500, detail="LLM generation failed")
        
        return {"text": text}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/pool")
async def get_pool():
    """Get the loaded pool data"""
    if _pool_data is None:
        raise HTTPException(status_code=500, detail="Pool data not loaded")
    return {"pool": _pool_data, "size": len(_pool_data)}

class PrefilterRequest(BaseModel):
    query_user: dict
    candidates: list
    percent: float = 0.02
    min_k: int = 80

@app.post("/prefilter")
async def ai_prefilter(request: PrefilterRequest):
    """AI prefilter using loaded BGE embeddings"""
    try:
        from main import ai_prefilter as main_ai_prefilter
        result = main_ai_prefilter(request.query_user, request.candidates, request.percent, request.min_k)
        return {"filtered_candidates": result, "count": len(result)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    print("🚀 Starting RoverMitra Llama API Server...")
    uvicorn.run(app, host="0.0.0.0", port=8002)
