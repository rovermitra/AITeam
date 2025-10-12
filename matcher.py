import json
import math
import re
import time
from typing import Any, Dict, List, Optional

# Suppress unimportant warnings for cleaner logs
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Attempt to import ML libraries, but allow the module to load without them
# This helps in environments where you might only want to test the non-ML parts.
try:
    import numpy as np
    import torch
    from sentence_transformers import SentenceTransformer
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
    _ML_OK = True
except ImportError as e:
    print(f"[Warning] ML libraries not found: {e}. AI features will be disabled.")
    _ML_OK = False

# Import paths and the setup function from our config file
from config import (
    USERS_CORE_PATH, MM_PATH, BGE_M3_PATH, LLAMA_FINETUNED_PATH,
    LLAMA_BASE_PATH, UIDS_PATH, EMB_PATH, setup_environment
)

# Set up environment variables before any heavy ML lifting
setup_environment()

# --- Global state for loaded models and data (to load only once) ---
CANDIDATE_POOL: List[Dict[str, Any]] = []
_EMB_MODEL: Optional['SentenceTransformer'] = None
_CACHED_IDS = None
_CACHED_EMBS = None
_LLM_FINETUNED: Optional[Dict[str, Any]] = None
_LLM_BASE: Optional[Dict[str, Any]] = None

# --- Helper Functions ---

def index_by(lst: list, key: str) -> Dict[str, Any]:
    """Creates a dictionary from a list of dictionaries, indexed by a given key."""
    return {x.get(key): x for x in lst if isinstance(x, dict) and x.get(key)}

def budget_band(amount: Optional[float]) -> str:
    """Categorizes a numeric budget into 'budget', 'mid', or 'lux'."""
    if amount is None: return "mid"
    if amount <= 90: return "budget"
    if amount >= 180: return "lux"
    return "mid"

def _faith_slug(s: str) -> str:
    """Converts a string to a snake_case slug."""
    return (s or "").strip().lower().replace(" ", "_")

def jaccard(a: List[str], b: List[str]) -> float:
    """Calculates the Jaccard similarity between two lists."""
    A, B = set(a or []), set(b or [])
    if not A and not B: return 0.0
    return len(A & B) / max(1, len(A | B))

# --- Data and Model Loading ---

def load_pool():
    """Loads the candidate user data from JSON files into the global state."""
    global CANDIDATE_POOL
    if CANDIDATE_POOL:
        return
    print("Loading candidate pool from JSON files...")
    if not USERS_CORE_PATH.exists() or not MM_PATH.exists():
        raise FileNotFoundError("User or Matchmaker data files not found. Please generate them first.")
    
    users = json.loads(USERS_CORE_PATH.read_text(encoding="utf-8"))
    mm = json.loads(MM_PATH.read_text(encoding="utf-8"))
    mm_by_uid = index_by(mm, "user_id")
    
    pool = []
    for u in users:
        uid = u.get("user_id")
        if not uid: continue
        pool.append({"user": u, "mm": mm_by_uid.get(uid)})
    CANDIDATE_POOL = pool
    print(f"✅ Candidate pool loaded with {len(CANDIDATE_POOL)} profiles.")

def _load_llama_4bit(path: 'Path') -> Dict[str, Any]:
    """Loads a Llama model with 4-bit quantization."""
    if not _ML_OK: raise ImportError("Cannot load Llama without ML libraries.")
    
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

def initialize_models():
    """Loads all ML models into memory. Should be called once at application startup."""
    global _EMB_MODEL, _CACHED_IDS, _CACHED_EMBS, _LLM_FINETUNED, _LLM_BASE
    
    if not _ML_OK:
        print("[Warning] ML libraries not installed. All AI ranking features will be disabled.")
        return

    # Load BGE model for embeddings
    print("Loading BGE-M3 embedding model...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    _EMB_MODEL = SentenceTransformer(str(BGE_M3_PATH), device=device)
    print(f"✅ BGE-M3 model loaded to device: {device}.")

    # Load pre-computed BGE embeddings cache
    print("Loading BGE cache...")
    if not UIDS_PATH.exists() or not EMB_PATH.exists():
        raise FileNotFoundError("BGE cache missing. Please run `build_bge_cache.py` first.")
    _CACHED_IDS = np.load(UIDS_PATH, allow_pickle=True)
    _CACHED_EMBS = np.load(EMB_PATH, allow_pickle=False)
    print("✅ BGE cache loaded.")

    # Load Llama models (fine-tuned with a base model as fallback)
    try:
        print("Loading fine-tuned Llama (4-bit)...")
        _LLM_FINETUNED = _load_llama_4bit(LLAMA_FINETUNED_PATH)
        print("✅ Fine-tuned Llama loaded.")
    except Exception as e:
        print(f"⚠️  Could not load fine-tuned Llama: {e}. Attempting to load base model as fallback.")
        _LLM_FINETUNED = None # Ensure it's None if loading fails
        try:
            print("Loading base Llama (4-bit)...")
            _LLM_BASE = _load_llama_4bit(LLAMA_BASE_PATH)
            print("✅ Base Llama loaded.")
        except Exception as e_base:
            _LLM_BASE = None
            print(f"🔥 CRITICAL: Could not load base Llama model: {e_base}. LLM ranking will be disabled.")

# --- Core Matching Logic ---

def summarize_user(u: Dict[str, Any], mm: Optional[Dict[str, Any]]) -> str:
    """Creates a compact, single-line summary of a user profile for model context."""
    parts = [
        f"{u.get('name', '')}, age {u.get('age', '')}",
        f"city={(u.get('home_base') or {}).get('city', '')}",
        f"langs={'/'.join(u.get('languages', []))}",
        f"budget={(u.get('budget') or {}).get('amount')}",
        f"diet={(u.get('diet_health') or {}).get('diet', 'none')}",
        f"alcohol={(u.get('comfort') or {}).get('alcohol', 'moderate')}",
        f"pace={(u.get('travel_prefs') or {}).get('pace', 'balanced')}",
        f"interests={','.join((u.get('interests') or [])[:10])}",
    ]
    if u.get('values'):
        parts.append(f"values={','.join(u.get('values', [])[:5])}")
    
    faith_info = u.get("faith") or {}
    if faith_info.get("consider_in_matching") and faith_info.get("religion"):
        parts.append(f"faith={faith_info['religion']}({faith_info.get('policy', 'open')})")
        
    return " | ".join(filter(None, parts))

def query_text(q: Dict[str, Any]) -> str:
    """Formats a query user's profile into a detailed block of text for embeddings and prompts."""
    hb = q.get("home_base") or {}
    fq = q.get("faith") or {}
    faith_str = ""
    if fq.get("consider_in_matching"):
        faith_str = f"\nFaithPolicy: {fq.get('policy', 'open')}\nFaith: {fq.get('religion', '')}"
    
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
        f"Bio: {q.get('bio', '')}{faith_str}"
    )

# --- Hard Filter Functions ---

def age_ok(query_age: int, cand_mm: Dict[str, Any]) -> bool:
    r = (cand_mm.get("preferred_companion") or {}).get("age_range")
    return not r or (isinstance(r, list) and len(r) == 2 and r[0] <= query_age <= r[1])

def gender_ok(query_gender: str, cand_mm: Dict[str, Any]) -> bool:
    allowed = [g.lower() for g in (cand_mm.get("preferred_companion") or {}).get("genders", ["any"])]
    return "any" in allowed or (query_gender or "").lower() in allowed

def companion_gender_ok(query_companion_pref: str, cand_user: Dict[str,Any]) -> bool:
    cand_gender = (cand_user.get("gender") or "").lower()
    pref_lower = query_companion_pref.lower()
    if "anyone" in pref_lower: return True
    if "men" in pref_lower: return cand_gender in ["male", "man"]
    if "women" in pref_lower: return cand_gender in ["female", "woman"]
    if "nonbinary" in pref_lower: return cand_gender in ["non-binary", "nonbinary", "other"]
    return True

def langs_ok(query_langs: List[str], cand_mm: Dict[str, Any], cand_user: Dict[str, Any]) -> bool:
    lp = cand_mm.get("language_policy") or {}
    need = lp.get("min_shared_languages", 1)
    if need <= 0: return True
    cand_pref = lp.get("preferred_chat_languages") or cand_user.get("languages", [])
    return len(set(query_langs or []) & set(cand_pref or [])) >= need

def budget_ok(query_amount: int, cand_user: Dict[str, Any]) -> bool:
    qa_band = budget_band(query_amount)
    ca_band = budget_band((cand_user.get("budget") or {}).get("amount"))
    order = ["budget", "mid", "lux"]
    return abs(order.index(qa_band) - order.index(ca_band)) <= 1

def pace_ok(query_pace: str, cand_user: Dict[str, Any], cand_mm: Dict[str, Any]) -> bool:
    pref = (cand_mm.get("soft_preferences") or {}).get("prefer_same_pace")
    if pref != "prefer_same": return True
    return (cand_user.get("travel_prefs") or {}).get("pace") == query_pace

def faith_ok(q: Dict[str, Any], cand_user: Dict[str, Any], cand_mm: Dict[str, Any]) -> bool:
    qf = q.get("faith") or {}
    if not qf.get("consider_in_matching") and "same_faith_required" not in (cand_mm.get("hard_dealbreakers") or []):
        return True
        
    q_token = _faith_slug(qf.get("religion", ""))
    c_token = (cand_mm.get("faith_preference") or {}).get("religion_token", "")

    if "same_faith_required" in (cand_mm.get("hard_dealbreakers") or []):
        return bool(q_token and c_token and q_token == c_token)
    
    if qf.get("policy") == "same_only":
        return bool(q_token and c_token and q_token == c_token)
        
    return True

def hard_prefilter(q: Dict[str, Any], pool: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Applies a series of non-negotiable filters to the candidate pool."""
    out = []
    q_age = int(q.get("age", 30))
    q_gender = q.get("gender", "Other")
    q_langs = q.get("languages", [])
    q_budget = (q.get("budget") or {}).get("amount", 150)
    q_pace = (q.get("travel_prefs") or {}).get("pace", "balanced")
    q_companion_gender = (q.get("companion_preferences", {}).get("genders_ok") or ["anyone"])[0]

    for rec in pool:
        cu, cm = rec["user"], rec.get("mm", {})
        if not cu or not cm: continue
        if not (
            langs_ok(q_langs, cm, cu) and
            age_ok(q_age, cm) and
            gender_ok(q_gender, cm) and
            companion_gender_ok(q_companion_gender, cu) and
            budget_ok(q_budget, cu) and
            pace_ok(q_pace, cu, cm) and
            faith_ok(q, cu, cm)
        ):
            continue
        out.append(rec)
    return out

# --- AI Prefilter (BGE Embeddings) ---

def _heuristic_shortlist(q_user: Dict[str, Any], cands: List[Dict[str, Any]], percent: float, min_k: int) -> List[Dict[str, Any]]:
    """A fallback scoring mechanism if ML models aren't available."""
    q_interests = set(q_user.get("interests", []))
    q_values = set(q_user.get("values", []))
    scored = []
    for rec in cands:
        u = rec["user"]
        score = 0.6 * jaccard(list(q_interests), u.get("interests", [])) + 0.4 * jaccard(list(q_values), u.get("values", []))
        scored.append((rec, score))

    scored.sort(key=lambda x: x[1], reverse=True)
    k = max(min_k, int(math.ceil(len(scored) * percent)))
    return [rec for rec, _ in scored[:k]]

def ai_prefilter(q_user: Dict[str, Any], cands: List[Dict[str, Any]], percent: float = 0.02, min_k: int = 80) -> List[Dict[str, Any]]:
    """Shortlists candidates using BGE embedding similarity combined with heuristic bonuses."""
    if not cands: return []
    if not _ML_OK or not _EMB_MODEL or _CACHED_EMBS is None or _CACHED_IDS is None:
        print("[Info] ML models/cache unavailable for pre-filtering; using heuristic shortlist.")
        return _heuristic_shortlist(q_user, cands, percent, min_k)

    uid2idx = {uid: i for i, uid in enumerate(_CACHED_IDS)}
    cand_uids = [rec["user"]["user_id"] for rec in cands]
    cand_row_idx = [uid2idx.get(u) for u in cand_uids]

    # Encode query user profile
    qv = _EMB_MODEL.encode(
        [query_text(q_user)], convert_to_numpy=True, normalize_embeddings=True
    )[0].astype("float16")

    # Calculate similarity scores
    sims_full = np.full(len(cands), 0.30, dtype="float32")
    valid_pairs = [(i, idx) for i, idx in enumerate(cand_row_idx) if idx is not None]
    if valid_pairs:
        sub_embs = _CACHED_EMBS[[idx for _, idx in valid_pairs]]
        sims = (sub_embs @ qv).astype("float32")
        s_min, s_max = sims.min(), sims.max()
        sims_norm = (sims - s_min) / (s_max - s_min) if s_max > s_min else 0.5
        for (pos, _), val in zip(valid_pairs, sims_norm):
            sims_full[pos] = float(val)

    # Heuristic bonuses
    q_interests = set(q_user.get("interests", []))
    q_values = set(q_user.get("values", []))
    bonuses = [
        0.6 * jaccard(list(q_interests), rec["user"].get("interests", [])) + 0.4 * jaccard(list(q_values), rec["user"].get("values", []))
        for rec in cands
    ]
    
    # Combine scores and sort
    combined = [(cands[i], float(sims_full[i]) + float(bonuses[i])) for i in range(len(cands))]
    combined.sort(key=lambda x: x[1], reverse=True)

    k = min(len(combined), max(min_k, int(math.ceil(len(combined) * percent))))
    return [rec for rec, _ in combined[:k]]

# --- Llama Ranking ---

def _extract_json(text: str) -> Optional[dict]:
    """Robustly extracts the first JSON object from a string."""
    match = re.search(r'\{.*\}', text, re.DOTALL)
    if not match: return None
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None

def build_llm_prompt(q_user: Dict[str, Any], shortlist: List[Dict[str, Any]]) -> str:
    """Constructs the full prompt for the Llama model."""
    head = (
        "You are a precise travel-match expert. Rank candidates for holistic trip compatibility.\n"
        "Consider: personality fit, conflict style, shared/complementary interests, languages, pace, budget, diet/substances, "
        "and values.\n"
        "Return ONLY valid JSON with key 'matches' as an array of objects with fields: user_id, name, explanation "
        "(ONE sentence, must start with 'For you,' and be specific), compatibility_score (0.0–1.0).\n\n"
        f"Query User:\n{query_text(q_user)}\n\nCandidates:\n"
    )
    body = "\n".join(
        f"[{i+1}] user_id={rec['user'].get('user_id')} | {summarize_user(rec['user'], rec.get('mm'))}"
        for i, rec in enumerate(shortlist)
    )
    return head + body

def get_llama_ranking_local(prompt: str, model_data: Dict, max_new_tokens=120, temperature=0.0) -> Optional[str]:
    """Generates a response from a locally loaded Llama model."""
    tok, mdl = model_data["tokenizer"], model_data["model"]
    inputs = tok(prompt, return_tensors="pt", truncation=True, max_length=4096)
    inputs = {k: v.to(mdl.device) for k, v in inputs.items()}

    try:
        with torch.inference_mode():
            out = mdl.generate(
                **inputs, max_new_tokens=max_new_tokens, do_sample=(temperature > 0.0),
                temperature=temperature, use_cache=True, pad_token_id=tok.eos_token_id
            )
        return tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    except Exception as e:
        print(f"[Error] Local LLM generation failed: {e}")
        return None

def craft_specific_reason(q: Dict[str, Any], u: Dict[str, Any]) -> str:
    """Creates a human-readable explanation for why two users are a good match."""
    shared_i = sorted(set(q.get("interests", [])) & set(u.get("interests", [])))
    hooks = []
    if shared_i: hooks.append(f"a shared love for {', '.join(shared_i[:2])}")
    if (q.get("travel_prefs") or {}).get("pace") == (u.get("travel_prefs") or {}).get("pace"):
        hooks.append(f"a matching {(q.get('travel_prefs') or {}).get('pace')} pace")
    if not hooks: hooks.append("complementary interests")
    return "For you, this match fits because of " + ", ".join(hooks) + "."

def llm_rank_fallback(q_user: Dict[str, Any], shortlist: List[Dict[str, Any]], out_top: int) -> List[Dict[str, Any]]:
    """A fallback ranking method using heuristics if the LLM fails."""
    results = []
    for rec in shortlist[:out_top]:
        u = rec["user"]
        score = 0.70 + 0.25 * jaccard(q_user.get("interests", []), u.get("interests", []))
        results.append({
            "user_id": u.get("user_id"), "name": u.get("name"),
            "explanation": craft_specific_reason(q_user, u),
            "compatibility_score": round(min(score, 0.99), 2)
        })
    return results

def llm_rank(q_user: Dict[str, Any], shortlist: List[Dict[str, Any]], out_top: int = 5) -> List[Dict[str, Any]]:
    """Ranks a shortlist of candidates using the loaded Llama model."""
    shortlist_for_llm = shortlist[:10]
    model_to_use = _LLM_FINETUNED or _LLM_BASE
    
    if not model_to_use:
        print("[Warning] No LLM loaded, using heuristic fallback for final ranking.")
        return llm_rank_fallback(q_user, shortlist_for_llm, out_top)

    prompt = build_llm_prompt(q_user, shortlist_for_llm)
    text = get_llama_ranking_local(prompt, model_to_use)

    if not text:
        return llm_rank_fallback(q_user, shortlist_for_llm, out_top)

    parsed = _extract_json(text)
    if parsed and isinstance(parsed.get("matches"), list):
        # Clean and validate the structured output from the LLM
        cleaned = []
        seen_ids = set()
        for m in parsed["matches"][:out_top]:
            uid = m.get("user_id")
            if uid and uid not in seen_ids:
                try:
                    cleaned.append({
                        "user_id": str(uid),
                        "name": str(m.get("name", "")),
                        "explanation": str(m.get("explanation", "")),
                        "compatibility_score": float(m.get("compatibility_score", 0.0))
                    })
                    seen_ids.add(uid)
                except (ValueError, TypeError):
                    continue # Skip malformed entries
        if cleaned:
            return cleaned

    print("[Warning] LLM response parsing failed. Using fallback.")
    return llm_rank_fallback(q_user, shortlist_for_llm, out_top)

# --- Main Orchestrator Function ---

def find_matches(query_user: Dict[str, Any]) -> List[Dict[str, Any]]:
    """The main pipeline function to find matches for a given user."""
    if not CANDIDATE_POOL:
        raise RuntimeError("Candidate pool is not loaded. Ensure load_pool() is called at startup.")

    print(f"\n🚀 Starting match process for: {query_user.get('name')}")
    
    # 1. Hard Prefilters
    t0 = time.time()
    hard_filtered = hard_prefilter(query_user, CANDIDATE_POOL)
    print(f"✅ Hard prefilter: {len(CANDIDATE_POOL)} -> {len(hard_filtered)} candidates (in {time.time()-t0:.2f}s)")
    if not hard_filtered:
        return []

    # 2. AI Prefilter (BGE)
    t1 = time.time()
    shortlist = ai_prefilter(query_user, hard_filtered)
    print(f"✅ AI prefilter: {len(hard_filtered)} -> {len(shortlist)} candidates (in {time.time()-t1:.2f}s)")
    if not shortlist:
        return []

    # 3. Final Ranking (Llama)
    t2 = time.time()
    final_matches = llm_rank(query_user, shortlist, out_top=5)
    print(f"✅ Llama ranking: {len(shortlist)} -> {len(final_matches)} matches (in {time.time()-t2:.2f}s)")

    # 4. Filter by quality score
    high_quality = [m for m in final_matches if m.get("compatibility_score", 0) >= 0.75]
    print(f"✅ Found {len(high_quality)} high-quality matches (score >= 75%).")
    
    return high_quality if high_quality else final_matches
