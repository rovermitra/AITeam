# config.py
import os
from pathlib import Path

# Base directory of the project
BASE_DIR = Path(__file__).resolve().parent

# --- Project paths ---
USERS_CORE_PATH  = BASE_DIR / "users/data/users_core.json"
MM_PATH          = BASE_DIR / "MatchMaker/data/matchmaker_profiles.json"
LOCAL_DB_PATH    = BASE_DIR / "data/travel_ready_user_profiles.json"

# --- Model paths ---
MODELS_DIR               = BASE_DIR / "models"
BGE_M3_PATH              = MODELS_DIR / "bge-m3"
LLAMA_FINETUNED_PATH     = MODELS_DIR / "llama-travel-matcher"
LLAMA_BASE_PATH          = MODELS_DIR / "llama-3.2-3b-instruct"

# --- Cache paths ---
CACHE_DIR = BASE_DIR / "models_cache"
UIDS_PATH = CACHE_DIR / "bge_user_ids.npy"
EMB_PATH  = CACHE_DIR / "bge_embeds_fp16.npy"

# --- Environment setup for ML libraries ---
def setup_environment():
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
    os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
    os.environ.setdefault("HF_HOME", str(BASE_DIR / "models_cache"))
    os.environ.setdefault("TRANSFORMERS_CACHE", str(BASE_DIR / "models_cache" / "transformers"))
    os.environ.setdefault("TORCH_HOME", str(BASE_DIR / "models_cache" / "torch"))
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")