# serve_llama.py
from __future__ import annotations
import os, typing as t, socket
from pathlib import Path
from fastapi import FastAPI, Body
from pydantic import BaseModel
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# ───── Warnings off ─────
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", message=".*TRANSFORMERS_CACHE.*")
warnings.filterwarnings("ignore", message=".*do_sample.*")
warnings.filterwarnings("ignore", message=".*temperature.*")
warnings.filterwarnings("ignore", message=".*top_p.*")

# ───── Env (offline-friendly caches) ─────
BASE_DIR = Path(__file__).resolve().parent
os.environ.setdefault("HF_HOME", os.getenv("HF_HOME", str(BASE_DIR / "models_cache")))
os.environ.setdefault("TRANSFORMERS_CACHE", os.getenv("TRANSFORMERS_CACHE", str(BASE_DIR / "models_cache" / "transformers")))
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", os.getenv("HF_HUB_ENABLE_HF_TRANSFER", "0"))
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", os.getenv("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1"))

# Hugging Face token
HF_TOKEN = os.getenv("HF_TOKEN")
if HF_TOKEN:
    os.environ["HF_TOKEN"] = HF_TOKEN

# ───── Model paths (Hugging Face Hub) ─────
FINETUNED_PATH = os.getenv("LLAMA_FINETUNED_MODEL_PATH", "abdulghaffaransari9/rovermitra-travel-matcher")
BASE_PATH      = os.getenv("LLAMA_BASE_MODEL_PATH", "abdulghaffaransari9/rovermitra-llama-base")

# ───── Globals ─────
_TOKENIZER = None
_MODEL = None
_DEVICE = "cpu"
_MODEL_NAME = "unloaded"
_MODEL_LOADING = False
_MODEL_LOADED = False

def _try_import_bnb():
    try:
        from transformers import BitsAndBytesConfig  # type: ignore
        return BitsAndBytesConfig
    except Exception:
        return None

def load_model_once():
    """Load finetuned model if available; otherwise base. 4-bit on CUDA if bitsandbytes exists."""
    global _TOKENIZER, _MODEL, _DEVICE, _MODEL_NAME, _MODEL_LOADING, _MODEL_LOADED
    if _MODEL is not None or _MODEL_LOADING:
        return
    
    _MODEL_LOADING = True

    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM

    use_cuda = torch.cuda.is_available()
    _DEVICE = "cuda:0" if use_cuda else "cpu"

    bnb_cfg_cls = _try_import_bnb()
    kwargs: dict = {
        "trust_remote_code": True, 
        "low_cpu_mem_usage": True,
        "torch_dtype": torch.float16 if use_cuda else torch.float32,
        "device_map": "cpu" if not use_cuda else "auto"
    }

    if use_cuda and bnb_cfg_cls is not None:
        quant_cfg = bnb_cfg_cls(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True)
        kwargs.update(dict(device_map="auto", quantization_config=quant_cfg))

    def _load(model_path: str):
        print(f"Loading model from Hugging Face: {model_path}...")
        tok = AutoTokenizer.from_pretrained(model_path, use_auth_token=HF_TOKEN)
        mdl = AutoModelForCausalLM.from_pretrained(model_path, use_auth_token=HF_TOKEN, **kwargs)
        if mdl.config.pad_token_id is None and tok.eos_token_id is not None:
            mdl.config.pad_token_id = tok.eos_token_id
        return tok, mdl

    # Try finetuned first
    try:
        print(f"Attempting to load fine-tuned model: {FINETUNED_PATH}")
        tok, mdl = _load(FINETUNED_PATH)
        _TOKENIZER, _MODEL = tok, mdl
        _MODEL_NAME = "finetuned"
        print(f"✅ Fine-tuned model loaded successfully")
    except Exception as e:
        print(f"❌ Fine-tuned model failed: {e}")
        print(f"Falling back to base model: {BASE_PATH}")
        try:
            tok, mdl = _load(BASE_PATH)
            _TOKENIZER, _MODEL = tok, mdl
            _MODEL_NAME = "base"
            print(f"✅ Base model loaded successfully")
        except Exception as e2:
            print(f"❌ Base model also failed: {e2}")
            print("🚨 No model could be loaded!")
            _MODEL_LOADING = False
            return

    if _DEVICE == "cpu":
        _MODEL = _MODEL.to("cpu")
    
    _MODEL_LOADED = True
    _MODEL_LOADING = False
    print(f"✅ Model loaded successfully: {_MODEL_NAME}")

# ───── FastAPI app ─────
app = FastAPI(title="RoverMitra Llama Server", version="1.0.0")

class RankReq(BaseModel):
    prompt: str
    max_new_tokens: int = 512
    temperature: float = 0.2
    top_p: float = 0.9

@app.on_event("startup")
def _startup():
    print("🚀 Server startup initiated")
    import threading
    # Load model in background thread to not block startup
    threading.Thread(target=load_model_once, daemon=True).start()
    print("🚀 Model loading started in background thread")

@app.get("/")
def root():
    return {"message": "RoverMitra Llama Server is running", "status": "ok"}

@app.get("/health")
def health():
    global _MODEL_LOADED, _MODEL_LOADING
    if _MODEL_LOADED:
        return {"ok": True, "device": _DEVICE, "model": _MODEL_NAME, "status": "ready"}
    elif _MODEL_LOADING:
        return {"ok": True, "device": _DEVICE, "model": "loading", "status": "loading"}
    else:
        return {"ok": True, "device": _DEVICE, "model": "unloaded", "status": "starting"}

@app.post("/rank")
def rank(req: RankReq = Body(...)):
    global _MODEL_LOADED, _MODEL_LOADING
    if not _MODEL_LOADED:
        if _MODEL_LOADING:
            return {"error": "Model is still loading, please wait"}
        else:
            return {"error": "Model not loaded"}
    
    import torch
    tok, mdl = _TOKENIZER, _MODEL

    inputs = tok(req.prompt, return_tensors="pt", truncation=True, max_length=4096)
    if _DEVICE.startswith("cuda"):
        inputs = {k: v.to(mdl.device) for k, v in inputs.items()}

    with torch.no_grad():
        out = mdl.generate(
            **inputs,
            max_new_tokens=req.max_new_tokens,
            do_sample=(req.temperature > 0.0),
            temperature=req.temperature,
            top_p=req.top_p,
            pad_token_id=tok.eos_token_id,
            eos_token_id=tok.eos_token_id,
        )
    gen_ids = out[0][inputs["input_ids"].shape[1]:]
    text = tok.decode(gen_ids, skip_special_tokens=True)
    return {"text": text}

# ───── Port selection helpers ─────
def _first_free_port(
    host: str = "0.0.0.0",
    start: int = 8000,
    end: int = 8010,
) -> int:
    """Return the first free TCP port in [start, end]. If none, return an ephemeral port."""
    for p in range(start, end + 1):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                s.bind((host, p))
                return p  # free
            except OSError:
                continue
    # fallback: ephemeral
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((host, 0))
        return s.getsockname()[1]

if __name__ == "__main__":
    import uvicorn
    
    # Railway provides PORT environment variable
    port = int(os.getenv("PORT", "8000"))
    host = os.getenv("HOST", "0.0.0.0")
    
    print(f"[serve_llama] Starting server on {host}:{port}")
    print(f"[serve_llama] Environment: PORT={port}, HOST={host}")
    
    # one worker so the model stays in a single process (keeps memory hot)
    uvicorn.run("serve_llama:app", host=host, port=port, workers=1)
