# serve_llama_CPU.py
from __future__ import annotations
import os, typing as t, socket
from pathlib import Path
from fastapi import FastAPI, Body
from pydantic import BaseModel

# ───── Warnings off ─────
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", message=".*TRANSFORMERS_CACHE.*")
warnings.filterwarnings("ignore", message=".*do_sample.*")
warnings.filterwarnings("ignore", message=".*temperature.*")
warnings.filterwarnings("ignore", message=".*top_p.*")

# ───── Env (offline-friendly caches) - FORCE CPU ─────
BASE_DIR = Path(__file__).resolve().parent
os.environ["CUDA_VISIBLE_DEVICES"] = ""  # Force CPU usage
os.environ.setdefault("HF_HOME", str(BASE_DIR / "models_cache"))
os.environ.setdefault("TRANSFORMERS_CACHE", str(BASE_DIR / "models_cache" / "transformers"))
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")

# ───── Model paths ─────
FINETUNED_PATH = BASE_DIR / "models" / "llama-travel-matcher"
BASE_PATH      = BASE_DIR / "models" / "llama-3.2-3b-instruct"

# ───── Globals ─────
_TOKENIZER = None
_MODEL = None
_DEVICE = "cpu"  # Force CPU
_MODEL_NAME = "unloaded"

def load_model_once():
    """Load finetuned model if available; otherwise base. CPU-only mode."""
    global _TOKENIZER, _MODEL, _DEVICE, _MODEL_NAME
    if _MODEL is not None:
        return

    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM

    # Force CPU usage
    _DEVICE = "cpu"
    print("🔄 Loading Llama model on CPU...")

    # CPU-only configuration
    kwargs: dict = {
        "trust_remote_code": True, 
        "low_cpu_mem_usage": True,
        "torch_dtype": torch.float32,  # Use float32 for CPU
        "device_map": "cpu"           # Force CPU
    }

    def _load(path: Path):
        print(f"Loading model from {path}...")
        tok = AutoTokenizer.from_pretrained(str(path), use_fast=True)
        mdl = AutoModelForCausalLM.from_pretrained(str(path), **kwargs)
        mdl.config.use_cache = True
        if mdl.config.pad_token_id is None and tok.eos_token_id is not None:
            mdl.config.pad_token_id = tok.eos_token_id
        print(f"✅ Model loaded successfully on CPU")
        return tok, mdl

    # Try finetuned first
    try:
        print("Attempting to load fine-tuned model...")
        tok, mdl = _load(FINETUNED_PATH)
        _TOKENIZER, _MODEL = tok, mdl
        _MODEL_NAME = FINETUNED_PATH.name
        print(f"✅ Fine-tuned Llama loaded: {_MODEL_NAME}")
    except Exception as e:
        print(f"Fine-tuned model failed: {e}")
        print("Attempting to load base model...")
        try:
            tok, mdl = _load(BASE_PATH)
            _TOKENIZER, _MODEL = tok, mdl
            _MODEL_NAME = BASE_PATH.name
            print(f"✅ Base Llama loaded: {_MODEL_NAME}")
        except Exception as e2:
            print(f"❌ Both models failed to load: {e2}")
            raise RuntimeError("No Llama model could be loaded on CPU")

    # Ensure model is on CPU
    _MODEL = _MODEL.to("cpu")
    print(f"🚀 Llama server ready on CPU with model: {_MODEL_NAME}")

# ───── FastAPI app ─────
app = FastAPI(title="RoverMitra Llama Server (CPU-only)", version="1.0.0")

class RankReq(BaseModel):
    prompt: str
    max_new_tokens: int = 512
    temperature: float = 0.2
    top_p: float = 0.9

@app.on_event("startup")
def _startup():
    print("🚀 Starting RoverMitra Llama Server (CPU-only)...")
    load_model_once()

@app.get("/health")
def health():
    return {"ok": True, "device": _DEVICE, "model": _MODEL_NAME, "mode": "cpu-only"}

@app.post("/rank")
def rank(req: RankReq = Body(...)):
    import torch
    load_model_once()
    tok, mdl = _TOKENIZER, _MODEL

    # Prepare inputs
    inputs = tok(req.prompt, return_tensors="pt", truncation=True, max_length=4096)
    # Force CPU device
    inputs = {k: v.to("cpu") for k, v in inputs.items()}

    try:
        with torch.inference_mode():
            out = mdl.generate(
                **inputs,
                max_new_tokens=req.max_new_tokens,
                do_sample=(req.temperature > 0.0),
                temperature=req.temperature,
                top_p=req.top_p,
                use_cache=True,
                pad_token_id=tok.eos_token_id,
                eos_token_id=tok.eos_token_id,
            )
        gen_ids = out[0][inputs["input_ids"].shape[1]:]
        text = tok.decode(gen_ids, skip_special_tokens=True)
        return {"text": text}
    except Exception as e:
        print(f"❌ Generation failed: {e}")
        return {"text": "", "error": str(e)}

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
    host = os.getenv("RM_LLAMA_HOST", "0.0.0.0")
    start = int(os.getenv("RM_LLAMA_PORT_START", "8000"))
    end   = int(os.getenv("RM_LLAMA_PORT_END", "8010"))
    port  = _first_free_port(host, start, end)

    print(f"🖥️  RoverMitra Llama Server (CPU-only)")
    print(f"📡 Host: {host}, Port: {port} (searched {start}..{end})")
    print(f"🔧 Mode: CPU-only (no GPU acceleration)")
    print(f"🌐 Server will be available at: http://{host}:{port}")
    print(f"📊 Health check: http://{host}:{port}/health")
    print(f"🤖 Ranking endpoint: http://{host}:{port}/rank")
    print("=" * 60)
    
    # one worker so the model stays in a single process (keeps memory hot)
    uvicorn.run("serve_llama_CPU:app", host=host, port=port, workers=1)
