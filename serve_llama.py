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

def _try_import_bnb():
    try:
        from transformers import BitsAndBytesConfig  # type: ignore
        return BitsAndBytesConfig
    except Exception:
        return None

def load_model_once():
    """Load finetuned model if available; otherwise base. 4-bit on CUDA if bitsandbytes exists."""
    global _TOKENIZER, _MODEL, _DEVICE, _MODEL_NAME
    if _MODEL is not None:
        return

    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM

    use_cuda = torch.cuda.is_available()
    _DEVICE = "cuda:0" if use_cuda else "cpu"

    bnb_cfg_cls = _try_import_bnb()
    kwargs: dict = {"trust_remote_code": True, "low_cpu_mem_usage": True}

    if use_cuda and bnb_cfg_cls is not None:
        quant_cfg = bnb_cfg_cls(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True)
        kwargs.update(dict(device_map="auto", quantization_config=quant_cfg))
    else:
        import torch as _torch
        kwargs.update(dict(torch_dtype=_torch.float16 if use_cuda else _torch.float32, device_map=None))

    def _load(model_path: str):
        print(f"Loading model from Hugging Face: {model_path}...")
        tok = AutoTokenizer.from_pretrained(model_path, use_auth_token=HF_TOKEN)
        mdl = AutoModelForCausalLM.from_pretrained(model_path, use_auth_token=HF_TOKEN, **kwargs)
        if mdl.config.pad_token_id is None and tok.eos_token_id is not None:
            mdl.config.pad_token_id = tok.eos_token_id
        return tok, mdl

    # Try finetuned first
    try:
        tok, mdl = _load(FINETUNED_PATH)
        _TOKENIZER, _MODEL = tok, mdl
        _MODEL_NAME = FINETUNED_PATH.name
    except Exception:
        tok, mdl = _load(BASE_PATH)
        _TOKENIZER, _MODEL = tok, mdl
        _MODEL_NAME = BASE_PATH.name

    if _DEVICE == "cpu":
        _MODEL = _MODEL.to("cpu")

# ───── FastAPI app ─────
app = FastAPI(title="RoverMitra Llama Server", version="1.0.0")

class RankReq(BaseModel):
    prompt: str
    max_new_tokens: int = 512
    temperature: float = 0.2
    top_p: float = 0.9

@app.on_event("startup")
def _startup():
    load_model_once()

@app.get("/health")
def health():
    return {"ok": True, "device": _DEVICE, "model": _MODEL_NAME}

@app.post("/rank")
def rank(req: RankReq = Body(...)):
    import torch
    load_model_once()
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
    host = os.getenv("RM_LLAMA_HOST", "0.0.0.0")
    start = int(os.getenv("RM_LLAMA_PORT_START", "8000"))
    end   = int(os.getenv("RM_LLAMA_PORT_END", "8010"))
    port  = _first_free_port(host, start, end)

    print(f"[serve_llama] Using host={host} port={port} (searched {start}..{end})")
    # one worker so the model stays in a single process (keeps memory hot)
    uvicorn.run("serve_llama:app", host=host, port=port, workers=1)
