# serve_llama.py
from __future__ import annotations
import os, json, typing as t
from pathlib import Path
from fastapi import FastAPI, Body
from pydantic import BaseModel

# Suppress warnings
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", message=".*TRANSFORMERS_CACHE.*")
warnings.filterwarnings("ignore", message=".*do_sample.*")
warnings.filterwarnings("ignore", message=".*temperature.*")
warnings.filterwarnings("ignore", message=".*top_p.*")

# ---------------- Env: safe defaults (no hub access needed at runtime) -------------
BASE_DIR = Path(__file__).resolve().parent
os.environ.setdefault("HF_HOME", str(BASE_DIR / "models_cache"))
os.environ.setdefault("TRANSFORMERS_CACHE", str(BASE_DIR / "models_cache" / "transformers"))
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")

# ---------------- Model paths ------------------------------------------------------
FINETUNED_PATH = BASE_DIR / "models" / "llama-travel-matcher"
BASE_PATH      = BASE_DIR / "models" / "llama-3.2-3b-instruct"

# ---------------- Globals (singleton) ---------------------------------------------
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
    global _TOKENIZER, _MODEL, _DEVICE, _MODEL_NAME
    if _MODEL is not None:
        return

    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM

    use_cuda = torch.cuda.is_available()
    _DEVICE = f"cuda:0" if use_cuda else "cpu"

    bnb_cfg_cls = _try_import_bnb()
    quant_cfg = None
    kwargs: dict = {
        "trust_remote_code": True,
        "low_cpu_mem_usage": True,
    }

    if use_cuda and bnb_cfg_cls is not None:
        # 4-bit quant for speed/VRAM when available
        quant_cfg = bnb_cfg_cls(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True)
        kwargs.update(dict(device_map="auto", quantization_config=quant_cfg))
    else:
        # CPU path or GPU without bitsandbytes
        import torch as _torch
        kwargs.update(dict(torch_dtype=_torch.float16 if use_cuda else _torch.float32, device_map=None))

    def _load(path: Path):
        tok = AutoTokenizer.from_pretrained(str(path))
        mdl = AutoModelForCausalLM.from_pretrained(str(path), **kwargs)
        # ensure pad token
        mdl.config.pad_token_id = tok.eos_token_id
        return tok, mdl

    # try finetuned first
    try:
        tok, mdl = _load(FINETUNED_PATH)
        _TOKENIZER, _MODEL = tok, mdl
        _MODEL_NAME = FINETUNED_PATH.name
    except Exception as e1:
        # fallback to base
        tok, mdl = _load(BASE_PATH)
        _TOKENIZER, _MODEL = tok, mdl
        _MODEL_NAME = BASE_PATH.name

    # if CPU path, move explicitly
    if _DEVICE == "cpu":
        _MODEL = _MODEL.to("cpu")

# ---------------- FastAPI app ------------------------------------------------------
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

    # If your tokenizer supports chat template, keep it simple:
    # Here we just pass the full prompt (you already build JSON-friendly prompts in your pipeline)
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
    # Return only the new text (after the prompt)
    gen_ids = out[0][inputs["input_ids"].shape[1]:]
    text = tok.decode(gen_ids, skip_special_tokens=True)
    return {"text": text}

if __name__ == "__main__":
    import uvicorn
    # one worker so the model stays in a single process (keeps memory hot)
    uvicorn.run("serve_llama:app", host="0.0.0.0", port=8000, workers=1)
