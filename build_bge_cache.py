#!/usr/bin/env python3
# Build and cache BGE-M3 embeddings for all candidates (one-time)

import os
import sys
import numpy as np
from pathlib import Path

# Reuse your project utils (imports only light helpers/paths)
from Updated_main import load_pool, summarize_user, MODELS_DIR, CACHE_DIR, UIDS_PATH, EMB_PATH

def pick_device() -> str:
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        # Apple Metal (MPS) fallback
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"

def get_model(device: str):
    from sentence_transformers import SentenceTransformer
    local_path = MODELS_DIR / "bge-m3"
    if local_path.exists():
        return SentenceTransformer(str(local_path), device=device)
    # Fallback to hub (will respect TRANSFORMERS_CACHE / HF_HOME envs)
    return SentenceTransformer("BAAI/bge-m3", device=device)

def main():
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    pool = load_pool()
    if not pool:
        raise SystemExit("No candidates found. Ensure users/data/users_core.json and MatchMaker/data/matchmaker_profiles.json exist.")

    users_txt, user_ids = [], []
    seen = set()
    for rec in pool:
        u, mm = rec["user"], rec.get("mm")
        uid = u.get("user_id")
        if not uid or uid in seen:
            continue
        seen.add(uid)
        users_txt.append(summarize_user(u, mm))
        user_ids.append(uid)

    if not users_txt or len(users_txt) != len(user_ids):
        raise SystemExit(f"Nothing to encode or length mismatch: texts={len(users_txt)} ids={len(user_ids)}")

    device = pick_device()
    print(f"🔌 Using device: {device}")

    try:
        model = get_model(device)
    except Exception as e:
        raise SystemExit(f"Failed to load BGE-M3 model: {e}")

    # Batch size heuristic
    if device == "cuda":
        batch_size = 512
    elif device == "mps":
        batch_size = 256
    else:
        batch_size = 128  # safer on CPU

    print(f"🧮 Encoding {len(users_txt)} candidates with batch_size={batch_size} (normalized)…")
    try:
        embs = model.encode(
            users_txt,
            batch_size=batch_size,
            convert_to_numpy=True,
            show_progress_bar=True,
            normalize_embeddings=True
        ).astype("float16")
    except Exception as e:
        raise SystemExit(f"Encoding failed: {e}")

    if embs.shape[0] != len(user_ids):
        raise SystemExit(f"Embedding count mismatch: embs={embs.shape[0]} ids={len(user_ids)}")

    # Save cache
    np.save(UIDS_PATH, np.array(user_ids, dtype=object))
    np.save(EMB_PATH,  embs)
    print("✅ BGE cache built:")
    print("   ", UIDS_PATH)
    print("   ", EMB_PATH)
    print("ℹ️  Rows (N), Dim (D):", embs.shape)

if __name__ == "__main__":
    main()
