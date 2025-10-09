#!/usr/bin/env python3
# Build and cache BGE-M3 embeddings for all candidates (one-time)
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer

# Reuse your project utils
from Updated_main import load_pool, summarize_user, MODELS_DIR, CACHE_DIR, UIDS_PATH, EMB_PATH

def main():
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    pool = load_pool()
    if not pool:
        raise SystemExit("No candidates found. Ensure users/data/users_core.json and MatchMaker/data/matchmaker_profiles.json exist.")

    users_txt, user_ids = [], []
    for rec in pool:
        u, mm = rec["user"], rec.get("mm")
        users_txt.append(summarize_user(u, mm))
        user_ids.append(u.get("user_id"))

    device = "cuda"  # fast
    model = SentenceTransformer(str(MODELS_DIR / "bge-m3"), device=device)

    # Big batches = fast; normalize for cosine
    embs = model.encode(
        users_txt,
        batch_size=512,
        convert_to_numpy=True,
        show_progress_bar=True,
        normalize_embeddings=True
    ).astype("float16")

    np.save(UIDS_PATH, np.array(user_ids, dtype=object))
    np.save(EMB_PATH,  embs)
    print("✅ BGE cache built:")
    print("   ", UIDS_PATH)
    print("   ", EMB_PATH)

if __name__ == "__main__":
    main()
