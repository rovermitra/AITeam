#!/usr/bin/env python3
"""
Automated test script for RoverMitra pipeline (strict Llama ranking)

- Asks which module to test (e.g., "main" or "main_fast").
- Loads candidate pool, runs hard prefilter, AI prefilter, then ALWAYS uses Llama (llm_rank).
- Prints compact results and basic timings.

Speed notes:
- Keeps AI prefilter relatively small for faster runs: percent=0.02, min_k=20
- Optionally caps pool via POOL_CAP (default 3000) to keep tests snappy.

Run:
  python test_automated_llama.py
"""

import os
import sys
import time
import json
import importlib
from pathlib import Path

# -----------------------------
# Config knobs for test speed
# -----------------------------
POOL_CAP = int(os.getenv("RM_TEST_POOL_CAP", "3000"))  # cap candidate pool for faster tests
PREFILTER_PERCENT = float(os.getenv("RM_TEST_PREFILTER_PERCENT", "0.02"))
PREFILTER_MIN_K = int(os.getenv("RM_TEST_PREFILTER_MIN_K", "20"))
TOP_K = int(os.getenv("RM_TEST_TOP_K", "5"))

def choose_module_name() -> str:
    print("🔧 Testing main module...")
    return "main"

def import_pipeline_module(mod_name: str):
    try:
        mod = importlib.import_module(mod_name)
    except Exception as e:
        print(f"❌ Could not import module '{mod_name}': {e}")
        sys.exit(1)

    required = ["load_pool", "hard_prefilter", "ai_prefilter", "llm_rank", "append_to_json", "LOCAL_DB_PATH"]
    missing = [name for name in required if not hasattr(mod, name)]
    if missing:
        print(f"❌ Module '{mod_name}' is missing required attributes: {', '.join(missing)}")
        sys.exit(1)

    return mod

def demo_test_users():
    """Four varied profiles to showcase different travel personalities and authentic adjectives."""
    return [
        {
            'email': 'test_user_1@rovermitra.example',
            'name': 'Alex Berlin',
            'age': 28,
            'gender': 'Male',
            'home_base': {'city': 'Berlin', 'country': 'Germany'},
            'languages': ['en', 'de'],
            'interests': ['museums', 'architecture', 'history', 'photography', 'bookstores'],
            'values': ['culture', 'learning'],
            'bio': 'Berlin-based culture enthusiast who loves museums, architecture, and capturing moments through photography',
            'travel_prefs': {'pace': 'balanced'},
            'budget': {'amount': 150, 'currency': 'EUR'},
            'diet_health': {'diet': 'none'},
            'comfort': {'smoking': 'never', 'alcohol': 'moderate'},
            'work': {'remote_work_ok': False},
            'companion_preferences': {
                'genders_ok': ['Men']
            },
            'faith': {
                'consider_in_matching': False,
                'religion': '',
                'policy': 'open',
                'visibility': 'private'
            },
            'privacy': {'share_home_city': True}
        },
        {
            'email': 'test_user_2@rovermitra.example',
            'name': 'Maria Munich',
            'age': 32,
            'gender': 'Female',
            'home_base': {'city': 'Munich', 'country': 'Germany'},
            'languages': ['de', 'en', 'it'],
            'interests': ['food tours', 'vineyards', 'scenic trains', 'coffee crawls', 'markets'],
            'values': ['food', 'luxury-taste'],
            'bio': 'Munich foodie and wine enthusiast who enjoys scenic train journeys and exploring local markets',
            'travel_prefs': {'pace': 'relaxed'},
            'budget': {'amount': 250, 'currency': 'EUR'},
            'diet_health': {'diet': 'vegetarian'},
            'comfort': {'smoking': 'never', 'alcohol': 'social'},
            'work': {'remote_work_ok': True},
            'companion_preferences': {
                'genders_ok': ['Women']
            },
            'faith': {
                'consider_in_matching': True,
                'religion': 'Christian',
                'policy': 'prefer_same',
                'visibility': 'private'
            },
            'privacy': {'share_home_city': True}
        },
        {
            'email': 'test_user_3@rovermitra.example',
            'name': 'Tom Karachi',
            'age': 24,
            'gender': 'Male',
            'home_base': {'city': 'Karachi', 'country': 'Pakistan'},
            'languages': ['en', 'ur'],
            'interests': ['festivals', 'live music', 'beaches', 'diving', 'nightlife'],
            'values': ['adventure', 'social'],
            'bio': 'Karachi adventure seeker who loves music festivals, beach diving, and vibrant nightlife',
            'travel_prefs': {'pace': 'packed'},
            'budget': {'amount': 80, 'currency': 'EUR'},
            'diet_health': {'diet': 'halal'},
            'comfort': {'smoking': 'occasionally', 'alcohol': 'none'},
            'work': {'remote_work_ok': False},
            'companion_preferences': {
                'genders_ok': ["I'm open to travel with anyone"]
            },
            'faith': {
                'consider_in_matching': True,
                'religion': 'Islam',
                'policy': 'same_only',
                'visibility': 'private'
            },
            'privacy': {'share_home_city': False}
        },
        {
            'email': 'test_user_4@rovermitra.example',
            'name': 'Priya Delhi',
            'age': 26,
            'gender': 'Female',
            'home_base': {'city': 'Delhi', 'country': 'India'},
            'languages': ['en', 'hi'],
            'interests': ['temples', 'yoga', 'meditation', 'wellness', 'thermal baths', 'spa'],
            'values': ['spirituality', 'wellness'],
            'bio': 'Delhi wellness warrior who finds peace in temples, yoga, meditation, and thermal spa experiences',
            'travel_prefs': {'pace': 'relaxed'},
            'budget': {'amount': 120, 'currency': 'EUR'},
            'diet_health': {'diet': 'vegetarian'},
            'comfort': {'smoking': 'never', 'alcohol': 'none'},
            'work': {'remote_work_ok': True},
            'companion_preferences': {
                'genders_ok': ["I'm open to travel with anyone"]
            },
            'faith': {
                'consider_in_matching': True,
                'religion': 'Hindu',
                'policy': 'prefer_same',
                'visibility': 'private'
            },
            'privacy': {'share_home_city': True}
        }
    ]

def run_core_tests(mod):
    print("🧪 Testing RoverMitra core functionality with faith preferences (strict Llama)…")

    # Show basic env hints (helps diagnose CPU/GPU issues quickly)
    print(f"⚙️  CUDA_VISIBLE_DEVICES = {os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')}")
    print(f"⚙️  TRANSFORMERS_CACHE   = {os.environ.get('TRANSFORMERS_CACHE', '<unset>')}")
    print(f"⚙️  HF_HOME              = {os.environ.get('HF_HOME', '<unset>')}")

    # 1) Load pool
    t0 = time.time()
    pool = mod.load_pool()
    t1 = time.time()
    print(f"\n1️⃣ Candidate pool loaded: {len(pool)} users (in {t1 - t0:.2f}s)")

    if not pool:
        print("❌ No candidates found. Please check users/data/users_core.json and MatchMaker/data/matchmaker_profiles.json")
        return False

    # (Optional) cap pool for faster testing
    if len(pool) > POOL_CAP:
        pool = pool[:POOL_CAP]
        print(f"⚡ POOL_CAP applied: testing with first {POOL_CAP} candidates")

    # 2) Test with 4 demo users (including faith preferences)
    users = demo_test_users()
    print(f"\n2️⃣ Running tests for {len(users)} demo users (AI prefilter {int(PREFILTER_PERCENT*100)}% / min_k={PREFILTER_MIN_K}, TOP_K={TOP_K})")
    print("   Testing: gender preferences, faith preferences (open/prefer_same/same_only)")

    for idx, q in enumerate(users, 1):
        print(f"\n--- User {idx}: {q['name']} ---")

        # Hard prefilter
        t0 = time.time()
        hard = mod.hard_prefilter(q, pool)
        t1 = time.time()
        print(f"✅ Hard prefilter: {len(hard)} candidates (in {t1 - t0:.2f}s)")
        if not hard:
            print("⚠️  No candidates passed hard prefilter; skipping this user.")
            continue

        # AI prefilter
        t0 = time.time()
        ai = mod.ai_prefilter(q, hard, percent=PREFILTER_PERCENT, min_k=PREFILTER_MIN_K)
        t1 = time.time()
        print(f"✅ AI prefilter: {len(ai)} candidates (in {t1 - t0:.2f}s)")
        if not ai:
            print("⚠️  No candidates passed AI prefilter; skipping this user.")
            continue

        # Llama ranking (STRICT: always call llm_rank)
        t0 = time.time()
        try:
            final = mod.llm_rank(q, ai, out_top=TOP_K)
        except Exception as e:
            print(f"❌ llm_rank raised an exception: {e}")
            return False
        t1 = time.time()
        print(f"✅ Llama ranking produced {len(final)} matches (in {t1 - t0:.2f}s)")

        # Show matches with new format
        print("— Top Recommendations —")
        for j, m in enumerate(final, 1):
            score = m.get("compatibility_score", 0.0)
            try:
                pct = int(round(float(score) * 100))
            except Exception:
                pct = 0
            
            # Extract 4 key adjectives from explanation
            explanation = m.get("explanation", "")
            adjectives = mod.extract_key_adjectives(explanation)
            
            print(f"{j}. {m.get('name','?')} - {pct}%")
            print(f"   {adjectives}")
        
        # Show faith-specific info for faith-conscious users
        if q.get('faith', {}).get('consider_in_matching'):
            faith_policy = q.get('faith', {}).get('policy', 'open')
            faith_religion = q.get('faith', {}).get('religion', '')
            print(f"  🕌 Faith preference: {faith_religion} ({faith_policy})")

    # 3) Append-to-JSON smoke test
    try:
        save_user = users[0].copy()
        save_user["email"] = "test_save_user_llama@rovermitra.example"
        mod.append_to_json(save_user, mod.LOCAL_DB_PATH)
        print(f"\n✅ JSON append OK → {mod.LOCAL_DB_PATH}")
    except Exception as e:
        print(f"\n❌ JSON append failed: {e}")
        return False

    print("\n🎉 Core tests completed.")
    return True

def run_perf_tests(mod):
    """Optional quick timing overview."""
    print("\n⏱️  Performance snapshot…")
    try:
        q = {
            'email': 'perf_user@rovermitra.example',
            'name': 'Perf User',
            'age': 30,
            'gender': 'Other',
            'home_base': {'city': 'Berlin', 'country': 'Germany'},
            'languages': ['en', 'de'],
            'interests': ['museum hopping', 'architecture walks'],
            'values': ['adventure', 'culture'],
            'bio': 'Performance smoke test user',
            'travel_prefs': {'pace': 'balanced'},
            'budget': {'amount': 150, 'currency': 'EUR'},
            'diet_health': {'diet': 'none'},
            'comfort': {'smoking': 'never', 'alcohol': 'moderate'},
            'work': {'remote_work_ok': False},
            'faith': {
                'consider_in_matching': False,
                'religion': '',
                'policy': 'open',
                'visibility': 'private'
            },
            'privacy': {'share_home_city': True}
        }

        t0 = time.time()
        pool = mod.load_pool()
        t1 = time.time()
        if len(pool) > POOL_CAP:
            pool = pool[:POOL_CAP]
        print(f"Pool: {t1 - t0:.2f}s ({len(pool)} candidates)")

        t0 = time.time()
        hard = mod.hard_prefilter(q, pool)
        t1 = time.time()
        print(f"Hard: {t1 - t0:.2f}s ({len(hard)} remain)")

        t0 = time.time()
        ai = mod.ai_prefilter(q, hard, percent=PREFILTER_PERCENT, min_k=PREFILTER_MIN_K)
        t1 = time.time()
        print(f"AI  : {t1 - t0:.2f}s ({len(ai)} remain)")

        t0 = time.time()
        final = mod.llm_rank(q, ai, out_top=TOP_K)
        t1 = time.time()
        print(f"LLM : {t1 - t0:.2f}s ({len(final)} returned)")

        total = (t1 - (t0 - (t1 - t0)))
        # (Not a perfect total; just a quick peek.)
        print("✅ Performance snapshot done.")

    except Exception as e:
        print(f"❌ Performance snapshot failed: {e}")

def main():
    print("🚀 RoverMitra Automated Test Suite (Strict Llama)")
    print("=" * 60)

    mod_name = choose_module_name()
    mod = import_pipeline_module(mod_name)

    ok = run_core_tests(mod)
    if ok:
        run_perf_tests(mod)
        print("\n" + "=" * 60)
        print("✅ All automated tests finished.")
        print("ℹ️  Final ranking used: Llama (llm_rank)")
    else:
        print("\n" + "=" * 60)
        print("❌ Some tests failed. See logs above.")

if __name__ == "__main__":
    main()
