#!/usr/bin/env python3
"""
Automated test script for RoverMitra CPU pipeline (strict Llama ranking)

- Tests the CPU-only version (main_CPU.py) with comprehensive performance metrics
- Loads candidate pool, runs hard prefilter, AI prefilter, then ALWAYS uses Llama (llm_rank)
- Prints detailed CPU performance results and timing analysis
- Includes server connection testing and fallback scenarios

Speed notes:
- Optimized for CPU testing with longer timeouts
- Keeps AI prefilter relatively small for faster runs: percent=0.02, min_k=20
- Optionally caps pool via POOL_CAP (default 3000) to keep tests manageable
- Includes server health checks and connection testing

Run:
  python test_automated_CPU.py
"""

import os
import sys
import time
import json
import importlib
import requests
from pathlib import Path

# -----------------------------
# Config knobs for CPU test speed
# -----------------------------
POOL_CAP = int(os.getenv("RM_TEST_POOL_CAP", "3000"))  # cap candidate pool for faster tests
PREFILTER_PERCENT = float(os.getenv("RM_TEST_PREFILTER_PERCENT", "0.02"))
PREFILTER_MIN_K = int(os.getenv("RM_TEST_PREFILTER_MIN_K", "20"))
TOP_K = int(os.getenv("RM_TEST_TOP_K", "5"))
SERVER_TIMEOUT = int(os.getenv("RM_TEST_SERVER_TIMEOUT", "300"))  # 5 minutes for CPU

def choose_module_name() -> str:
    print("🔧 Testing CPU-only module...")
    return "main_CPU"

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

def check_server_health():
    """Check if CPU Llama server is available"""
    ports_to_try = [8002, 8000, 8001, 8003, 8004, 8005]
    
    for port in ports_to_try:
        try:
            response = requests.get(f"http://localhost:{port}/health", timeout=5)
            if response.status_code == 200:
                data = response.json()
                if data.get("ok", False):
                    print(f"✅ CPU Llama server found on port {port}")
                    print(f"   Device: {data.get('device', 'unknown')}")
                    print(f"   Model: {data.get('model', 'unknown')}")
                    print(f"   Mode: {data.get('mode', 'unknown')}")
                    return True, port
        except Exception:
            continue
    
    print("⚠️  No CPU Llama server found - will use local CPU fallback")
    return False, None

def demo_test_users():
    """Four varied profiles to exercise the CPU pipeline with gender and faith preferences."""
    return [
        {
            'id': 'test_cpu_user_1',
            'name': 'Alex Berlin CPU',
            'age': 28,
            'gender': 'Other',
            'home_base': {'city': 'Berlin', 'country': 'Germany'},
            'languages': ['en', 'de'],
            'interests': ['museum hopping', 'architecture walks', 'history sites'],
            'values': ['adventure', 'culture'],
            'bio': 'Berlin-based traveler who loves history and architecture (CPU test)',
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
            'id': 'test_cpu_user_2',
            'name': 'Maria Munich CPU',
            'age': 32,
            'gender': 'Female',
            'home_base': {'city': 'Munich', 'country': 'Germany'},
            'languages': ['de', 'en', 'it'],
            'interests': ['food tours', 'vineyards', 'scenic trains'],
            'values': ['luxury-taste', 'nature'],
            'bio': 'Munich foodie who enjoys wine country and scenic journeys (CPU test)',
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
            'id': 'test_cpu_user_3',
            'name': 'Tom Karachi CPU',
            'age': 24,
            'gender': 'Male',
            'home_base': {'city': 'Karachi', 'country': 'Pakistan'},
            'languages': ['en', 'de'],
            'interests': ['festivals', 'live music', 'beach days'],
            'values': ['adventure', 'community'],
            'bio': 'Karachi student who loves music festivals and beach trips (CPU test)',
            'travel_prefs': {'pace': 'packed'},
            'budget': {'amount': 80, 'currency': 'EUR'},
            'diet_health': {'diet': 'none'},
            'comfort': {'smoking': 'occasionally', 'alcohol': 'social'},
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
            'id': 'test_cpu_user_4',
            'name': 'Priya Delhi CPU',
            'age': 26,
            'gender': 'Female',
            'home_base': {'city': 'Delhi', 'country': 'India'},
            'languages': ['en', 'hi'],
            'interests': ['temples', 'street food', 'yoga', 'meditation'],
            'values': ['spirituality', 'learning'],
            'bio': 'Delhi-based yoga instructor who loves spiritual journeys (CPU test)',
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
    print("🧪 Testing RoverMitra CPU pipeline with faith preferences (strict Llama)…")

    # Show CPU-specific env hints
    print(f"⚙️  CUDA_VISIBLE_DEVICES = {os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')}")
    print(f"⚙️  TRANSFORMERS_CACHE   = {os.environ.get('TRANSFORMERS_CACHE', '<unset>')}")
    print(f"⚙️  HF_HOME              = {os.environ.get('HF_HOME', '<unset>')}")
    print(f"⚙️  SERVER_TIMEOUT       = {SERVER_TIMEOUT}s")

    # Check server availability
    server_available, server_port = check_server_health()

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
    print(f"\n2️⃣ Running CPU tests for {len(users)} demo users (AI prefilter {int(PREFILTER_PERCENT*100)}% / min_k={PREFILTER_MIN_K}, TOP_K={TOP_K})")
    print("   Testing: gender preferences, faith preferences (open/prefer_same/same_only)")
    print("   CPU Mode: Server connection + local fallback")

    total_times = {
        'hard_prefilter': [],
        'ai_prefilter': [],
        'llm_rank': [],
        'total': []
    }

    for idx, q in enumerate(users, 1):
        print(f"\n--- User {idx}: {q['name']} ---")
        user_start = time.time()

        # Hard prefilter
        t0 = time.time()
        hard = mod.hard_prefilter(q, pool)
        t1 = time.time()
        hard_time = t1 - t0
        total_times['hard_prefilter'].append(hard_time)
        print(f"✅ Hard prefilter: {len(hard)} candidates (in {hard_time:.2f}s)")
        
        if not hard:
            print("⚠️  No candidates passed hard prefilter; skipping this user.")
            continue

        # AI prefilter
        t0 = time.time()
        ai = mod.ai_prefilter(q, hard, percent=PREFILTER_PERCENT, min_k=PREFILTER_MIN_K)
        t1 = time.time()
        ai_time = t1 - t0
        total_times['ai_prefilter'].append(ai_time)
        print(f"✅ AI prefilter: {len(ai)} candidates (in {ai_time:.2f}s)")
        
        if not ai:
            print("⚠️  No candidates passed AI prefilter; skipping this user.")
            continue

        # Llama ranking (STRICT: always call llm_rank)
        t0 = time.time()
        try:
            print(f"🔄 Starting Llama ranking (CPU mode - may take 30-60 seconds)...")
            final = mod.llm_rank(q, ai, out_top=TOP_K)
        except Exception as e:
            print(f"❌ llm_rank raised an exception: {e}")
            return False
        t1 = time.time()
        llm_time = t1 - t0
        total_times['llm_rank'].append(llm_time)
        print(f"✅ Llama ranking produced {len(final)} matches (in {llm_time:.2f}s)")

        # Show matches
        faith_matches = 0
        for j, m in enumerate(final, 1):
            score = m.get("compatibility_score", 0.0)
            try:
                pct = int(round(float(score) * 100))
            except Exception:
                pct = 0
            print(f"  {j}. {m.get('name','?')} — {pct}%")
            print(f"     {m.get('explanation','(no explanation)')}")
        
        # Show faith-specific info for faith-conscious users
        if q.get('faith', {}).get('consider_in_matching'):
            faith_policy = q.get('faith', {}).get('policy', 'open')
            faith_religion = q.get('faith', {}).get('religion', '')
            print(f"  🕌 Faith preference: {faith_religion} ({faith_policy})")

        user_total = time.time() - user_start
        total_times['total'].append(user_total)
        print(f"  ⏱️  User total time: {user_total:.2f}s")

    # 3) Append-to-JSON smoke test
    try:
        save_user = users[0].copy()
        save_user["id"] = "test_save_user_cpu"
        mod.append_to_json(save_user, mod.LOCAL_DB_PATH)
        print(f"\n✅ JSON append OK → {mod.LOCAL_DB_PATH}")
    except Exception as e:
        print(f"\n❌ JSON append failed: {e}")
        return False

    # 4) Performance summary
    print(f"\n📊 CPU PERFORMANCE SUMMARY:")
    if total_times['hard_prefilter']:
        avg_hard = sum(total_times['hard_prefilter']) / len(total_times['hard_prefilter'])
        print(f"   Hard prefilter: {avg_hard:.2f}s avg ({len(total_times['hard_prefilter'])} tests)")
    
    if total_times['ai_prefilter']:
        avg_ai = sum(total_times['ai_prefilter']) / len(total_times['ai_prefilter'])
        print(f"   AI prefilter:   {avg_ai:.2f}s avg ({len(total_times['ai_prefilter'])} tests)")
    
    if total_times['llm_rank']:
        avg_llm = sum(total_times['llm_rank']) / len(total_times['llm_rank'])
        print(f"   Llama ranking:  {avg_llm:.2f}s avg ({len(total_times['llm_rank'])} tests)")
    
    if total_times['total']:
        avg_total = sum(total_times['total']) / len(total_times['total'])
        print(f"   Total per user: {avg_total:.2f}s avg")
    
    print(f"   Server status: {'✅ Available' if server_available else '⚠️  Local fallback'}")

    print("\n🎉 CPU core tests completed.")
    return True

def run_perf_tests(mod):
    """CPU-specific performance snapshot."""
    print("\n⏱️  CPU Performance snapshot…")
    try:
        q = {
            'id': 'perf_cpu_user',
            'name': 'Perf CPU User',
            'age': 30,
            'gender': 'Other',
            'home_base': {'city': 'Berlin', 'country': 'Germany'},
            'languages': ['en', 'de'],
            'interests': ['museum hopping', 'architecture walks'],
            'values': ['adventure', 'culture'],
            'bio': 'CPU performance smoke test user',
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
        print("🔄 CPU Llama ranking (this may take 30-60 seconds)...")
        final = mod.llm_rank(q, ai, out_top=TOP_K)
        t1 = time.time()
        print(f"LLM : {t1 - t0:.2f}s ({len(final)} returned)")

        print("✅ CPU Performance snapshot done.")

    except Exception as e:
        print(f"❌ CPU Performance snapshot failed: {e}")

def main():
    print("🚀 RoverMitra CPU Automated Test Suite (Strict Llama)")
    print("=" * 70)
    print("🖥️  CPU-Only Mode: Testing performance on CPU hardware")
    print("⏱️  Expected: Slower inference but stable results")
    print("=" * 70)

    mod_name = choose_module_name()
    mod = import_pipeline_module(mod_name)

    ok = run_core_tests(mod)
    if ok:
        run_perf_tests(mod)
        print("\n" + "=" * 70)
        print("✅ All CPU automated tests finished.")
        print("ℹ️  Final ranking used: Llama (llm_rank) on CPU")
        print("📊 Performance: CPU inference is ~10x slower than GPU")
        print("💡 Tip: Start serve_llama_CPU.py for faster repeated tests")
    else:
        print("\n" + "=" * 70)
        print("❌ Some CPU tests failed. See logs above.")

if __name__ == "__main__":
    main()
