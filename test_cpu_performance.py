#!/usr/bin/env python3
"""
Test script for CPU-only RoverMitra without interactive input
"""

import sys
import time
from pathlib import Path

# Add the current directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

# Import the main functions
from main_CPU import (
    load_pool, hard_prefilter, ai_prefilter, llm_rank,
    append_to_json, LOCAL_DB_PATH
)

def create_test_user():
    """Create a test user profile"""
    return {
        "id": "u_test_cpu_001",
        "name": "Alex Rivera",
        "age": 27,
        "gender": "Other",
        "home_base": {"city": "Berlin", "country": "Germany", "nearby_nodes": [], "willing_radius_km": 40},
        "languages": ["en", "de"],
        "interests": ["museum hopping", "architecture walks", "city photography"],
        "values": ["adventure", "nature"],
        "bio": "Berlin-based planner who loves scenic trains + espresso.",

        "travel_prefs": {
            "pace": "balanced",
            "accommodation_types": ["hotel","apartment"],
            "room_setup": "twin",
            "transport_allowed": ["train", "plane"],
            "must_haves": ["wifi", "near_station"]
        },

        "budget": {"type":"per_day","amount": 150, "currency": "EUR", "split_rule": "each_own"},

        "diet_health": {"diet": "none", "allergies": ["none"], "accessibility": "none"},

        "comfort": {
            "risk_tolerance": "medium",
            "noise_tolerance": "medium",
            "cleanliness_preference": "medium",
            "chronotype": "flexible",
            "alcohol": "moderate",
            "smoking": "never"
        },

        "work": {
            "remote_work_ok": False,
            "hours_online_needed": 0,
            "wifi_quality_needed": "good"
        },

        "companion_preferences": {
            "genders_ok": ["I'm open to travel with anyone"]
        },

        "faith": {
            "consider_in_matching": False,
            "religion": "",
            "policy": "open",
            "visibility": "private"
        },

        "privacy": {
            "share_profile_with_matches": True,
            "share_home_city": True,
            "pre_meet_video_call_ok": True
        }
    }

def main():
    print("🚀 RoverMitra CPU Performance Test (Non-Interactive)")
    print("=" * 60)
    
    # Create test user
    q_user = create_test_user()
    print(f"✅ Created test user: {q_user['name']}")
    
    # Save test user
    append_to_json(q_user, LOCAL_DB_PATH)
    print(f"✅ Saved test profile to {LOCAL_DB_PATH}")

    # Load candidate pool
    pool = load_pool()
    if not pool:
        print("❌ No candidates found. Provide users_core.json and matchmaker_profiles.json.")
        return

    print(f"✅ Loaded {len(pool)} candidates from pool")

    # 1) Hard prefilters
    print("\n🔍 Step 1: Hard prefilters...")
    t0 = time.time()
    hard = hard_prefilter(q_user, pool)
    t_hard = time.time() - t0
    print(f"✅ Hard prefilter: {len(hard)} candidates (in {t_hard:.2f}s)")
    
    if not hard:
        print("❌ No candidates remained after hard prefilters.")
        return

    # 2) AI prefilter (BGE cache → fast)
    print("\n🤖 Step 2: AI prefilter...")
    t1 = time.time()
    shortlist = ai_prefilter(q_user, hard, percent=0.02, min_k=80)
    t_ai = time.time() - t1
    print(f"✅ AI prefilter: {len(shortlist)} candidates (in {t_ai:.2f}s)")
    
    if not shortlist:
        print("❌ No candidates after AI prefilter.")
        return

    # 3) Llama ranking (server or local)
    print("\n🧠 Step 3: Llama ranking...")
    print("   This will test the server connection...")
    t2 = time.time()
    final = llm_rank(q_user, shortlist, out_top=5)
    t_llm = time.time() - t2
    print(f"✅ Llama ranking produced {len(final)} matches (in {t_llm:.2f}s)")

    # 4) Results
    high_quality = [m for m in final if float(m.get("compatibility_score", 0)) >= 0.75]

    # Performance summary
    total_time = time.time() - t0
    print(f"\n📈 PERFORMANCE SUMMARY:")
    print(f"   Hard prefilter: {t_hard:.2f}s")
    print(f"   AI prefilter:   {t_ai:.2f}s")
    print(f"   Llama ranking:  {t_llm:.2f}s")
    print(f"   Total time:     {total_time:.2f}s")
    print(f"   Candidates processed: {len(pool)} → {len(hard)} → {len(shortlist)} → {len(final)}")

    print("\n— Results —")
    if not high_quality:
        print("No high-quality matches found (score >= 75%). Here are your top results:")
        for i, m in enumerate(final, 1):
            try:
                pct = int(round(float(m.get("compatibility_score",0))*100))
            except Exception:
                pct = 0
            print(f"{i}. {m.get('name')} (user_id: {m.get('user_id')})  —  {pct}%")
            print(f"   {m.get('explanation')}\n")
    else:
        for i, m in enumerate(high_quality, 1):
            try:
                pct = int(round(float(m.get("compatibility_score",0))*100))
            except Exception:
                pct = 0
            print(f"{i}. {m.get('name')} (user_id: {m.get('user_id')})  —  {pct}%")
            print(f"   {m.get('explanation')}\n")

if __name__ == "__main__":
    main()
