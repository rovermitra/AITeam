#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate 2,000 matchmaker profiles from users/users_core.json
- No duplication of core user data; everything links by user_id.
- Adds only match-specific preferences, constraints, weights, safety/visibility.
Output: matchmaker/matchmaker_profiles.json
"""

import os
import json
import uuid
import random
from datetime import datetime
from pathlib import Path

random.seed(101)

USERS_PATH = Path("users/data/users_core.json")
OUT_PATH = Path("MatchMaker/data/matchmaker_profiles.json")

# -------------------------
# Helpers
# -------------------------
def clamp(v, lo, hi):
    return max(lo, min(hi, v))

def norm_weights_dict(keys):
    vals = [random.random() for _ in keys]
    s = sum(vals) or 1.0
    return {k: round(v / s, 2) for k, v in zip(keys, vals)}

def pick_age_range(user_age, openness):
    # openness in [0,1] (fallback 0.5)
    # base band expands with openness
    span = 6 + int(openness * 12)         # ~6..18 width
    lo = clamp(user_age - (span // 2 + random.randint(0, 2)), 18, 75)
    hi = clamp(user_age + (span // 2 + random.randint(0, 3)), lo + 2, 80)
    return [lo, hi]

def derive_group_size(user_social):
    # Use user’s tolerated group sizes if present; else a reasonable default
    base = user_social.get("group_size_ok") if isinstance(user_social, dict) else None
    if isinstance(base, list) and base:
        return [min(base), max(base)]
    return [1, random.choice([2,3,4])]

def derive_languages_for_chat(user_langs):
    if not user_langs:
        return ["en"]
    # prefer their languages, but allow English if missing
    langs = list(user_langs)
    if "en" not in langs and random.random() < 0.6:
        langs.append("en")
    # pick 1–3 preferred chat languages
    k = clamp(random.randint(1, 3), 1, len(langs))
    return sorted(random.sample(langs, k=k))

def openness_from(user):
    return float(user.get("personality", {}).get("openness", 0.5))

def pace_from(user):
    return (user.get("travel_prefs") or {}).get("pace")

def budget_band_strictness(user):
    # If per-day budget is low → stricter; otherwise mixed
    per_day = (user.get("budget") or {}).get("amount", 120)
    if per_day <= 80:  return "strict"
    if per_day >= 200: return random.choice(["flexible", "moderate"])
    return random.choice(["moderate", "strict"])

def diet_strictness(user):
    diet = (user.get("diet_health") or {}).get("diet", "none")
    if diet in {"vegan","kosher","halal","gluten-free","lactose-free"}:
        return "strict"
    if diet in {"vegetarian","pescatarian","no pork"}:
        return random.choice(["moderate","strict"])
    return random.choice(["flexible","moderate"])

def chronotype_pref(user):
    chrono = (user.get("comfort") or {}).get("chronotype")
    if chrono in {"early bird","night owl"}:
        return "prefer_same"
    return random.choice(["prefer_same","open"])

def language_policy(user_langs):
    # Require at least 1 shared language in most cases
    return {
        "min_shared_languages": random.choice([0, 1, 1, 1, 2]),
        "preferred_chat_languages": derive_languages_for_chat(user_langs)
    }

def meeting_pref():
    return random.choice(["midpoint", "at_destination", "host_city", "no_preference"])

def availability_windows():
    # Simple seasonal windows for discovery (not bookings)
    seasons = [
        {"season": "spring", "months": [3,4,5]},
        {"season": "summer", "months": [6,7,8]},
        {"season": "autumn", "months": [9,10,11]},
        {"season": "winter", "months": [12,1,2]},
    ]
    pick = random.sample(seasons, k=random.choice([1,1,2,2,3]))
    # 50% add flexible length hint
    for w in pick:
        if random.random() < 0.5:
            w["preferred_trip_length_days"] = random.choice([3,5,7,10,14])
    return pick

def hard_dealbreakers_from_user(user):
    # Pull some social dealbreakers and add match-specific ones
    s = (user.get("social") or {}).get("dealbreakers_social", [])
    out = set(s)
    # Always include at least one “overlap” rule for the model to learn
    out.add(random.choice(["min_2_shared_interests","min_3_shared_interests","no_zero_shared_values"]))
    # Sometimes add specific lifestyle constraints
    if (user.get("comfort") or {}).get("smoking") == "never" and random.random() < 0.6:
        out.add("no_smokers")
    if (user.get("comfort") or {}).get("alcohol") == "none" and random.random() < 0.6:
        out.add("no_heavy_drinkers")
    return sorted(out)

def soft_prefs_from_user(user):
    return {
        "prefer_same_pace": random.choice(["prefer_same","open"]) if not pace_from(user) else "prefer_same",
        "prefer_same_budget_band": random.choice([True, False, True]),
        "prefer_same_chronotype": chronotype_pref(user),
        "prefer_language_practice": random.random() < 0.25,  # likes partners to practice languages
        "prefer_photography_buddy": "city photography" in (user.get("interests") or [])
    }

def visibility_block():
    return {
        "show_age_exact": random.random() < 0.85,
        "show_home_city": random.random() < 0.9,
        "show_values": random.random() < 0.75,
        "show_interests_list": random.random() < 0.95
    }

def communication_block():
    return {
        "primary": random.choice(["text","voice_notes","video_call","no_preference"]),
        "response_expectation_hours": random.choice([1,2,4,8,24]),
        "pre_meet_video_call_ok": random.random() < 0.6
    }

def preferred_companion(user):
    age = int(user.get("age", 30))
    openv = openness_from(user)
    genders = ["Male","Female","Non-binary","Other"]
    # Most people fine with "any"; sometimes pick subset
    gender_pref = random.choice([["any"], genders, ["Male","Female"], [random.choice(genders)]])
    return {
        "genders": gender_pref,
        "age_range": pick_age_range(age, openv),
        "group_size": derive_group_size(user.get("social") or {}),
        "max_origin_distance_km": random.choice([100, 200, 500, 1000, 2000, 5000]),
    }

def safety_block(user):
    return {
        "requires_verified_profile": random.random() < 0.65,
        "share_approx_location_only": True,
        "women_only_groups": (user.get("gender") == "Female" and random.random() < 0.25),
        "manual_review_required": random.random() < 0.05
    }

def match_intent():
    return random.sample(
        ["travel_buddy","co_work_trip","local_guide_exchange","weekend_getaway","festival_trip","hiking_partner"],
        k=random.choice([1,1,2,2,3])
    )

def compatibility_weights():
    keys = ["personality","interests","values","budget","languages","pace","cleanliness","risk_tolerance","chronotype","diet"]
    return norm_weights_dict(keys)

def blocklist(user_ids, self_id):
    # very rarely block 1-3 random users (simulates prior bad experiences)
    if random.random() < 0.06 and len(user_ids) > 5:
        pool = [u for u in user_ids if u != self_id]
        k = random.randint(1, min(3, len(pool)))
        return random.sample(pool, k=k)
    return []

# -------------------------
# Main
# -------------------------
def main():
    if not USERS_PATH.exists():
        raise FileNotFoundError(f"Core users not found: {USERS_PATH}")

    users = json.loads(USERS_PATH.read_text(encoding="utf-8"))
    user_ids = [u.get("user_id") for u in users if isinstance(u, dict) and u.get("user_id")]
    profiles = []

    for u in users:
        uid = u.get("user_id")
        if not uid: 
            continue

        profile = {
            "match_profile_id": f"mm_{uuid.uuid4().hex[:12]}",
            "user_id": uid,
            "status": "active",
            "created_at": datetime.utcnow().isoformat() + "Z",
            "updated_at": datetime.utcnow().isoformat() + "Z",

            # Discovery / privacy
            "visibility": visibility_block(),

            # What they’re looking for
            "match_intent": match_intent(),
            "preferred_companion": preferred_companion(u),

            # When/how to talk
            "communication_preferences": communication_block(),

            # When they’re generally free to travel
            "availability_windows": availability_windows(),

            # Matching logic knobs (used by the AI/ranker)
            "compatibility_weights": compatibility_weights(),
            "hard_dealbreakers": hard_dealbreakers_from_user(u),
            "soft_preferences": soft_prefs_from_user(u),
            "language_policy": language_policy(u.get("languages") or []),
            "meeting_preference": meeting_pref(),
            "budget_compatibility_strictness": budget_band_strictness(u),
            "diet_compatibility_strictness": diet_strictness(u),

            # Ops / trust & safety
            "safety_settings": safety_block(u),
            "blocklist_user_ids": blocklist(user_ids, uid),

            # Thresholds to keep model outputs clean
            "match_quality_threshold": random.choice([0.75, 0.78, 0.80, 0.82, 0.85])
        }

        profiles.append(profile)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(profiles, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"✅ Generated {len(profiles)} matchmaker profiles → {OUT_PATH}")
    print(json.dumps(profiles[0], indent=2)[:1000] + "\n...")

if __name__ == "__main__":
    main()
