#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate matchmaker profiles from users/data/users_core.json

- No duplication of core user data; each record links by user_id only.
- Derives match-specific preferences from rich user fields:
  companion_preferences, lifestyle, boundaries_safety, trip_intentions, causes,
  languages, pace/chronotype, budget, diet, comfort, social, etc.
- Produces knobs your prefilter + final LLM can use.

Output: MatchMaker/data/matchmaker_profiles.json
"""

import json
import uuid
import random
from datetime import datetime
from pathlib import Path

random.seed(101)

USERS_PATH = Path("users/data/users_core.json")
OUT_PATH   = Path("MatchMaker/data/matchmaker_profiles.json")

# -------------------------
# Helpers
# -------------------------
def clamp(v, lo, hi):
    return max(lo, min(hi, v))

def norm_weights_dict(keys):
    vals = [random.random() for _ in keys]
    s = sum(vals) or 1.0
    return {k: round(v / s, 2) for k, v in zip(keys, vals)}

def openness_from(user):
    return float((user.get("personality") or {}).get("openness", 0.5))

def user_pace(user):
    return (user.get("travel_prefs") or {}).get("pace")

def budget_band(amount):
    if amount is None: return "mid"
    if amount <= 90:   return "budget"
    if amount >= 180:  return "lux"
    return "mid"

def budget_strictness(user):
    per_day = (user.get("budget") or {}).get("amount", 120)
    band = budget_band(per_day)
    if band == "budget": return "strict"
    if band == "lux":    return random.choice(["moderate","flexible"])
    return random.choice(["moderate","strict"])

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

def derive_languages_for_chat(user_langs):
    if not user_langs:
        return ["en"]
    langs = list(user_langs)
    if "en" not in langs and random.random() < 0.6:
        langs.append("en")
    k = clamp(random.randint(1, 3), 1, len(langs))
    return sorted(random.sample(langs, k=k))

def language_policy(user_langs):
    # Require at least one shared language in most cases
    return {
        "min_shared_languages": random.choice([0, 1, 1, 1, 2]),
        "preferred_chat_languages": derive_languages_for_chat(user_langs)
    }

def availability_windows():
    seasons = [
        {"season": "spring", "months": [3,4,5]},
        {"season": "summer", "months": [6,7,8]},
        {"season": "autumn", "months": [9,10,11]},
        {"season": "winter", "months": [12,1,2]},
    ]
    pick = random.sample(seasons, k=random.choice([1,1,2,2,3]))
    for w in pick:
        if random.random() < 0.5:
            w["preferred_trip_length_days"] = random.choice([3,5,7,10,14])
    return pick

def meeting_pref():
    return random.choice(["midpoint", "at_destination", "host_city", "no_preference"])

def blocklist(user_ids, self_id):
    # rarely block 1-3 random users (simulate prior issues)
    if random.random() < 0.06 and len(user_ids) > 5:
        pool = [u for u in user_ids if u != self_id]
        k = random.randint(1, min(3, len(pool)))
        return random.sample(pool, k=k)
    return []

# -------------------------
# Derivations from new rich fields
# -------------------------
def preferred_companion(user):
    # Base from companion_preferences if present
    cp = user.get("companion_preferences") or {}
    age = int(user.get("age", 30))
    openv = openness_from(user)

    # Age range: use user-set range or derive from openness & age
    if isinstance(cp.get("age_range_preferred"), list) and len(cp["age_range_preferred"]) == 2:
        age_range = cp["age_range_preferred"]
    else:
        span = 6 + int(openv * 12)  # ~6..18
        lo = clamp(age - (span // 2 + random.randint(0, 2)), 18, 75)
        hi = clamp(age + (span // 2 + random.randint(0, 3)), lo + 2, 80)
        age_range = [lo, hi]

    # Gender preference
    genders_pref = cp.get("genders_ok")
    if not genders_pref:
        genders_pref = ["any"]  # default “any”

    # Group size from user social or companion group type
    gs = [1, random.choice([2,3,4])]
    social = user.get("social") or {}
    if isinstance(social.get("group_size_ok"), list) and social["group_size_ok"]:
        gs = [min(social["group_size_ok"]), max(social["group_size_ok"])]

    return {
        "genders": genders_pref,
        "age_range": age_range,
        "group_size": gs,
        "max_origin_distance_km": random.choice([100, 200, 500, 1000, 2000, 5000]),
    }

def communication_preferences(user):
    primary = "text"
    # If they added opening_move or prompts → likely text/voice friendly
    if user.get("opening_move"):
        primary = random.choice(["text","voice_notes"])
    return {
        "primary": primary,
        "response_expectation_hours": random.choice([1,2,4,8,24]),
        "pre_meet_video_call_ok": random.random() < 0.6
    }

def soft_prefs_from_user(user):
    return {
        "prefer_same_pace": "prefer_same" if user_pace(user) else random.choice(["prefer_same","open"]),
        "prefer_same_budget_band": random.choice([True, False, True]),
        "prefer_same_chronotype": chronotype_pref(user),
        "prefer_language_practice": random.random() < 0.25,
        "prefer_photography_buddy": "city photography" in (user.get("interests") or []),
        "prefer_similar_causes": bool(user.get("causes")) and random.random() < 0.5
    }

def visibility_block(user):
    b = user.get("boundaries_safety") or {}
    # Respect “photo / social” constraints where applicable
    photo_policy = b.get("photo_consent", "ask first")
    show_values = random.random() < 0.75
    if photo_policy == "no faces on public socials":
        show_values = False  # be slightly more private overall
    return {
        "show_age_exact": random.random() < 0.85,
        "show_home_city": random.random() < 0.9,
        "show_values": show_values,
        "show_interests_list": random.random() < 0.95
    }

def safety_block(user):
    b = user.get("boundaries_safety") or {}
    return {
        "requires_verified_profile": random.random() < 0.65,
        "share_approx_location_only": True,
        "women_only_groups": (user.get("gender") == "Female" and random.random() < 0.25),
        "manual_review_required": random.random() < 0.05,
        # propagate a bit of their boundary preferences
        "quiet_hours": b.get("quiet_hours", random.choice(["22:00–07:00","23:00–07:00","flexible"])),
        "photo_consent": b.get("photo_consent", "ask first"),
        "social_media": b.get("social_media", random.choice(["share occasionally","no tagging please","fine with tagging"]))
    }

def hard_dealbreakers(user):
    out = set()
    # Social dealbreakers from core
    for x in (user.get("social") or {}).get("dealbreakers_social", []):
        out.add(x)

    # Lifestyle boundaries → rules
    b = user.get("boundaries_safety") or {}
    if b.get("substance_boundaries") == "no cigarettes in room":
        out.add("no_smoking_roommates")
    comfort = user.get("comfort") or {}
    if comfort.get("smoking") == "never":
        out.add("no_smokers")
    if comfort.get("alcohol") == "none":
        out.add("no_heavy_drinkers")

    # Language & interests minimums help prefilter
    out.add(random.choice(["min_2_shared_interests","min_3_shared_interests","no_zero_shared_values"]))

    # Chronotype strictness
    if chronotype_pref(user) == "prefer_same":
        out.add("prefer_same_chronotype")

    return sorted(out)

def match_intent(user):
    # Tie to user trip_intentions when present
    ti = user.get("trip_intentions") or []
    base = ["travel_buddy","co_work_trip","local_guide_exchange","weekend_getaway","festival_trip","hiking_partner"]
    if ti:
        # lightly map
        mapped = []
        for t in ti:
            t = t.lower()
            if "work" in t or "wifi" in t: mapped.append("co_work_trip")
            elif "festival" in t or "event" in t: mapped.append("festival_trip")
            elif "hiking" in t or "outdoor" in t: mapped.append("hiking_partner")
            elif "weekend" in t or "city" in t: mapped.append("weekend_getaway")
            elif "local" in t: mapped.append("local_guide_exchange")
        base = list(set(base + mapped))
    k = random.choice([1,1,2,2,3])
    return random.sample(base, k=k)

def compatibility_weights():
    # Emphasize what matters for travel harmony
    keys = ["personality","interests","values","budget","languages","pace","cleanliness","risk_tolerance","chronotype","diet"]
    return norm_weights_dict(keys)

# -------------------------
# Main
# -------------------------
def main():
    if not USERS_PATH.exists():
        raise FileNotFoundError(f"Core users not found: {USERS_PATH}")

    users = json.loads(USERS_PATH.read_text(encoding="utf-8"))
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
            "visibility": visibility_block(u),

            # What they’re looking for (derived from their rich answers)
            "match_intent": match_intent(u),
            "preferred_companion": preferred_companion(u),

            # How to communicate / pre-meet comfort
            "communication_preferences": communication_preferences(u),

            # When they’re generally free to travel
            "availability_windows": availability_windows(),

            # Matching knobs (for AI prefilter & LLM)
            "compatibility_weights": compatibility_weights(),
            "hard_dealbreakers": hard_dealbreakers(u),
            "soft_preferences": soft_prefs_from_user(u),
            "language_policy": language_policy(u.get("languages") or []),
            "meeting_preference": meeting_pref(),
            "budget_compatibility_strictness": budget_strictness(u),
            "diet_compatibility_strictness": diet_strictness(u),

            # Trust & safety
            "safety_settings": safety_block(u),
            "blocklist_user_ids": blocklist([u.get("user_id") for u in users if u.get("user_id")], uid),

            # Keep low-quality matches out
            "match_quality_threshold": random.choice([0.75, 0.78, 0.80, 0.82, 0.85])
        }

        profiles.append(profile)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(profiles, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"✅ Generated {len(profiles)} matchmaker profiles → {OUT_PATH}")
    if profiles:
        print(json.dumps(profiles[0], indent=2, ensure_ascii=False)[:1000] + "\n...")
    else:
        print("No profiles generated.")

if __name__ == "__main__":
    main()
