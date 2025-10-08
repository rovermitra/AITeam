#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Hotel Booking Generator (VIP-grade)
Integrates:
  - users/data/users_core.json
  - MatchMaker/data/matchmaker_profiles.json
  - Flight/data/travel_groups_integrated_v3.json

Output (references users & hotel inventory, does not duplicate user data):
  - Hotels/data/hotel_bookings_integrated_v1.json
"""

import os
import json
import uuid
import math
import random
from datetime import datetime, date, timedelta
from pathlib import Path
from collections import defaultdict

random.seed(222)

USERS_PATH   = Path("users/data/users_core.json")
MM_PATH      = Path("MatchMaker/data/matchmaker_profiles.json")
GROUPS_PATH  = Path("Flight/data/travel_groups_integrated_v3.json")
OUT_PATH     = Path("Hotels/data/hotel_bookings_integrated_v1.json")

# -----------------------------
# Country taxes & currency map
# -----------------------------
CURRENCY_BY_COUNTRY = {
    "USA":"USD","Canada":"CAD","Mexico":"MXN","Brazil":"BRL","Argentina":"ARS","Chile":"CLP","Peru":"PEN","Colombia":"COP",
    "UK":"GBP","Ireland":"EUR","Germany":"EUR","France":"EUR","Spain":"EUR","Italy":"EUR","Portugal":"EUR",
    "Netherlands":"EUR","Belgium":"EUR","Switzerland":"CHF","Austria":"EUR","Poland":"PLN","Czechia":"CZK",
    "Denmark":"DKK","Sweden":"SEK","Norway":"NOK","Finland":"EUR","Greece":"EUR",
    "Japan":"JPY","South Korea":"KRW","China":"CNY","India":"INR","Thailand":"THB","Vietnam":"VND","Malaysia":"MYR",
    "Singapore":"SGD","Indonesia":"IDR","Philippines":"PHP","UAE":"AED","Saudi Arabia":"SAR","Qatar":"QAR","Israel":"ILS","Jordan":"JOD",
    "Morocco":"MAD","Egypt":"EGP","South Africa":"ZAR","Kenya":"KES","Australia":"AUD","New Zealand":"NZD","Iceland":"ISK"
}

# Reverse lookup for rough country from currency when city/country is unknown
COUNTRY_BY_CURRENCY = {v: k for k, v in CURRENCY_BY_COUNTRY.items()}

# Broadly realistic VAT + city tax assumptions by country (simplified)
TAX_RULES = {
    "Switzerland": {"vat_rate": 0.037, "city_tax_per_night": 3.1},
    "Germany":     {"vat_rate": 0.07,  "city_tax_per_night": 2.0},
    "France":      {"vat_rate": 0.10,  "city_tax_per_night": 2.6},
    "Italy":       {"vat_rate": 0.10,  "city_tax_per_night": 3.0},
    "Spain":       {"vat_rate": 0.10,  "city_tax_per_night": 2.0},
    "Portugal":    {"vat_rate": 0.06,  "city_tax_per_night": 2.0},
    "UK":          {"vat_rate": 0.20,  "city_tax_per_night": 0.0},
    "Netherlands": {"vat_rate": 0.09,  "city_tax_per_night": 3.0},
    "Belgium":     {"vat_rate": 0.06,  "city_tax_per_night": 3.0},
    "Austria":     {"vat_rate": 0.10,  "city_tax_per_night": 2.2},
    "Greece":      {"vat_rate": 0.13,  "city_tax_per_night": 1.5},
    "USA":         {"vat_rate": 0.00,  "city_tax_per_night": 5.0},
    "Canada":      {"vat_rate": 0.05,  "city_tax_per_night": 3.0},
    "Japan":       {"vat_rate": 0.10,  "city_tax_per_night": 2.0},
    "India":       {"vat_rate": 0.12,  "city_tax_per_night": 0.0},
}

# -----------------------------
# Helpers
# -----------------------------
def idx_by(lst, key):
    return {x.get(key): x for x in lst if isinstance(x, dict) and x.get(key)}

def safe_get(d, *keys, default=None):
    cur = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur

def weekend_uplift(dt: date) -> float:
    wd = dt.weekday()  # Mon=0
    if wd in (4, 5):  # Fri/Sat
        return 1.12
    if wd == 6:       # Sun
        return 0.96
    return 1.0

def cancellation_policy(refundable_prob=0.7):
    if random.random() < refundable_prob:
        hours = random.choice([24, 48, 72])
        return {"refundable": True, "free_cancel_until_hours": hours, "no_show_fee_percent": random.choice([80, 100])}
    else:
        return {"refundable": False, "free_cancel_until_hours": 0, "no_show_fee_percent": 100}

def jaccard(a, b):
    sa, sb = set(a or []), set(b or [])
    if not sa and not sb: return 0.0
    return len(sa & sb) / max(1, len(sa | sb))

def chronotype_score(a, b):
    pa = safe_get(a, "comfort", "chronotype")
    pb = safe_get(b, "comfort", "chronotype")
    if not pa or not pb: return 0.5
    if pa == pb: return 1.0
    if "flexible" in (pa, pb): return 0.7
    return 0.35

def smoking_compatible(a, b):
    sa = safe_get(a, "comfort", "smoking", default="never")
    sb = safe_get(b, "comfort", "smoking", default="never")
    if "regular" in (sa, sb) and ("never" in (sa, sb)):
        return False
    return True

def same_gender_or_open(a, b, mm_a=None, mm_b=None):
    ga = a.get("gender")
    gb = b.get("gender")
    women_only_a = safe_get(mm_a or {}, "safety_settings", "women_only_groups", default=False)
    women_only_b = safe_get(mm_b or {}, "safety_settings", "women_only_groups", default=False)
    if women_only_a or women_only_b:
        return (ga == "Female") and (gb == "Female")
    return True

def majority_split_rule(users):
    votes = defaultdict(int)
    for u in users:
        r = safe_get(u, "budget", "split_rule", default="each_own") or "each_own"
        votes[r] += 1
    return sorted(votes.items(), key=lambda x: (-x[1], x[0]))[0][0]

def contiguous_city_segments(itinerary):
    """Group consecutive days by base city."""
    if not itinerary: return []
    days = []
    for d in itinerary:
        iso = d.get("date")
        base = d.get("base")
        if not iso or not base:
            continue
        try:
            days.append((date.fromisoformat(iso), base))
        except Exception:
            continue
    if not days: return []
    days.sort(key=lambda x: x[0])

    segs = []
    cur_city, start_d, last_d = None, None, None
    for (dt, city) in days:
        if cur_city is None:
            cur_city, start_d, last_d = city, dt, dt
            continue
        if city == cur_city and dt == last_d + timedelta(days=1):
            last_d = dt
        else:
            segs.append({"city": cur_city, "check_in": start_d, "check_out": last_d + timedelta(days=1),
                         "nights": (last_d - start_d).days + 1})
            cur_city, start_d, last_d = city, dt, dt
    segs.append({"city": cur_city, "check_in": start_d, "check_out": last_d + timedelta(days=1),
                 "nights": (last_d - start_d).days + 1})
    return segs

# -----------------------------
# Hotel selection
# -----------------------------
def score_hotel(option, must_haves, target_price, needs_quiet, needs_workspace):
    score = 0.0
    ams = set(option.get("amenities", []))
    for m in must_haves:
        score += 1.5 if m in ams else -0.8
    if needs_quiet and "quiet_room" in ams: score += 0.4
    if needs_workspace and "workspace" in ams: score += 0.4
    p = option.get("price_per_night", 0)
    if target_price:
        score += math.exp(-abs(p - target_price) / max(30.0, target_price * 0.3))
    score += max(0, (2.0 - option.get("distance_to_station_km", 1.0))) * 0.6
    score += (option.get("stars", 3) - 3) * 0.4
    return score

def choose_hotel_for_segment(inv_for_city, users, mm_profiles):
    options = inv_for_city.get("options", [])
    if not options: return None

    must_haves = set()
    wants_quiet = False
    wants_workspace = False
    budgets = []
    values_all = []
    for u in users:
        for m in (safe_get(u, "travel_prefs", "must_haves", default=[]) or []):
            must_haves.add(m)
        if safe_get(u, "comfort", "noise_tolerance") == "low":
            wants_quiet = True
        if safe_get(u, "work", "hours_online_needed", default=0) > 0:
            wants_workspace = True
        per_day = safe_get(u, "budget", "amount", default=None)
        if per_day: budgets.append(per_day)
        values_all.extend(u.get("values", []) or [])

    target_price = sorted(budgets)[len(budgets)//2] if budgets else 150
    vip_bias = (values_all.count("luxury-taste") >= max(1, len(users)//3)) or (target_price >= 200)

    best, best_score = None, -1e9
    for opt in options:
        s = score_hotel(opt, must_haves, target_price, wants_quiet, wants_workspace)
        if vip_bias and opt.get("stars", 3) >= 4:
            s += 0.8
        if "hostel" in (opt.get("name","").lower()):
            s -= 0.6
        if s > best_score:
            best, best_score = opt, s
    return best

# -----------------------------
# Rooming list (pairing)
# -----------------------------
def build_rooming(users, mm_map, room_setup_hint=None):
    singles_only = (room_setup_hint == "2 rooms")

    pool = list(users)
    random.shuffle(pool)

    def pair_score(a, b):
        if not smoking_compatible(a, b): return -999
        if not same_gender_or_open(a, b, mm_map.get(a["user_id"]), mm_map.get(b["user_id"])): return -999
        s = 0.0
        s += 1.2 * jaccard(a.get("interests"), b.get("interests"))
        s += 0.8 * chronotype_score(a, b)
        if safe_get(a, "comfort", "noise_tolerance") == safe_get(b, "comfort", "noise_tolerance"): s += 0.35
        if safe_get(a, "comfort", "cleanliness_preference") == safe_get(b, "comfort", "cleanliness_preference"): s += 0.25
        return s

    assigned, rooms = set(), []

    for i in range(len(pool)):
        ua = pool[i]
        if ua["user_id"] in assigned: continue
        if singles_only:
            rooms.append([ua["user_id"]]); assigned.add(ua["user_id"]); continue

        best_j, best_sc = None, -9999
        for j in range(i+1, len(pool)):
            ub = pool[j]
            if ub["user_id"] in assigned: continue
            sc = pair_score(ua, ub)
            if sc > best_sc:
                best_sc, best_j = sc, j
        if best_j is not None and best_sc > -500:
            rooms.append([ua["user_id"], pool[best_j]["user_id"]])
            assigned.add(ua["user_id"]); assigned.add(pool[best_j]["user_id"])
        else:
            rooms.append([ua["user_id"]]); assigned.add(ua["user_id"])

    return rooms

# -----------------------------
# Pricing & payments
# -----------------------------
def nightly_prices(base_avg, start_date, nights):
    res = []
    for k in range(nights):
        d = start_date + timedelta(days=k)
        v = base_avg * random.uniform(0.9, 1.12) * weekend_uplift(d)
        res.append(round(v))
    return res

def compute_taxes(country, nightly, rooms_count):
    rule = TAX_RULES.get(country, {"vat_rate": 0.10, "city_tax_per_night": 2.0})
    base = sum(nightly)
    vat = round(base * rule["vat_rate"], 2)
    city_tax = round(rule["city_tax_per_night"] * len(nightly) * rooms_count, 2)
    return base, vat, city_tax

def pick_payer(users, split_rule):
    if split_rule == "each_own":
        return None
    best, best_c = None, -1
    for u in users:
        c = safe_get(u, "personality", "conscientiousness", default=0.5) or 0.5
        if c > best_c:
            best_c, best = c, u["user_id"]
    return best

def vip_tier(users):
    budgets = [safe_get(u, "budget", "amount", default=0) for u in users if isinstance(u, dict)]
    lux = sum(1 for u in users if "luxury-taste" in (u.get("values") or []))
    if any(b >= 240 for b in budgets) or lux >= max(1, len(users)//3):
        return random.choice(["Platinum","Gold"])
    if any(b >= 180 for b in budgets):
        return "Silver"
    return None

# -----------------------------
# Early check-in / late checkout
# -----------------------------
def earliest_arrival_iso(group_draft, user_id):
    ch = group_draft.get("chosen_flights", [])
    flights = group_draft.get("flight_offers", {})
    for x in ch:
        if x.get("user_id") == user_id:
            ofr = flights.get(user_id, {})
            for seg in ofr.get("outbound", []):
                if seg.get("id") == x.get("outbound"):
                    return seg.get("arrive_iso")
    return None

def should_request_early_checkin(arrivals_iso):
    times = []
    for iso in arrivals_iso:
        try:
            t = datetime.fromisoformat(iso)
            times.append(t.time())
        except Exception:
            pass
    if not times: return False
    early = [t for t in times if t < datetime.strptime("13:00","%H:%M").time()]
    return (len(early) / len(times)) >= 0.4

# -----------------------------
# Build bookings per group
# -----------------------------
def build_bookings_for_group(group, users_map, mm_map):
    bookings = []
    itinerary = safe_get(group, "draft_plan", "itinerary", default=[]) or []
    hotel_inventory = safe_get(group, "draft_plan", "hotel_inventory", default=[]) or []
    members = group.get("members", [])
    member_ids = [m.get("user_id") for m in members if m.get("user_id")]
    member_users = [users_map[uid] for uid in member_ids if uid in users_map]
    mm_for_member = {uid: mm_map.get(uid) for uid in member_ids}

    inv_by_city = {inv.get("city"): inv for inv in hotel_inventory if inv.get("city")}
    room_setup_hint = safe_get(group, "trip_context", "hard_constraints", "room_setup", default=None)

    segments = contiguous_city_segments(itinerary)

    for seg in segments:
        city = seg["city"]
        nights = seg["nights"]
        check_in = seg["check_in"]
        check_out = seg["check_out"]

        inv_city = inv_by_city.get(city)
        if not inv_city:
            continue

        chosen = choose_hotel_for_segment(inv_city, member_users, mm_for_member)
        if not chosen:
            continue

        currency = chosen.get("currency") or "EUR"
        # Choose a reasonable country for tax rules from currency (fallback to Germany/EUR)
        taxes_country = COUNTRY_BY_CURRENCY.get(currency, "Germany")

        rooms = build_rooming(member_users, mm_for_member, room_setup_hint=room_setup_hint)

        nightly = nightly_prices(chosen.get("price_per_night", 150), check_in, nights)
        base, vat, city_tax = compute_taxes(taxes_country, nightly, rooms_count=len(rooms))
        total = round(base + vat + city_tax, 2)

        split_rule = majority_split_rule(member_users)
        lead_payer = pick_payer(member_users, split_rule)

        arrivals = []
        for uid in member_ids:
            iso = earliest_arrival_iso(safe_get(group, "draft_plan", default={}), uid)
            if iso: arrivals.append(iso)
        req_eci = should_request_early_checkin(arrivals)
        req_lco = random.random() < 0.25

        vip = vip_tier(member_users)
        perks = []
        if vip == "Platinum":
            perks = ["room_upgrade_subject_to_availability","lounge_access","complimentary_breakfast","late_checkout_16"]
        elif vip == "Gold":
            perks = ["room_upgrade_subject_to_availability","complimentary_breakfast","late_checkout_14"]
        elif vip == "Silver":
            perks = ["late_checkout_13"]

        special_requests = set()
        diets = []
        for u in member_users:
            d = safe_get(u, "diet_health", "diet")
            if d and d != "none": diets.append(d)
            for a in (safe_get(u, "diet_health", "allergies", default=[]) or []):
                if a != "none":
                    special_requests.add(f"allergy_{a}")
        if diets:
            special_requests.add("breakfast_diet_friendly")
        if any(safe_get(u, "diet_health", "accessibility") in {"elevator_needed","reduced_mobility"} for u in member_users):
            special_requests.add("accessible_room_or_elevator_nearby")
        if any(safe_get(u, "comfort", "noise_tolerance") == "low" for u in member_users):
            special_requests.add("quiet_room_request")
        special_requests.add("non_smoking_rooms")

        room_blocks = []
        for occ in rooms:
            bed_cfg = "twin" if room_setup_hint == "twin" else ("double" if room_setup_hint == "double" else random.choice(["twin","double"]))
            rp = {
                "name": random.choice(["Flexible","Semi-Flexible","Advance Purchase"]),
                "board": random.choice(["Room Only","Breakfast Included","Half Board"]),
                "cancellation": cancellation_policy(refundable_prob=0.72),
                "currency": currency,
                "nightly_prices": nightly
            }
            room_blocks.append({
                "room_id": f"rm_{uuid.uuid4().hex[:10]}",
                "occupant_user_ids": occ,
                "bed_config": bed_cfg,
                "rate_plan": rp,
                "requests": sorted(list(special_requests))
            })

        booking = {
            "booking_id": f"hb_{uuid.uuid4().hex[:12]}",
            "group_id": group.get("group_id"),
            "trip_city": city,
            "hotel": {
                "hotel_id": chosen.get("id"),
                "name": chosen.get("name"),
                "stars": chosen.get("stars"),
                "distance_to_station_km": chosen.get("distance_to_station_km"),
                "amenities": chosen.get("amenities"),
            },
            "stay": {
                "check_in": check_in.isoformat(),
                "check_out": check_out.isoformat(),
                "nights": nights
            },
            "rooms": room_blocks,
            "payment": {
                "split_rule": split_rule,
                "lead_payer_user_id": lead_payer,
                "currency": currency,
                "price_summary": {
                    "room_subtotal": round(sum(nightly), 2),
                    "vat": vat,
                    "city_tax": city_tax,
                    "grand_total": total
                }
            },
            "policies": {
                "check_in_time": "15:00",
                "check_out_time": "11:00",
                "early_check_in_requested": req_eci,
                "late_checkout_requested": req_lco
            },
            "vip": {
                "tier": vip,
                "perks": perks
            },
            "operations": {
                "status": random.choice(["confirmed","confirmed","hold"]),
                "confirmation_code": f"RMH-{datetime.utcnow().strftime('%Y%m%d')}-{uuid.uuid4().hex[:6].upper()}",
                "supplier": {"channel": "RoverMitra", "supplier_ref": f"SUP-{uuid.uuid4().hex[:8].upper()}"},
                "audit": {"created_at": datetime.utcnow().isoformat() + "Z", "created_by": "booking_engine_v1", "version": "1.0"}
            },
            # Only references (no duplication of core user data)
            "members": [{"user_id": uid} for uid in member_ids],
            "inventory_ref": {
                "source_file": str(GROUPS_PATH),
                "hotel_inventory_city": city,
                "hotel_inventory_nights": safe_get(inv_city, "nights")
            }
        }

        bookings.append(booking)

    return bookings

# -----------------------------
# Main
# -----------------------------
def main():
    assert USERS_PATH.exists(), f"Missing {USERS_PATH}"
    assert MM_PATH.exists(), f"Missing {MM_PATH}"
    assert GROUPS_PATH.exists(), f"Missing {GROUPS_PATH}"

    users = json.loads(USERS_PATH.read_text(encoding="utf-8"))
    mm_profiles = json.loads(MM_PATH.read_text(encoding="utf-8"))
    groups = json.loads(GROUPS_PATH.read_text(encoding="utf-8"))

    users_map = idx_by(users, "user_id")
    mm_map = idx_by(mm_profiles, "user_id")

    all_bookings = []
    for g in groups:
        bks = build_bookings_for_group(g, users_map, mm_map)
        if bks:
            all_bookings.extend(bks)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(all_bookings, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"✅ Created {len(all_bookings)} hotel booking records → {OUT_PATH}")
    if all_bookings:
        print(json.dumps(all_bookings[0], indent=2)[:1400] + "\n...")

if __name__ == "__main__":
    main()
