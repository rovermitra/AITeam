#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Restaurant data generator (integrated)
- Reads: users/data/users_core.json, MatchMaker/data/matchmaker_profiles.json, Flight/data/travel_groups_integrated_v3.json
- Writes:
    Restaurants/data/restaurants_catalog.json
    Restaurants/data/restaurants_availability.json
    Restaurants/data/group_restaurant_reservations.json
- Design:
    * Does NOT repeat user data. Uses only user_id, group_id references.
    * City/country realism; cuisines, dietary tags, price levels, opening hours, amenities.
    * Availability slots + group-level AI-picked holds that align with diet/budget/party size.
"""

import os
import json
import uuid
import random
from pathlib import Path
from datetime import date, datetime, timedelta, time

random.seed(111)

# ------------------------------
# Inputs / Outputs
# ------------------------------
from pathlib import Path

# --- put this near the top ---
BASE_DIR = Path(__file__).resolve().parents[1]   # <project-root>, since this file is in Scripts/

USERS_PATH  = BASE_DIR / "users/data/users_core.json"
MM_PATH     = BASE_DIR / "MatchMaker/data/matchmaker_profiles.json"
GROUPS_PATH = BASE_DIR / "Flight/data/travel_groups_integrated_v3.json"

CATALOG_OUT = BASE_DIR / "Restaurants/data/restaurants_catalog.json"
AVAIL_OUT   = BASE_DIR / "Restaurants/data/restaurants_availability.json"
RESV_OUT    = BASE_DIR / "Restaurants/data/group_restaurant_reservations.json"


CATALOG_PER_CITY_RANGE = (24, 48)   # how many restaurants to create per destination city
AVAILABILITY_DAYS = 35              # horizon for slots
LUNCH_WINDOW = (time(12, 0), time(15, 0))
DINNER_WINDOW = (time(18, 0), time(22, 30))
SLOT_EVERY_MIN = 30

# ------------------------------
# Pools / Taxonomies
# ------------------------------
CUISINES = [
    "Swiss","Italian","French","German","Spanish","Greek","Indian","Japanese","Thai","Turkish",
    "Vegan","Vegetarian","Seafood","Steakhouse","Cafe","Bakery","Lebanese","Moroccan","Mexican",
    "Peruvian","Korean","Chinese","Vietnamese","Malaysian","Indonesian","Middle Eastern"
]
PRICE_LEVELS = ["€","€€","€€€"]  # ~ budget, mid, premium
DIET_TAGS = ["veg-friendly","vegan","halal","kosher","gluten-free","no-pork","pescatarian","lactose-free"]
SERVICE = ["dine-in","takeaway","delivery"]
SEATING = ["indoor","outdoor","bar","window","private-room"]
PAYMENT = ["visa","mastercard","amex","cash","apple-pay","google-pay"]
AMENITIES = ["wifi","wheelchair-access","family-friendly","pet-friendly","reservations","high-chairs","parking-nearby"]
LANGS = ["en","de","fr","it","es","pt","nl"]

DIET_STRICT = ["vegan","kosher","halal","gluten-free","lactose-free"]
DIET_MEDIUM = ["vegetarian","pescatarian","no pork"]
CITY_NAME_TOKENS = {
    "Italian": ["Trattoria","Osteria","Enoteca","Casa","Nonna","Roma","Milano","Napoli"],
    "French": ["Bistro","Brasserie","Maison","Côte","Bleu","Rouge","Provence"],
    "German": ["Alpen","Garten","Hof","Keller","Haus","Stube","Bahnhof"],
    "Spanish": ["Tapas","Cantina","Madrid","Barcelona","Sevilla","Ibérica"],
    "Greek": ["Taverna","Mykonos","Paros","Santorini","Olive","Agora"],
    "Turkish": ["Kebab","Meze","Anatolia","Istanbul","Bosporus","Mangal"],
    "Japanese": ["Sushi","Izakaya","Ramen","Tokyo","Kyoto","Sakura"],
    "Korean": ["Seoul","Bibim","Kimchi","Hansik","Gangnam"],
    "Chinese": ["Dim Sum","Wok","Dragon","Sichuan","Canton"],
    "Indian": ["Curry","Masala","Tandoor","Delhi","Bombay","Hyderabadi"],
    "Middle Eastern": ["Souk","Mezze","Shawarma","Zaatar","Cedar"],
    "Seafood": ["Harbor","Marina","Oyster","Fish","Catch"],
    "Vegan": ["Green","Plant","Leaf","Roots","Sprout"],
    "Vegetarian": ["Garden","Herb","Leafy","Greenhouse"],
    "Cafe": ["Roasters","Beans","Brew","Press","Corner"],
    "Bakery": ["Boulangerie","Patisserie","Bakehouse","Flour","Crust"]
}

CURRENCY_BY_COUNTRY = {
    "USA":"USD","Canada":"CAD","Mexico":"MXN","Brazil":"BRL","Argentina":"ARS","Chile":"CLP","Peru":"PEN","Colombia":"COP",
    "UK":"GBP","Ireland":"EUR","Germany":"EUR","France":"EUR","Spain":"EUR","Italy":"EUR","Portugal":"EUR",
    "Netherlands":"EUR","Belgium":"EUR","Switzerland":"CHF","Austria":"EUR","Poland":"PLN","Czechia":"CZK",
    "Denmark":"DKK","Sweden":"SEK","Norway":"NOK","Finland":"EUR","Greece":"EUR","Turkey":"TRY",
    "Japan":"JPY","South Korea":"KRW","China":"CNY","India":"INR","Pakistan":"PKR","Thailand":"THB","Vietnam":"VND","Malaysia":"MYR",
    "Singapore":"SGD","Indonesia":"IDR","Philippines":"PHP","UAE":"AED","Saudi Arabia":"SAR","Qatar":"QAR","Israel":"ILS","Jordan":"JOD",
    "Morocco":"MAD","Egypt":"EGP","South Africa":"ZAR","Kenya":"KES","Australia":"AUD","New Zealand":"NZD","Iceland":"ISK"
}

# ------------------------------
# Helpers
# ------------------------------
def idx_by(lst, key):
    return {x[key]: x for x in lst if isinstance(x, dict) and key in x}

def city_currency(city, location_data):
    for loc in location_data:
        if loc.get("city") == city:
            return CURRENCY_BY_COUNTRY.get(loc.get("country",""), "EUR")
    return "EUR"

def city_stub_latlon(city):
    # deterministic pseudo coordinates per city (do not claim to be real)
    r = random.Random(city)
    lat = r.uniform(-60.0, 60.0)
    lon = r.uniform(-120.0, 120.0)
    return round(lat, 5), round(lon, 5)

def random_name_for_city(city, cuisine):
    base_tokens = CITY_NAME_TOKENS.get(cuisine, []) or CITY_NAME_TOKENS.get("Cafe", [])
    if base_tokens and random.random() < 0.7:
        return f"{city} {random.choice(base_tokens)}"
    # generic
    return f"{random.choice(['Alpen','Lakeside','Old Town','Panorama','Central','Station','Vista','Riverside'])} {random.choice(['Kitchen','Bistro','House','Dining','Grill','Cafe','Table'])}"

def random_open_hours():
    # simple 7-day schedule (some days closed at lunch, open for dinner)
    oh = {}
    for i, day in enumerate(["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]):
        lunch_open = random.random() < 0.8
        dinner_open = True if day in ["Fri","Sat","Sun"] else random.random() < 0.95
        windows = []
        if lunch_open:
            windows.append({"start": "12:00", "end": "15:00"})
        if dinner_open:
            end = "23:00" if day in ["Fri","Sat"] else "22:30"
            windows.append({"start": "18:00", "end": end})
        oh[day] = windows
    return oh

def price_level_for_budget(avg_per_day):
    if avg_per_day <= 90:  return "€"
    if avg_per_day >= 180: return "€€€"
    return "€€"

def strictness_from_group_diets(diets):
    diets = [d for d in diets if d and d != "none"]
    if not diets: return "flexible"
    if any(d in DIET_STRICT for d in diets): return "strict"
    if any(d in DIET_MEDIUM for d in diets): return "moderate"
    return "flexible"

def diet_tags_for_cuisine(base_diet):
    tags = set()
    if base_diet and base_diet != "none":
        if base_diet == "vegetarian": tags.add("veg-friendly")
        elif base_diet in DIET_STRICT or base_diet in DIET_MEDIUM: tags.add(base_diet)
    # random extras for discoverability
    if random.random() < 0.35: tags.add(random.choice(DIET_TAGS))
    if random.random() < 0.20: tags.add(random.choice(DIET_TAGS))
    return sorted(tags)

def gen_menu(cuisine, price_level):
    # very small synthetic menu with allergens/diets
    base_prices = {"€": (9, 18), "€€": (15, 35), "€€€": (28, 65)}
    lo, hi = base_prices.get(price_level, (15, 35))
    names = {
        "Italian": ["Bruschetta","Pasta al Pomodoro","Risotto","Margherita Pizza","Tiramisu"],
        "French": ["Onion Soup","Coq au Vin","Nicoise Salad","Crème Brûlée","Steak Frites"],
        "German": ["Pretzel & Obatzda","Schnitzel","Käsespätzle","Apfelstrudel"],
        "Japanese": ["Miso Soup","Salmon Nigiri","Tempura","Tonkotsu Ramen","Matcha Mochi"],
        "Indian": ["Paneer Tikka","Chicken Tikka","Dal Makhani","Biryani","Gulab Jamun"],
        "Spanish": ["Patatas Bravas","Seafood Paella","Jamón Croquetas","Churros"],
        "Greek": ["Greek Salad","Souvlaki","Moussaka","Baklava"],
        "Turkish": ["Meze Platter","Adana Kebab","Lentil Soup","Kunefe"]
    }
    pool = names.get(cuisine, ["Chef Special","Seasonal Salad","Grilled Dish","House Dessert"])
    items = []
    for _ in range(random.randint(5, 9)):
        nm = random.choice(pool)
        items.append({
            "name": nm,
            "price": round(random.uniform(lo, hi), 2),
            "allergens": sorted(random.sample(["gluten","nuts","dairy","eggs","shellfish","soy"], k=random.randint(0,2))),
            "dietary": sorted(random.sample(["veg-friendly","vegan","gluten-free","pescatarian","halal","kosher","no-pork"], k=random.randint(0,2)))
        })
    return items

def slot_range(day: date, start: time, end: time, step_min: int):
    slots = []
    cur = datetime.combine(day, start)
    end_dt = datetime.combine(day, end)
    while cur <= end_dt:
        slots.append(cur.isoformat())
        cur += timedelta(minutes=step_min)
    return slots

# ------------------------------
# Load sources
# ------------------------------
def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else []

def all_destination_cities(groups):
    cities = set()
    for g in groups:
        dp = g.get("draft_plan") or {}
        # include any itinerary bases and hotel inventory cities
        for day in (dp.get("itinerary") or []):
            if "base" in day: cities.add(day["base"])
        for inv in (dp.get("hotel_inventory") or []):
            if "city" in inv: cities.add(inv["city"])
        # restaurants were previously generated inside groups; we ignore those objects and create a master catalog
    return sorted(cities)

def members_diets_and_budget(g):
    """Return (diets, avg_budget_per_day, currency_by_meet_city) for a group."""
    diets = []
    budgets = []
    meet_city = None
    dp = g.get("draft_plan") or {}
    mp = dp.get("meeting_plan") or {}
    if "meet_at" in mp: meet_city = mp["meet_at"]

    # budgets & diets from user records (if present)
    return meet_city, diets, budgets

# ------------------------------
# Build catalog per city
# ------------------------------
def build_city_catalog(city, country_guess, currency):
    # choose how many venues
    n = random.randint(*CATALOG_PER_CITY_RANGE)
    venues = []
    lat0, lon0 = city_stub_latlon(city)
    for _ in range(n):
        cuisine = random.choice(CUISINES)
        price = random.choice(PRICE_LEVELS)
        rid = f"rst_{uuid.uuid4().hex[:10]}"
        venues.append({
            "id": rid,
            "city": city,
            "country": country_guess,
            "name": random_name_for_city(city, cuisine if cuisine in CITY_NAME_TOKENS else "Cafe"),
            "cuisine": cuisine,
            "price_level": price,
            "dietary_tags": sorted(list({*diet_tags_for_cuisine(random.choice(["none","vegetarian","vegan","pescatarian","halal","kosher","gluten-free","no-pork"]))})),
            "rating": round(random.uniform(3.7, 4.9), 1),
            "reviews_count": random.randint(40, 3500),
            "distance_to_center_km": round(random.uniform(0.1, 3.5), 2),
            "open_hours": random_open_hours(),
            "menu_languages": sorted(random.sample(LANGS, k=random.randint(1, 3))),
            "amenities": sorted(random.sample(AMENITIES, k=random.randint(3, 6))),
            "seating": sorted(random.sample(SEATING, k=random.randint(1, 3))),
            "service_options": sorted(random.sample(SERVICE, k=random.randint(1, 3))),
            "payment_methods": sorted(random.sample(PAYMENT, k=random.randint(2, 5))),
            "popular_dishes": gen_menu(cuisine, price),
            "geo": {"approx_lat": lat0 + random.uniform(-0.03, 0.03), "approx_lon": lon0 + random.uniform(-0.03, 0.03)},
            "contact": {"phone": f"+{random.randint(11,99)} {random.randint(100,999)} {random.randint(1000000,9999999)}",
                        "website": None, "email": None},
            "reservation_policy": {
                "allow_online": True,
                "deposit_required": random.random() < 0.15,
                "cancellation_window_hours": random.choice([2, 4, 6, 24]),
                "min_party": 1,
                "max_party": random.choice([6, 8, 10, 12])
            },
            "currency": currency,
            "partner": random.random() < 0.35   # if True → instant confirm OK
        })
    return venues

def derive_location_data(groups):
    """Collect (city, country) hints from hotel inventory & itinerary (best-effort)."""
    seen = {}
    for g in groups:
        dp = g.get("draft_plan") or {}
        for inv in (dp.get("hotel_inventory") or []):
            c = inv.get("city"); 
            if c: seen[c] = seen.get(c, {"city": c, "country": None})
        for day in (dp.get("itinerary") or []):
            c = day.get("base")
            if c: seen[c] = seen.get(c, {"city": c, "country": None})
    return list(seen.values())

# ------------------------------
# Availability generator
# ------------------------------
def gen_availability_for_restaurant(rid, days=AVAILABILITY_DAYS):
    today = date.today()
    slots = []
    capacity_base = random.choice([40, 60, 80, 100])
    for d in range(days):
        day = today + timedelta(days=d)
        # lunch (not every day)
        if day.weekday() < 6 and random.random() < 0.85:
            for s in slot_range(day, LUNCH_WINDOW[0], LUNCH_WINDOW[1], SLOT_EVERY_MIN):
                slots.append({"restaurant_id": rid, "slot_iso": s, "capacity_left": random.randint(0, capacity_base//2)})
        # dinner (every day; busier Fri/Sat)
        dinner_end = DINNER_WINDOW[1]
        for s in slot_range(day, DINNER_WINDOW[0], dinner_end, SLOT_EVERY_MIN):
            fill = 0.6 if day.weekday() in (4,5) else 0.4
            left = max(0, int(capacity_base * (1 - fill) + random.randint(-8, 10)))
            slots.append({"restaurant_id": rid, "slot_iso": s, "capacity_left": left})
    return slots

# ------------------------------
# Matching & group reservations
# ------------------------------
def avg_group_budget_per_day(users_by_id, member_ids):
    vals = []
    for uid in member_ids:
        u = users_by_id.get(uid) or {}
        b = (u.get("budget") or {}).get("amount")
        if isinstance(b, (int, float)): vals.append(b)
    return sum(vals)/len(vals) if vals else 120

def group_diets(users_by_id, member_ids):
    diets = []
    for uid in member_ids:
        u = users_by_id.get(uid) or {}
        d = ((u.get("diet_health") or {}).get("diet"))
        if d: diets.append(d)
    return diets or ["none"]

def best_fit_restaurant(city_catalog, target_price, diets, party_size):
    """Pick best-fit by diet tags, price, max_party and distance."""
    strict = strictness_from_group_diets(diets)
    # score function
    best = None; best_score = -1
    for r in city_catalog:
        # party size feasibility
        maxp = (r.get("reservation_policy") or {}).get("max_party", 6)
        if party_size > maxp: continue
        score = 0.0
        # price fit
        score += 1.0 if r["price_level"] == target_price else (0.5 if target_price == "€€" and r["price_level"] in ["€","€€€"] else 0.2)
        # diet fit
        rtags = set(r.get("dietary_tags") or [])
        if strict == "strict":
            if not rtags.intersection(set(DIET_STRICT + DIET_MEDIUM)): score -= 10
            else: score += 1.2
        elif strict == "moderate":
            if rtags.intersection(set(DIET_STRICT + DIET_MEDIUM)): score += 0.8
            else: score += 0.2
        else:
            score += 0.2
        # distance
        dist = r.get("distance_to_center_km", 1.0)
        score += max(0, 1.0 - (dist/4.0))  # closer gets more
        # rating slight boost
        score += (r.get("rating", 4.2) - 4.0) * 0.4
        if score > best_score:
            best_score = score; best = r
    return best, round(best_score, 3)

# ------------------------------
# Main
# ------------------------------
def main():
    assert USERS_PATH.exists(), f"Missing {USERS_PATH}"
    assert GROUPS_PATH.exists(), f"Missing {GROUPS_PATH}"
    users = load_json(USERS_PATH)
    groups = load_json(GROUPS_PATH)
    mm_profiles = load_json(MM_PATH)  # not strictly needed now, but kept for future tie-ins

    users_by_id = idx_by(users, "user_id")

    # Collect destination cities and rough country hints
    locs = derive_location_data(groups)
    # For currency, try to infer via any group's hotel inventory hint (fall back EUR)
    # Build per-city catalog
    city_catalog_map = {}
    for loc in locs:
        city = loc["city"]; country = loc.get("country") or ""
        # pick currency via guess: if any group meets at this city, detect; else EUR
        currency = CURRENCY_BY_COUNTRY.get(country, "EUR")
        city_catalog_map[city] = build_city_catalog(city, country, currency)

    # Build availability for all restaurants
    availability = []
    for city, venues in city_catalog_map.items():
        for r in venues:
            availability.extend(gen_availability_for_restaurant(r["id"], AVAILABILITY_DAYS))

    # Create group reservations (holds) per group-day-city
    reservations = []
    for g in groups:
        gid = g.get("group_id")
        members = [m.get("user_id") for m in (g.get("members") or []) if m.get("user_id")]
        party_size = max(1, len(members))
        avg_budget = avg_group_budget_per_day(users_by_id, members)
        target_price = price_level_for_budget(avg_budget)
        diets = group_diets(users_by_id, members)

        dp = g.get("draft_plan") or {}
        itinerary = dp.get("itinerary") or []

        for day in itinerary:
            city = day.get("base")
            if not city or city not in city_catalog_map: 
                continue
            catalog = city_catalog_map[city]
            best, score = best_fit_restaurant(catalog, target_price, diets, party_size)
            if not best:
                continue

            # choose a realistic time (19:00–20:30)
            day_dt = datetime.fromisoformat(day["date"])
            timeslot = day_dt.replace(hour=random.choice([18, 19, 19, 20, 20, 20, 21]), minute=random.choice([0, 15, 30, 45]), second=0, microsecond=0)
            res_id = f"resv_{uuid.uuid4().hex[:12]}"

            reservations.append({
                "reservation_id": res_id,
                "group_id": gid,
                "city": city,
                "date": day["date"],
                "restaurant_id": best["id"],
                "slot_iso": timeslot.isoformat(),
                "party_size": party_size,
                "status": "hold" if not best.get("partner") else random.choice(["hold","confirmed"]),
                "payment": {
                    "split_rule": random.choice(["each_own","50/50","custom"]),
                    "currency": best["currency"]
                },
                "dietary_considerations": sorted(list(set([d for d in diets if d and d != "none"]))),
                "special_requests": sorted(random.sample(
                    ["quiet_table","birthday","anniversary","near_window","no_shellfish","no_nuts","wheelchair_access"],
                    k=random.choice([0,1,1,2]))),
                "matching_rationale": {
                    "budget_target": target_price,
                    "diet_strictness": strictness_from_group_diets(diets),
                    "distance_to_center_km": best.get("distance_to_center_km"),
                    "restaurant_rating": best.get("rating"),
                    "score": score,
                    "source": "AI_recommendation"
                }
            })

    # Write outputs
    CATALOG_OUT.parent.mkdir(parents=True, exist_ok=True)
    CATALOG_OUT.write_text(json.dumps(
        [{"city": c, "restaurants": v} for c, v in sorted(city_catalog_map.items())],
        indent=2, ensure_ascii=False
    ), encoding="utf-8")

    AVAIL_OUT.parent.mkdir(parents=True, exist_ok=True)
    AVAIL_OUT.write_text(json.dumps(availability, indent=2, ensure_ascii=False), encoding="utf-8")

    RESV_OUT.parent.mkdir(parents=True, exist_ok=True)
    RESV_OUT.write_text(json.dumps(reservations, indent=2, ensure_ascii=False), encoding="utf-8")

    # Quick peek
    print(f"✅ Catalog cities: {len(city_catalog_map)}  → {CATALOG_OUT}")
    print(f"✅ Availability slots: {len(availability)} → {AVAIL_OUT}")
    print(f"✅ Group reservations: {len(reservations)} → {RESV_OUT}")
    print("Reading:", USERS_PATH)
    print("Reading:", GROUPS_PATH)
    print("Writing to:", CATALOG_OUT, AVAIL_OUT, RESV_OUT)

    # Example peek (truncated)
    first_city = next(iter(city_catalog_map.keys()), None)
    if first_city:
        print(json.dumps(city_catalog_map[first_city][:2], indent=2)[:900] + " ...")
    if reservations:
        print(json.dumps(reservations[0], indent=2)[:600] + " ...")

if __name__ == "__main__":
    main()
