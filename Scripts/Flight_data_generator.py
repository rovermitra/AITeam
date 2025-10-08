#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Integrated generator that:
- Loads users_core.json (authoritative user data) and matchmaker_profiles.json (preferences)
- Creates realistic multi-person travel group requests without duplicating user data
- Builds inventories (flights/trains/hotels/restaurants/activities)
- Only references users by user_id/handle in the output

Output → Flight/data/travel_groups_integrated_v3.json
"""

import os
import json
import uuid
import random
from datetime import date, timedelta, datetime, time
from pathlib import Path

random.seed(73)

# ============================
# Config (paths & sizes)
# ============================
USERS_PATH = "users/data/users_core.json"
MM_PATH = "MatchMaker/data/matchmaker_profiles.json"
OUT_PATH = "Flight/data/travel_groups_integrated_v3.json"
N_GROUPS = 1000
RICH_GROUP_RATIO = 0.8   # share of groups with full context; others are lean

# ============================
# Catalogs (realistic)
# ============================
COUNTRIES = {
    # --- EUROPE ---
    "Germany": {
        "langs": ["de", "en"],
        "cities": {
            "Berlin": {"airports": ["BER"]},
            "Munich": {"airports": ["MUC"]},
            "Hamburg": {"airports": ["HAM"]},
            "Cologne": {"airports": ["CGN"]},
            "Frankfurt": {"airports": ["FRA"]},
            "Stuttgart": {"airports": ["STR"]},
            "Dusseldorf": {"airports": ["DUS"]},
            "Leipzig": {"airports": ["LEJ"]},
            "Dresden": {"airports": ["DRS"]},
            "Nuremberg": {"airports": ["NUE"]}
        }
    },
    "Switzerland": {
        "langs": ["de", "fr", "it", "en"],
        "cities": {
            "Zurich": {"airports": ["ZRH"]},
            "Geneva": {"airports": ["GVA"]},
            "Basel": {"airports": ["BSL"]},
            "Bern": {"airports": ["BRN"]},
            "Lucerne": {"airports": ["ZRH"]},
            "Interlaken": {"airports": ["BRN", "ZRH"]},
            "Zermatt": {"airports": ["GVA", "ZRH"]}
        }
    },
    "Austria": {"langs": ["de", "en"], "cities": {"Vienna": {"airports": ["VIE"]}, "Salzburg": {"airports": ["SZG"]}, "Graz": {"airports": ["GRZ"]}, "Innsbruck": {"airports": ["INN"]}}},
    "France": {"langs": ["fr", "en"], "cities": {"Paris": {"airports": ["CDG", "ORY"]}, "Lyon": {"airports": ["LYS"]}, "Nice": {"airports": ["NCE"]}, "Toulouse": {"airports": ["TLS"]}, "Bordeaux": {"airports": ["BOD"]}, "Marseille": {"airports": ["MRS"]}, "Nantes": {"airports": ["NTE"]}}},
    "UK": {"langs": ["en"], "cities": {"London": {"airports": ["LHR", "LGW", "LCY", "LTN", "STN"]}, "Manchester": {"airports": ["MAN"]}, "Edinburgh": {"airports": ["EDI"]}, "Glasgow": {"airports": ["GLA"]}, "Birmingham": {"airports": ["BHX"]}, "Bristol": {"airports": ["BRS"]}}},
    "Netherlands": {"langs": ["nl", "en"], "cities": {"Amsterdam": {"airports": ["AMS"]}, "Rotterdam": {"airports": ["RTM"]}}},
    "Belgium": {"langs": ["nl", "fr", "en"], "cities": {"Brussels": {"airports": ["BRU"]}, "Antwerp": {"airports": ["ANR"]}}},
    "Spain": {"langs": ["es", "en"], "cities": {"Madrid": {"airports": ["MAD"]}, "Barcelona": {"airports": ["BCN"]}, "Valencia": {"airports": ["VLC"]}, "Seville": {"airports": ["SVQ"]}, "Malaga": {"airports": ["AGP"]}, "Bilbao": {"airports": ["BIO"]}}},
    "Portugal": {"langs": ["pt", "en"], "cities": {"Lisbon": {"airports": ["LIS"]}, "Porto": {"airports": ["OPO"]}, "Faro": {"airports": ["FAO"]}}},
    "Italy": {"langs": ["it", "en"], "cities": {"Rome": {"airports": ["FCO", "CIA"]}, "Milan": {"airports": ["MXP", "LIN", "BGY"]}, "Venice": {"airports": ["VCE"]}, "Florence": {"airports": ["FLR"]}, "Naples": {"airports": ["NAP"]}, "Turin": {"airports": ["TRN"]}}},
    "Greece": {"langs": ["el", "en"], "cities": {"Athens": {"airports": ["ATH"]}, "Thessaloniki": {"airports": ["SKG"]}, "Heraklion": {"airports": ["HER"]}}},
    # --- AMERICAS (trimmed) ---
    "USA": {"langs": ["en", "es"], "cities": {"New York": {"airports": ["JFK", "EWR", "LGA"]}, "San Francisco": {"airports": ["SFO"]}, "Los Angeles": {"airports": ["LAX"]}, "Chicago": {"airports": ["ORD"]}, "Miami": {"airports": ["MIA"]}, "Seattle": {"airports": ["SEA"]}, "Boston": {"airports": ["BOS"]}, "Denver": {"airports": ["DEN"]}}},
    "Canada": {"langs": ["en", "fr"], "cities": {"Toronto": {"airports": ["YYZ"]}, "Vancouver": {"airports": ["YVR"]}, "Montreal": {"airports": ["YUL"]}}},
    # --- APAC (trimmed) ---
    "Japan": {"langs": ["ja", "en"], "cities": {"Tokyo": {"airports": ["HND", "NRT"]}, "Osaka": {"airports": ["KIX", "ITM"]}, "Kyoto": {"airports": ["KIX"]}}},
    "India": {"langs": ["en", "hi"], "cities": {"Delhi": {"airports": ["DEL"]}, "Mumbai": {"airports": ["BOM"]}, "Bangalore": {"airports": ["BLR"]}, "Hyderabad": {"airports": ["HYD"]}}},
}

LOCATION_DATA = [
    {"city": city, "country": country, "languages": cinfo["langs"], "airports": meta["airports"]}
    for country, cinfo in COUNTRIES.items()
    for city, meta in cinfo["cities"].items()
]

CURRENCY_BY_COUNTRY = {
    "USA":"USD","Canada":"CAD","UK":"GBP","Germany":"EUR","France":"EUR","Spain":"EUR",
    "Italy":"EUR","Portugal":"EUR","Netherlands":"EUR","Belgium":"EUR","Switzerland":"CHF",
    "Austria":"EUR","Greece":"EUR","Japan":"JPY","India":"INR"
}

AIRLINES = ["LH","LX","BA","AF","KL","U2","FR","TK","SN","SK","AY","OS","IB","AZ","EI","TP","LO","DL","UA"]
CABINS = ["economy","premium_economy","business"]
RAIL_CARRIERS = ["DB","SBB","SNCF","Trenitalia","Renfe","ÖBB","NS","SNCB"]
CUISINES = [
    "Swiss","Italian","French","German","Spanish","Greek","Indian","Japanese","Thai","Turkish","Vegan",
    "Vegetarian","Seafood","Steakhouse","Cafe","Bakery","Lebanese","Moroccan","Mexican","Peruvian","Korean","Chinese",
]

# Interest → destination bias (expanded keywords)
INTEREST_CITY_HINTS = {
    "mountain": ["Interlaken","Zermatt","Innsbruck","Lucerne"],
    "hike": ["Interlaken","Zermatt","Innsbruck"],
    "ski": ["Zermatt","Innsbruck"],
    "lake": ["Lucerne","Interlaken","Geneva"],
    "museum": ["Paris","Amsterdam","Berlin"],
    "old town": ["Florence","Bern","Prague"],
    "architecture": ["Barcelona","Paris","Rome"],
    "beach": ["Nice","Malaga","Lisbon"],
    "rail": ["Lucerne","Interlaken","Bern"],
    "photo": ["Paris","Amsterdam","Zurich"]
}

# ============================
# Utils
# ============================
def idx_by(lst, key):
    return {x[key]: x for x in lst if isinstance(x, dict) and key in x}

def nearest_airport_for_city(city):
    for loc in LOCATION_DATA:
        if loc["city"] == city:
            aps = loc.get("airports") or []
            return random.choice(aps) if aps else None
    return None

def currency_for_city(city):
    for loc in LOCATION_DATA:
        if loc["city"] == city:
            return CURRENCY_BY_COUNTRY.get(loc["country"], "EUR")
    return "EUR"

def future_date(min_days=10, max_days=240):
    start = date.today() + timedelta(days=random.randint(min_days, max_days))
    length = random.randint(4, 12)
    end = start + timedelta(days=length)
    return start, end, length

def time_on_date(d, hhmm):
    h, m = map(int, hhmm.split(":"))
    return datetime.combine(d, time(h, m))

def random_dep_time():
    return random.choice(["06:40","07:10","08:30","09:45","10:15","12:05","14:30","16:20","18:15"])

def random_arrival(dep_dt, min_hours=1.0, max_hours=6.5):
    hours = random.uniform(min_hours, max_hours)
    return dep_dt + timedelta(hours=hours)

def budget_band(amount):
    if amount is None: return "mid-range"
    if amount <= 90:   return "budget"
    if amount >= 180:  return "luxury"
    return "mid-range"

# ============================
# Inventory Generators
# ============================
def gen_flight_offers_for_user(user, meet_city, depart_date, return_date):
    origin_city = (user.get("home_base") or {}).get("city")
    origin_ap = nearest_airport_for_city(origin_city) or "XXX"
    meet_ap = nearest_airport_for_city(meet_city) or "XXX"
    currency = currency_for_city(meet_city)

    def mk_flight(_from, _to, d):
        dep_str = random_dep_time()
        dep_dt = time_on_date(d, dep_str)
        arr_dt = random_arrival(dep_dt)
        airline = random.choice(AIRLINES)
        cabin = random.choice(CABINS if random.random() < 0.2 else ["economy"])
        price = random.randint(60, 420) if currency in ["EUR","CHF","GBP"] else random.randint(150, 900)
        return {
            "id": f"flt_{uuid.uuid4().hex[:10]}",
            "airline": airline,
            "flight_number": f"{airline}{random.randint(100, 9999)}",
            "from_airport": _from,
            "to_airport": _to,
            "depart_iso": dep_dt.isoformat(),
            "arrive_iso": arr_dt.isoformat(),
            "duration_min": int((arr_dt - dep_dt).total_seconds() / 60),
            "cabin": cabin,
            "bag_allowance": random.choice(["cabin only","cabin+23kg"]),
            "price": price,
            "currency": currency,
            "co2_kg": random.randint(50, 220)
        }

    outbound = [mk_flight(origin_ap, meet_ap, depart_date) for _ in range(random.randint(2,4))]
    inbound  = [mk_flight(meet_ap, origin_ap, return_date) for _ in range(random.randint(2,4))]
    return {"outbound": outbound, "return": inbound}

def gen_train_offers(origin_city, dest_city, any_date):
    carrier = random.choice(RAIL_CARRIERS)
    dep_str = random_dep_time()
    dep_dt = time_on_date(any_date, dep_str)
    arr_dt = dep_dt + timedelta(hours=random.uniform(1.5, 7.0))
    price = random.randint(18, 120)
    return [{
        "id": f"rail_{uuid.uuid4().hex[:8]}",
        "carrier": carrier,
        "from": origin_city,
        "to": dest_city,
        "depart_iso": dep_dt.isoformat(),
        "arrive_iso": arr_dt.isoformat(),
        "transfers": random.choice([0,1,2]),
        "duration_min": int((arr_dt-dep_dt).total_seconds()/60),
        "price": price,
        "currency": currency_for_city(dest_city),
        "co2_kg": random.randint(3, 20)
    }]

def gen_hotels_for_city(city, nights, price_band="mid-range"):
    currency = currency_for_city(city)
    hotels = []
    for _ in range(random.randint(4,7)):
        price_base = {"budget": (50,120), "mid-range": (110,240), "luxury": (240,520)}.get(price_band, (110,220))
        price = random.randint(*price_base)
        hotels.append({
            "id": f"hot_{uuid.uuid4().hex[:10]}",
            "city": city,
            "name": f"{city} {random.choice(['Central','Grand','Garden','Station','Boutique','City','Plaza','Vista'])} Hotel",
            "stars": random.choice([3,4,5]) if price_band != "budget" else random.choice([2,3]),
            "price_per_night": price,
            "currency": currency,
            "distance_to_station_km": round(random.uniform(0.1, 1.8), 2),
            "amenities": sorted(random.sample(["wifi","private_bath","kitchen","workspace","gym","spa","breakfast","near_station","quiet_room","laundry"], k=random.randint(3,7)))
        })
    return {"city": city, "nights": nights, "options": hotels}

def gen_restaurants_for_city(city, diet="none"):
    price_level = ["€","€€","€€€"]
    diet_tags = {
        "none": [], "vegetarian": ["veg-friendly"], "vegan": ["vegan"], "halal": ["halal"], "kosher": ["kosher"],
        "gluten-free": ["gluten-free"], "no pork": ["no-pork"], "lactose-free": ["lactose-free"], "pescatarian": ["pescatarian"]
    }
    items = []
    for _ in range(random.randint(6,10)):
        tagset = set(diet_tags.get(diet, []))
        if random.random() < 0.35:
            tagset.add(random.choice(["veg-friendly","vegan","halal","gluten-free","no-pork","pescatarian"]))
        items.append({
            "id": f"rst_{uuid.uuid4().hex[:10]}",
            "city": city,
            "name": f"{random.choice(['Alpen','Lakeside','Station','Old Town','Panorama','Vista','Brasserie','Bistro','Cantina'])} {random.choice(['Kitchen','Bistro','House','Dining','Grill','Cafe','Table'])}",
            "cuisine": random.choice(CUISINES),
            "price_level": random.choice(price_level),
            "dietary_tags": sorted(list(tagset)),
            "rating": round(random.uniform(3.8, 4.8), 1),
            "distance_to_center_km": round(random.uniform(0.1, 3.0), 2)
        })
    return items

def gen_activities_for_city(city, interests):
    catalog = [
        ("viewpoint", "Panoramic viewpoint and photo walk", 1.5),
        ("museum", "Top museum with skip-the-line option", 2.0),
        ("hike", "Easy trail with lake/valley views", 3.0),
        ("lake", "Boat ride + lakeside stroll", 2.0),
        ("old town", "Guided old-town walk", 1.5),
        ("market", "Local market + street food tasting", 2.0),
        ("rail", "Scenic train segment with stops", 2.5)
    ]
    items = []
    for _ in range(random.randint(3,6)):
        t = random.choice(catalog)
        items.append({
            "id": f"act_{uuid.uuid4().hex[:10]}",
            "city": city,
            "type": t[0],
            "title": t[1],
            "duration_h": t[2],
            "suits": random.sample(interests, k=min(len(interests), random.randint(1,3))) if interests else [],
            "price": random.randint(0, 60),
            "currency": currency_for_city(city)
        })
    return items

# ============================
# Draft plan composer
# ============================
def choose_destinations_by_interests(users_list):
    # Map common interests (tokens) to hinted cities
    tokens = []
    for u in users_list:
        for it in (u.get("interests") or [])[:8]:
            low = it.lower()
            for key in INTEREST_CITY_HINTS.keys():
                if key in low:
                    tokens.append(key)
    pools = []
    for t in set(tokens):
        pools.extend(INTEREST_CITY_HINTS.get(t, []))
    if not pools:
        pools = [loc["city"] for loc in LOCATION_DATA]
    cities = list(dict.fromkeys(random.sample(pools, k=min(len(pools), random.randint(1,3)))))
    return cities or ["Zurich"]

def compose_draft_plan(member_users, member_mm, itinerary_style="multi-stop"):
    destinations = choose_destinations_by_interests(member_users)
    meet_city = destinations[0]
    start, end, pref_len = future_date(10, 240)

    # Nights split
    length = (end - start).days or pref_len
    if len(destinations) == 1:
        nights_per_city = {destinations[0]: max(3, length)}
    else:
        base = max(1, length // len(destinations))
        nights_per_city = {city: base for city in destinations}
        leftover = length - sum(nights_per_city.values())
        for i in range(leftover):
            nights_per_city[destinations[i % len(destinations)]] += 1

    # Inventories
    hotels_inventory = []
    restaurants_inventory = []
    activities_inventory = []

    # group diet (any strict member wins)
    group_diet = "none"
    for u in member_users:
        d = ((u.get("diet_health") or {}).get("diet"))
        if d and d != "none":
            group_diet = d
            break

    # hotel price band from budgets
    bands = []
    for u in member_users:
        amt = (u.get("budget") or {}).get("amount")
        bands.append(budget_band(amt))
    price_band = max(set(bands), key=bands.count) if bands else "mid-range"

    # aggregate interests
    all_interests = sorted({i for u in member_users for i in (u.get("interests") or [])})

    cur_date = start
    intercity_ground = []
    for city in destinations:
        hotels_inventory.append(gen_hotels_for_city(city, nights_per_city[city], price_band=price_band))
        restaurants_inventory.append({"city": city, "options": gen_restaurants_for_city(city, diet=group_diet)})
        activities_inventory.append({"city": city, "options": gen_activities_for_city(city, all_interests)})

    for i in range(len(destinations)-1):
        origin = destinations[i]
        dest = destinations[i+1]
        intercity_ground.append({"origin": origin, "dest": dest, "offers": gen_train_offers(origin, dest, cur_date)})
        cur_date += timedelta(days=nights_per_city[origin])

    # Flights by member (only user refs)
    flights_by_user = {}
    for u in member_users:
        flights_by_user[u["user_id"]] = gen_flight_offers_for_user(u, meet_city, start, end)

    chosen_flights = []
    for uid, offers in flights_by_user.items():
        ob = sorted(offers["outbound"], key=lambda x: x["price"])[0]
        ib = sorted(offers["return"],   key=lambda x: x["price"])[0]
        chosen_flights.append({"user_id": uid, "outbound": ob["id"], "return": ib["id"]})

    # Itinerary days
    itinerary = []
    d = start
    for city in destinations:
        for _ in range(nights_per_city[city]):
            plan = random.choice([
                "Old town walk + local food tour.",
                "Scenic viewpoint + short hike + cafe break.",
                "Museum morning + lakefront sunset.",
                "Day trip to nearby village + photo spots.",
                "Mountain excursion (easy trails) + early dinner.",
                "Market browse + street food sampler + rooftop view."
            ])
            itinerary.append({"date": d.isoformat(), "base": city, "plan": plan})
            d += timedelta(days=1)

    month = start.month
    season_hint = "Pack layers; mountain weather changes fast." if month in [5,6,7,8,9] else "Expect colder temps; consider thermal layers."

    return {
        "meeting_plan": {"meet_at": meet_city, "rationale": "Meet at first destination to minimize transfers."},
        "itinerary": itinerary,
        "intercity_ground_offers": intercity_ground,
        "flight_offers": flights_by_user,
        "chosen_flights": chosen_flights,
        "hotel_inventory": hotels_inventory,
        "restaurant_inventory": restaurants_inventory,
        "activities_inventory": activities_inventory,
        "hints": {"weather": season_hint, "safety": "Standard city precautions; watch bags in crowds."}
    }

# ============================
# Group assembly (no duplication of user data)
# ============================
def group_size():
    choices = list(range(1, 16))
    weights = [0.22, 0.28, 0.15, 0.10, 0.07, 0.05, 0.04, 0.03, 0.025, 0.02, 0.015, 0.01, 0.01, 0.01, 0.01]
    return random.choices(choices, weights=weights, k=1)[0]

def build_trip_context_from_mm(seed_mm):
    start, end, pref_len = future_date(10, 260)
    mp = seed_mm or {}

    meet_pref = mp.get("meeting_preference", "midpoint")
    if meet_pref == "midpoint":
        meeting_strategy_allowed = ["en_route_midpoint","at_destination","origin_A","origin_B"]
    elif meet_pref == "at_destination":
        meeting_strategy_allowed = ["at_destination","origin_A","origin_B"]
    else:
        meeting_strategy_allowed = ["origin_A","origin_B","at_destination"]

    cw = (mp.get("compatibility_weights") or {})
    objective = "minimize_total_time"
    if cw:
        if cw.get("budget", 0) >= max(cw.values() or [0]):
            objective = "minimize_total_cost"
        elif cw.get("pace", 0) + cw.get("cleanliness", 0) > 0.18:
            objective = "minimize_total_travel_time_and_cost"

    return {
        "title": "AI-Recommended Trip",
        "destinations": [],
        "date_window": {
            "earliest_departure": str(start),
            "latest_return": str(end),
            "preferred_trip_length_days": pref_len,
            "blackout_dates": []
        },
        "meeting_strategy_allowed": meeting_strategy_allowed,
        "meeting_priority_objective": objective,
        "itinerary_style": random.choice(["anchor_city","multi-stop"]),
        "min_time_per_stop_hours": random.choice([12, 24, 36, 48]),
        "luggage": {"carry_on_only": random.random() < 0.6, "special_gear": random.sample(["camera","skis","hiking poles","drone","none"], k=1)},
        "co2_preference": random.random() < 0.5,
        "tradeoff_weights": {"cost": round(random.random(),2), "time": round(random.random(),2), "comfort": round(random.random(),2), "scenery": round(random.random(),2), "co2": round(random.random(),2)},
        "hard_constraints": {
            "earliest_departure_time_local": random.choice(["07:00","08:30","09:00"]),
            "latest_arrival_time_local": random.choice(["20:00","21:30","22:00"]),
            "max_daily_travel_hours": random.choice([5,6,7]),
            "max_transfers": random.choice([1,2,3]),
            "room_setup": random.choice(["twin","double","2 rooms"])
        },
        "output_preferences": {"detail_level": "day-by-day","include_booking_links": True,"currency": "EUR","units": "metric","share_to_rovermitra_chat": True}
    }

# ============================
# Main
# ============================
def main():
    assert Path(USERS_PATH).exists(), f"Missing {USERS_PATH}"
    assert Path(MM_PATH).exists(), f"Missing {MM_PATH}"

    with open(USERS_PATH, "r", encoding="utf-8") as f:
        users = json.load(f)
    with open(MM_PATH, "r", encoding="utf-8") as f:
        mm_profiles = json.load(f)

    users_by_id = idx_by(users, "user_id")
    mm_by_uid = idx_by(mm_profiles, "user_id")

    user_ids = list(users_by_id.keys())
    groups = []

    for _ in range(N_GROUPS):
        size = group_size()
        if not user_ids:
            break
        chosen_ids = random.sample(user_ids, k=min(size, len(user_ids)))
        member_users = [users_by_id[uid] for uid in chosen_ids]
        member_mm = [mm_by_uid.get(uid) for uid in chosen_ids]

        # Trip context seeded from the first member's MM profile
        trip_ctx = build_trip_context_from_mm(member_mm[0] if member_mm else {})

        # Compose plan & inventories
        draft_plan = compose_draft_plan(member_users, member_mm, itinerary_style=trip_ctx["itinerary_style"])

        # Handle/IDs only (note: key is "contact" in core)
        members_ref = []
        for u in member_users:
            handle = ((u.get("contact") or {}).get("rovermitra_handle")) or f"rm_{u['user_id'][:6]}"
            members_ref.append({"user_id": u["user_id"], "rovermitra_handle": handle})

        group_doc = {
            "group_id": f"grp_{uuid.uuid4().hex[:10]}",
            "rovermitra_chat": {"room_id": f"rmr_{uuid.uuid4().hex[:10]}", "created_at": datetime.utcnow().isoformat() + "Z"},
            "members": members_ref,
            "trip_context": trip_ctx,
            "draft_plan": draft_plan
        }

        # Rich vs lean
        if random.random() > RICH_GROUP_RATIO:
            group_doc["trip_context"]["output_preferences"]["include_booking_links"] = False
            group_doc["trip_context"].pop("luggage", None)

        groups.append(group_doc)

    os.makedirs(os.path.dirname(OUT_PATH) or ".", exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(groups, f, indent=2, ensure_ascii=False)

    print(f"✅ Generated {len(groups)} integrated groups → {OUT_PATH}")
    print(json.dumps(groups[0], indent=2)[:1200] + "...\n")

if __name__ == "__main__":
    main()
