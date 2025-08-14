#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Car Rental data generator (integrated)
- Reads:
    users/data/users_core.json
    MatchMaker/data/matchmaker_profiles.json
    Flight/data/travel_groups_integrated_v3.json
- Writes:
    Rentals/data/carrental_catalog.json
    Rentals/data/carrental_availability.json
    Rentals/data/group_carrental_reservations.json

Design:
  * No user duplication: references by user_id and group_id only.
  * City/country realism, suppliers, vehicle classes, transmission, fuel type (incl. EV),
    mileage rules, deposits, insurance packs, add-ons (child seat, snow chains, GPS), and
    pick-up/return locations (airport vs. rail vs. downtown) aligned with group flight/itinerary.
  * Pricing & currency per city; CO2-friendly preference biases EV/hybrid when present.
  * Party size + luggage + trip style drive vehicle class choice.
"""

import os
import json
import uuid
import random
from pathlib import Path
from datetime import date, datetime, timedelta, time

random.seed(222)

# ------------------------------
# Paths (project-root aware)
# ------------------------------
BASE_DIR   = Path(__file__).resolve().parents[1]

USERS_PATH  = BASE_DIR / "users/data/users_core.json"
MM_PATH     = BASE_DIR / "MatchMaker/data/matchmaker_profiles.json"
GROUPS_PATH = BASE_DIR / "Flight/data/travel_groups_integrated_v3.json"

CATALOG_OUT = BASE_DIR / "Rentals/data/carrental_catalog.json"
AVAIL_OUT   = BASE_DIR / "Rentals/data/carrental_availability.json"
RESV_OUT    = BASE_DIR / "Rentals/data/group_carrental_reservations.json"

# ------------------------------
# Pools / taxonomies
# ------------------------------
SUPPLIERS = [
    "Hertz","Avis","Budget","Enterprise","Sixt","Europcar","Thrifty","Alamo","National","LocalCars"
]
VEHICLE_CLASSES = [
    # code, class, example models, seats, large bags (approx), min age
    ("MDMR","Mini","Fiat 500 / VW Up",4,1,18),
    ("EDMR","Economy","VW Polo / Ford Fiesta",5,1,21),
    ("CDMR","Compact","VW Golf / Toyota Corolla",5,2,21),
    ("IDMR","Intermediate","Skoda Octavia / Hyundai i30",5,3,23),
    ("SDMR","Standard","VW Passat / Mazda6",5,3,25),
    ("FWMR","Estate","VW Golf Variant / Skoda Fabia Combi",5,4,23),
    ("IFAR","Compact SUV","Nissan Qashqai / VW T-Roc",5,3,23),
    ("SFAR","SUV","Toyota RAV4 / VW Tiguan",5,4,25),
    ("PDAR","Premium","BMW 3-Series / Audi A4",5,3,27),
    ("LDAR","Luxury","BMW 5-Series / Mercedes E",5,4,27),
    ("FVMR","Minivan","VW Touran / Citroën Grand C4",7,4,25),
    ("PVAR","Passenger Van","VW Caravelle / Ford Transit",9,6,25)
]
TRANSMISSIONS = ["manual","automatic"]
FUELS = ["petrol","diesel","hybrid","ev"]
INSURANCE_PACKS = [
    {"code":"BASIC","desc":"CDW + TP, excess applies"},
    {"code":"PLUS","desc":"Lower excess + glass/tyres"},
    {"code":"PREMIUM","desc":"Zero excess, includes roadside"},
]
ADDONS = ["gps","child_seat","booster","additional_driver","snow_chains","ski_rack","wifi_hotspot"]
PAYMENT = ["visa","mastercard","amex","cash","apple-pay","google-pay"]
LOC_TYPES = ["airport","rail_station","downtown"]

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
def load_json(p: Path):
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else []

def idx_by(lst, key):
    return {x[key]: x for x in lst if isinstance(x, dict) and key in x}

def city_currency(city, location_data):
    for loc in location_data:
        if loc.get("city") == city:
            return CURRENCY_BY_COUNTRY.get(loc.get("country",""), "EUR")
    return "EUR"

def city_stub_latlon(city):
    r = random.Random(city)
    return round(r.uniform(-60,60),5), round(r.uniform(-120,120),5)

def derive_location_data(groups):
    """Collect (city,country) hints from hotel inventory & itinerary; lightweight and robust."""
    seen = {}
    for g in groups:
        dp = g.get("draft_plan") or {}
        for inv in (dp.get("hotel_inventory") or []):
            c = inv.get("city")
            if c: seen[c] = seen.get(c, {"city": c, "country": None})
        for day in (dp.get("itinerary") or []):
            c = day.get("base")
            if c: seen[c] = seen.get(c, {"city": c, "country": None})
    return list(seen.values())

def cheapest_flight_per_user(draft_plan, uid):
    """Return (outbound_iso, arrive_iso) for user's chosen cheapest outbound if present."""
    chosen = draft_plan.get("chosen_flights") or []
    fltmap = {x["user_id"]:x for x in chosen if "user_id" in x}
    sel = fltmap.get(uid)
    if not sel: return None, None
    offers = draft_plan.get("flight_offers") or {}
    uoffers = offers.get(uid) or {}
    ob = next((o for o in (uoffers.get("outbound") or []) if o.get("id")==sel.get("outbound")), None)
    ib = next((o for o in (uoffers.get("return") or []) if o.get("id")==sel.get("return")), None)
    return (ob.get("arrive_iso") if ob else None), (ib.get("depart_iso") if ib else None)

def price_per_day(vehicle_code, currency):
    # rough price ranges by class; returns a base that we’ll vary per availability slot
    table = {
        "MDMR": (18, 35), "EDMR": (22, 45), "CDMR": (28, 55),
        "IDMR": (35, 70), "SDMR": (45, 85), "FWMR": (42, 80),
        "IFAR": (48, 95), "SFAR": (55, 115), "PDAR": (70, 140),
        "LDAR": (95, 190), "FVMR": (65, 120), "PVAR": (85, 160)
    }
    lo, hi = table.get(vehicle_code, (30, 60))
    # non-EUR currencies sometimes run higher
    bump = 1.0
    if currency in ["CHF","GBP","USD","CAD","AUD","NZD","JPY","NOK","SEK","DKK"]: bump = 1.05
    return round(random.uniform(lo, hi) * bump, 2)

def co2_bias_to_ev(trip_ctx):
    return bool(trip_ctx.get("co2_preference"))

def group_vehicle_target(group, users_by_id):
    """Infer class target by party size + luggage + pace + road-trip-y interests."""
    members = [m.get("user_id") for m in (group.get("members") or []) if m.get("user_id")]
    size = max(1, len(members))
    # interests: if multiple cities or mountain interests → SUV/estate
    dp = group.get("draft_plan") or {}
    dests = {d.get("base") for d in (dp.get("itinerary") or []) if d.get("base")}
    multi_city = len(dests) >= 2
    # check any user must_haves (workspace not relevant)
    luggage_heavy = False
    for uid in members:
        # no luggage field in user; heuristics via travel_prefs.must_haves
        musts = ((users_by_id.get(uid) or {}).get("travel_prefs") or {}).get("must_haves") or []
        if "near_station" not in musts:  # weak signal
            luggage_heavy = luggage_heavy or random.random() < 0.4
    # pick class
    if size >= 7: return "FVMR" if size <= 7 else "PVAR"
    if size >= 5 and (luggage_heavy or multi_city): return random.choice(["FWMR","IFAR","SFAR"])
    if size == 4 and multi_city: return random.choice(["IDMR","IFAR","FWMR"])
    return random.choice(["EDMR","CDMR","IDMR"])

def pick_locations_for_city(city):
    lat, lon = city_stub_latlon(city)
    # synthesize 2–3 pickup spots
    spots = []
    for t in random.sample(LOC_TYPES, k=random.randint(2,3)):
        spots.append({
            "type": t,
            "name": f"{city} {t.replace('_',' ').title()} Rental Center",
            "approx_lat": lat + random.uniform(-0.02, 0.02),
            "approx_lon": lon + random.uniform(-0.02, 0.02)
        })
    return spots

# ------------------------------
# Catalog per city
# ------------------------------
def build_city_catalog(city, country_guess, currency):
    venues = []
    pickup_spots = pick_locations_for_city(city)
    # 4–8 supplier desks per city
    for supplier in random.sample(SUPPLIERS, k=random.randint(4, min(8,len(SUPPLIERS)))):
        stock = []
        # 6–12 vehicle classes stocked per supplier
        for code, klass, example, seats, bags, min_age in random.sample(VEHICLE_CLASSES, k=random.randint(6, 12)):
            # include some EV/hybrid in most suppliers
            fuels = list(FUELS)
            if random.random() < 0.4:
                fuels = list({*fuels, "ev"})
            stock.append({
                "vehicle_code": code,
                "class_name": klass,
                "example_models": example,
                "seats": seats,
                "large_bags": bags,
                "transmission": random.choice(TRANSMISSIONS if code not in ["LDAR","PDAR"] else ["automatic"]),
                "fuel_type": random.choice(fuels),
                "min_driver_age": min_age,
                "mileage": random.choice(["unlimited","200km/day","300km/day"]),
                "deposit": round(random.uniform(200, 1500), 2),
                "included_insurance": random.choice(INSURANCE_PACKS),
                "addons_available": sorted(random.sample(ADDONS, k=random.randint(2, 6))),
                "payment_methods": sorted(random.sample(PAYMENT, k=random.randint(2, 5))),
                "base_price_per_day": price_per_day(code, currency),
            })
        venues.append({
            "id": f"rent_{uuid.uuid4().hex[:10]}",
            "city": city,
            "country": country_guess,
            "supplier": supplier,
            "pickup_locations": pickup_spots,
            "currency": currency,
            "partner": random.random() < 0.45,  # instant confirm
            "fleet": stock
        })
    return venues

# ------------------------------
# Availability
# ------------------------------
AVAIL_DAYS = 120

def gen_availability_for_supplier(supplier, currency):
    today = date.today()
    slots = []
    for d in range(AVAIL_DAYS):
        day = today + timedelta(days=d)
        # prices vary by day-of-week and seasonality hint
        dow = day.weekday()  # 0=Mon
        demand_bump = 1.15 if dow in (4,5) else 1.0
        for f in supplier["fleet"]:
            price = round(f["base_price_per_day"] * demand_bump * random.uniform(0.9, 1.15), 2)
            qty = random.randint(0, 14)  # daily availability per class
            slots.append({
                "supplier_id": supplier["id"],
                "vehicle_code": f["vehicle_code"],
                "date": str(day),
                "price_per_day": price,
                "currency": currency,
                "qty_left": qty
            })
    return slots

# ------------------------------
# Group reservations (AI holds)
# ------------------------------
def avg_group_age(users_by_id, member_ids):
    vals = []
    for uid in member_ids:
        u = users_by_id.get(uid) or {}
        a = u.get("age")
        if isinstance(a, int): vals.append(a)
    return sum(vals)/len(vals) if vals else 29

def license_ok_prob(age):
    # heuristic only; no license field in core data
    if age < 21: return 0.3
    if age < 23: return 0.6
    if age < 25: return 0.8
    return 0.92

def party_driver_candidates(users_by_id, member_ids):
    cands = []
    for uid in member_ids:
        age = (users_by_id.get(uid) or {}).get("age", 28)
        if random.random() < license_ok_prob(age):
            cands.append(uid)
    return cands

def pick_pickup_drop_times(group):
    """Try to align pickup near first flight arrival / first itinerary day, and drop near last day."""
    dp = group.get("draft_plan") or {}
    members = [m.get("user_id") for m in (group.get("members") or []) if m.get("user_id")]
    arrive_isos = []
    depart_isos = []
    for uid in members:
        ob_arr, ib_dep = cheapest_flight_per_user(dp, uid)
        if ob_arr: arrive_isos.append(ob_arr)
        if ib_dep: depart_isos.append(ib_dep)
    if arrive_isos:
        first_arr = min(datetime.fromisoformat(x) for x in arrive_isos)
        pickup_time = first_arr + timedelta(minutes=random.choice([30, 45, 60, 75]))
    else:
        # fallback: 10:00 on first itinerary date
        first_day = next((d.get("date") for d in (dp.get("itinerary") or [])), None)
        pickup_time = datetime.fromisoformat(first_day + "T10:00:00") if first_day else datetime.now() + timedelta(days=7)

    if depart_isos:
        last_dep = max(datetime.fromisoformat(x) for x in depart_isos)
        drop_time = last_dep - timedelta(hours=random.choice([3, 4, 5]))
    else:
        last_day = None
        for d in (dp.get("itinerary") or []):
            last_day = d.get("date")
        drop_time = datetime.fromisoformat(last_day + "T17:00:00") if last_day else pickup_time + timedelta(days=3)

    return pickup_time, drop_time

def should_offer_car(group):
    """Offer logic: multi-city or mountain/outskirts activities get car; small city single-stop less likely."""
    dp = group.get("draft_plan") or {}
    itinerary = dp.get("itinerary") or []
    cities = [d.get("base") for d in itinerary if d.get("base")]
    multi = len(set(cities)) >= 2
    # heuristics via plan text
    text = " ".join([d.get("plan","").lower() for d in itinerary])
    scenic = any(k in text for k in ["day trip","mountain","excursion","village","scenic","lake"])
    return multi or scenic or random.random() < 0.35

def best_supplier_for_city(catalog, want_ev=False, want_code=None, party_size=2):
    best = None; best_score = -1.0
    for s in catalog:
        score = 0.0
        if want_ev:
            if any(f.get("fuel_type")=="ev" for f in s["fleet"]): score += 1.0
        if want_code:
            if any(f.get("vehicle_code")==want_code for f in s["fleet"]): score += 1.0
        # partner gets slight boost
        if s.get("partner"): score += 0.3
        # availability proxy: fleet size
        score += len(s["fleet"]) * 0.01
        if score > best_score:
            best = s; best_score = score
    return best

# ------------------------------
# Main
# ------------------------------
def main():
    assert USERS_PATH.exists(), f"Missing {USERS_PATH}"
    assert GROUPS_PATH.exists(), f"Missing {GROUPS_PATH}"

    users = load_json(USERS_PATH)
    groups = load_json(GROUPS_PATH)
    mm_profiles = load_json(MM_PATH)  # reserved for future filters (e.g., response expectations)

    users_by_id = idx_by(users, "user_id")

    # 1) Build city catalogs (suppliers + fleet) inferred from itinerary/hotels
    locs = derive_location_data(groups)
    city_catalog_map = {}
    for loc in locs:
        city = loc["city"]; country = loc.get("country") or ""
        currency = CURRENCY_BY_COUNTRY.get(country, "EUR")
        city_catalog_map[city] = build_city_catalog(city, country, currency)

    # 2) Availability per supplier/fleet per day
    availability = []
    for city, suppliers in city_catalog_map.items():
        for sup in suppliers:
            availability.extend(gen_availability_for_supplier(sup, sup["currency"]))

    # 3) Group-level AI holds (references only group_id + candidate driver user_ids)
    reservations = []
    for g in groups:
        if not should_offer_car(g):
            continue

        gid = g.get("group_id")
        dp = g.get("draft_plan") or {}
        meet_city = (dp.get("meeting_plan") or {}).get("meet_at") or (dp.get("itinerary") or [{}])[0].get("base")
        if not meet_city or meet_city not in city_catalog_map:
            continue
        catalog = city_catalog_map[meet_city]
        trip_ctx = g.get("trip_context") or {}

        # timing aligned with flights/itinerary
        pickup_time, drop_time = pick_pickup_drop_times(g)

        # party and driver candidates
        member_ids = [m.get("user_id") for m in (g.get("members") or []) if m.get("user_id")]
        drivers = party_driver_candidates(users_by_id, member_ids)
        if not drivers:
            # skip if no plausible licensed driver
            continue

        # vehicle target + EV bias
        want_code = group_vehicle_target(g, users_by_id)
        want_ev = co2_bias_to_ev(trip_ctx)
        supplier = best_supplier_for_city(catalog, want_ev=want_ev, want_code=want_code, party_size=len(member_ids))
        if not supplier:
            continue

        # choose a vehicle from supplier close to want_code
        fleet = supplier["fleet"]
        # try exact vehicle code, else closest by seats/bags
        pick = next((f for f in fleet if f["vehicle_code"] == want_code), None)
        if not pick:
            # rough: pick by seats >= party size then by bags
            viable = [f for f in fleet if f.get("seats",5) >= max(2, len(member_ids))]
            pick = viable[0] if viable else random.choice(fleet)

        # price estimate by availability table for pickup date
        avail_for_day = [a for a in availability
                         if a["supplier_id"] == supplier["id"]
                         and a["vehicle_code"] == pick["vehicle_code"]
                         and a["date"] == pickup_time.date().isoformat()]
        if avail_for_day:
            day_price = min(avail_for_day, key=lambda x: x["price_per_day"])["price_per_day"]
        else:
            day_price = pick["base_price_per_day"]

        # length in days (min 1)
        days = max(1, (drop_time.date() - pickup_time.date()).days)
        est_total = round(day_price * days, 2)

        resv = {
            "reservation_id": f"cr_{uuid.uuid4().hex[:12]}",
            "group_id": gid,
            "city": meet_city,
            "supplier_id": supplier["id"],
            "vehicle_code": pick["vehicle_code"],
            "class_name": pick["class_name"],
            "seats": pick["seats"],
            "large_bags": pick["large_bags"],
            "transmission": pick["transmission"],
            "fuel_type": pick["fuel_type"],
            "insurance_pack": pick["included_insurance"]["code"],
            "addons_requested": sorted(random.sample(pick["addons_available"], k=min(2, len(pick["addons_available"])))),
            "pickup": {
                "when_iso": pickup_time.isoformat(),
                "location_hint": random.choice(supplier["pickup_locations"])["name"],
                "location_type": random.choice([pl["type"] for pl in supplier["pickup_locations"]]),
            },
            "dropoff": {
                "when_iso": (pickup_time + timedelta(days=days, hours=random.choice([0, 1]))).isoformat(),
                "location_hint": random.choice(supplier["pickup_locations"])["name"],
                "location_type": random.choice([pl["type"] for pl in supplier["pickup_locations"]]),
            },
            "payment": {
                "currency": supplier["currency"],
                "price_per_day": day_price,
                "days": days,
                "estimated_total": est_total,
                "deposit": pick["deposit"],
                "payment_methods": pick["payment_methods"]
            },
            "drivers_user_ids": sorted(random.sample(drivers, k=min(len(drivers), random.choice([1,1,2])))),
            "status": "hold" if not supplier.get("partner") else random.choice(["hold","confirmed"]),
            "matching_rationale": {
                "party_size": len(member_ids),
                "target_vehicle_code": want_code,
                "co2_ev_bias": want_ev,
                "supplier_partner": supplier.get("partner"),
                "reason": "Multi-city / scenic itinerary benefits from flexibility" if should_offer_car(g) else "Optional convenience"
            }
        }
        reservations.append(resv)

    # --------------------------
    # Write outputs
    # --------------------------
    CATALOG_OUT.parent.mkdir(parents=True, exist_ok=True)
    AVAIL_OUT.parent.mkdir(parents=True, exist_ok=True)
    RESV_OUT.parent.mkdir(parents=True, exist_ok=True)

    CATALOG_OUT.write_text(json.dumps(
        [{"city": c, "suppliers": v} for c, v in sorted(city_catalog_map.items())],
        indent=2, ensure_ascii=False
    ), encoding="utf-8")
    AVAIL_OUT.write_text(json.dumps(availability, indent=2, ensure_ascii=False), encoding="utf-8")
    RESV_OUT.write_text(json.dumps(reservations, indent=2, ensure_ascii=False), encoding="utf-8")

    # quick peek
    print(f"✅ Cities with suppliers: {len(city_catalog_map)} → {CATALOG_OUT}")
    print(f"✅ Availability rows: {len(availability)} → {AVAIL_OUT}")
    print(f"✅ Group car holds: {len(reservations)} → {RESV_OUT}")
    # sample peeks
    first_city = next(iter(city_catalog_map.keys()), None)
    if first_city:
        print(json.dumps(city_catalog_map[first_city][:1], indent=2)[:900] + " ...")
    if reservations:
        print(json.dumps(reservations[0], indent=2)[:700] + " ...")

if __name__ == "__main__":
    main()
