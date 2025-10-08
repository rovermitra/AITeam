#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Car Rental data generator (integrated, smarter v2)
- Reads:
    users/data/users_core.json
    MatchMaker/data/matchmaker_profiles.json
    Flight/data/travel_groups_integrated_v3.json
- Writes:
    Rentals/data/carrental_catalog.json
    Rentals/data/carrental_availability.json
    Rentals/data/group_carrental_reservations.json

Notes:
  * No user/profile duplication; references by user_id/group_id only.
  * Currency per city inferred from group hotel inventory (mode), else EUR.
  * Pickup/return location chosen by real group context (airport if flights, rail if intercity trains).
  * EV/hybrid bias if trip_context.co2_preference=True, with fallback if stock unavailable.
  * Add-ons from season/itinerary keywords (snow chains, ski rack, child seat, GPS).
  * Region-aware transmissions (US/CA/AU/NZ ≈ automatic; most of Europe allows manual).
  * Young-driver fee & one-way fee modeled; insurance packs and deposits realistic.
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
    ("FWMR","Estate","VW Golf Variant / Skoda Octavia Combi",5,4,23),
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

HIGH_AUTO_COUNTRIES = {"USA","Canada","Australia","New Zealand","UAE","Saudi Arabia","Qatar","Japan"}  # bias to automatic

# ------------------------------
# Helpers
# ------------------------------
def load_json(p: Path):
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else []

def idx_by(lst, key):
    return {x[key]: x for x in lst if isinstance(x, dict) and key in x}

def city_stub_latlon(city):
    r = random.Random("geo-"+city)
    return round(r.uniform(-60,60),5), round(r.uniform(-120,120),5)

def month_of(dt: datetime) -> int:
    try:
        return dt.month
    except Exception:
        return date.today().month

def itinerary_keywords(itinerary):
    text = " ".join((d.get("plan","") or "").lower() for d in itinerary)
    keys = set()
    for k in ["mountain","ski","snow","village","scenic","lake","day trip","child","family","hike","winter","summer"]:
        if k in text: keys.add(k)
    return keys

def infer_city_currencies(groups):
    """
    Infer per-city currency from hotel inventory option currencies (mode).
    Fallback to EUR if unknown.
    Returns dict: city -> currency
    """
    result = {}
    for g in groups:
        dp = g.get("draft_plan") or {}
        for inv in (dp.get("hotel_inventory") or []):
            city = inv.get("city")
            votes = {}
            for opt in (inv.get("options") or []):
                cur = opt.get("currency")
                if cur:
                    votes[cur] = votes.get(cur, 0) + 1
            if city:
                if votes:
                    result[city] = max(votes.items(), key=lambda x:x[1])[0]
                else:
                    result.setdefault(city, "EUR")
        for d in (dp.get("itinerary") or []):
            c = d.get("base")
            if c and c not in result:
                result[c] = "EUR"
    return result

def flights_touch_airport(dp):
    """Return True if any chosen flight exists (airport context likely)."""
    ch = dp.get("chosen_flights") or []
    return len(ch) > 0

def pick_locations_for_city(city):
    lat, lon = city_stub_latlon(city)
    spots = []
    for t in random.sample(LOC_TYPES, k=random.randint(2,3)):
        spots.append({
            "type": t,
            "name": f"{city} {t.replace('_',' ').title()} Rental Center",
            "approx_lat": lat + random.uniform(-0.02, 0.02),
            "approx_lon": lon + random.uniform(-0.02, 0.02)
        })
    return spots

def regional_transmission_bias(country_guess, default="manual"):
    if country_guess in HIGH_AUTO_COUNTRIES:
        return "automatic"
    return default

def price_per_day(vehicle_code, currency):
    table = {
        "MDMR": (18, 35), "EDMR": (22, 45), "CDMR": (28, 55),
        "IDMR": (35, 70), "SDMR": (45, 85), "FWMR": (42, 80),
        "IFAR": (48, 95), "SFAR": (55, 115), "PDAR": (70, 140),
        "LDAR": (95, 190), "FVMR": (65, 120), "PVAR": (85, 160)
    }
    lo, hi = table.get(vehicle_code, (30, 60))
    bump = 1.0
    if currency in ["CHF","GBP","USD","CAD","AUD","NZD","JPY","NOK","SEK","DKK"]:
        bump = 1.05
    return round(random.uniform(lo, hi) * bump, 2)

def license_ok_prob(age):
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

def cheapest_flight_per_user(draft_plan, uid):
    chosen = draft_plan.get("chosen_flights") or []
    fltmap = {x["user_id"]:x for x in chosen if "user_id" in x}
    sel = fltmap.get(uid)
    if not sel: return None, None
    offers = draft_plan.get("flight_offers") or {}
    uoffers = offers.get(uid) or {}
    ob = next((o for o in (uoffers.get("outbound") or []) if o.get("id")==sel.get("outbound")), None)
    ib = next((o for o in (uoffers.get("return") or []) if o.get("id")==sel.get("return")), None)
    return (ob.get("arrive_iso") if ob else None), (ib.get("depart_iso") if ib else None)

def pick_pickup_drop_times(group):
    dp = group.get("draft_plan") or {}
    members = [m.get("user_id") for m in (group.get("members") or []) if m.get("user_id")]
    arrive_isos, depart_isos = [], []
    for uid in members:
        ob_arr, ib_dep = cheapest_flight_per_user(dp, uid)
        if ob_arr: arrive_isos.append(ob_arr)
        if ib_dep: depart_isos.append(ib_dep)
    if arrive_isos:
        first_arr = min(datetime.fromisoformat(x) for x in arrive_isos)
        pickup_time = first_arr + timedelta(minutes=random.choice([30, 45, 60, 75]))
    else:
        first_day = next((d.get("date") for d in (dp.get("itinerary") or [])), None)
        pickup_time = datetime.fromisoformat(first_day + "T10:00:00") if first_day else datetime.now() + timedelta(days=7)
    if depart_isos:
        last_dep = max(datetime.fromisoformat(x) for x in depart_isos)
        drop_time = last_dep - timedelta(hours=random.choice([3, 4, 5]))
    else:
        last_day = None
        for d in (dp.get("itinerary") or []): last_day = d.get("date")
        drop_time = datetime.fromisoformat(last_day + "T17:00:00") if last_day else pickup_time + timedelta(days=3)
    return pickup_time, drop_time

def should_offer_car(group):
    dp = group.get("draft_plan") or {}
    itinerary = dp.get("itinerary") or []
    cities = [d.get("base") for d in itinerary if d.get("base")]
    multi = len(set(cities)) >= 2
    text = " ".join([d.get("plan","").lower() for d in itinerary])
    scenic = any(k in text for k in ["day trip","mountain","excursion","village","scenic","lake","ski"])
    return multi or scenic or random.random() < 0.35

def group_vehicle_target(group, users_by_id):
    members = [m.get("user_id") for m in (group.get("members") or []) if m.get("user_id")]
    size = max(1, len(members))
    dp = group.get("draft_plan") or {}
    dests = {d.get("base") for d in (dp.get("itinerary") or []) if d.get("base")}
    multi_city = len(dests) >= 2
    if size >= 8: return "PVAR"
    if size == 7: return "FVMR"
    if size >= 5 and multi_city: return random.choice(["FWMR","IFAR","SFAR"])
    if size == 4 and multi_city: return random.choice(["IDMR","IFAR","FWMR"])
    return random.choice(["EDMR","CDMR","IDMR"])

def best_supplier_for_city(catalog, want_ev=False, want_code=None):
    best, best_score = None, -1.0
    for s in catalog:
        score = 0.0
        if want_ev and any(f.get("fuel_type")=="ev" for f in s["fleet"]): score += 1.0
        if want_code and any(f.get("vehicle_code")==want_code for f in s["fleet"]): score += 1.0
        if s.get("partner"): score += 0.3
        score += len(s["fleet"]) * 0.01
        if score > best_score: best, best_score = s, score
    return best

# ------------------------------
# Catalog per city
# ------------------------------
def build_city_catalog(city, country_guess, currency):
    venues = []
    pickup_spots = pick_locations_for_city(city)
    auto_bias = regional_transmission_bias(country_guess or "")
    for supplier in random.sample(SUPPLIERS, k=random.randint(4, min(8,len(SUPPLIERS)))):
        stock = []
        for code, klass, example, seats, bags, min_age in random.sample(VEHICLE_CLASSES, k=random.randint(6, 12)):
            fuels = list(FUELS)
            if random.random() < 0.45:
                fuels = list({*fuels, "ev"})
            # region transmission bias
            trans = "automatic" if (auto_bias=="automatic" or code in {"LDAR","PDAR","PVAR"}) else random.choice(TRANSMISSIONS)
            stock.append({
                "vehicle_code": code,
                "class_name": klass,
                "example_models": example,
                "seats": seats,
                "large_bags": bags,
                "transmission": trans,
                "fuel_type": random.choice(fuels),
                "min_driver_age": min_age,
                "mileage": random.choice(["unlimited","200km/day","300km/day"]),
                "deposit": round(random.uniform(300, 1600), 2),
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
            "partner": random.random() < 0.5,  # instant confirm
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
        dow = day.weekday()  # Fri/Sat higher
        demand_bump = 1.15 if dow in (4,5) else 1.0
        for f in supplier["fleet"]:
            price = round(f["base_price_per_day"] * demand_bump * random.uniform(0.9, 1.16), 2)
            qty = random.randint(0, 14)
            # young-driver surcharge advertised hint
            ydf = 0
            if f["min_driver_age"] >= 25:
                ydf = random.choice([0,0,5,10])  # per-day
            slots.append({
                "supplier_id": supplier["id"],
                "vehicle_code": f["vehicle_code"],
                "date": str(day),
                "price_per_day": price,
                "currency": currency,
                "qty_left": qty,
                "fees": {
                    "young_driver_per_day": ydf,
                    "one_way_fee": random.choice([0,0,0,25,40,60])  # only assessed when different return location
                }
            })
    return slots

# ------------------------------
# Main
# ------------------------------
def main():
    assert USERS_PATH.exists(), f"Missing {USERS_PATH}"
    assert GROUPS_PATH.exists(), f"Missing {GROUPS_PATH}"

    users = load_json(USERS_PATH)
    groups = load_json(GROUPS_PATH)
    _mm = load_json(MM_PATH)  # reserved for future use

    users_by_id = idx_by(users, "user_id")

    # 1) Infer per-city currencies and rough country guesses from strings in hotel names (best effort not required here)
    city_currency_map = infer_city_currencies(groups)  # city -> currency
    # naive country guess map (optional; improves transmission bias). If not known, leave None.
    city_country_guess = {c: ( "Switzerland" if c in {"Zurich","Geneva","Basel","Bern","Lucerne","Interlaken","Zermatt"} else None )
                          for c in city_currency_map.keys()}

    # 2) Build city catalogs
    city_catalog_map = {}
    for city, currency in city_currency_map.items():
        city_catalog_map[city] = build_city_catalog(city, city_country_guess.get(city), currency or "EUR")

    # 3) Availability per supplier/fleet per day
    availability = []
    for city, suppliers in city_catalog_map.items():
        for sup in suppliers:
            availability.extend(gen_availability_for_supplier(sup, sup["currency"]))

    # 4) Group-level AI holds (references only)
    reservations = []
    for g in groups:
        if not should_offer_car(g):
            continue

        gid = g.get("group_id")
        dp = g.get("draft_plan") or {}
        members = [m.get("user_id") for m in (g.get("members") or []) if m.get("user_id")]
        if not members:
            continue

        meet_city = (dp.get("meeting_plan") or {}).get("meet_at") or (dp.get("itinerary") or [{}])[0].get("base")
        if not meet_city or meet_city not in city_catalog_map:
            continue
        catalog = city_catalog_map[meet_city]
        trip_ctx = g.get("trip_context") or {}

        pickup_time, drop_time = pick_pickup_drop_times(g)
        drivers = party_driver_candidates(users_by_id, members)
        if not drivers:
            continue

        want_code = group_vehicle_target(g, users_by_id)
        want_ev = bool(trip_ctx.get("co2_preference"))
        supplier = best_supplier_for_city(catalog, want_ev=want_ev, want_code=want_code)
        if not supplier:
            continue

        # choose vehicle
        fleet = supplier["fleet"]
        pick = next((f for f in fleet if f["vehicle_code"] == want_code), None)
        if not pick:
            viable = [f for f in fleet if f.get("seats",5) >= max(2, len(members))]
            if want_ev:
                ev_viable = [f for f in viable if f.get("fuel_type")=="ev"]
                pick = ev_viable[0] if ev_viable else (viable[0] if viable else random.choice(fleet))
            else:
                pick = viable[0] if viable else random.choice(fleet)

        # availability lookup for price
        avail_for_day = [a for a in availability
                         if a["supplier_id"] == supplier["id"]
                         and a["vehicle_code"] == pick["vehicle_code"]
                         and a["date"] == pickup_time.date().isoformat()]
        if avail_for_day:
            day_row = min(avail_for_day, key=lambda x: x["price_per_day"])
            day_price = day_row["price_per_day"]
            fees_template = day_row.get("fees", {"young_driver_per_day":0,"one_way_fee":0})
        else:
            day_price = pick["base_price_per_day"]
            fees_template = {"young_driver_per_day": 0, "one_way_fee": 0}

        days = max(1, (drop_time.date() - pickup_time.date()).days)
        # one-way decision (30% if multi-stop)
        is_one_way = random.random() < (0.30 if len(set(d.get("base") for d in (dp.get("itinerary") or []) if d.get("base"))) > 1 else 0.05)
        one_way_fee = fees_template.get("one_way_fee", 0) if is_one_way else 0

        # add-ons from itinerary & season
        keys = itinerary_keywords(dp.get("itinerary") or [])
        month = month_of(pickup_time)
        addons_req = set()
        if "mountain" in keys or "ski" in keys or month in {12,1,2,3}:
            if "snow_chains" in pick["addons_available"]: addons_req.add("snow_chains")
            if "ski_rack" in pick["addons_available"] and "ski" in keys: addons_req.add("ski_rack")
        if "family" in " ".join((users_by_id.get(uid,{}).get("values") or []) for uid in members):
            if "child_seat" in pick["addons_available"]: addons_req.add("child_seat")
        if "gps" in pick["addons_available"]: addons_req.add("gps")
        # second driver if group >=4
        if len(members) >= 4 and "additional_driver" in pick["addons_available"]:
            addons_req.add("additional_driver")

        # pickup/return location preference (airport if flights chosen)
        pickup_loc = random.choice([pl for pl in supplier["pickup_locations"]
                                    if pl["type"]=="airport"] or supplier["pickup_locations"]) \
                     if flights_touch_airport(dp) else \
                     random.choice([pl for pl in supplier["pickup_locations"]
                                    if pl["type"] in {"rail_station","downtown"}] or supplier["pickup_locations"])
        drop_loc = pickup_loc if not is_one_way else random.choice(supplier["pickup_locations"])

        # young-driver fee if all drivers under threshold for class
        min_age = pick.get("min_driver_age", 21)
        driver_ages = [(users_by_id.get(uid) or {}).get("age", 28) for uid in drivers]
        ydf_applies = all(a < min_age for a in driver_ages)
        ydf_total = fees_template.get("young_driver_per_day", 0) * days if ydf_applies else 0

        est_total = round(day_price * days + one_way_fee + ydf_total, 2)

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
            "addons_requested": sorted(addons_req),
            "pickup": {
                "when_iso": pickup_time.isoformat(),
                "location_hint": pickup_loc["name"],
                "location_type": pickup_loc["type"],
            },
            "dropoff": {
                "when_iso": (pickup_time + timedelta(days=days, hours=random.choice([0, 1]))).isoformat(),
                "location_hint": drop_loc["name"],
                "location_type": drop_loc["type"],
                "one_way": is_one_way
            },
            "payment": {
                "currency": supplier["currency"],
                "price_per_day": day_price,
                "days": days,
                "estimated_total": est_total,
                "deposit": pick["deposit"],
                "payment_methods": pick["payment_methods"],
                "fees": {
                    "young_driver_total": ydf_total,
                    "one_way_fee": one_way_fee
                }
            },
            "drivers_user_ids": sorted(random.sample(drivers, k=min(len(drivers), random.choice([1,1,2])))),
            "status": "hold" if not supplier.get("partner") else random.choice(["hold","confirmed"]),
            "matching_rationale": {
                "party_size": len(members),
                "target_vehicle_code": want_code,
                "co2_ev_bias": want_ev,
                "supplier_partner": supplier.get("partner"),
                "airport_pickup": flights_touch_airport(dp),
                "itinerary_signals": sorted(list(keys))
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
    first_city = next(iter(city_catalog_map.keys()), None)
    if first_city:
        print(json.dumps(city_catalog_map[first_city][:1], indent=2)[:900] + " ...")
    if reservations:
        print(json.dumps(reservations[0], indent=2)[:700] + " ...")

if __name__ == "__main__":
    main()
