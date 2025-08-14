#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Activities, Attractions & Events generator (integrated, VIP)
Reads:
  users/data/users_core.json
  MatchMaker/data/matchmaker_profiles.json
  Flight/data/travel_groups_integrated_v3.json
Writes:
  Activities/data/attractions_catalog.json
  Activities/data/availability_slots.json
  Activities/data/group_activity_holds.json
  Events/data/events_catalog.json
  Events/data/group_event_holds.json

Design:
  * No user/profile duplication; references by user_id, group_id only.
  * City-&-date realistic attractions with pricing, duration, capacities, ops policy, geo, languages, accessibility.
  * Availability windows + group-level “holds” ranked by interests/pace/budget/risk/accessibility/language.
  * Events/festivals with calendar rules; future-proofed with provider placeholders (for later real-API swap).
"""

import json, uuid, random
from pathlib import Path
from datetime import date, datetime, timedelta, time

random.seed(212)

# --------------------------------------------------
# Project paths (root-anchored)
# --------------------------------------------------
BASE_DIR   = Path(__file__).resolve().parents[1]

USERS_PATH  = BASE_DIR / "users/data/users_core.json"
MM_PATH     = BASE_DIR / "MatchMaker/data/matchmaker_profiles.json"
GROUPS_PATH = BASE_DIR / "Flight/data/travel_groups_integrated_v3.json"

ATTRACTIONS_OUT   = BASE_DIR / "Activities/data/attractions_catalog.json"
AVAIL_SLOTS_OUT   = BASE_DIR / "Activities/data/availability_slots.json"
ACTIVITY_HOLDS_OUT= BASE_DIR / "Activities/data/group_activity_holds.json"

EVENTS_OUT        = BASE_DIR / "Events/data/events_catalog.json"
EVENT_HOLDS_OUT   = BASE_DIR / "Events/data/group_event_holds.json"

# --------------------------------------------------
# Tunables
# --------------------------------------------------
CATALOG_PER_CITY_RANGE = (28, 55)  # attractions per city
AVAILABILITY_DAYS      = 180       # horizon for bookable slots
SLOT_STEP_MIN          = 30
CITY_WIDE_PASS_SHARE   = 0.08      # small share of “city pass / combo” items

# --------------------------------------------------
# Taxonomies / pools
# --------------------------------------------------
CATEGORIES = [
    "museum","landmark","viewpoint","castle","old-town-walk","river-cruise","lake-cruise",
    "scenic-train","day-trip","food-tour","market","workshop","winery","brewery",
    "hike-easy","hike-moderate","bike-tour","ski","spa-thermal","theme-park","zoo-aquarium"
]
TAGS = [
    "skip-the-line","family-friendly","instagrammable","audio-guide","guided","free-entry",
    "city-pass-included","evening","sunrise","sunset","wheelchair-access","quiet-option",
    "indoors","outdoors","weather-dependent","pet-friendly","photography","local-expert",
    "small-group","private","romantic","history","architecture","nature","wildlife"
]
LANGS = ["en","de","fr","it","es","pt","nl","ja","ko","zh","ru"]
ACCESS = ["none","wheelchair","reduced-mobility","elevator-required"]
RISK = ["low","medium","high"]
PACE = ["relaxed","balanced","packed"]
WEATHER = ["any","good-weather","clear-sky","no-rain"]
PROVIDERS = [
  {"name":"Viator","type":"ota","id_field":"product_code"},
  {"name":"GetYourGuide","type":"ota","id_field":"gyg_id"},
  {"name":"Klook","type":"ota","id_field":"klook_id"},
  {"name":"LocalDMO","type":"dmo","id_field":"listing_id"},
]

CURRENCY_BY_COUNTRY = {
    "USA":"USD","Canada":"CAD","Mexico":"MXN","Brazil":"BRL","Argentina":"ARS","Chile":"CLP","Peru":"PEN","Colombia":"COP",
    "UK":"GBP","Ireland":"EUR","Germany":"EUR","France":"EUR","Spain":"EUR","Italy":"EUR","Portugal":"EUR",
    "Netherlands":"EUR","Belgium":"EUR","Switzerland":"CHF","Austria":"EUR","Poland":"PLN","Czechia":"CZK",
    "Denmark":"DKK","Sweden":"SEK","Norway":"NOK","Finland":"EUR","Greece":"EUR","Turkey":"TRY",
    "Japan":"JPY","South Korea":"KRW","China":"CNY","India":"INR","Pakistan":"PKR","Thailand":"THB","Vietnam":"VND","Malaysia":"MYR",
    "Singapore":"SGD","Indonesia":"IDR","Philippines":"PHP","UAE":"AED","Saudi Arabia":"SAR","Qatar":"QAR","Israel":"ILS","Jordan":"JOD",
    "Morocco":"MAD","Egypt":"EGP","South Africa":"ZAR","Kenya":"KES","Australia":"AUD","New Zealand":"NZD","Iceland":"ISK"
}

# light cuisine/interest → category hints
INTEREST_TO_CATS = {
  "mountain hiking": ["hike-easy","hike-moderate","viewpoint","scenic-train"],
  "short hikes": ["hike-easy","viewpoint"],
  "long hikes": ["hike-moderate","viewpoint","day-trip"],
  "city photography": ["viewpoint","old-town-walk","landmark","river-cruise","market"],
  "architecture walks": ["old-town-walk","landmark","castle","museum"],
  "history sites": ["museum","castle","old-town-walk","landmark"],
  "museum hopping": ["museum","landmark"],
  "scenic trains": ["scenic-train","viewpoint","day-trip"],
  "thermal baths": ["spa-thermal","wellness"],
  "vineyards": ["winery","day-trip"],
  "food tours": ["food-tour","market","workshop"],
  "street food": ["food-tour","market"],
  "cycling": ["bike-tour","day-trip"],
  "yoga": ["spa-thermal","wellness"],
  "wildlife watching": ["zoo-aquarium","day-trip","nature"],
  "festivals": ["event"],
  "beach days": ["day-trip"],
  "lake swims": ["lake-cruise","day-trip"],
  "rooftop views": ["viewpoint","landmark"],
  "local crafts": ["workshop","market"]
}

# Signature event templates (recurring; we generate plausible occurrences around trip windows)
EVENT_TEMPLATES = [
  {"slug":"summer-music-fest","title":"Summer Music Fest","category":"festival","month_window":[6,7,8],"days":3,"tags":["music","outdoors","evening"],"risk":"low"},
  {"slug":"winter-markets","title":"Winter Markets","category":"market-festival","month_window":[11,12],"days":15,"tags":["market","seasonal","food"],"risk":"low"},
  {"slug":"film-week","title":"City Film Week","category":"culture","month_window":[9,10],"days":7,"tags":["cinema","culture"],"risk":"low"},
  {"slug":"light-festival","title":"Festival of Lights","category":"culture","month_window":[1,2,3,10,11],"days":10,"tags":["night","photography"],"risk":"low"},
  {"slug":"marathon","title":"City Marathon","category":"sports","month_window":[4,5,9,10],"days":2,"tags":["sports","outdoors"],"risk":"medium"},
]

# --------------------------------------------------
# Utils
# --------------------------------------------------
def load_json(p: Path):
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else []

def idx_by(lst, key):
    return {x[key]: x for x in lst if isinstance(x, dict) and key in x}

def derive_location_hints(groups):
    """Collect (city, country_guess) from hotel_inventory & itinerary."""
    seen = {}
    for g in groups:
        dp = g.get("draft_plan") or {}
        for inv in (dp.get("hotel_inventory") or []):
            c = inv.get("city")
            if c:
                seen[c] = seen.get(c, {"city": c, "country": None})
        for day in (dp.get("itinerary") or []):
            c = day.get("base")
            if c:
                seen[c] = seen.get(c, {"city": c, "country": None})
    return list(seen.values())

def city_currency(city, location_hints):
    for loc in location_hints:
        if loc.get("city") == city:
            return CURRENCY_BY_COUNTRY.get(loc.get("country",""), "EUR")
    return "EUR"

def city_stub_latlon(city):
    r = random.Random(f"geo-{city}")
    lat = r.uniform(-60.0, 60.0)
    lon = r.uniform(-120.0, 120.0)
    return round(lat, 5), round(lon, 5)

def rand_name(city, cat):
    base = {
        "museum": ["City Museum","Modern Art Hall","History Pavilion","Science Forum"],
        "viewpoint": ["Skyline Terrace","Panorama Deck","Rooftop Vista","Hill Lookout"],
        "old-town-walk": ["Heritage Walk","Old Town Stroll","City Core Walk"],
        "landmark": ["Grand Cathedral","Royal Palace","River Gate","Clock Tower"],
        "river-cruise": ["River Cruise","Twilight Cruise","City Lights Cruise"],
        "lake-cruise": ["Lake Circuit","Island Hops","Blue Waters"],
        "scenic-train": ["Scenic Rail","Panorama Express","Valley Line"],
        "day-trip": ["Countryside Day Trip","Mountain Valley Day","Village Circuit"],
        "food-tour": ["Tastes of", "Gourmet Walk", "Street Eats"],
        "market": ["Central Market","Artisan Market","Farmers Market"],
        "workshop": ["Chocolate Workshop","Pasta Class","Photo Masterclass","Cheese Making"],
        "winery": ["Vineyard Visit","Wine Cellar Tasting","Estate Tour"],
        "brewery": ["Brewery Tour","Craft Beer Tasting","Brew Lab"],
        "hike-easy": ["Forest Path","Lake Trail","City Ridge"],
        "hike-moderate": ["Peak Trail","Alpine Route","Gorge Walk"],
        "bike-tour": ["City Bike Tour","Riverside Cycle","Sunset Ride"],
        "spa-thermal": ["Thermal Baths","City Spa Ritual","Wellness Circuit"],
        "theme-park": ["Adventure Park","Thrill World","Family Park"],
        "zoo-aquarium": ["City Zoo","Urban Aquarium","Wildlife Park"],
        "castle": ["Hilltop Castle","Fortress Tour","Royal Keep"]
    }
    stem = random.choice(base.get(cat, ["Local Experience"]))
    # small chance to include city name
    if random.random() < 0.5:
        if "of" in stem:
            return f"{stem} {city}"
        return f"{city} {stem}"
    return stem

def slot_range(day: date, start: time, end: time, step_min: int):
    slots = []
    cur = datetime.combine(day, start)
    end_dt = datetime.combine(day, end)
    while cur <= end_dt:
        slots.append(cur.isoformat())
        cur += timedelta(minutes=step_min)
    return slots

# --------------------------------------------------
# Generators
# --------------------------------------------------
def build_attractions_for_city(city, currency):
    n = random.randint(*CATALOG_PER_CITY_RANGE)
    lat0, lon0 = city_stub_latlon(city)
    items = []
    for i in range(n):
        cat = random.choice(CATEGORIES)
        # Some items are “city pass” style
        city_pass = (random.random() < CITY_WIDE_PASS_SHARE) and cat in {"museum","landmark","old-town-walk"}
        base_price = {"museum": (8, 25), "landmark": (0, 20), "viewpoint": (5, 18), "castle": (10, 25),
                      "old-town-walk": (0, 20), "river-cruise": (18, 45), "lake-cruise": (20, 55),
                      "scenic-train": (30, 120), "day-trip": (45, 160), "food-tour": (35, 95), "market": (0, 10),
                      "workshop": (30, 120), "winery": (25, 60), "brewery": (15, 40),
                      "hike-easy": (0, 10), "hike-moderate": (0, 15), "bike-tour": (25, 60),
                      "spa-thermal": (20, 70), "theme-park": (30, 90), "zoo-aquarium": (10, 35)}
        lo, hi = base_price.get(cat, (15, 50))
        price = 0 if city_pass else round(random.uniform(lo, hi), 2)
        dur_map = {"museum": (1.5, 3.0),"landmark": (0.5, 1.5),"viewpoint": (0.5, 1.5),"castle": (1.5, 3.0),
                   "old-town-walk": (1.5, 2.5),"river-cruise": (1.0, 2.0),"lake-cruise": (1.0, 2.5),
                   "scenic-train": (2.0, 4.0),"day-trip": (6.0, 10.0),"food-tour": (2.0, 3.5),"market": (1.0, 2.0),
                   "workshop": (1.5, 3.0),"winery": (1.5, 3.0),"brewery": (1.0, 2.0),"hike-easy": (1.5, 3.0),
                   "hike-moderate": (2.5, 5.0),"bike-tour": (2.0, 4.0),"spa-thermal": (1.5, 3.0),
                   "theme-park": (4.0, 8.0),"zoo-aquarium": (2.0, 4.0)}
        dmin, dmax = dur_map.get(cat, (1.5, 3.0))
        duration_h = round(random.uniform(dmin, dmax), 1)

        # provider placeholder for future live swap
        prov = random.choice(PROVIDERS)
        provider_ref = {
            "provider": prov["name"], "type": prov["type"],
            prov["id_field"]: f"{prov['name'].lower()}_{uuid.uuid4().hex[:10]}"
        }

        item = {
            "id": f"act_{uuid.uuid4().hex[:12]}",
            "city": city,
            "name": rand_name(city, cat),
            "category": cat,
            "tags": sorted(random.sample(TAGS, k=random.randint(2, 6))),
            "duration_h": duration_h,
            "price": price,
            "currency": currency,
            "rating": round(random.uniform(3.9, 4.9), 1),
            "reviews_count": random.randint(25, 12000),
            "languages": sorted(random.sample(LANGS, k=random.randint(1, 3))),
            "accessibility": random.choice(ACCESS),
            "risk_level": random.choice(RISK) if cat in {"hike-moderate","bike-tour","theme-park"} else "low",
            "weather_dependency": random.choice(WEATHER if cat in {"viewpoint","hike-easy","hike-moderate","bike-tour","river-cruise","lake-cruise"} else ["any"]),
            "amenities": sorted(random.sample(["toilets","locker","cloakroom","wifi","charging","cafe","souvenir-shop"], k=random.randint(1,4))),
            "meeting_point": random.choice(["hotel-pickup","main-station","old-town-plaza","harbor-pier","tourist-info"]),
            "open_hours": {
                "Mon":{"start":"09:00","end":"18:00"},
                "Tue":{"start":"09:00","end":"18:00"},
                "Wed":{"start":"09:00","end":"18:00"},
                "Thu":{"start":"09:00","end":"18:00"},
                "Fri":{"start":"09:00","end":"19:00"},
                "Sat":{"start":"09:00","end":"19:00"},
                "Sun":{"start":"10:00","end":"17:00"}
            },
            "capacity": random.choice([12,16,20,25,30,50]),
            "min_party": 1,
            "max_party": random.choice([6,8,10,12,16]),
            "cancellation_window_hours": random.choice([4,6,12,24,48]),
            "lead_time_hours": random.choice([0,2,6,12,24,48]),
            "geo": {
                "approx_lat": lat0 + random.uniform(-0.03,0.03),
                "approx_lon": lon0 + random.uniform(-0.03,0.03)
            },
            "best_seasons": random.sample(["spring","summer","autumn","winter"], k=random.randint(1,3)),
            "city_pass": city_pass,
            "provider_ref": provider_ref
        }
        items.append(item)
    return items

def gen_availability_for_item(item_id, days=AVAILABILITY_DAYS, open_hours=None):
    today = date.today()
    slots = []
    cap = random.choice([12,16,20,25,30,40,50])
    for d in range(days):
        day = today + timedelta(days=d)
        # derive windows from open_hours (default)
        start = time(9, 0); end = time(18, 0)
        if open_hours:
            weekday = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"][day.weekday()]
            wnd = open_hours.get(weekday, {"start":"09:00","end":"18:00"})
            try:
                sH,sM = map(int, wnd["start"].split(":")); eH,eM = map(int, wnd["end"].split(":"))
                start = time(sH, sM); end = time(eH, eM)
            except Exception:
                pass
        # slice to 30-min slots, mark capacity_left (busier weekends)
        wday = day.weekday()
        base_fill = 0.55 if wday in (4,5) else 0.35
        for s in slot_range(day, start, end, SLOT_STEP_MIN):
            left = max(0, int(cap*(1-base_fill) + random.randint(-5, 5)))
            slots.append({
                "activity_id": item_id,
                "slot_iso": s,
                "capacity_left": left,
                "cutoff_hours": random.choice([2,4,6,12,24])
            })
    return slots

def build_events_for_city(city, currency):
    """Generate city event occurrences over next ~180 days based on templates."""
    today = date.today()
    end = today + timedelta(days=AVAILABILITY_DAYS)
    events = []
    for tpl in EVENT_TEMPLATES:
        # pick one or two occurrences inside window if month matches
        for _ in range(random.choice([0,1,1,2])):
            # choose a start date in allowed months within window
            m = random.choice(tpl["month_window"])
            # anchor to current or next window
            anchor_year = today.year if m >= today.month else today.year + 1
            start = date(anchor_year, m, random.randint(1, 20))
            if not (today <= start <= end):
                continue
            days = tpl["days"]
            ev = {
                "event_id": f"evt_{uuid.uuid4().hex[:12]}",
                "city": city,
                "title": f"{city} {tpl['title']}",
                "category": tpl["category"],
                "starts_on": start.isoformat(),
                "ends_on": (start + timedelta(days=days-1)).isoformat(),
                "tags": tpl["tags"],
                "risk_level": tpl["risk"],
                "languages": sorted(random.sample(LANGS, k=random.randint(1, 3))),
                "ticket": {
                    "price_min": random.choice([0, 5, 10, 15, 20]),
                    "price_max": random.choice([25, 35, 50, 80, 120]),
                    "currency": currency,
                    "provider_ref": {
                        "provider": "LocalDMO",
                        "type": "dmo",
                        "listing_id": f"dmo_{uuid.uuid4().hex[:8]}"
                    }
                },
                "geo_hint": {
                    "approx_lat": city_stub_latlon(city)[0] + random.uniform(-0.05,0.05),
                    "approx_lon": city_stub_latlon(city)[1] + random.uniform(-0.05,0.05)
                }
            }
            events.append(ev)
    return events

# --------------------------------------------------
# Matching
# --------------------------------------------------
def traveler_prefs_from_users(users_by_id, member_ids):
    langs = set(); diets = set(); budgets = []
    interests = set(); pace_votes = []
    risk_max = "low"; access_needs = set()
    chronos = []

    for uid in member_ids:
        u = users_by_id.get(uid) or {}
        langs.update(u.get("languages") or [])
        diets.add(((u.get("diet_health") or {}).get("diet") or "none"))
        b = (u.get("budget") or {}).get("amount")
        if isinstance(b, (int,float)): budgets.append(b)
        interests.update(u.get("interests") or [])
        pace_votes.append(((u.get("travel_prefs") or {}).get("pace") or "balanced"))
        risk_max = max([risk_max, (u.get("comfort") or {}).get("risk_tolerance","low")], key=["low","medium","high"].index)
        dn = (u.get("diet_health") or {}).get("accessibility")
        if dn and dn != "none": access_needs.add(dn)
        chrono = (u.get("comfort") or {}).get("chronotype")
        if chrono: chronos.append(chrono)

    avg_budget = sum(budgets)/len(budgets) if budgets else 120
    pace = max(set(pace_votes), key=pace_votes.count) if pace_votes else "balanced"
    return {
        "languages": sorted(langs)[:4],
        "diets": sorted(d for d in diets if d and d!="none"),
        "avg_budget": avg_budget,
        "interests": sorted(list(interests))[:12],
        "pace": pace,
        "risk_max": risk_max,
        "access_needs": sorted(list(access_needs)),
        "chronos": chronos
    }

def price_band_for_budget(avg):
    if avg <= 90: return "budget"
    if avg >= 200: return "premium"
    return "mid"

def score_activity(a, prefs, party_size):
    score = 0.0
    # category match via interests
    wanted_cats = set()
    for it in prefs["interests"]:
        wanted_cats.update(INTEREST_TO_CATS.get(it, []))
    if not wanted_cats:
        wanted_cats = {"old-town-walk","museum","landmark","viewpoint"}
    if a["category"] in wanted_cats:
        score += 1.2
    # pace alignment (shorter durations for relaxed, longer ok for packed)
    if prefs["pace"] == "relaxed" and a["duration_h"] <= 2.5: score += 0.6
    if prefs["pace"] == "packed" and a["duration_h"] >= 2.0: score += 0.6
    if prefs["pace"] == "balanced" and 1.5 <= a["duration_h"] <= 4.0: score += 0.6
    # risk tolerance
    order = {"low":0,"medium":1,"high":2}
    if order.get(a["risk_level"],0) <= order.get(prefs["risk_max"],0): score += 0.5
    else: score -= 1.0
    # accessibility
    if prefs["access_needs"]:
        if a.get("accessibility") in {"wheelchair","reduced-mobility"}: score += 0.7
        else: score -= 0.7
    # language intersection
    if set(a.get("languages",[])) & set(prefs["languages"]): score += 0.4
    # price band soft fit
    band = price_band_for_budget(prefs["avg_budget"])
    if band == "budget" and a["price"] <= 25: score += 0.5
    if band == "mid" and 10 <= a["price"] <= 70: score += 0.4
    if band == "premium" and a["price"] >= 40: score += 0.4
    # capacity feasibility
    if party_size <= a.get("max_party", 8): score += 0.3
    else: score -= 1.0
    # city pass bonus if price sensitive
    if band == "budget" and a.get("city_pass"): score += 0.3
    # rating slight boost
    score += (a.get("rating",4.2)-4.0)*0.4
    return round(score, 3)

# --------------------------------------------------
# Main
# --------------------------------------------------
def main():
    assert USERS_PATH.exists(), f"Missing {USERS_PATH}"
    assert GROUPS_PATH.exists(), f"Missing {GROUPS_PATH}"

    users = load_json(USERS_PATH)
    groups = load_json(GROUPS_PATH)
    mm_profiles = load_json(MM_PATH)  # reserved for future tie-ins if needed

    users_by_id = idx_by(users, "user_id")
    locs = derive_location_hints(groups)

    # Build per-city attractions and availability; events
    city_catalog_map = {}   # city -> [attractions]
    availability = []       # [{activity_id, slot_iso, ...}]
    city_events_map = {}    # city -> [events]

    for loc in locs:
        city = loc["city"]
        currency = city_currency(city, locs)
        catalog = build_attractions_for_city(city, currency)
        city_catalog_map[city] = catalog
        # availability per attraction
        for a in catalog:
            availability.extend(gen_availability_for_item(a["id"], AVAILABILITY_DAYS, a.get("open_hours")))
        # events
        city_events_map[city] = build_events_for_city(city, currency)

    # Build group holds per itinerary day
    activity_holds = []
    event_holds = []

    for g in groups:
        gid = g.get("group_id")
        members = [m.get("user_id") for m in (g.get("members") or []) if m.get("user_id")]
        party_size = max(1, len(members))
        prefs = traveler_prefs_from_users(users_by_id, members)

        dp = g.get("draft_plan") or {}
        itin = dp.get("itinerary") or []

        for day in itin:
            city = day.get("base")
            date_iso = day.get("date")
            if not city or city not in city_catalog_map or not date_iso:
                continue

            # candidate activities (filter by basic feasibility)
            candidates = []
            for a in city_catalog_map[city]:
                # very light filter by weather dependency vs random “good weather” chance
                if a.get("weather_dependency") in {"good-weather","clear-sky"} and random.random() < 0.2:
                    continue
                candidates.append(a)

            # score & pick top 2-3
            scored = sorted(
                ((score_activity(a, prefs, party_size), a) for a in candidates),
                key=lambda x: x[0], reverse=True
            )[:3]

            for score, a in scored:
                # realistic times: morning 09–11, afternoon 13–16, evening 18–20 for viewpoints/river-cruise
                base_dt = datetime.fromisoformat(date_iso + "T09:00:00")
                start_hour = random.choice(
                    [9,10,11,13,14,15] + ([18,19,20] if a["category"] in {"viewpoint","river-cruise","lake-cruise","old-town-walk"} else [])
                )
                start_time = base_dt.replace(hour=start_hour, minute=random.choice([0,15,30,45]))
                hold_id = f"act_hold_{uuid.uuid4().hex[:10]}"
                activity_holds.append({
                    "hold_id": hold_id,
                    "group_id": gid,
                    "city": city,
                    "date": date_iso,
                    "activity_id": a["id"],
                    "party_size": party_size,
                    "planned_start_iso": start_time.isoformat(),
                    "status": random.choice(["hold","hold","confirmed"]) if a["rating"] >= 4.6 else "hold",
                    "payment": {
                        "split_rule": random.choice(["each_own","50/50","custom"]),
                        "currency": a["currency"]
                    },
                    "matching_rationale": {
                        "score": score,
                        "pace": prefs["pace"],
                        "risk_ok": a["risk_level"],
                        "languages_overlap": sorted(list(set(a.get("languages",[])) & set(prefs["languages"]))),
                        "access_need_met": prefs["access_needs"] and a.get("accessibility") in {"wheelchair","reduced-mobility"},
                        "price_band": price_band_for_budget(prefs["avg_budget"]),
                        "interests_used": sorted(list(set(prefs["interests"])))[0:5]
                    }
                })

            # Event holds if an event overlaps this day
            for ev in city_events_map.get(city, []):
                if ev["starts_on"] <= date_iso <= ev["ends_on"]:
                    eh_id = f"evt_hold_{uuid.uuid4().hex[:10]}"
                    event_holds.append({
                        "hold_id": eh_id,
                        "group_id": gid,
                        "city": city,
                        "date": date_iso,
                        "event_id": ev["event_id"],
                        "party_size": party_size,
                        "status": random.choice(["interest","hold"]),
                        "rationale": {
                            "tags": ev["tags"],
                            "risk": ev["risk_level"]
                        }
                    })
                    # one event per day is enough
                    break

    # --------------------------------------------------
    # Write outputs
    # --------------------------------------------------
    ATTRACTIONS_OUT.parent.mkdir(parents=True, exist_ok=True)
    AVAIL_SLOTS_OUT.parent.mkdir(parents=True, exist_ok=True)
    ACTIVITY_HOLDS_OUT.parent.mkdir(parents=True, exist_ok=True)
    EVENTS_OUT.parent.mkdir(parents=True, exist_ok=True)
    EVENT_HOLDS_OUT.parent.mkdir(parents=True, exist_ok=True)

    ATTRACTIONS_OUT.write_text(json.dumps(
        [{"city": c, "attractions": v} for c, v in sorted(city_catalog_map.items())],
        indent=2, ensure_ascii=False
    ), encoding="utf-8")

    AVAIL_SLOTS_OUT.write_text(json.dumps(availability, indent=2, ensure_ascii=False), encoding="utf-8")

    ACTIVITY_HOLDS_OUT.write_text(json.dumps(activity_holds, indent=2, ensure_ascii=False), encoding="utf-8")

    EVENTS_OUT.write_text(json.dumps(
        [{"city": c, "events": v} for c, v in sorted(city_events_map.items())],
        indent=2, ensure_ascii=False
    ), encoding="utf-8")

    EVENT_HOLDS_OUT.write_text(json.dumps(event_holds, indent=2, ensure_ascii=False), encoding="utf-8")

    # Quick peek
    print(f"✅ Cities with attractions: {len(city_catalog_map)} → {ATTRACTIONS_OUT}")
    print(f"✅ Availability slots: {len(availability)} → {AVAIL_SLOTS_OUT}")
    print(f"✅ Group activity holds: {len(activity_holds)} → {ACTIVITY_HOLDS_OUT}")
    print(f"✅ Events catalog entries (city buckets): {len(city_events_map)} → {EVENTS_OUT}")
    print(f"✅ Group event holds: {len(event_holds)} → {EVENT_HOLDS_OUT}")
    # Traces
    one_city = next(iter(city_catalog_map.keys()), None)
    if one_city:
        print(json.dumps(city_catalog_map[one_city][:2], indent=2)[:900], "...\n")
    if activity_holds:
        print(json.dumps(activity_holds[0], indent=2)[:600], "...\n")
    if event_holds:
        print(json.dumps(event_holds[0], indent=2)[:600], "...\n")

if __name__ == "__main__":
    main()
