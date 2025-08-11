import os
import json
import uuid
import random
from datetime import date, timedelta, datetime

# ----------------------------
# Config
# ----------------------------
N_GROUPS = 1000
RICH_GROUP_RATIO = 0.75
OUT_PATH = "Flight/data/travel_group_requests.json"
random.seed(42)

# ----------------------------
# Location catalog
# ----------------------------
COUNTRIES = {
    "Germany": {"langs": ["de", "en"], "cities": ["Berlin","Munich","Hamburg","Cologne","Frankfurt","Stuttgart","Dusseldorf","Leipzig","Dresden","Nuremberg"]},
    "UK": {"langs": ["en"], "cities": ["London","Manchester","Birmingham","Leeds","Edinburgh","Glasgow","Bristol","Liverpool","Newcastle","Belfast"]},
    "France": {"langs": ["fr","en"], "cities": ["Paris","Lyon","Marseille","Nice","Toulouse","Bordeaux","Lille","Nantes","Strasbourg","Montpellier"]},
    "Spain": {"langs": ["es","en"], "cities": ["Madrid","Barcelona","Valencia","Seville","Bilbao","Malaga","Zaragoza","Palma","Alicante","Granada"]},
    "Italy": {"langs": ["it","en"], "cities": ["Rome","Milan","Naples","Turin","Florence","Bologna","Genoa","Venice","Verona","Palermo"]},
    "Netherlands": {"langs": ["nl","en"], "cities": ["Amsterdam","Rotterdam","The Hague","Utrecht","Eindhoven"]},
    "Belgium": {"langs": ["nl","fr","en"], "cities": ["Brussels","Antwerp","Ghent","Bruges","Leuven"]},
    "Switzerland": {"langs": ["de","fr","it","en"], "cities": ["Zurich","Geneva","Basel","Lausanne","Bern","Lucerne","Lugano","St. Gallen"]},
    "Austria": {"langs": ["de","en"], "cities": ["Vienna","Salzburg","Graz","Innsbruck","Linz"]},
    "Portugal": {"langs": ["pt","en"], "cities": ["Lisbon","Porto","Faro","Coimbra","Braga"]},
    "Ireland": {"langs": ["en"], "cities": ["Dublin","Cork","Galway","Limerick","Waterford"]},
    "Poland": {"langs": ["pl","en"], "cities": ["Warsaw","Krakow","Gdansk","Wroclaw","Poznan"]},
    "Czechia": {"langs": ["cs","en"], "cities": ["Prague","Brno","Ostrava","Plzen","Liberec"]},
    "Greece": {"langs": ["el","en"], "cities": ["Athens","Thessaloniki","Heraklion","Chania","Rhodes"]},
    "Norway": {"langs": ["no","en"], "cities": ["Oslo","Bergen","Trondheim","Stavanger"]},
    "Sweden": {"langs": ["sv","en"], "cities": ["Stockholm","Gothenburg","Malmo","Uppsala"]},
    "Denmark": {"langs": ["da","en"], "cities": ["Copenhagen","Aarhus","Odense","Aalborg"]},
    "Finland": {"langs": ["fi","en"], "cities": ["Helsinki","Espoo","Tampere","Turku"]},
    "Iceland": {"langs": ["is","en"], "cities": ["Reykjavik"]},
    "USA": {"langs": ["en","es"], "cities": ["New York","San Francisco","Los Angeles","Chicago","Miami","Seattle","Boston","Austin","Denver","Atlanta"]},
    "Canada": {"langs": ["en","fr"], "cities": ["Toronto","Vancouver","Montreal","Ottawa","Calgary"]},
    "Mexico": {"langs": ["es","en"], "cities": ["Mexico City","Guadalajara","Monterrey","Cancun","Puebla"]},
    "Argentina": {"langs": ["es","en"], "cities": ["Buenos Aires","Cordoba","Mendoza","Rosario"]},
    "Chile": {"langs": ["es","en"], "cities": ["Santiago","Valparaiso","Concepcion"]},
    "Peru": {"langs": ["es","en"], "cities": ["Lima","Cusco","Arequipa"]},
    "Brazil": {"langs": ["pt","en"], "cities": ["Sao Paulo","Rio de Janeiro","Brasilia","Belo Horizonte","Curitiba"]},
    "Japan": {"langs": ["ja","en"], "cities": ["Tokyo","Osaka","Kyoto","Nagoya","Fukuoka"]},
    "South Korea": {"langs": ["ko","en"], "cities": ["Seoul","Busan","Incheon","Daegu"]},
    "China": {"langs": ["zh","en"], "cities": ["Beijing","Shanghai","Shenzhen","Guangzhou","Chengdu"]},
    "India": {"langs": ["en","hi"], "cities": ["Delhi","Mumbai","Bangalore","Chennai","Hyderabad","Pune","Kolkata"]},
    "Turkey": {"langs": ["tr","en"], "cities": ["Istanbul","Ankara","Izmir","Antalya"]},
    "UAE": {"langs": ["ar","en"], "cities": ["Dubai","Abu Dhabi","Sharjah"]},
    "Saudi Arabia": {"langs": ["ar","en"], "cities": ["Riyadh","Jeddah","Dammam"]},
    "Qatar": {"langs": ["ar","en"], "cities": ["Doha"]},
    "Kuwait": {"langs": ["ar","en"], "cities": ["Kuwait City"]},
    "South Africa": {"langs": ["en","af","zu"], "cities": ["Cape Town","Johannesburg","Durban","Pretoria"]},
    "Kenya": {"langs": ["sw","en"], "cities": ["Nairobi","Mombasa"]},
    "Nigeria": {"langs": ["en"], "cities": ["Lagos","Abuja"]},
    "Morocco": {"langs": ["ar","fr","en"], "cities": ["Casablanca","Marrakech","Rabat"]},
    "New Zealand": {"langs": ["en"], "cities": ["Auckland","Wellington","Christchurch"]},
}
LOCATION_DATA = [{"city": c, "country": k, "languages": v["langs"]} for k, v in COUNTRIES.items() for c in v["cities"]]

# ----------------------------
# Pools
# ----------------------------
GENDERS = ["Male","Female","Non-binary","Other"]
EDU_LEVELS = ["High School","Bachelor","Master","PhD","Vocational","Associate"]
OCCUPS = ["Engineer","Designer","Artist","Researcher","Student","Developer","Manager","Teacher","Doctor",
          "Consultant","Freelancer","Entrepreneur","Nurse","Writer","Architect","Lawyer","Scientist",
          "Chef","Musician","Photographer","Guide","Data Scientist","Analyst","Marketer","Product Manager"]
MARITAL = ["Single","Married","In a relationship","Divorced","Widowed"]
LEARNING_STYLES = ["Visual","Auditory","Kinesthetic","Reading/Writing"]
HUMOR = ["Dry","Witty","Slapstick","Sarcastic","Playful","Observational"]
INTERESTS = ["mountains","lakes","beaches","museums","old towns","food tours","street food",
             "scenic trains","short hikes","long hikes","nightlife","shopping","photography",
             "architecture","history","skiing","diving","sailing","cycling","festivals","thermal baths"]
DIET = ["none","vegetarian","vegan","halal","kosher","gluten-free","no pork","lactose-free"]
RISK = ["low","medium","high"]
NOISE = ["low","medium","high"]
CLEAN = ["low","medium","high"]
TRANSPORT_MODES = ["train","plane","bus","car"]
ACCOM_TYPES = ["hotel","apartment","hostel","guesthouse","boutique"]
PRICE_BANDS = ["budget","mid-range","luxury"]
ROOM_SETUP = ["twin","double","2 rooms","dorm"]
PACE = ["relaxed","balanced","packed"]
PLANNING = ["rigid planner","flexible","spontaneous"]
MONEY_ATT = ["split-everything","treat-and-be-treated","flexible"]
WORK_MODES = ["offline only","remote-work friendly","digital nomad"]

CURRENCY_BY_COUNTRY = {
    "USA":"USD","Canada":"CAD","Mexico":"MXN","Brazil":"BRL","Argentina":"ARS","Chile":"CLP","Peru":"PEN",
    "UK":"GBP","Ireland":"EUR","Germany":"EUR","France":"EUR","Spain":"EUR","Italy":"EUR","Portugal":"EUR",
    "Netherlands":"EUR","Belgium":"EUR","Switzerland":"CHF","Austria":"EUR","Poland":"PLN","Czechia":"CZK",
    "Denmark":"DKK","Sweden":"SEK","Norway":"NOK","Finland":"EUR","Greece":"EUR","Turkey":"TRY",
    "Japan":"JPY","South Korea":"KRW","China":"CNY","India":"INR","UAE":"AED","Saudi Arabia":"SAR","Qatar":"QAR","Kuwait":"KWD",
    "South Africa":"ZAR","Kenya":"KES","Nigeria":"NGN","Morocco":"MAD","New Zealand":"NZD","Iceland":"ISK"
}

# ----------------------------
# Helpers
# ----------------------------
def pick_location():
    return random.choice(LOCATION_DATA)

def future_date(days_from_today_min=10, days_from_today_max=200):
    start = date.today() + timedelta(days=random.randint(days_from_today_min, days_from_today_max))
    length = random.randint(3, 14)
    end = start + timedelta(days=length)
    return start, end, length

def normalized_weights():
    weights = [random.random() for _ in range(5)]
    s = sum(weights)
    return [round(w/s, 2) for w in weights]

def age_based_edu_occ(age):
    if age < 23:
        edu = random.choice(["High School","Bachelor"])
        occ = random.choice(["Student","Intern","Junior " + random.choice(OCCUPS)])
    elif age < 30:
        edu = random.choice(["Bachelor","Master"])
        occ = random.choice(OCCUPS)
    else:
        edu = random.choice(EDU_LEVELS)
        occ = random.choice(OCCUPS)
    return edu, occ

def build_traveler(idx):
    loc = pick_location()
    city, country, base_langs = loc["city"], loc["country"], loc["languages"]
    age = random.randint(20, 60)
    edu, occ = age_based_edu_occ(age)
    langs = list(set(random.sample(base_langs + ["en"], k=min(len(set(base_langs+['en'])), random.randint(1, 3)))))
    per_day = random.randint(70, 220)
    top_ints = random.sample(INTERESTS, k=random.randint(3, 6))
    eu_countries = {"Germany","France","Spain","Italy","Netherlands","Belgium","Switzerland","Austria","Portugal","Ireland",
                    "Poland","Czechia","Denmark","Sweden","Norway","Finland","Greece"}
    if country in eu_countries:
        allowed_modes = random.sample(["train","plane","bus"], k=random.randint(1, 3))
        if "train" not in allowed_modes and random.random() < 0.6:
            allowed_modes.append("train")
    else:
        allowed_modes = random.sample(TRANSPORT_MODES, k=random.randint(1, 3))

    t = {
        "name": f"Traveler-{idx}",
        "rovermitra_contact": {
            "channel": "RoverMitra",
            "user_handle": f"rm_{uuid.uuid4().hex[:8]}"
        },
        "home_base": {
            "city": city,
            "nearby_nodes": [f"{city} Central", f"{country[:3].upper()}-INTL"],
            "willing_radius_km": random.choice([30,50,60,80])
        },
        "age": age,
        "gender": random.choice(GENDERS),
        "education": edu,
        "occupation": occ,
        "marital_status": random.choice(MARITAL),
        "learning_style": random.choice(LEARNING_STYLES),
        "humor_style": random.choice(HUMOR),
        "budget": {
            "type": "per_day",
            "amount": per_day,
            "currency": CURRENCY_BY_COUNTRY.get(country, "USD"),
            "split_rule": random.choice(["each_own","50/50","custom"])
        },
        "transport": {
            "allowed_modes": list(set(allowed_modes)),
            "max_transfers": random.choice([1,2,3]),
            "max_leg_hours": random.choice([3,4,5,6]),
            "night_travel_ok": random.random() < 0.2
        },
        "accommodation": {
            "types": random.sample(ACCOM_TYPES, k=random.randint(1, 2)),
            "price_band": random.choice(PRICE_BANDS),
            "room_setup": random.choice(ROOM_SETUP),
            "must_haves": random.sample(["private_bath","wifi","kitchen","workspace","near_station","quiet_room"], k=random.randint(1, 3))
        },
        "pace_and_interests": {"pace": random.choice(PACE), "top_interests": top_ints},
        "diet_health": {
            "diet": random.choice(DIET),
            "allergies": random.sample(["nuts","shellfish","pollen","none"], k=1 if random.random()<0.8 else 2),
            "accessibility": random.sample(["stairs_ok","elevator_needed","reduced_mobility","none"], k=1)
        },
        "comfort": {
            "risk_tolerance": random.choice(RISK),
            "noise_tolerance": random.choice(NOISE),
            "cleanliness_preference": random.choice(CLEAN)
        },
        "work": {
            "hours_online_needed": random.choice([0, 1, 2]),
            "fixed_meetings": [],
            "wifi_quality_needed": random.choice(["normal","good","excellent"])
        },
        "documents": {
            "passport_valid_until": f"{random.randint(2027, 2032)}-{random.randint(1,12):02d}-{random.randint(1,28):02d}",
            "visa_status": random.choice(["Schengen OK","Visa-free","eVisa needed","Visa required"]),
            "insurance": random.random() < 0.8
        },
        "languages": langs,
        "dealbreakers": random.sample(
            ["no red-eye travel","no >2 transfers","no dorm rooms","no overnight trains","no party hostels","no smoking room"],
            k=random.randint(0, 3)
        )
    }
    return t

def build_trip_context(group_idx):
    dest_candidates = [loc["city"] for loc in random.sample(LOCATION_DATA, k=random.randint(1, 4))]
    if random.random() < 0.15:
        dest_candidates = random.sample(["Zurich","Lucerne","Interlaken","Zermatt","Bern","Basel"], k=random.randint(1,3))
    start, end, pref_len = future_date(10, 220)
    w_cost, w_time, w_comf, w_scene, w_co2 = normalized_weights()
    return {
        "title": f"Group-{group_idx} Trip",
        "destinations": dest_candidates,
        "date_window": {
            "earliest_departure": str(start),
            "latest_return": str(end),
            "preferred_trip_length_days": pref_len,
            "blackout_dates": []
        },
        "meeting_strategy_allowed": ["en_route_midpoint","at_destination","origin_A","origin_B"],
        "meeting_priority_objective": "minimize_total_travel_time_and_cost",
        "itinerary_style": random.choice(["anchor_city","multi-stop"]),
        "min_time_per_stop_hours": random.choice([12, 24, 36, 48]),
        "luggage": { "carry_on_only": random.random() < 0.6, "special_gear": random.sample(["camera","skis","hiking poles","none"], k=1) },
        "co2_preference": random.random() < 0.5,
        "tradeoff_weights": { "cost": w_cost, "time": w_time, "comfort": w_comf, "scenery": w_scene, "co2": w_co2 },
        "hard_constraints": {
            "earliest_departure_time_local": random.choice(["07:00","08:30","09:00"]),
            "latest_arrival_time_local": random.choice(["20:00","21:30","22:00"]),
            "max_daily_travel_hours": random.choice([5,6,7]),
            "max_transfers": random.choice([1,2,3]),
            "room_setup": random.choice(["twin","double","2 rooms"])
        },
        "output_preferences": {
            "detail_level": "day-by-day",
            "include_booking_links": True,
            "currency": "EUR",
            "units": "metric",
            "share_to_rovermitra_chat": True
        }
    }

def group_size():
    choices = list(range(1, 16))
    weights = [0.22, 0.28, 0.15, 0.10, 0.07, 0.05, 0.04, 0.03, 0.025, 0.02, 0.015, 0.01, 0.01, 0.01, 0.01]
    return random.choices(choices, weights=weights, k=1)[0]

def degrade_group_randomly(group):
    for t in group["travelers"]:
        if random.random() < 0.35:
            t.pop("learning_style", None)
        if random.random() < 0.35:
            t.pop("humor_style", None)
        if random.random() < 0.25:
            t.pop("dealbreakers", None)
        if random.random() < 0.25:
            t["budget"]["type"] = "total"
    tc = group["trip_context"]
    if random.random() < 0.35:
        tc.pop("co2_preference", None)
    if random.random() < 0.30:
        tc.pop("min_time_per_stop_hours", None)
    if random.random() < 0.30:
        tc.pop("luggage", None)
    return group

# ----------------------------
# Generate groups
# ----------------------------
groups = []
for g in range(N_GROUPS):
    size = group_size()
    travelers = [build_traveler(idx=f"{g}-{i}") for i in range(size)]
    trip_ctx = build_trip_context(g)
    group = {
        "group_id": str(uuid.uuid4()),
        "rovermitra_chat": {
            "room_id": f"rmr_{uuid.uuid4().hex[:10]}",
            "created_at": datetime.utcnow().isoformat() + "Z"
        },
        "trip_context": trip_ctx,
        "travelers": travelers
    }
    if random.random() > RICH_GROUP_RATIO:
        group = degrade_group_randomly(group)
    groups.append(group)

os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
with open(OUT_PATH, "w", encoding="utf-8") as f:
    json.dump(groups, f, indent=2, ensure_ascii=False)

# Summary
sizes = {}
for g in groups:
    s = len(g["travelers"])
    sizes[s] = sizes.get(s, 0) + 1
print(f"✅ Generated {len(groups)} travel group requests to {OUT_PATH}")
print("Group-size distribution (count):", dict(sorted(sizes.items())))
print("Example group (truncated):")
print(json.dumps(groups[0], indent=2)[:1200], "...\n")
