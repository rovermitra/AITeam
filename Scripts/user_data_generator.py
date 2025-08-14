#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate 2,000 realistic 'users_core.json' profiles for RoverMitra.
- Single source of truth for user data shared by Matchmaker and Trip Planner.
- Cities/countries/airports are consistent and realistic.
- Languages reflect country defaults (+ optional English).
- Includes identity, home base, budget, interests, diet/health, comfort, personality, values,
  social/work habits, stable travel preferences, and a short natural bio.
- ~20% "lean" profiles to simulate incomplete data.

Output: users/Data/users_core.json
"""

import os
import json
import uuid
import random
from datetime import datetime

random.seed(77)

# ------------------------------
# Output path
# ------------------------------
OUT_PATH = "users/data/users_core.json"
N_USERS = 2000
LEAN_RATIO = 0.20  # ~20% intentionally less complete

# ------------------------------
# Country/City/Airports (realistic)
# ------------------------------
COUNTRIES = {
    # EUROPE (selection)
    "Germany": {
        "langs": ["de", "en"],
        "cities": {
            "Berlin": ["BER"], "Munich": ["MUC"], "Hamburg": ["HAM"], "Cologne": ["CGN"],
            "Frankfurt": ["FRA"], "Stuttgart": ["STR"], "Dusseldorf": ["DUS"], "Leipzig": ["LEJ"],
            "Dresden": ["DRS"], "Nuremberg": ["NUE"]
        }
    },
    "Switzerland": {
        "langs": ["de", "fr", "it", "en"],
        "cities": {
            "Zurich": ["ZRH"], "Geneva": ["GVA"], "Basel": ["BSL"], "Bern": ["BRN"],
            "Lucerne": ["ZRH"], "Interlaken": ["BRN", "ZRH"], "Zermatt": ["GVA", "ZRH"]
        }
    },
    "Austria": {"langs": ["de", "en"], "cities": {"Vienna": ["VIE"], "Salzburg": ["SZG"], "Graz": ["GRZ"], "Innsbruck": ["INN"]}},
    "France": {"langs": ["fr", "en"], "cities": {"Paris": ["CDG","ORY"], "Lyon": ["LYS"], "Nice": ["NCE"], "Toulouse": ["TLS"], "Bordeaux": ["BOD"], "Marseille": ["MRS"], "Nantes": ["NTE"]}},
    "UK": {"langs": ["en"], "cities": {"London": ["LHR","LGW","LCY","LTN","STN"], "Manchester": ["MAN"], "Edinburgh": ["EDI"], "Glasgow": ["GLA"], "Birmingham": ["BHX"], "Bristol": ["BRS"]}},
    "Ireland": {"langs": ["en"], "cities": {"Dublin": ["DUB"], "Cork": ["ORK"]}},
    "Netherlands": {"langs": ["nl","en"], "cities": {"Amsterdam": ["AMS"], "Rotterdam": ["RTM"]}},
    "Belgium": {"langs": ["nl","fr","en"], "cities": {"Brussels": ["BRU"], "Antwerp": ["ANR"]}},
    "Spain": {"langs": ["es","en"], "cities": {"Madrid": ["MAD"], "Barcelona": ["BCN"], "Valencia": ["VLC"], "Seville": ["SVQ"], "Malaga": ["AGP"], "Bilbao": ["BIO"]}},
    "Portugal": {"langs": ["pt","en"], "cities": {"Lisbon": ["LIS"], "Porto": ["OPO"], "Faro": ["FAO"]}},
    "Italy": {"langs": ["it","en"], "cities": {"Rome": ["FCO","CIA"], "Milan": ["MXP","LIN","BGY"], "Venice": ["VCE"], "Florence": ["FLR"], "Naples": ["NAP"], "Turin": ["TRN"]}},
    "Greece": {"langs": ["el","en"], "cities": {"Athens": ["ATH"], "Thessaloniki": ["SKG"], "Heraklion": ["HER"]}},
    "Poland": {"langs": ["pl","en"], "cities": {"Warsaw": ["WAW"], "Krakow": ["KRK"], "Gdansk": ["GDN"]}},
    "Czechia": {"langs": ["cs","en"], "cities": {"Prague": ["PRG"], "Brno": ["BRQ"]}},
    "Denmark": {"langs": ["da","en"], "cities": {"Copenhagen": ["CPH"], "Aarhus": ["AAR"]}},
    "Sweden": {"langs": ["sv","en"], "cities": {"Stockholm": ["ARN","BMA"], "Gothenburg": ["GOT"]}},
    "Norway": {"langs": ["no","en"], "cities": {"Oslo": ["OSL"], "Bergen": ["BGO"]}},
    "Finland": {"langs": ["fi","en"], "cities": {"Helsinki": ["HEL"]}},
    "Iceland": {"langs": ["is","en"], "cities": {"Reykjavik": ["KEF"]}},

    # AMERICAS (selection)
    "USA": {"langs": ["en","es"], "cities": {"New York": ["JFK","EWR","LGA"], "San Francisco": ["SFO"], "Los Angeles": ["LAX"], "Chicago": ["ORD"], "Miami": ["MIA"], "Seattle": ["SEA"], "Boston": ["BOS"], "Denver": ["DEN"]}},
    "Canada": {"langs": ["en","fr"], "cities": {"Toronto": ["YYZ"], "Vancouver": ["YVR"], "Montreal": ["YUL"]}},
    "Mexico": {"langs": ["es","en"], "cities": {"Mexico City": ["MEX"], "Guadalajara": ["GDL"], "Monterrey": ["MTY"]}},
    "Brazil": {"langs": ["pt","en"], "cities": {"Sao Paulo": ["GRU","CGH"], "Rio de Janeiro": ["GIG","SDU"], "Brasilia": ["BSB"]}},
    "Argentina": {"langs": ["es","en"], "cities": {"Buenos Aires": ["EZE","AEP"], "Cordoba": ["COR"]}},
    "Chile": {"langs": ["es","en"], "cities": {"Santiago": ["SCL"]}},
    "Peru": {"langs": ["es","en"], "cities": {"Lima": ["LIM"], "Cusco": ["CUZ"]}},
    "Colombia": {"langs": ["es","en"], "cities": {"Bogota": ["BOG"], "Medellin": ["MDE"]}},

    # MIDDLE EAST / AFRICA (selection)
    "UAE": {"langs": ["ar","en"], "cities": {"Dubai": ["DXB"], "Abu Dhabi": ["AUH"]}},
    "Saudi Arabia": {"langs": ["ar","en"], "cities": {"Riyadh": ["RUH"], "Jeddah": ["JED"]}},
    "Qatar": {"langs": ["ar","en"], "cities": {"Doha": ["DOH"]}},
    "Israel": {"langs": ["he","en"], "cities": {"Tel Aviv": ["TLV"]}},
    "Jordan": {"langs": ["ar","en"], "cities": {"Amman": ["AMM"]}},
    "Morocco": {"langs": ["ar","fr","en"], "cities": {"Marrakech": ["RAK"], "Casablanca": ["CMN"]}},
    "Egypt": {"langs": ["ar","en"], "cities": {"Cairo": ["CAI"]}},
    "South Africa": {"langs": ["en","af","zu"], "cities": {"Cape Town": ["CPT"], "Johannesburg": ["JNB"]}},
    "Kenya": {"langs": ["sw","en"], "cities": {"Nairobi": ["NBO"]}},

    # ASIA (selection)
    "Japan": {"langs": ["ja","en"], "cities": {"Tokyo": ["HND","NRT"], "Osaka": ["KIX","ITM"], "Kyoto": ["KIX"]}},
    "South Korea": {"langs": ["ko","en"], "cities": {"Seoul": ["ICN","GMP"], "Busan": ["PUS"]}},
    "China": {"langs": ["zh","en"], "cities": {"Beijing": ["PEK","PKX"], "Shanghai": ["PVG","SHA"], "Shenzhen": ["SZX"]}},
    "India": {"langs": ["en","hi"], "cities": {"Delhi": ["DEL"], "Mumbai": ["BOM"], "Bangalore": ["BLR"], "Hyderabad": ["HYD"]}},
    "Pakistan": {"langs": ["ur","en"], "cities": {"Karachi": ["KHI"], "Lahore": ["LHE"], "Islamabad": ["ISB"]}},
    "Thailand": {"langs": ["th","en"], "cities": {"Bangkok": ["BKK","DMK"], "Chiang Mai": ["CNX"], "Phuket": ["HKT"]}},
    "Vietnam": {"langs": ["vi","en"], "cities": {"Hanoi": ["HAN"], "Ho Chi Minh City": ["SGN"]}},
    "Malaysia": {"langs": ["ms","en"], "cities": {"Kuala Lumpur": ["KUL"], "Penang": ["PEN"]}},
    "Singapore": {"langs": ["en"], "cities": {"Singapore": ["SIN"]}},
    "Indonesia": {"langs": ["id","en"], "cities": {"Jakarta": ["CGK"], "Bali": ["DPS"]}},
    "Philippines": {"langs": ["en","tl"], "cities": {"Manila": ["MNL"], "Cebu": ["CEB"]}},

    # OCEANIA
    "Australia": {"langs": ["en"], "cities": {"Sydney": ["SYD"], "Melbourne": ["MEL"], "Brisbane": ["BNE"]}},
    "New Zealand": {"langs": ["en"], "cities": {"Auckland": ["AKL"], "Wellington": ["WLG"], "Christchurch": ["CHC"]}}
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

# Flatten locations
LOCATION_DATA = []
for country, cinfo in COUNTRIES.items():
    for city, airports in cinfo["cities"].items():
        LOCATION_DATA.append({
            "city": city, "country": country, "airports": airports, "languages": cinfo["langs"]
        })

# ------------------------------
# Pools
# ------------------------------
FIRST_NAMES = [
    "Abdul","Aisha","Akira","Alex","Alice","Amir","Ana","Andreas","Arjun","Ava","Ben","Bianca","Carlos","Chloe",
    "Daniel","Diana","Elena","Emil","Emma","Fatima","Felix","Franz","Grace","Hana","Hassan","Helena","Hiro",
    "Imran","Ivy","Jack","Jana","Jonas","Karim","Kate","Khalid","Laura","Leo","Liam","Lucia","Mateo","Maya",
    "Mei","Milan","Nadia","Noah","Omar","Olivia","Paul","Priya","Qasim","Riya","Rosa","Sam","Sofia","Sven",
    "Tariq","Tina","Tom","Valerie","Vihaan","Wei","Xenia","Yara","Yuki","Zahra","Zane","Zoe"
]
LAST_NAMES = [
    "Ahmed","Ali","Anderson","Aoki","Bauer","Bianchi","Brown","Carter","Chen","Costa","Das","Diaz","Dubois","Eriksen",
    "Fischer","Garcia","Ghosh","Gruber","Hernandez","Hussain","Ibrahim","Ivanov","Jackson","Johansson","Jones","Khan",
    "Kim","Kumar","Larsson","Lee","Lopez","Martin","Meier","Meyer","Miller","Mori","Müller","Nguyen","Novak","Nowak",
    "O'Connor","Olsen","Park","Patel","Pereira","Petrov","Rodriguez","Santos","Schmidt","Schneider","Singh","Smith",
    "Tanaka","Thompson","Wang","Williams","Yamamoto","Zhang"
]

GENDERS = ["Male","Female","Non-binary","Other"]
VALUES = ["adventure","stability","learning","family","budget-minded","luxury-taste","nature","culture","community","fitness","spirituality"]
INTERESTS = [
    "mountain hiking","city photography","food tours","street food","coffee crawls","scenic trains","short hikes","long hikes",
    "nightlife","museum hopping","architecture walks","history sites","skiing","diving","sailing","cycling","festivals",
    "thermal baths","vineyards","wildlife watching","markets","street art","rooftop views","bookstores","local crafts","castles",
    "beach days","lake swims","yoga","trail running","road trips","camping","chess","board games","cinema","live music",
    "theatre","coding","robotics","blogging","vlogging","language exchange"
]
LEARNING_STYLES = ["Visual","Auditory","Kinesthetic","Reading/Writing"]
HUMOR_STYLES = ["Dry","Witty","Slapstick","Sarcastic","Playful","Observational"]
DIET = ["none","vegetarian","vegan","halal","kosher","gluten-free","no pork","lactose-free","pescatarian"]
RISK = ["low","medium","high"]
NOISE = ["low","medium","high"]
CLEAN = ["low","medium","high"]
PACE = ["relaxed","balanced","packed"]
ACCOM_TYPES = ["hotel","apartment","guesthouse","boutique","hostel"]
ROOM_SETUP = ["twin","double","2 rooms","dorm"]
TRANSPORT = ["train","plane","bus","car"]
CHRONO = ["early bird","night owl","flexible"]
ALCOHOL = ["none","moderate","social"]
SMOKING = ["never","occasionally","regular"]
WIFI_NEED = ["normal","good","excellent"]

# ------------------------------
# Helpers
# ------------------------------
def pick_location():
    return random.choice(LOCATION_DATA)

def station_name(city, country):
    if country in ["Germany","Austria","Switzerland"]:
        return f"{city} Hbf"
    return f"{city} Central Station"

def sample_languages(default_langs):
    base = list(set(default_langs + (["en"] if "en" not in default_langs and random.random() < 0.6 else [])))
    k = random.randint(1, min(3, len(base)))
    return random.sample(base, k=k)

def personality_block():
    return {
        "openness": round(random.uniform(0.15, 0.98), 2),
        "conscientiousness": round(random.uniform(0.15, 0.98), 2),
        "extraversion": round(random.uniform(0.15, 0.98), 2),
        "agreeableness": round(random.uniform(0.15, 0.98), 2),
        "neuroticism": round(random.uniform(0.05, 0.85), 2),
        "creativity": round(random.uniform(0.2, 0.98), 2),
        "empathy": round(random.uniform(0.2, 0.98), 2)
    }

def short_bio(name, city, interests):
    two = ", ".join(random.sample(interests, k=min(2, len(interests))))
    verbs = ["exploring", "learning", "creating", "documenting", "sharing", "planning"]
    return f"{name} from {city} loves {two} and enjoys {random.choice(verbs)} memorable trips."

def values_pick():
    k = random.choice([2,3])
    return random.sample(VALUES, k=k)

def budget_block(country):
    currency = CURRENCY_BY_COUNTRY.get(country, "EUR")
    per_day = random.randint(60, 260)
    return {"type": "per_day", "amount": per_day, "currency": currency, "split_rule": random.choice(["each_own","50/50","custom"])}

def travel_prefs_block():
    return {
        "pace": random.choice(PACE),
        "accommodation_types": random.sample(ACCOM_TYPES, k=random.randint(1,2)),
        "room_setup": random.choice(ROOM_SETUP),
        "transport_allowed": list(set(random.sample(TRANSPORT, k=random.randint(1,3)))),
        "must_haves": random.sample(["wifi","private_bath","kitchen","workspace","near_station","quiet_room","breakfast"], k=random.randint(1,4))
    }

def diet_health_block():
    return {
        "diet": random.choice(DIET),
        "allergies": random.sample(["none","nuts","shellfish","pollen","gluten","lactose"], k=1),
        "accessibility": random.choice(["none","elevator_needed","reduced_mobility"])
    }

def comfort_block():
    return {
        "risk_tolerance": random.choice(RISK),
        "noise_tolerance": random.choice(NOISE),
        "cleanliness_preference": random.choice(CLEAN),
        "chronotype": random.choice(CHRONO),
        "alcohol": random.choice(ALCOHOL),
        "smoking": random.choice(SMOKING)
    }

def work_block():
    return {
        "remote_work_ok": random.random() < 0.55,
        "hours_online_needed": random.choice([0,1,2]),
        "wifi_quality_needed": random.choice(WIFI_NEED)
    }

def id_contact(name):
    first, last = name.split(" ", 1)
    handle = f"rm_{first.lower()}.{last.lower()[:8]}"
    email = f"{first.lower()}.{last.lower()}@rovermitra.example".replace(" ", "")
    return handle, email

# ------------------------------
# Builder
# ------------------------------
def build_user(idx):
    # Location
    loc = pick_location()
    city, country, airports, langs = loc["city"], loc["country"], loc["airports"], loc["languages"]

    # Name & IDs
    name = f"{random.choice(FIRST_NAMES)} {random.choice(LAST_NAMES)}"
    handle, email = id_contact(name)

    # Core profile
    age = random.randint(19, 62)
    languages = sample_languages(langs)
    interests = random.sample(INTERESTS, k=random.randint(5, 10))

    profile = {
        "user_id": f"u_{uuid.uuid4().hex[:12]}",
        "created_at": datetime.utcnow().isoformat() + "Z",
        "name": name,
        "age": age,
        "gender": random.choice(GENDERS),

        "contact": {
            "rovermitra_handle": handle,
            "email": email
        },

        "home_base": {
            "city": city,
            "country": country,
            "nearby_nodes": [station_name(city, country), random.choice(airports)],
            "willing_radius_km": random.choice([25, 40, 60, 80])
        },

        "languages": languages,
        "interests": interests,
        "values": values_pick(),
        "personality": personality_block(),
        "bio": short_bio(name, city, interests),

        # Financial baseline for planning / matching
        "budget": budget_block(country),

        # Health & comfort
        "diet_health": diet_health_block(),
        "comfort": comfort_block(),

        # Social/learning preferences (useful for matchmaker & group travel)
        "social": {
            "group_size_ok": random.choice([[1,2],[1,2,3],[2,3,4],[1,2,3,4,5]]),
            "learning_style": random.choice(LEARNING_STYLES),
            "humor_style": random.choice(HUMOR_STYLES),
            "dealbreakers_social": random.sample(
                ["no party hostels","no smoking room","no red-eye travel","no >2 transfers","no dorm rooms"], 
                k=random.randint(0,2)
            )
        },

        # Work style relevant for on-trip planning
        "work": work_block(),

        # Stable travel preferences that rarely change
        "travel_prefs": travel_prefs_block(),

        # Privacy/consent flags
        "privacy": {
            "share_profile_with_matches": True,
            "share_itinerary_with_group": random.random() < 0.9,
            "marketing_opt_in": random.random() < 0.3
        }
    }

    return profile

def degrade_profile_randomly(profile):
    """Make some profiles intentionally less complete."""
    removable_top = [
        "values","personality","bio","diet_health","comfort","social","work","travel_prefs","privacy"
    ]
    k = random.randint(2, 5)
    for key in random.sample(removable_top, k=k):
        profile.pop(key, None)
    # Keep core identity, home_base, budget, languages, interests intact
    return profile

# ------------------------------
# Generate
# ------------------------------
def main():
    users = []
    for i in range(N_USERS):
        p = build_user(i)
        if random.random() < LEAN_RATIO:
            p = degrade_profile_randomly(p)
        users.append(p)

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(users, f, indent=2, ensure_ascii=False)

    # quick sanity report
    print(f"✅ Generated {len(users)} users → {OUT_PATH}")
    # Example peek
    print(json.dumps(users[0], indent=2)[:1200] + "\n...")

if __name__ == "__main__":
    main()
