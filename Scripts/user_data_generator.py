#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RoverMitra – Rich & Diverse User Generator (schema-compatible)


What’s improved:
- Balanced distributions (gender, ages, pace, budget bands).
- Language mix tuned so de/en and other combos are common → more hard-filter passes.
- Region-aware budgets but with target shares for budget/mid/lux.
- Deterministic seed (override via RM_GEN_SEED).

Usage:
  RM_GEN_N=10000 python Scripts/user_data_generator_rich.py
Output:
  users/data/users_core.json
"""

import os
import json
import uuid
import random
import unicodedata
from datetime import datetime, date, timedelta

# ========= Config =========
SEED = int(os.getenv("RM_GEN_SEED", "77"))
random.seed(SEED)

OUT_PATH   = "users/data/users_core.json"
N_USERS    = int(os.getenv("RM_GEN_N", "10000"))
LEAN_RATIO = float(os.getenv("RM_GEN_LEAN_RATIO", "0.20"))  # keep as in your pipeline

# Target shares to increase chance of hard-filter matches
# (tune these to influence how many pass languages/budget/pace filters)
TARGET_GENDER_SPLIT = {"Male": 0.34, "Female": 0.34, "Non-binary": 0.16, "Other": 0.16}
TARGET_PACE_SPLIT   = {"relaxed": 0.35, "balanced": 0.40, "packed": 0.25}
TARGET_BUDGET_BAND  = {"budget": 0.30, "mid": 0.45, "lux": 0.25}

# Increase likelihood that many users speak en/de (helps when your query users are in Germany)
# Still region-aware, but boosts shared-language probability.
GLOBAL_LANG_BONUS = {"en": 0.70, "de": 0.55}

# ========= Geography & Pools (same core as your current file; trimmed comments) =========
COUNTRIES = {
    "Germany": {"langs":["de","en"],"cities":{
        "Berlin":["BER"],"Munich":["MUC"],"Hamburg":["HAM"],"Cologne":["CGN"],"Frankfurt":["FRA"],"Stuttgart":["STR"],"Düsseldorf":["DUS"],"Leipzig":["LEJ"]
    }},
    "Switzerland":{"langs":["de","fr","it","en"],"cities":{
        "Zurich":["ZRH"],"Geneva":["GVA"],"Basel":["BSL"],"Bern":["BRN"],"Lugano":["LUG"]
    }},
    "France":{"langs":["fr","en"],"cities":{
        "Paris":["CDG","ORY"],"Lyon":["LYS"],"Nice":["NCE"],"Marseille":["MRS"],"Toulouse":["TLS"],"Bordeaux":["BOD"]
    }},
    "UK":{"langs":["en"],"cities":{
        "London":["LHR","LGW","LCY","LTN","STN"],"Manchester":["MAN"],"Edinburgh":["EDI"],"Birmingham":["BHX"],"Glasgow":["GLA"],"Bristol":["BRS"]
    }},
    "Ireland":{"langs":["en","ga"],"cities":{"Dublin":["DUB"],"Cork":["ORK"],"Shannon":["SNN"]}},
    "Spain":{"langs":["es","en"],"cities":{"Madrid":["MAD"],"Barcelona":["BCN"],"Valencia":["VLC"],"Seville":["SVQ"],"Malaga":["AGP"],"Bilbao":["BIO"],"Palma de Mallorca":["PMI"]}},
    "Italy":{"langs":["it","en"],"cities":{"Rome":["FCO","CIA"],"Milan":["MXP","LIN","BGY"],"Florence":["FLR"],"Venice":["VCE"],"Naples":["NAP"],"Bologna":["BLQ"],"Turin":["TRN"]}},
    "Netherlands":{"langs":["nl","en"],"cities":{"Amsterdam":["AMS"],"Rotterdam":["RTM"],"Eindhoven":["EIN"]}},
    "Belgium":{"langs":["nl","fr","de","en"],"cities":{"Brussels":["BRU","CRL"],"Antwerp":["ANR"]}},
    "Portugal":{"langs":["pt","en"],"cities":{"Lisbon":["LIS"],"Porto":["OPO"],"Faro":["FAO"]}},
    "Greece":{"langs":["el","en"],"cities":{"Athens":["ATH"],"Thessaloniki":["SKG"],"Heraklion":["HER"]}},
    "Poland":{"langs":["pl","en"],"cities":{"Warsaw":["WAW","WMI"],"Krakow":["KRK"],"Gdansk":["GDN"],"Wroclaw":["WRO"],"Poznan":["POZ"]}},
    "Austria":{"langs":["de","en"],"cities":{"Vienna":["VIE"],"Salzburg":["SZG"],"Graz":["GRZ"],"Innsbruck":["INN"]}},
    "Czechia":{"langs":["cs","en"],"cities":{"Prague":["PRG"],"Brno":["BRQ"]}},
    "Hungary":{"langs":["hu","en"],"cities":{"Budapest":["BUD"],"Debrecen":["DEB"]}},
    "Slovakia":{"langs":["sk","en"],"cities":{"Bratislava":["BTS"],"Kosice":["KSC"]}},
    "Slovenia":{"langs":["sl","en"],"cities":{"Ljubljana":["LJU"]}},
    "Croatia":{"langs":["hr","en"],"cities":{"Zagreb":["ZAG"],"Split":["SPU"],"Dubrovnik":["DBV"]}},
    "Romania":{"langs":["ro","en"],"cities":{"Bucharest":["OTP","BBU"],"Cluj-Napoca":["CLJ"]}},
    "Bulgaria":{"langs":["bg","en"],"cities":{"Sofia":["SOF"],"Varna":["VAR"]}},
    "Serbia":{"langs":["sr","en"],"cities":{"Belgrade":["BEG"],"Nis":["INI"]}},
    "Bosnia and Herzegovina":{"langs":["bs","hr","sr","en"],"cities":{"Sarajevo":["SJJ"]}},
    "Albania":{"langs":["sq","en"],"cities":{"Tirana":["TIA"]}},
    "North Macedonia":{"langs":["mk","en"],"cities":{"Skopje":["SKP"]}},
    "Montenegro":{"langs":["sr","en"],"cities":{"Podgorica":["TGD"]}},
    "Lithuania":{"langs":["lt","en"],"cities":{"Vilnius":["VNO"],"Kaunas":["KUN"]}},
    "Latvia":{"langs":["lv","en"],"cities":{"Riga":["RIX"]}},
    "Estonia":{"langs":["et","en"],"cities":{"Tallinn":["TLL"]}},
    "Finland":{"langs":["fi","sv","en"],"cities":{"Helsinki":["HEL"],"Tampere":["TMP"]}},
    "Sweden":{"langs":["sv","en"],"cities":{"Stockholm":["ARN","BMA"],"Gothenburg":["GOT"],"Malmo":["MMX"]}},
    "Norway":{"langs":["no","en"],"cities":{"Oslo":["OSL","TRF"],"Bergen":["BGO"],"Stavanger":["SVG"]}},
    "Denmark":{"langs":["da","en"],"cities":{"Copenhagen":["CPH","RKE"],"Aarhus":["AAR"],"Billund":["BLL"]}},

    "USA":{"langs":["en","es"],"cities":{
        "New York":["JFK","EWR","LGA"],"San Francisco":["SFO"],"Los Angeles":["LAX"],"Chicago":["ORD","MDW"],"Miami":["MIA"],"Seattle":["SEA"],"Boston":["BOS"],"Dallas":["DFW","DAL"],"Atlanta":["ATL"],"Washington":["IAD","DCA"]
    }},
    "Canada":{"langs":["en","fr"],"cities":{"Toronto":["YYZ","YTZ"],"Vancouver":["YVR"],"Montreal":["YUL","YHU"],"Calgary":["YYC"],"Ottawa":["YOW"]}},
    "Mexico":{"langs":["es","en"],"cities":{"Mexico City":["MEX","NLU","TLC"],"Guadalajara":["GDL"],"Monterrey":["MTY"],"Cancun":["CUN"]}},
    "Brazil":{"langs":["pt","en"],"cities":{"Sao Paulo":["GRU","CGH","VCP"],"Rio de Janeiro":["GIG","SDU"],"Brasilia":["BSB"],"Belo Horizonte":["CNF"],"Porto Alegre":["POA"]}},
    "Argentina":{"langs":["es","en"],"cities":{"Buenos Aires":["EZE","AEP"],"Cordoba":["COR"]}},
    "Chile":{"langs":["es","en"],"cities":{"Santiago":["SCL"]}},
    "Colombia":{"langs":["es","en"],"cities":{"Bogota":["BOG"],"Medellin":["MDE"],"Cartagena":["CTG"]}},
    "Peru":{"langs":["es","en"],"cities":{"Lima":["LIM"],"Cusco":["CUZ"]}},

    "UAE":{"langs":["ar","en"],"cities":{"Dubai":["DXB","DWC"],"Abu Dhabi":["AUH"],"Sharjah":["SHJ"]}},
    "Saudi Arabia":{"langs":["ar","en"],"cities":{"Riyadh":["RUH"],"Jeddah":["JED"],"Dammam":["DMM"]}},
    "Qatar":{"langs":["ar","en"],"cities":{"Doha":["DOH"]}},
    "Jordan":{"langs":["ar","en"],"cities":{"Amman":["AMM"]}},
    "Lebanon":{"langs":["ar","fr","en"],"cities":{"Beirut":["BEY"]}},
    "Israel":{"langs":["he","en"],"cities":{"Tel Aviv":["TLV"],"Eilat":["ETM"]}},
    "Pakistan":{"langs":["ur","en"],"cities":{"Karachi":["KHI"],"Lahore":["LHE"],"Islamabad":["ISB"],"Peshawar":["PEW"],"Quetta":["UET"]}},
    "India":{"langs":["hi","en"],"cities":{"Delhi":["DEL"],"Mumbai":["BOM"],"Bangalore":["BLR"],"Hyderabad":["HYD"],"Chennai":["MAA"],"Kolkata":["CCU"],"Pune":["PNQ"]}},
    "Bangladesh":{"langs":["bn","en"],"cities":{"Dhaka":["DAC"],"Chittagong":["CGP"]}},
    "Sri Lanka":{"langs":["si","ta","en"],"cities":{"Colombo":["CMB"],"Mattala":["HRI"]}},
    "Nepal":{"langs":["ne","en"],"cities":{"Kathmandu":["KTM"]}},
    "Turkey":{"langs":["tr","en"],"cities":{"Istanbul":["IST","SAW"],"Ankara":["ESB"],"Izmir":["ADB"],"Antalya":["AYT"]}},
    "Iran":{"langs":["fa","en"],"cities":{"Tehran":["IKA","THR"],"Shiraz":["SYZ"],"Mashhad":["MHD"]}},
    "Morocco":{"langs":["ar","fr","en"],"cities":{"Marrakech":["RAK"],"Casablanca":["CMN"],"Rabat":["RBA"]}},
    "Tunisia":{"langs":["ar","fr","en"],"cities":{"Tunis":["TUN"]}},
    "Algeria":{"langs":["ar","fr","en"],"cities":{"Algiers":["ALG"]}},
    "Egypt":{"langs":["ar","en"],"cities":{"Cairo":["CAI"],"Alexandria":["HBE"]}},
    "South Africa":{"langs":["en","zu","xh","af"],"cities":{"Johannesburg":["JNB"],"Cape Town":["CPT"],"Durban":["DUR"]}},
    "Kenya":{"langs":["en","sw"],"cities":{"Nairobi":["NBO","WIL"],"Mombasa":["MBA"]}},

    "China":{"langs":["zh","en"],"cities":{"Beijing":["PEK","PKX"],"Shanghai":["PVG","SHA"],"Shenzhen":["SZX"],"Guangzhou":["CAN"],"Chengdu":["TFU","CTU"]}},
    "Hong Kong":{"langs":["zh","en"],"cities":{"Hong Kong":["HKG"]}},
    "Taiwan":{"langs":["zh","en"],"cities":{"Taipei":["TPE","TSA"],"Kaohsiung":["KHH"]}},
    "Japan":{"langs":["ja","en"],"cities":{"Tokyo":["HND","NRT"],"Osaka":["KIX","ITM"],"Nagoya":["NGO"],"Fukuoka":["FUK"],"Sapporo":["CTS"]}},
    "South Korea":{"langs":["ko","en"],"cities":{"Seoul":["ICN","GMP"],"Busan":["PUS"]}},
    "Thailand":{"langs":["th","en"],"cities":{"Bangkok":["BKK","DMK"],"Chiang Mai":["CNX"],"Phuket":["HKT"]}},
    "Vietnam":{"langs":["vi","en"],"cities":{"Hanoi":["HAN"],"Ho Chi Minh City":["SGN"],"Da Nang":["DAD"]}},
    "Malaysia":{"langs":["ms","en"],"cities":{"Kuala Lumpur":["KUL","SZB"],"Penang":["PEN"],"Kuching":["KCH"]}},
    "Singapore":{"langs":["en","ms","zh","ta"],"cities":{"Singapore":["SIN"]}},
    "Indonesia":{"langs":["id","en"],"cities":{"Jakarta":["CGK","HLP"],"Bali":["DPS"],"Surabaya":["SUB"]}},
    "Philippines":{"langs":["en","tl"],"cities":{"Manila":["MNL"],"Cebu":["CEB"],"Davao":["DVO"]}},
    "Cambodia":{"langs":["km","en"],"cities":{"Phnom Penh":["PNH"],"Siem Reap":["SAI","REP"]}},
    "Laos":{"langs":["lo","en"],"cities":{"Vientiane":["VTE"],"Luang Prabang":["LPQ"]}},
    "Myanmar":{"langs":["my","en"],"cities":{"Yangon":["RGN"]}},
    "Brunei":{"langs":["ms","en"],"cities":{"Bandar Seri Begawan":["BWN"]}},
    "Australia":{"langs":["en"],"cities":{"Sydney":["SYD"],"Melbourne":["MEL"],"Brisbane":["BNE"],"Perth":["PER"],"Adelaide":["ADL"]}},
    "New Zealand":{"langs":["en"],"cities":{"Auckland":["AKL"],"Wellington":["WLG"],"Christchurch":["CHC"]}},
    "Fiji":{"langs":["en","fj","hi"],"cities":{"Nadi":["NAN"],"Suva":["SUV"]}}
}

# Currency map (same as your file)
CURRENCY_BY_COUNTRY = {
    "UK":"GBP","Germany":"EUR","France":"EUR","Spain":"EUR","Italy":"EUR","Portugal":"EUR","Netherlands":"EUR","Belgium":"EUR",
    "Switzerland":"CHF","Austria":"EUR","Greece":"EUR","Poland":"PLN","Czechia":"CZK","Hungary":"HUF","Slovakia":"EUR","Slovenia":"EUR","Croatia":"EUR",
    "Romania":"RON","Bulgaria":"BGN","Serbia":"RSD","Bosnia and Herzegovina":"BAM","Albania":"ALL","North Macedonia":"MKD","Montenegro":"EUR",
    "Lithuania":"EUR","Latvia":"EUR","Estonia":"EUR","Finland":"EUR","Sweden":"SEK","Norway":"NOK","Denmark":"DKK","Ireland":"EUR",
    "USA":"USD","Canada":"CAD","Mexico":"MXN","Brazil":"BRL","Argentina":"ARS","Chile":"CLP","Colombia":"COP","Peru":"PEN",
    "UAE":"AED","Saudi Arabia":"SAR","Qatar":"QAR","Jordan":"JOD","Lebanon":"LBP","Israel":"ILS","Pakistan":"PKR","India":"INR","Bangladesh":"BDT",
    "Sri Lanka":"LKR","Nepal":"NPR","Turkey":"TRY","Iran":"IRR","Morocco":"MAD","Tunisia":"TND","Algeria":"DZD","Egypt":"EGP","South Africa":"ZAR","Kenya":"KES",
    "China":"CNY","Hong Kong":"HKD","Taiwan":"TWD","Japan":"JPY","South Korea":"KRW","Thailand":"THB","Vietnam":"VND","Malaysia":"MYR",
    "Singapore":"SGD","Indonesia":"IDR","Philippines":"PHP","Cambodia":"KHR","Laos":"LAK","Myanmar":"MMK","Brunei":"BND",
    "Australia":"AUD","New Zealand":"NZD","Fiji":"FJD"
}

# Flatten for sampling
LOCATION_DATA = []
for country, c in COUNTRIES.items():
    for city, airports in c["cities"].items():
        LOCATION_DATA.append({"city": city, "country": country, "airports": airports, "languages": c["langs"]})

# Country phone codes (needed by fake_phone function)
COUNTRY_PHONE_CODE = {
    "Germany": "+49", "France": "+33", "UK": "+44", "USA": "+1", "Canada": "+1",
    "Italy": "+39", "Spain": "+34", "Netherlands": "+31", "Belgium": "+32",
    "Switzerland": "+41", "Austria": "+43", "Sweden": "+46", "Norway": "+47",
    "Denmark": "+45", "Finland": "+358", "Poland": "+48", "Czechia": "+420",
    "Hungary": "+36", "Slovakia": "+421", "Slovenia": "+386", "Croatia": "+385",
    "Romania": "+40", "Bulgaria": "+359", "Greece": "+30", "Turkey": "+90",
    "Russia": "+7", "China": "+86", "Japan": "+81", "South Korea": "+82",
    "India": "+91", "Australia": "+61", "Brazil": "+55", "Mexico": "+52",
    "Argentina": "+54", "Chile": "+56", "Colombia": "+57", "Peru": "+51"
}

# (Name banks, country→culture, phone codes, pools) – re-use from your current script.
# ↓↓↓ Paste your big lists EXACTLY like in the existing generator ↓↓↓
# NAMES, COUNTRY_CULTURE, SWISS_LANG_TO_CULT, COUNTRY_PHONE_CODE,
# GENDERS, VALUES, CAUSES, INTERESTS, LEARNING_STYLES, HUMOR_STYLES,
# DIET, RISK, NOISE, CLEAN, PACE, ACCOM_TYPES, ROOM_SETUP, TRANSPORT,
# CHRONO, ALCOHOL, SMOKING, TRIP_INTENTIONS, COMPANION_TYPES,
# COMPANION_GENDER_PREF, OPENING_MOVE_TEMPLATES, PROMPTS
# -----------------------------------------------------------

# ========== Helpers (unchanged API) ==========
def station_name(city, country):
    if country in ["Germany","Austria","Switzerland"]:
        return f"{city} Hbf"
    if country in ["Poland","Czechia","Hungary","Slovakia","Slovenia","Croatia","Serbia"]:
        return f"{city} Centralna"
    return f"{city} Central Station"

def _strip_accents(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c))

def split_name(full_name: str):
    parts = full_name.replace("  ", " ").split()
    first = parts[0] if parts else ""
    last = parts[-1] if len(parts) > 1 else ""
    middle = " ".join(parts[1:-1]) if len(parts) > 2 else ""
    return first, middle, last

def email_from_name(full_name: str):
    base = _strip_accents(full_name).replace("'", "").replace("’", "")
    parts = base.lower().replace("-", " ").split()
    first_token = parts[0] if parts else "user"
    last_token  = parts[-1] if len(parts) > 1 else "rm"
    return f"{first_token}.{last_token}@rovermitra.example"

def fake_phone(country: str):
    # COUNTRY_PHONE_CODE is imported in main() before we call this
    cc = COUNTRY_PHONE_CODE.get(country, "+1")
    return f"{cc}-{random.randint(200,999)}-{random.randint(200,999)}-{random.randint(1000,9999)}"

def fake_street():
    return f"{random.randint(12, 9999)} {random.choice(['Maple','Oak','Cedar','Lake','Sunset','Riverside','Hill','Garden','Park','King','Queen','Pine','Elm','Birch'])} {random.choice(['St','Ave','Rd','Blvd','Ln'])}"

def fake_postal(country: str):
    if country == "USA":
        return f"{random.randint(10000, 99999)}"
    if country in {"India","Pakistan","Bangladesh"}:
        return f"{random.randint(100000, 999999)}"
    if country == "UK":
        return random.choice(["SW1A 1AA","EC1A 1BB","B33 8TH","W1A 0AX","M1 1AE"])
    if country == "Canada":
        return random.choice(["M5V 2T6","H3Z 2Y7","V6B 1V2","K1A 0B1"])
    if country == "Ireland":
        return random.choice(["D02 X285","D08 VF8H","T12 X70A"])
    if country in {"Germany","Austria","Switzerland"}:
        return f"{random.randint(1000, 99999)}"
    if country == "Netherlands":
        return f"{random.randint(1000,9999)} {random.choice(['AA','AB','AC','BA','BB','BC'])}"
    if country == "France":
        return f"{random.randint(10000, 95999)}"
    if country == "Australia":
        return f"{random.randint(200, 9944)}"
    return f"{random.randint(1000, 999999)}"

def dob_from_age(age: int):
    today = date.today()
    year = today.year - age
    day_of_year = random.randint(1, 365)
    try:
        born = date(year, 1, 1) + timedelta(days=day_of_year-1)
    except Exception:
        born = date(year, 6, 15)
    return datetime(born.year, born.month, born.day).isoformat(timespec="milliseconds") + "Z"

# ==== Distributions to “guide” variety ====
def pick_with_shares(shares: dict):
    r = random.random()
    cum = 0.0
    for k, p in shares.items():
        cum += p
        if r <= cum:
            return k
    return list(shares.keys())[-1]

def choose_gender():
    return pick_with_shares(TARGET_GENDER_SPLIT)

def choose_pace():
    return pick_with_shares(TARGET_PACE_SPLIT)

def choose_budget_band():
    return pick_with_shares(TARGET_BUDGET_BAND)

def budget_for_country_and_band(country: str, band: str):
    # region anchors
    low=(25, 90); mid=(90, 180); lux=(180, 340)
    if country in ["Switzerland","Norway","Denmark"]:
        low=(90,150); mid=(150,240); lux=(240,400)
    if country in ["India","Pakistan","Bangladesh","Vietnam","Indonesia","Philippines"]:
        low=(20,60); mid=(60,120); lux=(120,220)
    lo, hi = {"budget":low, "mid":mid, "lux":lux}[band]
    return random.randint(lo, hi)

def sample_languages(default_langs):
    # Start with region langs + maybe English
    base = set(default_langs)
    if random.random() < 0.7:
        base.add("en")
    # Randomly include bonus languages to improve matchability
    for lang, p in GLOBAL_LANG_BONUS.items():
        if random.random() < p:
            base.add(lang)
    # pick 1..3
    k = random.randint(1, min(3, len(base)))
    return sorted(random.sample(list(base), k=k))

def values_pick(values):
    return random.sample(values, k=random.choice([2,3]))

def personality_block():
    ru = random.uniform
    return {
        "openness": round(ru(0.15, 0.98), 2),
        "conscientiousness": round(ru(0.15, 0.98), 2),
        "extraversion": round(ru(0.15, 0.98), 2),
        "agreeableness": round(ru(0.15, 0.98), 2),
        "neuroticism": round(ru(0.05, 0.85), 2),
        "creativity": round(ru(0.2, 0.98), 2),
        "empathy": round(ru(0.2, 0.98), 2)
    }

def short_bio(name, city, interests):
    verbs = ["exploring","learning","documenting","sharing","planning","daydreaming about"]
    two = ", ".join(random.sample(interests, k=min(2, len(interests)))) if interests else "good coffee and trains"
    return f"{name} from {city} loves {two} and enjoys {random.choice(verbs)} memorable trips."

def travel_prefs_block(pace, transport_pool, accom_pool):
    # ensure 1–3 modes; prefer train/plane to mimic Europe
    modes = []
    if "train" in transport_pool: modes.append("train")
    if "plane" in transport_pool and random.random() < 0.8: modes.append("plane")
    others = [m for m in transport_pool if m not in modes]
    modes += random.sample(others, k=max(0, random.randint(0,2) - len(modes)))
    modes = list(dict.fromkeys(modes))[:3] or ["train"]

    return {
        "pace": pace,
        "accommodation_types": random.sample(accom_pool, k=random.randint(1,2)),
        "room_setup": random.choice(["twin","double","2 rooms","dorm"]),
        "transport_allowed": modes,
        "must_haves": random.sample(
            ["wifi","private_bath","kitchen","workspace","near_station","quiet_room","breakfast"],
            k=random.randint(1,4)
        ),
    }

def diet_health_block(DIET):
    return {
        "diet": random.choice(DIET),
        "allergies": random.sample(["none","nuts","shellfish","pollen","gluten","lactose"], k=1),
        "accessibility": random.choice(["none","elevator_needed","reduced_mobility"])
    }

def comfort_block(ALCOHOL, SMOKING):
    return {
        "risk_tolerance": random.choice(["low","medium","high"]),
        "noise_tolerance": random.choice(["low","medium","high"]),
        "cleanliness_preference": random.choice(["low","medium","high"]),
        "chronotype": random.choice(["early bird","night owl","flexible"]),
        "alcohol": random.choice(ALCOHOL),
        "smoking": random.choice(SMOKING)
    }

def work_block():
    return {
        "remote_work_ok": random.random() < 0.55,
        "hours_online_needed": random.choice([0,1,2]),
        "wifi_quality_needed": random.choice(["normal","good","excellent"])
    }

def random_prompts(PROMPTS):
    out = []
    for section, opts in PROMPTS.items():
        if random.random() < 0.75:
            prompt = random.choice(opts)
            ans = random.choice([
                "coffee at sunrise","train-window playlists","budget spreadsheets","offline maps wizard",
                "asking locals for food tips","packing light","too many photos","always early for trains"
            ])
            out.append({"section": section, "prompt": prompt, "answer": ans})
    return out[:3]

def opening_move(OPENING_MOVE_TEMPLATES):
    return random.choice(OPENING_MOVE_TEMPLATES)

def companion_preferences(COMPANION_TYPES, COMPANION_GENDER_PREF):
    age_low = random.randint(20, 28)
    age_high = age_low + random.choice([6,8,10,12,15])
    return {
        "group_preference": random.choice(COMPANION_TYPES),
        "genders_ok": random.sample(COMPANION_GENDER_PREF, k=random.randint(1, len(COMPANION_GENDER_PREF))),
        "age_range_preferred": [age_low, age_high],
        "kids_ok_in_group": random.choice([True, False]),
        "pets_ok_in_group": random.choice([True, False])
    }

def lifestyle_block():
    return {
        "fitness_level": random.choice(["light","moderate","active"]),
        "daily_steps_goal": random.choice(["5-8k","8-12k","12k+"]),
        "caffeine": random.choice(["tea","coffee","either","none"]),
        "sleeping_habits": random.choice(["light sleeper","heavy sleeper","depends"]),
        "snoring_tolerance": random.choice(["fine","prefer not","earplugs ready"])
    }

def boundaries_safety():
    return {
        "quiet_hours": random.choice(["22:00–07:00","23:00–07:00","flexible"]),
        "photo_consent": random.choice(["ask first","ok for private albums","no faces on public socials"]),
        "social_media": random.choice(["share occasionally","no tagging please","fine with tagging"]),
        "substance_boundaries": random.choice(["no drugs","alcohol in moderation","no cigarettes in room"]),
        "share_location_with_group": random.choice([True, False])
    }

def maybe_faith_block():
    # Optional + private; only included when opted-in
    if random.random() < 0.25:  # adjust sampling rate as needed
        return {
            "consider_in_matching": True,
            "religion": random.choice([
                "Islam","Hindu","Christian","Jewish","Buddhist","Sikh","Other"
            ]),
            "policy": random.choice(["same_only","prefer_same","open"]),
            "visibility": "private"
        }
    return None

def culture_for(country, city_langs, COUNTRY_CULTURE, SWISS_LANG_TO_CULT):
    if country == "Switzerland":
        lang = next((l for l in city_langs if l in ("de","fr","it")), "de")
        return SWISS_LANG_TO_CULT.get(lang, "germanic")
    return COUNTRY_CULTURE.get(country, "anglo")

def build_name(country, gender, LOCATION_DATA, COUNTRY_CULTURE, SWISS_LANG_TO_CULT, NAMES):
    loc = next((x for x in LOCATION_DATA if x["country"] == country), None)
    city_langs = loc["languages"] if loc else ["en"]
    cult = culture_for(country, city_langs, COUNTRY_CULTURE, SWISS_LANG_TO_CULT)
    bank = NAMES.get(cult, NAMES["anglo"])

    if gender.lower().startswith("male"):
        given_pool = bank["male_given"]
    elif gender.lower().startswith("female"):
        given_pool = bank["female_given"]
    else:
        given_pool = random.choice([bank["male_given"], bank["female_given"]])

    given = random.choice(given_pool)
    family = random.choice(bank["surnames"])
    if cult in ("chinese","japanese","korean","vietnamese","thai"):
        full = f"{family} {given}"
    else:
        full = f"{given} {family}"
    return full

# ====== MAIN BUILD ======
def build_user(
    LOCATION_DATA, CURRENCY_BY_COUNTRY,
    GENDERS, VALUES, INTERESTS, DIET, ALCOHOL, SMOKING,
    ACCOM_TYPES, TRANSPORT, PROMPTS, OPENING_MOVE_TEMPLATES,
    COMPANION_TYPES, COMPANION_GENDER_PREF,
    COUNTRY_CULTURE, SWISS_LANG_TO_CULT, NAMES
):
    loc = random.choice(LOCATION_DATA)
    city, country, airports, langs = loc["city"], loc["country"], loc["airports"], loc["languages"]

    gender = choose_gender()
    name   = build_name(country, gender, LOCATION_DATA, COUNTRY_CULTURE, SWISS_LANG_TO_CULT, NAMES)

    email  = email_from_name(name)
    handle = "rm_" + _strip_accents(name).lower().replace(" ", ".").replace("-", ".")

    # Ages broader (19–62) so more pairings pass companion age bands
    age = random.randint(19, 62)

    # Languages tuned for matchability but region-aware
    languages = sample_languages(langs)

    # Interests: 6–10 to give LLM more hooks
    interests = random.sample(INTERESTS, k=random.randint(6, 10))

    # Mandatory IDs (your downstream expects user_id at top level)
    user_id = f"user_{uuid.uuid4().hex[:12]}"

    firstName, middleName, lastName = split_name(name)
    password = "Pass@" + str(random.randint(100000, 999999))
    signup_block = {
        "email": email,
        "password": password,
        "confirmPassword": password,
        "firstName": firstName,
        "lastName": lastName,
        "middleName": middleName,
        "phoneNumber": fake_phone(country),
        "dateOfBirth": dob_from_age(age),
        "address": fake_street(),
        "city": city,
        "state": "",
        "postalCode": fake_postal(country),
        "country": country
    }

    pace = choose_pace()
    band = choose_budget_band()
    amount = budget_for_country_and_band(country, band)
    currency = CURRENCY_BY_COUNTRY.get(country, "EUR")

    profile = {
        "user_id": user_id,
        **signup_block,

        "name": name,
        "age": age,
        "gender": gender,

        "contact": {"rovermitra_handle": handle, "email": email},

        "home_base": {
            "city": city, "country": country,
            "nearby_nodes": [station_name(city, country), random.choice(airports)],
            "willing_radius_km": random.choice([25, 40, 60, 80])
        },

        "languages": languages,
        "interests": interests,
        "values": values_pick(VALUES),
        "personality": personality_block(),
        "bio": short_bio(name, city, interests),

        "budget": {"type": "per_day", "amount": amount, "currency": currency, "split_rule": random.choice(["each_own","50/50","custom"])},
        "diet_health": diet_health_block(DIET),
        "comfort": comfort_block(ALCOHOL, SMOKING),

        "social": {
            "group_size_ok": random.choice([[1,2],[1,2,3],[2,3,4],[1,2,3,4,5]]),
            "learning_style": random.choice(["Visual","Auditory","Kinesthetic","Reading/Writing"]),
            "humor_style": random.choice(["Dry","Witty","Slapstick","Sarcastic","Playful","Observational"]),
            "dealbreakers_social": random.sample(
                ["no party hostels","no smoking room","no red-eye travel","no >2 transfers","no dorm rooms"],
                k=random.randint(0,2)
            )
        },

        "work": work_block(),
        "travel_prefs": travel_prefs_block(pace, TRANSPORT, ACCOM_TYPES),

        "trip_intentions": random.sample([
            "weekend city breaks","slow travel & cafes","country-hopping adventure","workation with good wifi",
            "food & culture deep dive","outdoor & hiking focus","beaches & warm weather","photography missions",
            "ski or snow trips","festival/ events chasing","history & museums track","road trips & scenic drives"
        ], k=random.randint(1,3)),

        "companion_preferences": companion_preferences(COMPANION_TYPES, COMPANION_GENDER_PREF),
        "lifestyle": lifestyle_block(),
        "boundaries_safety": boundaries_safety(),
        "causes": random.sample([
            "environmentalism","human rights","disability rights","immigrant rights","LGBTQ+ rights",
            "voter rights","reproductive rights","neurodiversity","end religious hate","stop asian hate","volunteering"
        ], k=random.randint(0,3)),
        "prompts": random_prompts(PROMPTS),
        "opening_move": opening_move(OPENING_MOVE_TEMPLATES),

        # Looking-for (same field you already output)
        "looking_to_meet": random.sample(["men","women","nonbinary","everyone"], k=1)[0],

        "privacy": {
            "share_profile_with_matches": True,
            "share_itinerary_with_group": random.random() < 0.9,
            "marketing_opt_in": random.random() < 0.3
        }
    }
    
    fb = maybe_faith_block()
    if fb:
        profile["faith"] = fb
    
    return profile

def degrade_profile_randomly(profile):
    # Keep same behavior: some lean profiles for realism (do NOT remove the new signup fields)
    removable = [
        "values","personality","bio","diet_health","comfort","social","work","travel_prefs",
        "trip_intentions","companion_preferences","lifestyle","boundaries_safety","causes","prompts","opening_move"
    ]
    k = random.randint(3, 6)
    for key in random.sample(removable, k=min(k, len(removable))):
        profile.pop(key, None)
    return profile

def main():
    # === Define constant pools ===
    # Basic constants
    GENDERS = ["Male", "Female", "Non-binary", "Other"]
    VALUES = ["adventure", "culture", "nature", "food", "relaxation", "learning", "social", "solo"]
    CAUSES = ["environment", "social", "education", "health", "arts", "technology"]
    INTERESTS = [
        "mountains", "lakes", "beaches", "museums", "old towns", "food tours", "street food", "coffee crawls",
        "scenic trains", "short hikes", "long hikes", "nightlife", "shopping", "photography", "architecture", "history",
        "skiing", "diving", "sailing", "cycling", "festivals", "thermal baths", "vineyards", "wildlife",
        "markets", "street art", "rooftop views", "bookstores", "local crafts", "castles"
    ]
    LEARNING_STYLES = ["Visual", "Auditory", "Kinesthetic", "Reading/Writing"]
    HUMOR_STYLES = ["Dry", "Witty", "Slapstick", "Sarcastic", "Playful", "Observational"]
    DIET = ["none", "vegetarian", "vegan", "halal", "kosher", "gluten-free", "no pork", "lactose-free", "pescatarian"]
    RISK = ["low", "medium", "high"]
    NOISE = ["low", "medium", "high"]
    CLEAN = ["low", "medium", "high"]
    PACE = ["relaxed", "balanced", "packed"]
    ACCOM_TYPES = ["hotel", "hostel", "airbnb", "camping", "guesthouse"]
    ROOM_SETUP = ["single", "shared", "private", "family"]
    TRANSPORT = ["train", "plane", "bus", "car", "bike", "walking"]
    CHRONO = ["morning", "afternoon", "evening", "night"]
    ALCOHOL = ["none", "light", "moderate", "heavy"]
    SMOKING = ["non-smoker", "occasional", "regular"]
    TRIP_INTENTIONS = ["leisure", "business", "adventure", "cultural", "romantic", "family"]
    COMPANION_TYPES = ["solo", "couple", "friends", "family", "group"]
    COMPANION_GENDER_PREF = ["any", "same", "mixed"]
    OPENING_MOVE_TEMPLATES = [
        "Hi! I'm {name} from {city}. Looking forward to exploring together!",
        "Hello! Excited to meet fellow travelers. I'm {name}.",
        "Hey there! {name} here, ready for some great adventures!"
    ]
    PROMPTS = {
        "travel_memories": [
            "Tell me about your favorite travel memory",
            "What's the most interesting place you've visited?",
            "Share a memorable travel experience"
        ],
        "food_recommendations": [
            "Share a local food recommendation",
            "What's the best meal you've had while traveling?",
            "Any must-try local dishes?"
        ],
        "travel_tips": [
            "What's your best travel tip?",
            "How do you plan your trips?",
            "What do you always pack?"
        ]
    }
    
    # Name banks by culture
    NAMES = {
        "anglo": {
            "male_given": ["James", "John", "Robert", "Michael", "William", "David", "Richard", "Charles", "Joseph", "Thomas"],
            "female_given": ["Mary", "Patricia", "Jennifer", "Linda", "Elizabeth", "Barbara", "Susan", "Jessica", "Sarah", "Karen"],
            "surnames": ["Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller", "Davis", "Rodriguez", "Martinez"]
        },
        "german": {
            "male_given": ["Hans", "Peter", "Klaus", "Wolfgang", "Jürgen", "Helmut", "Dieter", "Günther", "Manfred", "Rainer"],
            "female_given": ["Anna", "Maria", "Elisabeth", "Ursula", "Monika", "Petra", "Sabine", "Andrea", "Silvia", "Birgit"],
            "surnames": ["Müller", "Schmidt", "Schneider", "Fischer", "Weber", "Meyer", "Wagner", "Becker", "Schulz", "Hoffmann"]
        },
        "french": {
            "male_given": ["Pierre", "Jean", "Michel", "Philippe", "Alain", "Bernard", "André", "François", "Claude", "Daniel"],
            "female_given": ["Marie", "Nathalie", "Isabelle", "Sylvie", "Christine", "Monique", "Catherine", "Françoise", "Martine", "Patricia"],
            "surnames": ["Martin", "Bernard", "Thomas", "Petit", "Robert", "Richard", "Durand", "Dubois", "Moreau", "Laurent"]
        }
    }
    
    # Country culture mapping
    COUNTRY_CULTURE = {
        "Germany": "german", "Austria": "german", "Switzerland": "german",
        "France": "french", "Belgium": "french", "Luxembourg": "french",
        "UK": "anglo", "Ireland": "anglo", "USA": "anglo", "Canada": "anglo", "Australia": "anglo", "New Zealand": "anglo"
    }
    
    # Swiss language to culture mapping
    SWISS_LANG_TO_CULT = {"de": "german", "fr": "french", "it": "anglo", "en": "anglo"}
    

    users = []
    for _ in range(N_USERS):
        p = build_user(
            LOCATION_DATA, CURRENCY_BY_COUNTRY,
            GENDERS, VALUES, INTERESTS, DIET, ALCOHOL, SMOKING,
            ACCOM_TYPES, TRANSPORT, PROMPTS, OPENING_MOVE_TEMPLATES,
            COMPANION_TYPES, COMPANION_GENDER_PREF,
            COUNTRY_CULTURE, SWISS_LANG_TO_CULT, NAMES
        )
        if random.random() < LEAN_RATIO:
            p = degrade_profile_randomly(p)
        users.append(p)

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(users, f, indent=2, ensure_ascii=False)

    print(f"✅ Generated {len(users)} users → {OUT_PATH}")
    print(json.dumps(users[0], indent=2, ensure_ascii=False)[:1200] + "\n...")

if __name__ == "__main__":
    main()
