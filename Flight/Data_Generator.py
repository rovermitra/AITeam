import os
import json
import uuid
import random
from datetime import date, timedelta, datetime, time
from pathlib import Path

random.seed(52)

# ============================
# Config
# ============================
N_GROUPS = 10000  # change as needed
RICH_GROUP_RATIO = 0.75
OUT_PATH = "Flight/data/travel_group_requests_with_inventory_v2.json"

# ============================
# Expanded Location / Airports Catalog (60+ countries, 180+ cities)
# Keep pairs realistic: city belongs to its country, sensible airport codes
# ============================
COUNTRIES = {
    # EUROPE (core focus)
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
    "Austria": {
        "langs": ["de", "en"],
        "cities": {
            "Vienna": {"airports": ["VIE"]},
            "Salzburg": {"airports": ["SZG"]},
            "Graz": {"airports": ["GRZ"]},
            "Innsbruck": {"airports": ["INN"]}
        }
    },
    "France": {
        "langs": ["fr", "en"],
        "cities": {
            "Paris": {"airports": ["CDG", "ORY"]},
            "Lyon": {"airports": ["LYS"]},
            "Nice": {"airports": ["NCE"]},
            "Toulouse": {"airports": ["TLS"]},
            "Bordeaux": {"airports": ["BOD"]},
            "Marseille": {"airports": ["MRS"]},
            "Nantes": {"airports": ["NTE"]}
        }
    },
    "UK": {
        "langs": ["en"],
        "cities": {
            "London": {"airports": ["LHR", "LGW", "LCY", "LTN", "STN"]},
            "Manchester": {"airports": ["MAN"]},
            "Edinburgh": {"airports": ["EDI"]},
            "Glasgow": {"airports": ["GLA"]},
            "Birmingham": {"airports": ["BHX"]},
            "Bristol": {"airports": ["BRS"]}
        }
    },
    "Ireland": {"langs": ["en"], "cities": {"Dublin": {"airports": ["DUB"]}, "Cork": {"airports": ["ORK"]}}},
    "Netherlands": {"langs": ["nl", "en"], "cities": {"Amsterdam": {"airports": ["AMS"]}, "Rotterdam": {"airports": ["RTM"]}}},
    "Belgium": {"langs": ["nl", "fr", "en"], "cities": {"Brussels": {"airports": ["BRU"]}, "Antwerp": {"airports": ["ANR"]}}},
    "Spain": {
        "langs": ["es", "en"],
        "cities": {
            "Madrid": {"airports": ["MAD"]},
            "Barcelona": {"airports": ["BCN"]},
            "Valencia": {"airports": ["VLC"]},
            "Seville": {"airports": ["SVQ"]},
            "Malaga": {"airports": ["AGP"]},
            "Bilbao": {"airports": ["BIO"]}
        }
    },
    "Portugal": {"langs": ["pt", "en"], "cities": {"Lisbon": {"airports": ["LIS"]}, "Porto": {"airports": ["OPO"]}, "Faro": {"airports": ["FAO"]}}},
    "Italy": {
        "langs": ["it", "en"],
        "cities": {
            "Rome": {"airports": ["FCO", "CIA"]},
            "Milan": {"airports": ["MXP", "LIN", "BGY"]},
            "Venice": {"airports": ["VCE"]},
            "Florence": {"airports": ["FLR"]},
            "Naples": {"airports": ["NAP"]},
            "Turin": {"airports": ["TRN"]}
        }
    },
    "Greece": {"langs": ["el", "en"], "cities": {"Athens": {"airports": ["ATH"]}, "Thessaloniki": {"airports": ["SKG"]}, "Heraklion": {"airports": ["HER"]}}},
    "Poland": {"langs": ["pl", "en"], "cities": {"Warsaw": {"airports": ["WAW"]}, "Krakow": {"airports": ["KRK"]}, "Gdansk": {"airports": ["GDN"]}}},
    "Czechia": {"langs": ["cs", "en"], "cities": {"Prague": {"airports": ["PRG"]}, "Brno": {"airports": ["BRQ"]}}},
    "Denmark": {"langs": ["da", "en"], "cities": {"Copenhagen": {"airports": ["CPH"]}, "Aarhus": {"airports": ["AAR"]}}},
    "Sweden": {"langs": ["sv", "en"], "cities": {"Stockholm": {"airports": ["ARN", "BMA"]}, "Gothenburg": {"airports": ["GOT"]}}},
    "Norway": {"langs": ["no", "en"], "cities": {"Oslo": {"airports": ["OSL"]}, "Bergen": {"airports": ["BGO"]}}},
    "Finland": {"langs": ["fi", "en"], "cities": {"Helsinki": {"airports": ["HEL"]}}},
    "Iceland": {"langs": ["is", "en"], "cities": {"Reykjavik": {"airports": ["KEF"]}}},

    # AMERICAS
    "USA": {
        "langs": ["en", "es"],
        "cities": {
            "New York": {"airports": ["JFK", "EWR", "LGA"]},
            "San Francisco": {"airports": ["SFO"]},
            "Los Angeles": {"airports": ["LAX"]},
            "Chicago": {"airports": ["ORD"]},
            "Miami": {"airports": ["MIA"]},
            "Seattle": {"airports": ["SEA"]},
            "Boston": {"airports": ["BOS"]},
            "Denver": {"airports": ["DEN"]}
        }
    },
    "Canada": {"langs": ["en", "fr"], "cities": {"Toronto": {"airports": ["YYZ"]}, "Vancouver": {"airports": ["YVR"]}, "Montreal": {"airports": ["YUL"]}}},
    "Mexico": {"langs": ["es", "en"], "cities": {"Mexico City": {"airports": ["MEX"]}, "Guadalajara": {"airports": ["GDL"]}, "Monterrey": {"airports": ["MTY"]}}},
    "Brazil": {"langs": ["pt", "en"], "cities": {"Sao Paulo": {"airports": ["GRU", "CGH"]}, "Rio de Janeiro": {"airports": ["GIG", "SDU"]}, "Brasilia": {"airports": ["BSB"]}}},
    "Argentina": {"langs": ["es", "en"], "cities": {"Buenos Aires": {"airports": ["EZE", "AEP"]}, "Cordoba": {"airports": ["COR"]}}},
    "Chile": {"langs": ["es", "en"], "cities": {"Santiago": {"airports": ["SCL"]}}},
    "Peru": {"langs": ["es", "en"], "cities": {"Lima": {"airports": ["LIM"]}, "Cusco": {"airports": ["CUZ"]}}},
    "Colombia": {"langs": ["es", "en"], "cities": {"Bogota": {"airports": ["BOG"]}, "Medellin": {"airports": ["MDE"]}}},

    # MIDDLE EAST / AFRICA
    "UAE": {"langs": ["ar", "en"], "cities": {"Dubai": {"airports": ["DXB"]}, "Abu Dhabi": {"airports": ["AUH"]}}},
    "Saudi Arabia": {"langs": ["ar", "en"], "cities": {"Riyadh": {"airports": ["RUH"]}, "Jeddah": {"airports": ["JED"]}}},
    "Qatar": {"langs": ["ar", "en"], "cities": {"Doha": {"airports": ["DOH"]}}},
    "Israel": {"langs": ["he", "en"], "cities": {"Tel Aviv": {"airports": ["TLV"]}}},
    "Jordan": {"langs": ["ar", "en"], "cities": {"Amman": {"airports": ["AMM"]}}},
    "Morocco": {"langs": ["ar", "fr", "en"], "cities": {"Marrakech": {"airports": ["RAK"]}, "Casablanca": {"airports": ["CMN"]}}},
    "Egypt": {"langs": ["ar", "en"], "cities": {"Cairo": {"airports": ["CAI"]}},},
    "South Africa": {"langs": ["en", "af", "zu"], "cities": {"Cape Town": {"airports": ["CPT"]}, "Johannesburg": {"airports": ["JNB"]}}},
    "Kenya": {"langs": ["sw", "en"], "cities": {"Nairobi": {"airports": ["NBO"]}}},
    "Tanzania": {"langs": ["sw", "en"], "cities": {"Dar es Salaam": {"airports": ["DAR"]}, "Zanzibar": {"airports": ["ZNZ"]}}},
    "Nigeria": {"langs": ["en"], "cities": {"Lagos": {"airports": ["LOS"]}, "Abuja": {"airports": ["ABV"]}}},
    "Ghana": {"langs": ["en"], "cities": {"Accra": {"airports": ["ACC"]}}},

    # ASIA
    "Japan": {"langs": ["ja", "en"], "cities": {"Tokyo": {"airports": ["HND", "NRT"]}, "Osaka": {"airports": ["KIX", "ITM"]}, "Kyoto": {"airports": ["KIX"]}}},
    "South Korea": {"langs": ["ko", "en"], "cities": {"Seoul": {"airports": ["ICN", "GMP"]}, "Busan": {"airports": ["PUS"]}}},
    "China": {"langs": ["zh", "en"], "cities": {"Beijing": {"airports": ["PEK", "PKX"]}, "Shanghai": {"airports": ["PVG", "SHA"]}, "Shenzhen": {"airports": ["SZX"]}}},
    "India": {"langs": ["en", "hi"], "cities": {"Delhi": {"airports": ["DEL"]}, "Mumbai": {"airports": ["BOM"]}, "Bangalore": {"airports": ["BLR"]}, "Hyderabad": {"airports": ["HYD"]}}},
    "Pakistan": {"langs": ["ur", "en"], "cities": {"Karachi": {"airports": ["KHI"]}, "Lahore": {"airports": ["LHE"]}, "Islamabad": {"airports": ["ISB"]}}},
    "Bangladesh": {"langs": ["bn", "en"], "cities": {"Dhaka": {"airports": ["DAC"]}}},
    "Sri Lanka": {"langs": ["si", "ta", "en"], "cities": {"Colombo": {"airports": ["CMB"]}}},
    "Nepal": {"langs": ["ne", "en"], "cities": {"Kathmandu": {"airports": ["KTM"]}}},
    "Thailand": {"langs": ["th", "en"], "cities": {"Bangkok": {"airports": ["BKK", "DMK"]}, "Chiang Mai": {"airports": ["CNX"]}, "Phuket": {"airports": ["HKT"]}}},
    "Vietnam": {"langs": ["vi", "en"], "cities": {"Hanoi": {"airports": ["HAN"]}, "Ho Chi Minh City": {"airports": ["SGN"]}}},
    "Malaysia": {"langs": ["ms", "en"], "cities": {"Kuala Lumpur": {"airports": ["KUL"]}, "Penang": {"airports": ["PEN"]}}},
    "Singapore": {"langs": ["en"], "cities": {"Singapore": {"airports": ["SIN"]}}},
    "Indonesia": {"langs": ["id", "en"], "cities": {"Jakarta": {"airports": ["CGK"]}, "Bali": {"airports": ["DPS"]}}},
    "Philippines": {"langs": ["en", "tl"], "cities": {"Manila": {"airports": ["MNL"]}, "Cebu": {"airports": ["CEB"]}}},

    # OCEANIA
    "Australia": {"langs": ["en"], "cities": {"Sydney": {"airports": ["SYD"]}, "Melbourne": {"airports": ["MEL"]}, "Brisbane": {"airports": ["BNE"]}}},
    "New Zealand": {"langs": ["en"], "cities": {"Auckland": {"airports": ["AKL"]}, "Wellington": {"airports": ["WLG"]}, "Christchurch": {"airports": ["CHC"]}}}
}

# Flatten locations
LOCATION_DATA = []
for country, cinfo in COUNTRIES.items():
    for city, meta in cinfo["cities"].items():
        LOCATION_DATA.append({"city": city, "country": country, "languages": cinfo["langs"], "airports": meta["airports"]})

# ============================
# Pools (expanded)
# ============================
GENDERS = ["Male","Female","Non-binary","Other"]
EDU_LEVELS = ["High School","Bachelor","Master","PhD","Vocational","Associate"]
OCCUPS = [
    "Engineer","Designer","Artist","Researcher","Student","Developer","Manager","Teacher","Doctor",
    "Consultant","Freelancer","Entrepreneur","Nurse","Writer","Architect","Lawyer","Scientist",
    "Chef","Musician","Photographer","Guide","Data Scientist","Analyst","Marketer","Product Manager",
    "Content Creator","Videographer","Tour Operator","Civil Servant","Sales Lead"
]
MARITAL = ["Single","Married","In a relationship","Divorced","Widowed"]
LEARNING_STYLES = ["Visual","Auditory","Kinesthetic","Reading/Writing"]
HUMOR = ["Dry","Witty","Slapstick","Sarcastic","Playful","Observational"]

INTERESTS = [
    "mountains","lakes","beaches","museums","old towns","food tours","street food","coffee crawls",
    "scenic trains","short hikes","long hikes","nightlife","shopping","photography","architecture","history",
    "skiing","diving","sailing","cycling","festivals","thermal baths","vineyards","wildlife",
    "markets","street art","rooftop views","bookstores","local crafts","castles"
]

DIET = ["none","vegetarian","vegan","halal","kosher","gluten-free","no pork","lactose-free","pescatarian"]
RISK = ["low","medium","high"]
NOISE = ["low","medium","high"]
CLEAN = ["low","medium","high"]
TRANSPORT_MODES = ["train","plane","bus","car"]
ACCOM_TYPES = ["hotel","apartment","guesthouse","boutique","hostel"]
PRICE_BANDS = ["budget","mid-range","luxury"]
ROOM_SETUP = ["twin","double","2 rooms","dorm"]
PACE = ["relaxed","balanced","packed"]

AIRLINES = ["LH","LX","BA","AF","KL","U2","FR","TK","SN","SK","AY","OS","IB","AZ","EI","TP","LO","OK","JJ","DL","UA"]
CABINS = ["economy","premium_economy","business"]

RAIL_CARRIERS = ["DB","SBB","SNCF","Trenitalia","Renfe","ÖBB","NS","SNCB","CFL","SJ","DSB","VR","CP","CD"]
BUS_CARRIERS = ["FlixBus","ALSA","Ouibus","National Express","Megabus","Eurolines"]

CUISINES = [
    "Swiss","Italian","French","German","Spanish","Greek","Indian","Japanese","Thai","Turkish","Vegan",
    "Vegetarian","Seafood","Steakhouse","Cafe","Bakery","Lebanese","Moroccan","Mexican","Peruvian","Korean","Chinese",
    "Vietnamese","Malaysian","Indonesian","Middle Eastern"
]

CURRENCY_BY_COUNTRY = {
    "USA":"USD","Canada":"CAD","Mexico":"MXN","Brazil":"BRL","Argentina":"ARS","Chile":"CLP","Peru":"PEN","Colombia":"COP",
    "UK":"GBP","Ireland":"EUR","Germany":"EUR","France":"EUR","Spain":"EUR","Italy":"EUR","Portugal":"EUR",
    "Netherlands":"EUR","Belgium":"EUR","Switzerland":"CHF","Austria":"EUR","Poland":"PLN","Czechia":"CZK",
    "Denmark":"DKK","Sweden":"SEK","Norway":"NOK","Finland":"EUR","Greece":"EUR","Turkey":"TRY",
    "Japan":"JPY","South Korea":"KRW","China":"CNY","India":"INR","Pakistan":"PKR","Bangladesh":"BDT","Sri Lanka":"LKR","Nepal":"NPR",
    "Thailand":"THB","Vietnam":"VND","Malaysia":"MYR","Singapore":"SGD","Indonesia":"IDR","Philippines":"PHP",
    "UAE":"AED","Saudi Arabia":"SAR","Qatar":"QAR","Israel":"ILS","Jordan":"JOD","Morocco":"MAD","Egypt":"EGP",
    "South Africa":"ZAR","Kenya":"KES","Tanzania":"TZS","Nigeria":"NGN","Ghana":"GHS",
    "Australia":"AUD","New Zealand":"NZD","Iceland":"ISK"
}

# ============================
# Helpers
# ============================
def pick_location():
    return random.choice(LOCATION_DATA)

def future_date(days_from_today_min=10, days_from_today_max=240):
    start = date.today() + timedelta(days=random.randint(days_from_today_min, days_from_today_max))
    length = random.randint(4, 12)
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

def sample_languages(base_langs):
    base = list(set(base_langs + ["en"]))
    k = random.randint(1, min(3, len(base)))
    return random.sample(base, k=k)

def nearest_airport_for_city(city):
    for loc in LOCATION_DATA:
        if loc["city"] == city:
            return random.choice(loc["airports"])
    return "XXX"

def time_on_date(d, hhmm):
    h, m = map(int, hhmm.split(":"))
    return datetime.combine(d, time(h, m))

def random_dep_time():
    return random.choice(["06:40","07:10","08:30","09:45","10:15","12:05","14:30","16:20","18:15"]) 

def random_arrival(dep_dt, min_hours=1.0, max_hours=9.0):
    hours = random.uniform(min_hours, max_hours)
    return dep_dt + timedelta(hours=hours)

def city_currency(city):
    for loc in LOCATION_DATA:
        if loc["city"] == city:
            return CURRENCY_BY_COUNTRY.get(loc["country"], "EUR")
    return "EUR"

# ============================
# Builders (Traveler / Context)
# ============================
def build_traveler(idx):
    loc = pick_location()
    city, country, langs, airports = loc["city"], loc["country"], loc["languages"], loc["airports"]
    age = random.randint(20, 62)
    edu, occ = age_based_edu_occ(age)
    per_day = random.randint(60, 260)
    top_ints = random.sample(INTERESTS, k=random.randint(3, 7))

    traveler = {
        "name": f"Traveler-{idx}",
        "rovermitra_contact": {"channel": "RoverMitra", "user_handle": f"rm_{uuid.uuid4().hex[:8]}"},
        "home_base": {"city": city, "nearby_nodes": [f"{city} Hbf", random.choice(airports)], "willing_radius_km": random.choice([25,40,60,80])},
        "age": age,
        "gender": random.choice(GENDERS),
        "education": edu,
        "occupation": occ,
        "marital_status": random.choice(MARITAL),
        "learning_style": random.choice(LEARNING_STYLES),
        "humor_style": random.choice(HUMOR),
        "budget": {"type": "per_day", "amount": per_day, "currency": CURRENCY_BY_COUNTRY.get(country, "EUR"), "split_rule": random.choice(["each_own","50/50","custom"])},
        "transport": {"allowed_modes": list(set(random.sample(["train","plane","bus"], k=random.randint(1,3)))), "max_transfers": random.choice([1,2,3]), "max_leg_hours": random.choice([3,4,5,6]), "night_travel_ok": random.random() < 0.2},
        "accommodation": {"types": random.sample(ACCOM_TYPES, k=random.randint(1,2)), "price_band": random.choice(PRICE_BANDS), "room_setup": random.choice(ROOM_SETUP), "must_haves": random.sample(["private_bath","wifi","kitchen","workspace","near_station","quiet_room","breakfast"], k=random.randint(1,4))},
        "pace_and_interests": {"pace": random.choice(PACE), "top_interests": top_ints},
        "diet_health": {"diet": random.choice(DIET), "allergies": random.sample(["none","nuts","shellfish","pollen","gluten"], k=1), "accessibility": random.sample(["none","elevator_needed","reduced_mobility"], k=1)},
        "comfort": {"risk_tolerance": random.choice(RISK), "noise_tolerance": random.choice(NOISE), "cleanliness_preference": random.choice(CLEAN)},
        "work": {"hours_online_needed": random.choice([0,1,2]), "fixed_meetings": [], "wifi_quality_needed": random.choice(["normal","good","excellent"])},
        "documents": {"passport_valid_until": f"{random.randint(2027,2033)}-{random.randint(1,12):02d}-{random.randint(1,28):02d}", "visa_status": random.choice(["Schengen OK","Visa-free","eVisa needed","Visa required"]), "insurance": random.random() < 0.85},
        "languages": sample_languages(langs),
        "dealbreakers": random.sample(["no red-eye travel","no >2 transfers","no dorm rooms","no overnight trains","no party hostels","no smoking room"], k=random.randint(0,2))
    }
    return traveler


def build_trip_context(group_idx):
    # Destinations: sometimes bias to Switzerland, otherwise mix across catalog
    if random.random() < 0.25:
        destinations = random.sample(["Zurich","Lucerne","Interlaken","Zermatt","Basel","Bern","Geneva"], k=random.randint(1,3))
    else:
        # choose unique cities
        destinations = []
        for _ in range(random.randint(1,3)):
            destinations.append(random.choice(LOCATION_DATA)["city"])
        destinations = list(dict.fromkeys(destinations))
    start, end, pref_len = future_date(10, 260)
    w_cost, w_time, w_comf, w_scene, w_co2 = normalized_weights()
    return {
        "title": f"Group-{group_idx} Trip",
        "destinations": destinations,
        "date_window": {
            "earliest_departure": str(start),
            "latest_return": str(end),
            "preferred_trip_length_days": pref_len,
            "blackout_dates": []
        },
        "meeting_strategy_allowed": ["en_route_midpoint","at_destination","origin_A","origin_B"],
        "meeting_priority_objective": random.choice(["minimize_total_travel_time_and_cost","minimize_total_time","minimize_total_cost"]),
        "itinerary_style": random.choice(["anchor_city","multi-stop"]),
        "min_time_per_stop_hours": random.choice([12, 24, 36, 48]),
        "luggage": {"carry_on_only": random.random() < 0.6, "special_gear": random.sample(["camera","skis","hiking poles","drone","none"], k=1)},
        "co2_preference": random.random() < 0.5,
        "tradeoff_weights": {"cost": w_cost, "time": w_time, "comfort": w_comf, "scenery": w_scene, "co2": w_co2},
        "hard_constraints": {"earliest_departure_time_local": random.choice(["07:00","08:30","09:00"]), "latest_arrival_time_local": random.choice(["20:00","21:30","22:00"]), "max_daily_travel_hours": random.choice([5,6,7]), "max_transfers": random.choice([1,2,3]), "room_setup": random.choice(["twin","double","2 rooms"])},
        "output_preferences": {"detail_level": "day-by-day","include_booking_links": True,"currency": "EUR","units": "metric","share_to_rovermitra_chat": True}
    }

# ============================
# Inventory Generators (Flights / Trains / Hotels / Restaurants / Activities)
# ============================

def gen_flight_offers_for_traveler(traveler, meet_city, depart_date, return_date):
    origin_city = traveler["home_base"]["city"]
    origin_ap = nearest_airport_for_city(origin_city)
    meet_ap = nearest_airport_for_city(meet_city)
    currency = city_currency(meet_city)

    def mk_flight(_from, _to, d):
        dep_str = random_dep_time()
        dep_dt = time_on_date(d, dep_str)
        arr_dt = random_arrival(dep_dt, min_hours=1.0, max_hours=6.5)
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
    inbound = [mk_flight(meet_ap, origin_ap, return_date) for _ in range(random.randint(2,4))]
    return {"outbound": outbound, "return": inbound}


def gen_train_offers(origin_city, dest_city, any_date):
    carrier = random.choice(RAIL_CARRIERS)
    dep_str = random_dep_time()
    dep_dt = time_on_date(any_date, dep_str)
    # EU intra-city typical durations
    base_dur = random.uniform(1.5, 7.5)
    arr_dt = dep_dt + timedelta(hours=base_dur)
    price = random.randint(18, 120)
    return [{
        "id": f"rail_{uuid.uuid4().hex[:8]}",
        "carrier": carrier,
        "from": origin_city,
        "to": dest_city,
        "depart_iso": dep_dt.isoformat(),
        "arrive_iso": arr_dt.isoformat(),
        "transfers": random.choice([0,1,2]),
        "duration_min": int(base_dur*60),
        "price": price,
        "currency": city_currency(dest_city),
        "co2_kg": random.randint(3, 20)
    }]


def gen_hotels_for_city(city, nights, price_band="mid-range"):
    currency = city_currency(city)
    hotels = []
    for i in range(random.randint(4,7)):
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
            "suits": random.sample(interests, k=min(len(interests), random.randint(1,3))),
            "price": random.randint(0, 60),
            "currency": city_currency(city)
        })
    return items

# ============================
# Draft plan composer
# ============================

def compose_draft_plan(travelers, trip_ctx):
    destinations = list(dict.fromkeys(trip_ctx["destinations"]))  # ensure unique
    meet_city = destinations[0]
    start = date.fromisoformat(trip_ctx["date_window"]["earliest_departure"]) 
    end = date.fromisoformat(trip_ctx["date_window"]["latest_return"]) 
    length = (end - start).days or trip_ctx["date_window"]["preferred_trip_length_days"]

    # Nights distribution across cities
    if len(destinations) == 1:
        nights_per_city = {destinations[0]: max(3, length)}
    else:
        base = max(1, length // len(destinations))
        nights_per_city = {city: base for city in destinations}
        leftover = length - sum(nights_per_city.values())
        for i in range(leftover):
            nights_per_city[destinations[i % len(destinations)]] += 1

    # Inventories per city
    hotels_inventory = []
    restaurants_inventory = []
    activities_inventory = []
    for city in destinations:
        pbands = [t.get("accommodation",{}).get("price_band","mid-range") for t in travelers]
        band = random.choice(pbands) if pbands else "mid-range"
        hotels_inventory.append(gen_hotels_for_city(city, nights_per_city[city], price_band=band))
        diet = travelers[0].get("diet_health", {}).get("diet", "none") if travelers else "none"
        restaurants_inventory.append({"city": city, "options": gen_restaurants_for_city(city, diet=diet)})
        all_interests = sorted({i for t in travelers for i in t.get("pace_and_interests",{}).get("top_interests", [])}) or INTERESTS
        activities_inventory.append({"city": city, "options": gen_activities_for_city(city, all_interests)})

    # Flights per traveler
    flights_by_traveler = {}
    for t in travelers:
        flights_by_traveler[t["rovermitra_contact"]["user_handle"]] = gen_flight_offers_for_traveler(
            t, meet_city, start, end
        )

    # Choose cheapest outbound/return per traveler
    chosen_flights = []
    for uid, offers in flights_by_traveler.items():
        ob = sorted(offers["outbound"], key=lambda x: x["price"])[0]
        ib = sorted(offers["return"], key=lambda x: x["price"])[0]
        chosen_flights.append({"user_handle": uid, "outbound": ob["id"], "return": ib["id"]})

    # Simple inter-city ground offers (train) between consecutive destinations
    intercity_ground = []
    cur_date = start
    for i in range(len(destinations)-1):
        origin = destinations[i]
        dest = destinations[i+1]
        offers = gen_train_offers(origin, dest, cur_date)
        intercity_ground.append({"origin": origin, "dest": dest, "offers": offers})
        cur_date += timedelta(days=nights_per_city[origin])

    # Hotel reservations: pick first option per city
    hotel_reservations = []
    for inv in hotels_inventory:
        option = sorted(inv["options"], key=lambda h: (h["distance_to_station_km"], h["price_per_night"]))[0]
        hotel_reservations.append({
            "city": inv["city"],
            "hotel_id": option["id"],
            "nights": inv["nights"],
            "status": "hold"
        })

    # Restaurant holds: 1 per city at 19:30
    restaurant_reservations = []
    for rinv in restaurants_inventory:
        option = sorted(rinv["options"], key=lambda r: (-r["rating"], r["distance_to_center_km"]))[0]
        restaurant_reservations.append({
            "city": rinv["city"],
            "restaurant_id": option["id"],
            "when": f"{start.isoformat()}T19:30:00",
            "status": "hold"
        })

    # Itinerary skeleton
    itinerary = []
    cur_day = start
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
            itinerary.append({"date": cur_day.isoformat(), "base": city, "plan": plan})
            cur_day += timedelta(days=1)

    # Simple seasonal/weather hint
    month = start.month
    season_hint = "Pack layers; mountain weather changes fast." if month in [5,6,7,8,9] else "Expect colder temps; consider thermal layers."

    return {
        "meeting_plan": {"meet_at": meet_city, "rationale": "Meet at first destination to simplify and minimize connections."},
        "itinerary": itinerary,
        "intercity_ground_offers": intercity_ground,
        "flight_offers": flights_by_traveler,
        "chosen_flights": chosen_flights,
        "hotel_inventory": hotels_inventory,
        "hotel_reservations": hotel_reservations,
        "restaurant_inventory": restaurants_inventory,
        "restaurant_reservations": restaurant_reservations,
        "activities_inventory": activities_inventory,
        "hints": {"weather": season_hint, "safety": "Standard city precautions; watch bags in crowded areas."}
    }

# ============================
# Lean vs Rich groups
# ============================

def degrade_group_randomly(group):
    for t in group["travelers"]:
        if random.random() < 0.35:
            t.pop("learning_style", None)
        if random.random() < 0.35:
            t.pop("humor_style", None)
        if random.random() < 0.25:
            t.pop("dealbreakers", None)
        if random.random() < 0.25 and "budget" in t:
            t["budget"]["type"] = "total"
            t["budget"]["amount"] = t["budget"]["amount"] * random.randint(4, 9)
    tc = group["trip_context"]
    if random.random() < 0.35:
        tc.pop("co2_preference", None)
    if random.random() < 0.30:
        tc.pop("min_time_per_stop_hours", None)
    if random.random() < 0.30:
        tc.pop("luggage", None)
    return group

# ============================
# Generate groups
# ============================

def group_size():
    choices = list(range(1, 16))
    weights = [0.22, 0.28, 0.15, 0.10, 0.07, 0.05, 0.04, 0.03, 0.025, 0.02, 0.015, 0.01, 0.01, 0.01, 0.01]
    return random.choices(choices, weights=weights, k=1)[0]


groups = []
for g in range(N_GROUPS):
    size = group_size()
    travelers = [build_traveler(idx=f"{g}-{i}") for i in range(size)]
    trip_ctx = build_trip_context(g)
    draft_plan = compose_draft_plan(travelers, trip_ctx)

    group = {
        "group_id": str(uuid.uuid4()),
        "rovermitra_chat": {"room_id": f"rmr_{uuid.uuid4().hex[:10]}", "created_at": datetime.utcnow().isoformat() + "Z"},
        "trip_context": trip_ctx,
        "travelers": travelers,
        "draft_plan": draft_plan
    }

    if random.random() > RICH_GROUP_RATIO:
        group = degrade_group_randomly(group)

    groups.append(group)

os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
with open(OUT_PATH, "w", encoding="utf-8") as f:
    json.dump(groups, f, indent=2, ensure_ascii=False)

sizes = {}
for g in groups:
    s = len(g["travelers"])
    sizes[s] = sizes.get(s, 0) + 1
print(f"✅ Generated {len(groups)} groups with expanded countries + flights/trains/hotels/restaurants/activities → {OUT_PATH}")
print("Group-size distribution:", dict(sorted(sizes.items())))
print("Example (truncated):")
print(json.dumps(groups[0], indent=2)[:1400], "...\n")

p = Path("Flight/data/travel_group_requests_with_inventory_v2.json")
with p.open("r", encoding="utf-8") as f:
    data = json.load(f)

print(len(data))          # number of groups

# quick sanity peek
print(data[0]["group_id"], len(data[0]["travelers"]))