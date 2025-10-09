#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Flight Data Generator for RoverMitra
Generates realistic flight data for travel groups
"""

import os
import json
import uuid
import random
from datetime import datetime, timedelta
from pathlib import Path

# Configuration
SEED = int(os.getenv("RM_GEN_SEED", "77"))
random.seed(SEED)

OUT_PATH = "Flight/data/flight_data.json"
N_FLIGHTS = int(os.getenv("RM_GEN_FLIGHTS", "1000"))

# Airport codes and cities
AIRPORTS = {
    "BER": {"city": "Berlin", "country": "Germany"},
    "MUC": {"city": "Munich", "country": "Germany"},
    "HAM": {"city": "Hamburg", "country": "Germany"},
    "FRA": {"city": "Frankfurt", "country": "Germany"},
    "CDG": {"city": "Paris", "country": "France"},
    "ORY": {"city": "Paris", "country": "France"},
    "LHR": {"city": "London", "country": "UK"},
    "LGW": {"city": "London", "country": "UK"},
    "FCO": {"city": "Rome", "country": "Italy"},
    "MXP": {"city": "Milan", "country": "Italy"},
    "MAD": {"city": "Madrid", "country": "Spain"},
    "BCN": {"city": "Barcelona", "country": "Spain"},
    "AMS": {"city": "Amsterdam", "country": "Netherlands"},
    "ZUR": {"city": "Zurich", "country": "Switzerland"},
    "VIE": {"city": "Vienna", "country": "Austria"},
    "PRG": {"city": "Prague", "country": "Czechia"},
    "WAW": {"city": "Warsaw", "country": "Poland"},
    "BUD": {"city": "Budapest", "country": "Hungary"},
    "IST": {"city": "Istanbul", "country": "Turkey"},
    "ATH": {"city": "Athens", "country": "Greece"}
}

AIRLINES = [
    "Lufthansa", "Air France", "British Airways", "KLM", "Swiss",
    "Austrian Airlines", "SAS", "Finnair", "Iberia", "Alitalia",
    "Turkish Airlines", "Aegean Airlines", "Ryanair", "EasyJet",
    "Wizz Air", "Eurowings", "TAP Air Portugal", "LOT Polish Airlines"
]

def generate_flight():
    """Generate a single flight record"""
    origin = random.choice(list(AIRPORTS.keys()))
    destination = random.choice([airport for airport in AIRPORTS.keys() if airport != origin])
    
    # Generate departure time (next 30 days)
    departure = datetime.now() + timedelta(days=random.randint(1, 30), hours=random.randint(0, 23))
    
    # Flight duration (1-8 hours)
    duration_hours = random.randint(1, 8)
    arrival = departure + timedelta(hours=duration_hours)
    
    # Price range based on distance
    base_price = random.randint(50, 500)
    if duration_hours > 4:
        base_price += random.randint(100, 300)
    
    return {
        "flight_id": f"FL_{uuid.uuid4().hex[:8]}",
        "airline": random.choice(AIRLINES),
        "origin": {
            "code": origin,
            "city": AIRPORTS[origin]["city"],
            "country": AIRPORTS[origin]["country"]
        },
        "destination": {
            "code": destination,
            "city": AIRPORTS[destination]["city"],
            "country": AIRPORTS[destination]["country"]
        },
        "departure_time": departure.isoformat(),
        "arrival_time": arrival.isoformat(),
        "duration_hours": duration_hours,
        "price_euro": base_price,
        "available_seats": random.randint(5, 200),
        "aircraft_type": random.choice(["Boeing 737", "Airbus A320", "Boeing 777", "Airbus A330"]),
        "flight_class": random.choice(["Economy", "Premium Economy", "Business", "First"])
    }

def main():
    """Generate flight data"""
    print(f"✈️ Generating {N_FLIGHTS} flights...")
    
    flights = []
    for _ in range(N_FLIGHTS):
        flights.append(generate_flight())
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    
    # Write to file
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(flights, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Generated {len(flights)} flights → {OUT_PATH}")
    
    # Show sample
    if flights:
        sample = flights[0]
        print(f"Sample flight: {sample['airline']} {sample['origin']['code']} → {sample['destination']['code']}")

if __name__ == "__main__":
    main()
