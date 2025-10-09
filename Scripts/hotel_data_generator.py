#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Hotel Data Generator for RoverMitra
Generates realistic hotel data for travel groups
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

OUT_PATH = "Hotels/data/hotel_data.json"
N_HOTELS = int(os.getenv("RM_GEN_HOTELS", "800"))

# Hotel categories and amenities
HOTEL_TYPES = [
    "Boutique Hotel", "Business Hotel", "Resort", "Hostel", "Bed & Breakfast",
    "Luxury Hotel", "Budget Hotel", "Eco Hotel", "Historic Hotel", "Design Hotel"
]

AMENITIES = [
    "Free WiFi", "Air Conditioning", "Room Service", "Fitness Center", "Spa",
    "Restaurant", "Bar", "Pool", "Parking", "Airport Shuttle", "Concierge",
    "Business Center", "Pet Friendly", "Non-Smoking", "Laundry Service",
    "Currency Exchange", "Tour Desk", "Garden", "Terrace", "Library"
]

CITIES = [
    {"name": "Berlin", "country": "Germany", "district": "Mitte"},
    {"name": "Munich", "country": "Germany", "district": "Altstadt"},
    {"name": "Paris", "country": "France", "district": "Marais"},
    {"name": "London", "country": "UK", "district": "Covent Garden"},
    {"name": "Rome", "country": "Italy", "district": "Trastevere"},
    {"name": "Madrid", "country": "Spain", "district": "Salamanca"},
    {"name": "Amsterdam", "country": "Netherlands", "district": "Jordaan"},
    {"name": "Zurich", "country": "Switzerland", "district": "Old Town"},
    {"name": "Vienna", "country": "Austria", "district": "Innere Stadt"},
    {"name": "Prague", "country": "Czechia", "district": "Mala Strana"}
]

def generate_hotel():
    """Generate a single hotel record"""
    city = random.choice(CITIES)
    hotel_type = random.choice(HOTEL_TYPES)
    
    # Price range based on hotel type
    if hotel_type in ["Luxury Hotel", "Resort"]:
        base_price = random.randint(150, 400)
    elif hotel_type in ["Budget Hotel", "Hostel"]:
        base_price = random.randint(30, 80)
    else:
        base_price = random.randint(80, 200)
    
    # Star rating
    if hotel_type in ["Luxury Hotel", "Resort"]:
        stars = random.choice([4, 5])
    elif hotel_type in ["Budget Hotel", "Hostel"]:
        stars = random.choice([2, 3])
    else:
        stars = random.choice([3, 4])
    
    # Available rooms
    total_rooms = random.randint(20, 200)
    available_rooms = random.randint(5, total_rooms)
    
    return {
        "hotel_id": f"HT_{uuid.uuid4().hex[:8]}",
        "name": f"{random.choice(['Grand', 'Royal', 'Plaza', 'Central', 'Garden', 'Park'])} {hotel_type}",
        "type": hotel_type,
        "stars": stars,
        "description": f"A {hotel_type.lower()} located in the heart of {city['district']}, {city['name']}",
        "location": {
            "city": city["name"],
            "country": city["country"],
            "district": city["district"],
            "address": f"{random.randint(1, 200)} {random.choice(['Main', 'Central', 'Park', 'Garden', 'Royal'])} Street",
            "postal_code": f"{random.randint(10000, 99999)}",
            "coordinates": {
                "lat": round(random.uniform(40.0, 60.0), 6),
                "lng": round(random.uniform(-10.0, 20.0), 6)
            }
        },
        "pricing": {
            "price_per_night_euro": base_price,
            "currency": "EUR",
            "taxes_included": random.choice([True, False]),
            "breakfast_included": random.choice([True, False])
        },
        "availability": {
            "total_rooms": total_rooms,
            "available_rooms": available_rooms,
            "check_in_time": "15:00",
            "check_out_time": "11:00"
        },
        "amenities": random.sample(AMENITIES, k=random.randint(5, 12)),
        "room_types": [
            {
                "type": "Standard Room",
                "max_occupancy": random.randint(1, 3),
                "price_per_night": base_price,
                "available": random.randint(2, 10)
            },
            {
                "type": "Deluxe Room",
                "max_occupancy": random.randint(2, 4),
                "price_per_night": base_price + random.randint(20, 50),
                "available": random.randint(1, 5)
            }
        ],
        "rating": {
            "overall": round(random.uniform(3.0, 5.0), 1),
            "cleanliness": round(random.uniform(3.0, 5.0), 1),
            "location": round(random.uniform(3.0, 5.0), 1),
            "service": round(random.uniform(3.0, 5.0), 1),
            "value": round(random.uniform(3.0, 5.0), 1),
            "total_reviews": random.randint(50, 1000)
        },
        "policies": {
            "cancellation": random.choice(["Free cancellation", "Non-refundable", "Partial refund"]),
            "pets_allowed": random.choice([True, False]),
            "smoking_allowed": random.choice([True, False]),
            "age_restriction": random.choice([None, 18, 21])
        },
        "contact": {
            "phone": f"+{random.randint(1, 99)} {random.randint(100000000, 999999999)}",
            "email": f"info@{random.choice(['hotel', 'grand', 'royal'])}.com",
            "website": f"https://www.{random.choice(['hotel', 'grand', 'royal'])}.com"
        }
    }

def main():
    """Generate hotel data"""
    print(f"🏨 Generating {N_HOTELS} hotels...")
    
    hotels = []
    for _ in range(N_HOTELS):
        hotels.append(generate_hotel())
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    
    # Write to file
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(hotels, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Generated {len(hotels)} hotels → {OUT_PATH}")
    
    # Show sample
    if hotels:
        sample = hotels[0]
        print(f"Sample hotel: {sample['name']} in {sample['location']['city']}")

if __name__ == "__main__":
    main()
