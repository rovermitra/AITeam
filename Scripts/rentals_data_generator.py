#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Rentals Data Generator for RoverMitra
Generates realistic rental data for travel groups
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

OUT_PATH = "Rentals/data/rental_data.json"
N_RENTALS = int(os.getenv("RM_GEN_RENTALS", "600"))

# Rental types and categories
RENTAL_TYPES = [
    "Apartment", "House", "Studio", "Villa", "Cottage", "Loft", "Penthouse", "Townhouse"
]

RENTAL_CATEGORIES = {
    "Budget": {"price_range": (30, 80), "amenities": ["Basic Kitchen", "WiFi", "Heating"]},
    "Mid-range": {"price_range": (80, 150), "amenities": ["Full Kitchen", "WiFi", "AC", "Washing Machine"]},
    "Luxury": {"price_range": (150, 400), "amenities": ["Full Kitchen", "WiFi", "AC", "Washing Machine", "Pool", "Garden", "Parking"]}
}

AMENITIES = [
    "WiFi", "Air Conditioning", "Heating", "Kitchen", "Washing Machine", "Dryer",
    "Parking", "Pool", "Garden", "Terrace", "Balcony", "Fireplace", "TV", "Sound System",
    "Coffee Maker", "Dishwasher", "Microwave", "Refrigerator", "Oven", "Stove"
]

CITIES = [
    {"name": "Berlin", "country": "Germany", "neighborhood": "Kreuzberg"},
    {"name": "Munich", "country": "Germany", "neighborhood": "Schwabing"},
    {"name": "Paris", "country": "France", "neighborhood": "Le Marais"},
    {"name": "London", "country": "UK", "neighborhood": "Shoreditch"},
    {"name": "Rome", "country": "Italy", "neighborhood": "Trastevere"},
    {"name": "Madrid", "country": "Spain", "neighborhood": "Malasaña"},
    {"name": "Amsterdam", "country": "Netherlands", "neighborhood": "Jordaan"},
    {"name": "Zurich", "country": "Switzerland", "neighborhood": "Niederdorf"},
    {"name": "Vienna", "country": "Austria", "neighborhood": "Neubau"},
    {"name": "Prague", "country": "Czechia", "neighborhood": "Vinohrady"}
]

def generate_rental():
    """Generate a single rental record"""
    city = random.choice(CITIES)
    rental_type = random.choice(RENTAL_TYPES)
    category = random.choice(list(RENTAL_CATEGORIES.keys()))
    
    # Price based on category
    price_range = RENTAL_CATEGORIES[category]["price_range"]
    base_price = random.randint(price_range[0], price_range[1])
    
    # Size based on rental type
    if rental_type in ["Studio", "Loft"]:
        size_sqm = random.randint(25, 50)
        bedrooms = 0
    elif rental_type in ["Apartment", "Penthouse"]:
        size_sqm = random.randint(50, 120)
        bedrooms = random.randint(1, 3)
    else:  # House, Villa, Cottage, Townhouse
        size_sqm = random.randint(80, 200)
        bedrooms = random.randint(2, 5)
    
    bathrooms = random.randint(1, bedrooms + 1)
    max_guests = bedrooms * 2 + random.randint(0, 2)
    
    # Available dates (next 30 days)
    available_from = datetime.now() + timedelta(days=random.randint(1, 7))
    available_until = available_from + timedelta(days=random.randint(7, 30))
    
    return {
        "rental_id": f"RT_{uuid.uuid4().hex[:8]}",
        "title": f"{rental_type} in {city['neighborhood']}, {city['name']}",
        "type": rental_type,
        "category": category,
        "description": f"Beautiful {rental_type.lower()} in the heart of {city['neighborhood']}, perfect for exploring {city['name']}",
        "location": {
            "city": city["name"],
            "country": city["country"],
            "neighborhood": city["neighborhood"],
            "address": f"{random.randint(1, 200)} {random.choice(['Main', 'Central', 'Park', 'Garden', 'Royal'])} Street",
            "postal_code": f"{random.randint(10000, 99999)}",
            "coordinates": {
                "lat": round(random.uniform(40.0, 60.0), 6),
                "lng": round(random.uniform(-10.0, 20.0), 6)
            }
        },
        "property_details": {
            "size_sqm": size_sqm,
            "bedrooms": bedrooms,
            "bathrooms": bathrooms,
            "max_guests": max_guests,
            "floor": random.randint(1, 10) if rental_type in ["Apartment", "Penthouse", "Loft"] else None
        },
        "pricing": {
            "price_per_night_euro": base_price,
            "currency": "EUR",
            "cleaning_fee": random.randint(20, 50),
            "security_deposit": random.randint(100, 500),
            "minimum_nights": random.randint(1, 7),
            "maximum_nights": random.randint(30, 90)
        },
        "availability": {
            "available_from": available_from.isoformat(),
            "available_until": available_until.isoformat(),
            "instant_booking": random.choice([True, False])
        },
        "amenities": random.sample(AMENITIES, k=random.randint(3, 8)),
        "host": {
            "name": f"{random.choice(['Alex', 'Sarah', 'Mike', 'Emma', 'David', 'Lisa'])}",
            "rating": round(random.uniform(4.0, 5.0), 1),
            "response_rate": random.randint(80, 100),
            "response_time": random.choice(["within an hour", "within a few hours", "within a day"]),
            "superhost": random.choice([True, False])
        },
        "rating": {
            "overall": round(random.uniform(3.5, 5.0), 1),
            "cleanliness": round(random.uniform(3.5, 5.0), 1),
            "location": round(random.uniform(3.5, 5.0), 1),
            "communication": round(random.uniform(3.5, 5.0), 1),
            "check_in": round(random.uniform(3.5, 5.0), 1),
            "value": round(random.uniform(3.5, 5.0), 1),
            "total_reviews": random.randint(5, 200)
        },
        "policies": {
            "cancellation": random.choice(["Flexible", "Moderate", "Strict"]),
            "pets_allowed": random.choice([True, False]),
            "smoking_allowed": random.choice([True, False]),
            "parties_allowed": random.choice([True, False]),
            "age_restriction": random.choice([None, 18, 21])
        },
        "images": [
            f"https://example.com/images/{uuid.uuid4().hex[:8]}.jpg"
            for _ in range(random.randint(3, 8))
        ]
    }

def main():
    """Generate rental data"""
    print(f"🏠 Generating {N_RENTALS} rentals...")
    
    rentals = []
    for _ in range(N_RENTALS):
        rentals.append(generate_rental())
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    
    # Write to file
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(rentals, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Generated {len(rentals)} rentals → {OUT_PATH}")
    
    # Show sample
    if rentals:
        sample = rentals[0]
        print(f"Sample rental: {sample['title']}")

if __name__ == "__main__":
    main()
