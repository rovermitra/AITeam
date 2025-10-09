#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Restaurant Data Generator for RoverMitra
Generates realistic restaurant data for travel groups
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

OUT_PATH = "Restaurants/data/restaurant_data.json"
N_RESTAURANTS = int(os.getenv("RM_GEN_RESTAURANTS", "700"))

# Restaurant categories and cuisines
RESTAURANT_TYPES = [
    "Fine Dining", "Casual Dining", "Fast Food", "Cafe", "Bistro", "Bar & Grill",
    "Food Truck", "Street Food", "Buffet", "Family Restaurant", "Seafood", "Steakhouse"
]

CUISINES = [
    "Italian", "French", "German", "Spanish", "Mediterranean", "Asian", "Chinese",
    "Japanese", "Thai", "Indian", "Mexican", "American", "British", "Turkish",
    "Greek", "Lebanese", "Moroccan", "Fusion", "Vegetarian", "Vegan", "Seafood"
]

PRICE_RANGES = {
    "Budget": {"min": 10, "max": 25},
    "Mid-range": {"min": 25, "max": 50},
    "Upscale": {"min": 50, "max": 100},
    "Fine Dining": {"min": 100, "max": 200}
}

CITIES = [
    {"name": "Berlin", "country": "Germany", "district": "Mitte"},
    {"name": "Munich", "country": "Germany", "district": "Altstadt"},
    {"name": "Paris", "country": "France", "district": "Le Marais"},
    {"name": "London", "country": "UK", "district": "Covent Garden"},
    {"name": "Rome", "country": "Italy", "district": "Trastevere"},
    {"name": "Madrid", "country": "Spain", "district": "Salamanca"},
    {"name": "Amsterdam", "country": "Netherlands", "district": "Jordaan"},
    {"name": "Zurich", "country": "Switzerland", "district": "Old Town"},
    {"name": "Vienna", "country": "Austria", "district": "Innere Stadt"},
    {"name": "Prague", "country": "Czechia", "district": "Mala Strana"}
]

def generate_restaurant():
    """Generate a single restaurant record"""
    city = random.choice(CITIES)
    restaurant_type = random.choice(RESTAURANT_TYPES)
    cuisine = random.choice(CUISINES)
    price_range = random.choice(list(PRICE_RANGES.keys()))
    
    # Price per person based on price range
    price_info = PRICE_RANGES[price_range]
    price_per_person = random.randint(price_info["min"], price_info["max"])
    
    # Operating hours
    opening_hour = random.randint(7, 11)
    closing_hour = random.randint(18, 23)
    
    return {
        "restaurant_id": f"RS_{uuid.uuid4().hex[:8]}",
        "name": f"{random.choice(['Bella', 'Grand', 'Royal', 'Golden', 'Silver', 'Blue', 'Red'])} {cuisine}",
        "type": restaurant_type,
        "cuisine": cuisine,
        "description": f"Authentic {cuisine.lower()} cuisine in the heart of {city['district']}, {city['name']}",
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
            "price_range": price_range,
            "price_per_person_euro": price_per_person,
            "currency": "EUR"
        },
        "operating_hours": {
            "monday": f"{opening_hour:02d}:00 - {closing_hour:02d}:00",
            "tuesday": f"{opening_hour:02d}:00 - {closing_hour:02d}:00",
            "wednesday": f"{opening_hour:02d}:00 - {closing_hour:02d}:00",
            "thursday": f"{opening_hour:02d}:00 - {closing_hour:02d}:00",
            "friday": f"{opening_hour:02d}:00 - {closing_hour:02d}:00",
            "saturday": f"{opening_hour:02d}:00 - {closing_hour:02d}:00",
            "sunday": f"{opening_hour:02d}:00 - {closing_hour:02d}:00"
        },
        "capacity": {
            "indoor_seating": random.randint(20, 150),
            "outdoor_seating": random.randint(0, 50),
            "private_dining": random.choice([True, False]),
            "group_friendly": random.choice([True, False])
        },
        "amenities": random.sample([
            "WiFi", "Parking", "Wheelchair Accessible", "Outdoor Seating",
            "Private Dining", "Live Music", "Bar", "Wine List", "Takeout",
            "Delivery", "Reservations", "Group Bookings"
        ], k=random.randint(3, 8)),
        "dietary_options": random.sample([
            "Vegetarian", "Vegan", "Gluten-Free", "Halal", "Kosher",
            "Dairy-Free", "Nut-Free", "Low-Carb", "Organic"
        ], k=random.randint(2, 5)),
        "rating": {
            "overall": round(random.uniform(3.0, 5.0), 1),
            "food": round(random.uniform(3.0, 5.0), 1),
            "service": round(random.uniform(3.0, 5.0), 1),
            "ambiance": round(random.uniform(3.0, 5.0), 1),
            "value": round(random.uniform(3.0, 5.0), 1),
            "total_reviews": random.randint(20, 500)
        },
        "features": {
            "reservations_required": random.choice([True, False]),
            "walk_ins_welcome": random.choice([True, False]),
            "group_bookings": random.choice([True, False]),
            "private_events": random.choice([True, False]),
            "catering": random.choice([True, False])
        },
        "contact": {
            "phone": f"+{random.randint(1, 99)} {random.randint(100000000, 999999999)}",
            "email": f"info@{random.choice(['restaurant', 'bella', 'grand'])}.com",
            "website": f"https://www.{random.choice(['restaurant', 'bella', 'grand'])}.com"
        },
        "images": [
            f"https://example.com/images/{uuid.uuid4().hex[:8]}.jpg"
            for _ in range(random.randint(2, 6))
        ]
    }

def main():
    """Generate restaurant data"""
    print(f"🍽️ Generating {N_RESTAURANTS} restaurants...")
    
    restaurants = []
    for _ in range(N_RESTAURANTS):
        restaurants.append(generate_restaurant())
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    
    # Write to file
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(restaurants, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Generated {len(restaurants)} restaurants → {OUT_PATH}")
    
    # Show sample
    if restaurants:
        sample = restaurants[0]
        print(f"Sample restaurant: {sample['name']} ({sample['cuisine']}) in {sample['location']['city']}")

if __name__ == "__main__":
    main()
