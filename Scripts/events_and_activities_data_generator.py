#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Events and Activities Data Generator for RoverMitra
Generates realistic events and activities data for travel groups
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

OUT_PATH = "Events/data/events_and_activities.json"
N_EVENTS = int(os.getenv("RM_GEN_EVENTS", "500"))

# Event categories and types
EVENT_CATEGORIES = {
    "Cultural": ["Museum Exhibition", "Art Gallery Opening", "Historical Tour", "Cultural Festival", "Theater Performance"],
    "Food & Drink": ["Food Tour", "Wine Tasting", "Cooking Class", "Local Market Visit", "Restaurant Crawl"],
    "Outdoor": ["Hiking Tour", "City Walking Tour", "Bike Tour", "Park Visit", "Scenic Viewpoint"],
    "Entertainment": ["Live Music", "Comedy Show", "Dance Performance", "Film Screening", "Nightlife Tour"],
    "Shopping": ["Local Crafts Market", "Shopping District Tour", "Antique Market", "Designer Boutique Visit"],
    "Adventure": ["Adventure Park", "Water Sports", "Rock Climbing", "Zip Line", "Bungee Jumping"],
    "Wellness": ["Spa Day", "Yoga Session", "Meditation Class", "Thermal Baths", "Massage Therapy"],
    "Educational": ["Language Exchange", "Workshop", "Lecture", "Guided Tour", "Cultural Exchange"]
}

CITIES = [
    {"name": "Berlin", "country": "Germany"},
    {"name": "Munich", "country": "Germany"},
    {"name": "Paris", "country": "France"},
    {"name": "London", "country": "UK"},
    {"name": "Rome", "country": "Italy"},
    {"name": "Madrid", "country": "Spain"},
    {"name": "Amsterdam", "country": "Netherlands"},
    {"name": "Zurich", "country": "Switzerland"},
    {"name": "Vienna", "country": "Austria"},
    {"name": "Prague", "country": "Czechia"}
]

def generate_event():
    """Generate a single event record"""
    category = random.choice(list(EVENT_CATEGORIES.keys()))
    event_type = random.choice(EVENT_CATEGORIES[category])
    city = random.choice(CITIES)
    
    # Generate event time (next 30 days)
    event_time = datetime.now() + timedelta(days=random.randint(1, 30), hours=random.randint(9, 20))
    
    # Duration (1-8 hours)
    duration_hours = random.randint(1, 8)
    
    # Price range
    base_price = random.randint(10, 150)
    if category == "Adventure":
        base_price += random.randint(50, 100)
    elif category == "Wellness":
        base_price += random.randint(30, 80)
    
    return {
        "event_id": f"EV_{uuid.uuid4().hex[:8]}",
        "name": f"{event_type} in {city['name']}",
        "category": category,
        "type": event_type,
        "description": f"Join us for an amazing {event_type.lower()} experience in {city['name']}",
        "location": {
            "city": city["name"],
            "country": city["country"],
            "venue": f"{random.choice(['Central', 'Historic', 'Modern', 'Cultural'])} {random.choice(['Center', 'Hall', 'Plaza', 'Square'])}"
        },
        "event_time": event_time.isoformat(),
        "duration_hours": duration_hours,
        "price_euro": base_price,
        "max_participants": random.randint(5, 50),
        "current_participants": random.randint(0, 20),
        "difficulty_level": random.choice(["Easy", "Moderate", "Challenging"]),
        "language": random.choice(["English", "German", "French", "Spanish", "Italian"]),
        "includes": random.sample([
            "Professional Guide", "Equipment", "Transportation", "Meals", 
            "Drinks", "Insurance", "Materials", "Certificate"
        ], k=random.randint(1, 4)),
        "requirements": random.sample([
            "Comfortable Shoes", "Weather-Appropriate Clothing", "Camera", 
            "Water Bottle", "Valid ID", "Basic Fitness Level"
        ], k=random.randint(1, 3)),
        "organizer": {
            "name": f"{random.choice(['Local', 'Cultural', 'Adventure', 'Experience'])} Tours",
            "rating": round(random.uniform(3.5, 5.0), 1),
            "verified": random.choice([True, False])
        }
    }

def main():
    """Generate events and activities data"""
    print(f"🎭 Generating {N_EVENTS} events and activities...")
    
    events = []
    for _ in range(N_EVENTS):
        events.append(generate_event())
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    
    # Write to file
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(events, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Generated {len(events)} events → {OUT_PATH}")
    
    # Show sample
    if events:
        sample = events[0]
        print(f"Sample event: {sample['name']} ({sample['category']})")

if __name__ == "__main__":
    main()
