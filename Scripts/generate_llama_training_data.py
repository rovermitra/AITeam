#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate Llama Training Data for RoverMitra
Creates training data for fine-tuning Llama models on travel matching
"""

import os
import json
import random
from datetime import datetime
from pathlib import Path

# Configuration
SEED = int(os.getenv("RM_GEN_SEED", "77"))
random.seed(SEED)

OUT_PATH = "data/llama_training_data.json"
N_SAMPLES = int(os.getenv("RM_GEN_LLAMA_SAMPLES", "1000"))

# Sample user profiles for training
SAMPLE_USERS = [
    {
        "name": "Alex",
        "age": 28,
        "interests": ["photography", "hiking", "local cuisine"],
        "travel_style": "budget",
        "personality": "adventurous"
    },
    {
        "name": "Sarah",
        "age": 32,
        "interests": ["art", "museums", "wine tasting"],
        "travel_style": "luxury",
        "personality": "cultural"
    },
    {
        "name": "Mike",
        "age": 25,
        "interests": ["nightlife", "music", "partying"],
        "travel_style": "budget",
        "personality": "social"
    },
    {
        "name": "Emma",
        "age": 30,
        "interests": ["yoga", "wellness", "nature"],
        "travel_style": "mid-range",
        "personality": "peaceful"
    }
]

# Sample destinations
DESTINATIONS = [
    "Berlin", "Paris", "Rome", "Amsterdam", "Prague", "Barcelona", "Vienna", "Zurich"
]

def generate_training_sample():
    """Generate a single training sample"""
    user = random.choice(SAMPLE_USERS)
    destination = random.choice(DESTINATIONS)
    
    # Generate different types of training prompts
    prompt_types = [
        "match_explanation",
        "preference_analysis", 
        "compatibility_assessment",
        "travel_recommendation"
    ]
    
    prompt_type = random.choice(prompt_types)
    
    if prompt_type == "match_explanation":
        instruction = f"Explain why {user['name']} would be a good travel match for someone interested in {', '.join(user['interests'])} in {destination}."
        response = f"{user['name']} would be an excellent travel match because they share a passion for {', '.join(user['interests'])}. Their {user['personality']} personality and {user['travel_style']} travel style make them ideal for exploring {destination}. They would bring enthusiasm and knowledge about local experiences."
    
    elif prompt_type == "preference_analysis":
        instruction = f"Analyze the travel preferences of a {user['age']}-year-old who likes {', '.join(user['interests'])} and prefers {user['travel_style']} travel."
        response = f"This traveler shows a preference for {user['travel_style']} experiences with a focus on {', '.join(user['interests'])}. Their age suggests they're looking for authentic, meaningful travel experiences. They would appreciate recommendations that balance cost-effectiveness with quality experiences."
    
    elif prompt_type == "compatibility_assessment":
        instruction = f"Assess the compatibility between someone with {user['personality']} personality and {user['travel_style']} travel style for a trip to {destination}."
        response = f"A {user['personality']} personality with {user['travel_style']} preferences would thrive in {destination}. The city offers perfect opportunities for {', '.join(user['interests'])} activities. This combination suggests high compatibility for an enjoyable travel experience."
    
    else:  # travel_recommendation
        instruction = f"Recommend activities in {destination} for someone interested in {', '.join(user['interests'])} with a {user['travel_style']} budget."
        response = f"For {destination}, I recommend focusing on {', '.join(user['interests'])} activities. Given your {user['travel_style']} budget, look for free walking tours, local markets, and cultural sites. Consider {random.choice(['museums', 'parks', 'local restaurants', 'historical sites'])} as they align perfectly with your interests."
    
    return {
        "instruction": instruction,
        "input": "",
        "output": response,
        "metadata": {
            "user_profile": user,
            "destination": destination,
            "prompt_type": prompt_type,
            "generated_at": datetime.now().isoformat()
        }
    }

def main():
    """Generate Llama training data"""
    print(f"🦙 Generating {N_SAMPLES} Llama training samples...")
    
    samples = []
    for _ in range(N_SAMPLES):
        samples.append(generate_training_sample())
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    
    # Write to file
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(samples, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Generated {len(samples)} training samples → {OUT_PATH}")
    
    # Show sample
    if samples:
        sample = samples[0]
        print(f"Sample instruction: {sample['instruction'][:100]}...")

if __name__ == "__main__":
    main()
