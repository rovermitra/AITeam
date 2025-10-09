#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Llama Fine-Tuning Data Generator for RoverMitra

This script generates instruction-following training data for fine-tuning
Llama models on travel companion matching tasks.

Usage:
    python Scripts/generate_llama_training_data.py --output artifacts/llama_training_data.jsonl
"""

import json
import random
import argparse
from pathlib import Path
from typing import List, Dict, Any, Tuple
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from main import (
    load_json, load_pool, hard_prefilter, ai_prefilter, 
    craft_specific_reason, summarize_user, query_text
)

def create_instruction_template() -> str:
    """Create the instruction template for fine-tuning."""
    return """You are a psychologist + travel-match expert. Rank candidates for holistic trip compatibility.
Consider personality fit, conflict style, shared & complementary interests, languages/communication, pace, budget, diet/substances,
safety/risk tolerance, work needs, and values. Trip context: weekend to multi-week.
Return JSON with: user_id, name, explanation (ONE sentence, must start with 'For you,' and be specific), compatibility_score (0.0–1.0)."""

def generate_training_example(query_user: Dict[str, Any], 
                            candidates: List[Dict[str, Any]], 
                            num_matches: int = 3) -> Dict[str, Any]:
    """Generate a single training example."""
    
    # Create input text
    query_text_content = query_text(query_user)
    
    # Create candidates text
    candidates_text = []
    for i, rec in enumerate(candidates):
        u, m = rec["user"], rec["mm"]
        candidates_text.append(f"[{i+1}] user_id={u.get('user_id')} | {summarize_user(u, m)}")
    
    candidates_str = "\n".join(candidates_text)
    
    # Create the full input
    full_input = f"{query_text_content}\n\nCandidates:\n{candidates_str}"
    
    # Generate ground truth matches using our existing logic
    matches = []
    for i, rec in enumerate(candidates[:num_matches]):
        u = rec["user"]
        explanation = craft_specific_reason(query_user, u, rec.get("mm"))
        
        # Calculate compatibility score based on shared interests and other factors
        shared_interests = len(set(query_user.get("interests", [])) & set(u.get("interests", [])))
        total_interests = len(set(query_user.get("interests", [])) | set(u.get("interests", [])))
        interest_score = shared_interests / max(1, total_interests)
        
        # Add bonuses for other compatibility factors
        score = 0.5 + 0.3 * interest_score
        
        # Language bonus
        shared_langs = len(set(query_user.get("languages", [])) & set(u.get("languages", [])))
        if shared_langs > 0:
            score += 0.1
        
        # Pace bonus
        if (query_user.get("travel_prefs", {}).get("pace") == 
            u.get("travel_prefs", {}).get("pace")):
            score += 0.1
        
        # Budget compatibility bonus
        q_budget = query_user.get("budget", {}).get("amount", 150)
        u_budget = u.get("budget", {}).get("amount", 150)
        budget_diff = abs(q_budget - u_budget)
        if budget_diff <= 30:
            score += 0.1
        
        # Ensure score is between 0.0 and 1.0
        score = min(max(score, 0.0), 0.99)
        
        matches.append({
            "user_id": u.get("user_id"),
            "name": u.get("name"),
            "explanation": explanation,
            "compatibility_score": round(score, 2)
        })
    
    # Create the training example
    training_example = {
        "instruction": create_instruction_template(),
        "input": full_input,
        "output": json.dumps({"matches": matches}, ensure_ascii=False)
    }
    
    return training_example

def generate_training_data(num_examples: int = 1000, 
                          output_path: Path = None) -> List[Dict[str, Any]]:
    """Generate training data for Llama fine-tuning."""
    
    print(f"🔄 Generating {num_examples} training examples...")
    
    # Load data
    print("📊 Loading user data...")
    pool = load_pool()
    if not pool:
        raise ValueError("No candidate pool found. Please generate users_core.json + matchmaker_profiles.json first.")
    
    print(f"✅ Loaded {len(pool)} candidates")
    
    # Load existing users for query generation
    users_core_path = Path(__file__).parent.parent / "users/data/users_core.json"
    users = load_json(users_core_path)
    
    if not users:
        raise ValueError("No users found in users_core.json")
    
    print(f"✅ Loaded {len(users)} users for query generation")
    
    training_data = []
    
    for i in range(num_examples):
        if (i + 1) % 100 == 0:
            print(f"   Generated {i + 1}/{num_examples} examples...")
        
        # Randomly select a query user
        query_user = random.choice(users)
        
        # Apply hard prefilters to get realistic candidates
        hard_filtered = hard_prefilter(query_user, pool)
        
        if len(hard_filtered) < 3:
            # Skip if not enough candidates
            continue
        
        # Apply AI prefilter to get top candidates
        shortlist = ai_prefilter(query_user, hard_filtered, percent=0.05, min_k=5)
        
        if len(shortlist) < 3:
            # Use hard filtered if AI prefilter doesn't give enough
            shortlist = hard_filtered[:10]
        
        # Generate training example
        try:
            example = generate_training_example(query_user, shortlist, num_matches=3)
            training_data.append(example)
        except Exception as e:
            print(f"   Skipping example {i+1} due to error: {e}")
            continue
    
    print(f"✅ Generated {len(training_data)} training examples")
    print(f"📊 Note: Some duplicates are expected and beneficial for model generalization")
    
    # Save training data
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save as JSONL (one JSON object per line)
        with open(output_path, 'w', encoding='utf-8') as f:
            for example in training_data:
                f.write(json.dumps(example, ensure_ascii=False) + '\n')
        
        print(f"💾 Saved training data to {output_path}")
        
        # Also save as regular JSON for inspection
        json_path = output_path.with_suffix('.json')
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(training_data, f, indent=2, ensure_ascii=False)
        
        print(f"💾 Also saved as JSON to {json_path}")
    
    return training_data

def validate_training_data(training_data: List[Dict[str, Any]]) -> None:
    """Validate the generated training data."""
    print("\n🔍 Validating training data...")
    
    if not training_data:
        print("❌ No training data generated!")
        return
    
    print(f"✅ Total examples: {len(training_data)}")
    
    # Check data structure
    required_keys = ["instruction", "input", "output"]
    valid_examples = 0
    
    for example in training_data:
        if all(key in example for key in required_keys):
            try:
                # Validate JSON output
                output_data = json.loads(example["output"])
                if "matches" in output_data and isinstance(output_data["matches"], list):
                    valid_examples += 1
            except json.JSONDecodeError:
                continue
    
    print(f"✅ Valid examples: {valid_examples}/{len(training_data)}")
    
    # Show sample
    if training_data:
        print("\n📝 Sample training example:")
        sample = training_data[0]
        print(f"   Instruction: {sample['instruction'][:100]}...")
        print(f"   Input length: {len(sample['input'])} characters")
        print(f"   Output: {sample['output'][:100]}...")

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Generate Llama training data for RoverMitra")
    parser.add_argument("--output", "-o", 
                       default="artifacts/llama_training_data.jsonl",
                       help="Output path for training data (default: artifacts/llama_training_data.jsonl)")
    parser.add_argument("--num-examples", "-n", 
                       type=int, default=1000,
                       help="Number of training examples to generate (default: 1000)")
    parser.add_argument("--validate", "-v", 
                       action="store_true",
                       help="Validate generated training data")
    
    args = parser.parse_args()
    
    output_path = Path(args.output)
    
    print("🚀 RoverMitra Llama Training Data Generator")
    print("=" * 50)
    
    try:
        # Generate training data
        training_data = generate_training_data(
            num_examples=args.num_examples,
            output_path=output_path
        )
        
        # Validate if requested
        if args.validate:
            validate_training_data(training_data)
        
        print(f"\n🎉 Successfully generated {len(training_data)} training examples!")
        print(f"📁 Output saved to: {output_path}")
        print("\n💡 Next steps:")
        print("   1. Review the generated training data")
        print("   2. Run the fine-tuning script: python Scripts/finetune_llama.py")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
