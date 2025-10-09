#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Llama Fine-Tuning Script for RoverMitra

This script fine-tunes Llama models on travel companion matching tasks
using LoRA (Low-Rank Adaptation) for efficient training.

Usage:
    python Scripts/finetune_llama.py --training-data artifacts/llama_training_data.jsonl
"""

import json
import argparse
import torch
from pathlib import Path
from typing import List, Dict, Any
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from transformers import (
        AutoTokenizer, AutoModelForCausalLM, 
        TrainingArguments, Trainer, DataCollatorForLanguageModeling
    )
    from peft import LoraConfig, get_peft_model, TaskType
    from datasets import Dataset
    import numpy as np
    _TRANSFORMERS_OK = True
except ImportError as e:
    print(f"❌ Missing dependencies: {e}")
    print("   Install with: pip install transformers peft datasets accelerate")
    _TRANSFORMERS_OK = False

def load_training_data(data_path: Path) -> List[Dict[str, Any]]:
    """Load training data from JSONL file."""
    print(f"📊 Loading training data from {data_path}")
    
    training_data = []
    
    if data_path.suffix == '.jsonl':
        # Load JSONL format
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    training_data.append(json.loads(line))
    else:
        # Load JSON format
        with open(data_path, 'r', encoding='utf-8') as f:
            training_data = json.load(f)
    
    print(f"✅ Loaded {len(training_data)} training examples")
    return training_data

def format_training_example(example: Dict[str, Any], tokenizer) -> Dict[str, str]:
    """Format a training example for Llama."""
    
    # Create the chat format
    messages = [
        {"role": "system", "content": example["instruction"]},
        {"role": "user", "content": example["input"]},
        {"role": "assistant", "content": example["output"]}
    ]
    
    # Apply chat template
    formatted_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False
    )
    
    return {"text": formatted_text}

def prepare_dataset(training_data: List[Dict[str, Any]], 
                   tokenizer,
                   max_length: int = 2048) -> Dataset:
    """Prepare dataset for training."""
    print("🔄 Preparing dataset...")
    
    # Format examples
    formatted_examples = []
    for example in training_data:
        try:
            formatted = format_training_example(example, tokenizer)
            formatted_examples.append(formatted)
        except Exception as e:
            print(f"   Skipping example due to error: {e}")
            continue
    
    print(f"✅ Formatted {len(formatted_examples)} examples")
    
    # Create dataset
    dataset = Dataset.from_list(formatted_examples)
    
    # Tokenize dataset
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            padding=True,
            max_length=max_length,
            return_tensors=None  # Don't return tensors here, let DataCollator handle it
        )
    
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset.column_names  # Remove original text column
    )
    
    print(f"✅ Tokenized dataset with max_length={max_length}")
    return tokenized_dataset

def setup_lora_config() -> LoraConfig:
    """Setup LoRA configuration for efficient fine-tuning."""
    return LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=16,  # Rank
        lora_alpha=32,  # Scaling parameter
        lora_dropout=0.1,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ],
        bias="none",
    )

# Removed detect_gpu_setup function - using simple CPU training

def setup_model_and_tokenizer(model_path: Path,
                               use_lora: bool = True) -> tuple:
    """Setup model and tokenizer for CPU fine-tuning."""
    print(f"🔄 Loading model from {model_path}")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(model_path), use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load model for CPU training
    print("🖥️  Loading model for CPU training")
    model = AutoModelForCausalLM.from_pretrained(
        str(model_path),
        torch_dtype=torch.float32,
        device_map="cpu",
        low_cpu_mem_usage=True
    )

    # Apply LoRA if requested
    if use_lora:
        print("🔧 Applying LoRA configuration...")
        try:
            lora_config = setup_lora_config()
            model = get_peft_model(model, lora_config)
            model.print_trainable_parameters()
            print("✅ LoRA applied successfully")
        except Exception as e:
            print(f"⚠️  LoRA failed, using base model: {str(e)[:100]}...")
            print("✅ Model loaded without LoRA (will still work)")

    print("✅ Model and tokenizer loaded successfully")
    return model, tokenizer

def setup_training_args(output_dir: Path,
                       num_epochs: int = 3,
                       batch_size: int = 2,
                       learning_rate: float = 2e-4,
                       warmup_steps: int = 100) -> TrainingArguments:
    """Setup training arguments for CPU training."""
    return TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        warmup_steps=warmup_steps,
        learning_rate=learning_rate,
        logging_steps=10,
        save_steps=500,
        save_total_limit=3,
        eval_strategy="no",
        save_strategy="steps",
        load_best_model_at_end=False,
        report_to=None,  # Disable wandb/tensorboard
        remove_unused_columns=False,
        dataloader_pin_memory=False,
        fp16=False,  # Disable mixed precision for CPU training
        gradient_accumulation_steps=4,
        lr_scheduler_type="cosine",
        weight_decay=0.01,
        gradient_checkpointing=False, # Disable gradient checkpointing for CPU training
    )

def fine_tune_model(model, tokenizer, dataset, training_args) -> None:
    """Fine-tune the model."""
    print("🚀 Starting fine-tuning...")
    
    # Setup data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # Causal LM, not masked LM
    )
    
    # Setup trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )
    
    # Train
    trainer.train()
    
    print("✅ Fine-tuning completed!")

def save_model(model, tokenizer, output_dir: Path) -> None:
    """Save the fine-tuned model."""
    print(f"💾 Saving model to {output_dir}")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save LoRA weights
    model.save_pretrained(str(output_dir))
    
    # Save tokenizer
    tokenizer.save_pretrained(str(output_dir))
    
    # Save training config
    config = {
        "model_type": "llama_travel_matcher",
        "base_model": "llama-3.1-8b-instruct",
        "fine_tuned_on": "rovermitra_travel_matching",
        "training_examples": len(dataset) if 'dataset' in locals() else "unknown"
    }
    
    with open(output_dir / "training_config.json", 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"✅ Model saved to {output_dir}")

def test_model(model, tokenizer, test_input: str) -> str:
    """Test the fine-tuned model."""
    print("🧪 Testing fine-tuned model...")
    
    # Format test input
    messages = [
        {"role": "system", "content": "You are a psychologist + travel-match expert. Rank candidates for holistic trip compatibility."},
        {"role": "user", "content": test_input}
    ]
    
    formatted_prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    # Tokenize
    inputs = tokenizer(formatted_prompt, return_tensors="pt")
    if torch.cuda.is_available():
        inputs = {k: v.cuda() for k, v in inputs.items()}
    
    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=500,
            temperature=0.25,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    
    # Decode response
    response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    
    print("📝 Test output:")
    print(response)
    
    return response

def main():
    """Main function for CPU fine-tuning."""
    if not _TRANSFORMERS_OK:
        return 1
    
    parser = argparse.ArgumentParser(description="Fine-tune Llama for RoverMitra travel matching")
    parser.add_argument("--training-data", "-t", 
                       required=True,
                       help="Path to training data JSONL file")
    parser.add_argument("--base-model", "-m",
                       default="models/llama-3.1-8b-instruct",
                       help="Path to base Llama model (default: models/llama-3.1-8b-instruct)")
    parser.add_argument("--output-dir", "-o",
                       default="models/llama-travel-matcher",
                       help="Output directory for fine-tuned model (default: models/llama-travel-matcher)")
    parser.add_argument("--epochs", "-e",
                       type=int, default=3,
                       help="Number of training epochs (default: 3)")
    parser.add_argument("--batch-size", "-b",
                       type=int, default=2,
                       help="Training batch size (default: 2)")
    parser.add_argument("--learning-rate", "-lr",
                       type=float, default=2e-4,
                       help="Learning rate (default: 2e-4)")
    parser.add_argument("--max-length", "-l",
                       type=int, default=2048,
                       help="Maximum sequence length (default: 2048)")
    parser.add_argument("--test", 
                       action="store_true",
                       help="Test the model after training")
    
    args = parser.parse_args()
    
    training_data_path = Path(args.training_data)
    base_model_path = Path(args.base_model)
    output_dir = Path(args.output_dir)
    
    print("🚀 RoverMitra Llama Fine-Tuning")
    print("=" * 40)
    
    try:
        # Check if base model exists
        if not base_model_path.exists():
            print(f"❌ Base model not found at {base_model_path}")
            print("   Please ensure you have llama-3.1-8b-instruct or llama-3.2-3b-instruct in models/")
            return 1
        
        # Load training data
        training_data = load_training_data(training_data_path)
        if not training_data:
            print("❌ No training data loaded!")
            return 1
        
        # Setup model and tokenizer
        model, tokenizer = setup_model_and_tokenizer(base_model_path, use_lora=True)
        
        # Prepare dataset
        dataset = prepare_dataset(training_data, tokenizer, max_length=args.max_length)
        
        # Setup training arguments
        training_args = setup_training_args(
            output_dir=output_dir,
            num_epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate
        )
        
        # Fine-tune
        fine_tune_model(model, tokenizer, dataset, training_args)
        
        # Save model
        save_model(model, tokenizer, output_dir)
        
        # Test if requested
        if args.test:
            test_input = "Query user: Alex, 27, Berlin, interests: museums, food tours, languages: en/de, pace: balanced, budget: €150/day"
            test_model(model, tokenizer, test_input)
        
        print(f"\n🎉 Fine-tuning completed successfully!")
        print(f"📁 Model saved to: {output_dir}")
        print("\n💡 Next steps:")
        print("   1. Update main.py to use the fine-tuned model")
        print("   2. Test the model with real user queries")
        print("   3. Evaluate performance and adjust if needed")
        
    except Exception as e:
        print(f"❌ Error during fine-tuning: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
