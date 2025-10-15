#!/usr/bin/env python3
"""
Test script to verify Hugging Face model loading
"""

import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def test_hf_token():
    """Test if Hugging Face token is loaded"""
    token = os.getenv("HF_TOKEN")
    if token:
        print(f"✅ HF Token loaded: {token[:10]}...")
        return True
    else:
        print("❌ HF Token not found")
        return False

def test_model_paths():
    """Test if model paths are configured"""
    paths = {
        "BGE_M3": os.getenv("BGE_M3_MODEL_PATH"),
        "LLAMA_FINETUNED": os.getenv("LLAMA_FINETUNED_MODEL_PATH"),
        "LLAMA_BASE": os.getenv("LLAMA_BASE_MODEL_PATH"),
        "BGE_CACHE": os.getenv("BGE_CACHE_MODEL_PATH"),
    }
    
    all_good = True
    for name, path in paths.items():
        if path:
            print(f"✅ {name}: {path}")
        else:
            print(f"❌ {name}: Not configured")
            all_good = False
    
    return all_good

def test_environment():
    """Test environment configuration"""
    env_vars = [
        "CUDA_VISIBLE_DEVICES",
        "TRANSFORMERS_NO_ADVISORY_WARNINGS",
        "HF_HUB_ENABLE_HF_TRANSFER",
        "TOKENIZERS_PARALLELISM",
        "HF_HOME",
        "TRANSFORMERS_CACHE",
        "TORCH_HOME"
    ]
    
    print("\n🔧 Environment Variables:")
    for var in env_vars:
        value = os.getenv(var)
        print(f"   {var}: {value}")

def main():
    print("🧪 Testing RoverMitra Hugging Face Configuration")
    print("=" * 50)
    
    # Test token
    token_ok = test_hf_token()
    
    # Test model paths
    paths_ok = test_model_paths()
    
    # Test environment
    test_environment()
    
    print("\n" + "=" * 50)
    if token_ok and paths_ok:
        print("✅ All tests passed! Ready for deployment.")
    else:
        print("❌ Some tests failed. Check configuration.")

if __name__ == "__main__":
    main()
