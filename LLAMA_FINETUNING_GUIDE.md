# 🚀 RoverMitra Llama Fine-Tuning Guide

## 📁 Files Created

### 1. **Data Generation Script**: `Scripts/generate_llama_training_data.py`
- Generates instruction-following training data for Llama fine-tuning
- Uses your existing user data to create realistic training examples
- Outputs JSONL format suitable for fine-tuning

### 2. **Fine-Tuning Script**: `Scripts/finetune_llama.py`
- Fine-tunes Llama models using LoRA (Low-Rank Adaptation)
- Efficient training with minimal resource requirements
- Saves fine-tuned weights to `models/` folder

## 🎯 **Quick Start**

### **Step 1: Generate Training Data**
```bash
# Generate 1000 training examples (recommended)
python Scripts/generate_llama_training_data.py --num-examples 1000 --output artifacts/llama_training_data.jsonl --validate

# Generate smaller test set
python Scripts/generate_llama_training_data.py --num-examples 100 --output artifacts/test_data.jsonl
```

### **Step 2: Fine-Tune Llama Model**
```bash
# Fine-tune with default settings
python Scripts/finetune_llama.py --training-data artifacts/llama_training_data.jsonl

# Fine-tune with custom settings
python Scripts/finetune_llama.py \
    --training-data artifacts/llama_training_data.jsonl \
    --base-model models/llama-3.1-8b-instruct \
    --output-dir models/llama-travel-matcher \
    --epochs 3 \
    --batch-size 4 \
    --learning-rate 2e-4 \
    --test
```

## 📊 **Training Data Format**

Each training example contains:
```json
{
  "instruction": "You are a psychologist + travel-match expert...",
  "input": "Query user: Alex, 27, Berlin, interests: museums...\n\nCandidates:\n[1] user_id=123 | Summary...",
  "output": "{\"matches\": [{\"user_id\": \"123\", \"name\": \"Sarah\", \"explanation\": \"For you, this match fits because...\", \"compatibility_score\": 0.85}]}"
}
```

## ⚙️ **Fine-Tuning Parameters**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--epochs` | 3 | Training epochs |
| `--batch-size` | 4 | Batch size per device |
| `--learning-rate` | 2e-4 | Learning rate |
| `--max-length` | 2048 | Max sequence length |

## 🎯 **Expected Results**

After fine-tuning, you'll get:
- **Better explanations**: More accurate and specific match reasoning
- **Higher accuracy**: Better understanding of travel compatibility
- **Domain expertise**: Specialized knowledge of travel matching
- **Reduced hallucination**: More consistent results

## 🔧 **Integration**

To use the fine-tuned model in your system:

1. **Update `main.py`** to point to the fine-tuned model:
```python
model_paths = [
    BASE_DIR / "models/llama-travel-matcher",  # Fine-tuned model first
    BASE_DIR / "models/llama-3.1-8b-instruct",  # Fallback
    BASE_DIR / "models/llama-3.2-3b-instruct"
]
```

2. **Test the improvements**:
```bash
python main.py  # Should now use fine-tuned model
```

## 📈 **Performance Tips**

- **Start small**: Test with 100 examples first
- **Monitor training**: Watch for overfitting
- **Validate results**: Test on held-out data
- **Iterate**: Adjust parameters based on results

## 🎉 **Success Metrics**

- **Explanation quality**: More specific and accurate
- **Match relevance**: Better compatibility scores
- **Consistency**: Similar inputs produce similar outputs
- **User satisfaction**: Better travel buddy matches

---

**Ready to fine-tune? Start with Step 1!** 🚀

