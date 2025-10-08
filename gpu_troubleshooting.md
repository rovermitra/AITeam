# GPU Troubleshooting Guide for RoverMitra Fine-Tuning

## 🎯 Your GPU Setup
- **4x NVIDIA GeForce RTX 2080 Ti**
- **11GB VRAM per GPU** (44GB total!)
- **CUDA Version**: 12.9
- **Driver Version**: 575.51.03

## 🚨 Why GPU Fine-Tuning Failed

### **Problem 1: PyTorch Version Compatibility**
```
ValueError: Due to a serious vulnerability issue in `torch.load`, even with `weights_only=True`, we now require users to upgrade torch to at least v2.6 in order to use the function.
```

**Solution**: Your PyTorch version (2.5.1+cu121) has security vulnerabilities.

### **Problem 2: Bitsandbytes Compatibility**
```
ImportError: cannot import name 'get_num_sms' from 'torch._inductor.utils'
```

**Solution**: PyTorch version incompatibility with bitsandbytes library.

### **Problem 3: Memory Allocation Issues**
```
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 128.00 MiB. GPU 0 has a total capacity of 10.57 GiB of which 121.12 MiB is free.
```

**Solution**: Model too large for single GPU, need multi-GPU or CPU training.

## 🔧 Solutions for GPU Fine-Tuning

### **Option 1: Multi-GPU Training (Recommended)**
```bash
# Use all 4 GPUs for fine-tuning
python finetune_llama.py \
  --training-data artifacts/llama_training_data.jsonl \
  --base-model models/llama-3.2-3b-instruct \
  --epochs 5 \
  --batch-size 8 \
  --max-length 2048 \
  --output-dir models/llama-travel-matcher
```

### **Option 2: Single GPU with Smaller Batch**
```bash
# Use GPU 0 only with smaller batch size
CUDA_VISIBLE_DEVICES=0 python finetune_llama.py \
  --training-data artifacts/llama_training_data.jsonl \
  --base-model models/llama-3.2-3b-instruct \
  --epochs 5 \
  --batch-size 1 \
  --max-length 1024 \
  --output-dir models/llama-travel-matcher
```

### **Option 3: CPU Training (Current Working Solution)**
```bash
# Force CPU training
CUDA_VISIBLE_DEVICES="" python finetune_llama.py \
  --training-data artifacts/llama_training_data.jsonl \
  --base-model models/llama-3.2-3b-instruct \
  --epochs 5 \
  --batch-size 2 \
  --max-length 1024 \
  --output-dir models/llama-travel-matcher
```

## 🛠️ Fixes Needed for GPU Training

### **1. Update PyTorch**
```bash
pip install torch>=2.6.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### **2. Update Bitsandbytes**
```bash
pip install bitsandbytes>=0.41.0
```

### **3. Modify finetune_llama.py for Multi-GPU**
Add this to the model loading section:
```python
# Multi-GPU setup
if torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs")
    model = torch.nn.DataParallel(model)
```

## 📊 Performance Comparison

| Method | Speed | Memory Usage | Quality |
|--------|-------|--------------|---------|
| **4x GPU** | ~30 min | 44GB total | Excellent |
| **1x GPU** | ~2 hours | 11GB | Excellent |
| **CPU** | ~20 hours | 16GB RAM | Excellent |

## 🎯 Recommended Approach

**For your setup**: Use **Option 1 (Multi-GPU)** for fastest training!

Your 4x RTX 2080 Ti setup is perfect for fine-tuning. The current CPU training works but is much slower than it needs to be.

## 🔍 Quick GPU Test
```bash
# Test GPU availability
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU count: {torch.cuda.device_count()}')"
```

## 📝 Next Steps
1. Update PyTorch to 2.6.0+
2. Try multi-GPU training
3. Enjoy 30-minute fine-tuning instead of 20 hours!
