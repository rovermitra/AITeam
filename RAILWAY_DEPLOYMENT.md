# RoverMitra Railway Deployment Guide

## 🚀 Quick Deployment Steps

### 1. **Set Environment Variables in Railway**
Go to your Railway project dashboard and add these environment variables:

```bash
# Hugging Face Token
HF_TOKEN=your_hf_token_here

# Model Paths
BGE_M3_MODEL_PATH=abdulghaffaransari9/rovermitra-bge-m3
LLAMA_FINETUNED_MODEL_PATH=abdulghaffaransari9/rovermitra-travel-matcher
LLAMA_BASE_MODEL_PATH=abdulghaffaransari9/rovermitra-llama-base
BGE_CACHE_MODEL_PATH=abdulghaffaransari9/rovermitra-bge-cache

# Railway Settings (CPU-only)
CUDA_VISIBLE_DEVICES=""
TRANSFORMERS_NO_ADVISORY_WARNINGS=1
HF_HUB_ENABLE_HF_TRANSFER=0
TOKENIZERS_PARALLELISM=false

# Cache Directories
HF_HOME=/tmp/hf_home
TRANSFORMERS_CACHE=/tmp/transformers_cache
TORCH_HOME=/tmp/torch_home
```

### 2. **Deploy Commands**
```bash
# Push to Railway
git add .
git commit -m "Deploy RoverMitra with Hugging Face models"
git push origin main
```

### 3. **Monitor Deployment**
- Check Railway logs for model download progress
- First deployment will take 5-10 minutes (downloading models)
- Subsequent deployments will be faster (cached models)

## 🔧 **Key Changes Made**

### **Environment Configuration**
- ✅ Created `.env` file with Hugging Face token
- ✅ Updated `main.py` to use environment variables
- ✅ Updated `serve_llama.py` to use Hugging Face Hub
- ✅ Created Railway-specific configuration

### **Model Loading**
- ✅ Changed from local paths to Hugging Face Hub paths
- ✅ Added authentication token support
- ✅ Updated all model loading functions

### **Deployment Fixes**
- ✅ Removed CUDA-specific dependencies from requirements.txt
- ✅ Created CPU-compatible torch versions
- ✅ Added Railway configuration file
- ✅ Set proper cache directories for Railway

## 📊 **Expected Performance on Railway**

| Component | Local GPU | Railway CPU |
|-----------|-----------|-------------|
| **Model Loading** | 30-60s | 2-5 minutes |
| **Hard Prefilter** | <0.1s | <0.1s |
| **AI Prefilter** | 1-2s | 3-5s |
| **Llama Ranking** | 3-8s | 30-60s |
| **Total Time** | 5-10s | 35-70s |

## 🚨 **Troubleshooting**

### **Common Issues:**

1. **"Could not find torch==2.4.0+cu121"**
   - ✅ Fixed: Updated requirements.txt with CPU-compatible versions

2. **"No module named 'bitsandbytes'"**
   - ✅ Fixed: Added CPU-compatible bitsandbytes

3. **"CUDA out of memory"**
   - ✅ Fixed: Set `CUDA_VISIBLE_DEVICES=""` for CPU-only mode

4. **"Model not found"**
   - ✅ Fixed: Using Hugging Face Hub paths with authentication

### **Railway-Specific Issues:**

1. **Build timeout**
   - Models download during build - this is normal
   - First build takes longer, subsequent builds are faster

2. **Memory limits**
   - Railway has memory limits - CPU models use more RAM
   - Consider upgrading Railway plan if needed

3. **Cold starts**
   - First request after inactivity takes longer
   - Consider keeping service warm with health checks

## 🔍 **Monitoring**

### **Health Check Endpoint**
```bash
curl https://your-railway-app.railway.app/health
```

### **Expected Response**
```json
{
  "ok": true,
  "device": "cpu",
  "model": "rovermitra-travel-matcher",
  "mode": "cpu-only"
}
```

## 📝 **Next Steps**

1. **Deploy to Railway** using the environment variables above
2. **Test the health endpoint** to ensure models loaded
3. **Run a test query** to verify functionality
4. **Monitor performance** and adjust Railway plan if needed

## 💡 **Tips for Railway**

- **Use Railway's environment variables** instead of .env file
- **Monitor memory usage** - CPU models use more RAM
- **Consider upgrading plan** for better performance
- **Set up health checks** to keep service warm
- **Use Railway's logs** to debug any issues
