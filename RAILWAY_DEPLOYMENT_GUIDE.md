# 🚀 RoverMitra Complete Deployment & Data Upload Guide

This guide provides step-by-step instructions to deploy the RoverMitra .NET backend to Railway and upload your data to the Postgres database. This backend handles user authentication, profiles, preferences, and integrates with the Python AI matching service.

## Prerequisites

1.  **Railway Account**: Ensure you have an active Railway account.
2.  **GitHub Repository**: Your `rovermitra-backend` project should be pushed to a GitHub repository.
3.  **PostgreSQL Database**: Railway will automatically provision a PostgreSQL database for you, but you need to ensure your application is configured to use it.
4.  **Python AI Service URL**: The URL of your deployed Python AI matching service (e.g., `https://ai.rovermitra.com`).
5.  **Generated Data**: Your `users/data/users_core.json` and `MatchMaker/data/matchmaker_profiles.json` files.

## **📊 How to Upload Data to Railway Postgres**

### **Method 1: Using Python Scripts (Recommended)**

#### **Step 1: Set Environment Variables**
```bash
# Set your Railway API URL
export RAILWAY_API_URL="https://api.rovermitra.com"  # Replace with your actual URL

# Optional: Set other variables
export RM_VERIFY_SSL="true"
export RM_RATE_LIMIT="0.1"
```

#### **Step 2: Upload User Data**
```bash
cd /data/abdul/RoverMitra/RoverMitraDataCreation

# Upload users to Railway Postgres
python UsersCreation.py \
    --users-json ../users/data/users_core.json \
    --base-url $RAILWAY_API_URL \
    --register-endpoint /User/register \
    --login-endpoint /User/login \
    --prefs-endpoint /User/preferences \
    --verify-ssl true \
    --rate-limit 0.1 \
    --limit 100  # Start with 100 users to test
```

#### **Step 3: Upload Matchmaker Data**
```bash
# Upload matchmaker profiles to Railway Postgres
python upload_matchmaker_data.py
```

### **Method 2: Direct Database Connection (Advanced)**

#### **Step 1: Get Database Connection String**
1. Go to your Railway project dashboard
2. Click on your Postgres service
3. Go to "Variables" tab
4. Copy the `DATABASE_URL` value

#### **Step 2: Connect to Database**
```bash
# Install PostgreSQL client
sudo apt-get install postgresql-client

# Connect to Railway Postgres
psql "postgresql://username:password@host:port/database"
```

#### **Step 3: Upload Data via SQL**
```sql
-- Create tables (if not exists)
CREATE TABLE IF NOT EXISTS users (
    id SERIAL PRIMARY KEY,
    email VARCHAR(255) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    first_name VARCHAR(100),
    last_name VARCHAR(100),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Insert users (example)
INSERT INTO users (email, password_hash, first_name, last_name) 
VALUES ('user1@example.com', 'hashed_password', 'John', 'Doe');
```

### **Method 3: Using Railway CLI (If Available)**

```bash
# Install Railway CLI
npm install -g @railway/cli

# Login to Railway
railway login

# Connect to your project
railway link

# Run database commands
railway run psql -c "SELECT COUNT(*) FROM users;"
```

## **🔍 Verify Data Upload**

### **Check User Count**
```bash
# Test API connection
curl -X GET "https://api.rovermitra.com/api/health"

# Check if users are registered
curl -X POST "https://api.rovermitra.com/User/login" \
  -H "Content-Type: application/json" \
  -d '{"email": "test@example.com", "password": "Test123!"}'
```

### **Check Database Directly**
```bash
# Connect to Railway Postgres
psql $DATABASE_URL

# Check user count
SELECT COUNT(*) FROM users;

# Check matchmaker data
SELECT COUNT(*) FROM matchmaker_profiles;
```

## **🚨 Troubleshooting Data Upload**

### **Common Issues:**

#### **1. Connection Refused**
```bash
# Check if Railway API is running
curl -X GET "https://api.rovermitra.com/api/health"

# If not responding, check Railway logs
# Go to Railway dashboard → Deployments → View logs
```

#### **2. Authentication Errors**
```bash
# Check JWT configuration
# Ensure JWT_SECRET_KEY is set in Railway environment variables
```

#### **3. Database Connection Issues**
```bash
# Check DATABASE_URL in Railway environment variables
# Ensure Postgres service is running
```

#### **4. Rate Limiting**
```bash
# Add delays between requests
python UsersCreation.py --rate-limit 0.5  # 0.5 second delay
```

### **Debug Commands:**
```bash
# Test with dry run first
python UsersCreation.py --dry-run --users-json ../users/data/users_core.json

# Check specific user
python UsersCreation.py --start 0 --limit 1 --users-json ../users/data/users_core.json

# Verbose logging
python UsersCreation.py --log-level DEBUG --users-json ../users/data/users_core.json
```

## **📈 Monitoring Upload Progress**

### **Real-time Monitoring:**
```bash
# Watch Railway logs
# Go to Railway dashboard → Deployments → View logs

# Monitor API calls
tail -f /var/log/railway-api.log  # If available
```

### **Check Upload Status:**
```bash
# Count successful uploads
grep "Preferences OK" upload_log.txt | wc -l

# Check failed uploads
grep "Preferences failed" upload_log.txt
```

## **🎯 Expected Results**

After successful data upload:

1. **✅ Users Table**: 10,000+ user records in Railway Postgres
2. **✅ Matchmaker Table**: 10,000+ matchmaker profiles in Railway Postgres
3. **✅ API Endpoints**: All endpoints responding correctly
4. **✅ Authentication**: Users can login and get JWT tokens
5. **✅ Preferences**: User preferences stored and retrievable

---

## **Step 1: Deploy .NET Backend to Railway**

### **Manual Deployment (Recommended)**

1. **Go to Railway Dashboard**: https://railway.app/dashboard
2. **Create New Project**: Click "New Project"
3. **Connect GitHub Repository**: 
   - Select "Deploy from GitHub repo"
   - Choose your `rovermitra-backend` repository
4. **Configure Environment Variables**:
   ```
   ASPNETCORE_ENVIRONMENT=Production
   JWT_SECRET_KEY=YourSuperSecretKeyThatIsAtLeast32CharactersLong!
   JWT_ISSUER=RoverMitra
   JWT_AUDIENCE=RoverMitra-Users
   SMTP_HOST=smtp.gmail.com
   SMTP_PORT=587
   SMTP_USERNAME=your-email@gmail.com
   SMTP_PASSWORD=your-app-password
   FROM_EMAIL=noreply@rovermitra.com
   FROM_NAME=RoverMitra
   ```
5. **Deploy**: Railway will automatically build and deploy your .NET app

### **Get Your Railway API URL**
After deployment, you'll get a URL like: `https://rovermitra-backend-production.railway.app`

## **Step 2: Update UsersCreation.py for Railway**

### **Update the script to use Railway URL:**

```bash
cd /data/abdul/RoverMitra/RoverMitraDataCreation

# Upload user data to Railway
python UsersCreation.py \
    --users-json ../users/data/users_core.json \
    --base-url https://your-railway-app.railway.app \
    --register-endpoint /User/register \
    --login-endpoint /User/login \
    --prefs-endpoint /User/preferences \
    --verify-ssl true \
    --rate-limit 0.1 \
    --limit 100
```

## **Step 3: Upload Matchmaker Data**

### **Create a script to upload matchmaker data:**

```python
# upload_matchmaker_data.py
import requests
import json
import os

def upload_matchmaker_data():
    # Your Railway API URL
    base_url = "https://your-railway-app.railway.app"
    
    # Load matchmaker data
    with open("../MatchMaker/data/matchmaker_profiles.json", "r") as f:
        matchmaker_data = json.load(f)
    
    # Upload each matchmaker profile
    for profile in matchmaker_data:
        try:
            response = requests.post(
                f"{base_url}/Matching/matchmaker-profile",
                json=profile,
                headers={"Content-Type": "application/json"}
            )
            if response.status_code == 200:
                print(f"✅ Uploaded matchmaker profile: {profile.get('email', 'unknown')}")
            else:
                print(f"❌ Failed to upload: {response.text}")
        except Exception as e:
            print(f"❌ Error uploading {profile.get('email', 'unknown')}: {e}")

if __name__ == "__main__":
    upload_matchmaker_data()
```

## **Step 4: Integrate Llama Model into .NET**

### **Add Llama Integration to .NET Backend:**

1. **Add Python Integration Service**:
   ```csharp
   // Services/PythonIntegrationService.cs
   public class PythonIntegrationService
   {
       private readonly string _pythonApiUrl;
       
       public PythonIntegrationService(IConfiguration config)
       {
           _pythonApiUrl = config["PythonApiUrl"] ?? "http://localhost:8000";
       }
       
       public async Task<string> GetLlamaRanking(string prompt)
       {
           using var client = new HttpClient();
           var request = new { prompt = prompt, max_new_tokens = 512 };
           var response = await client.PostAsJsonAsync($"{_pythonApiUrl}/rank", request);
           var result = await response.Content.ReadFromJsonAsync<dynamic>();
           return result?.text ?? "";
       }
   }
   ```

2. **Update MatchingController**:
   ```csharp
   // In MatchingController.cs
   private readonly PythonIntegrationService _pythonService;
   
   [HttpPost("ai-match")]
   public async Task<ActionResult> GetAiMatches([FromBody] MatchRequest request)
   {
       var prompt = BuildMatchingPrompt(request);
       var aiResponse = await _pythonService.GetLlamaRanking(prompt);
       return Ok(ParseAiResponse(aiResponse));
   }
   ```

## **Step 5: Test the Complete System**

### **Test User Registration:**
```bash
curl -X POST https://your-railway-app.railway.app/User/register \
  -H "Content-Type: application/json" \
  -d '{
    "email": "test@example.com",
    "password": "Test123!",
    "confirmPassword": "Test123!",
    "firstName": "Test",
    "lastName": "User"
  }'
```

### **Test Matching API:**
```bash
curl -X POST https://your-railway-app.railway.app/Matching/ai-match \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_JWT_TOKEN" \
  -d '{
    "userId": "user-id",
    "preferences": {...}
  }'
```

## **Step 6: Environment Variables for Railway**

### **Required Environment Variables:**
```
# .NET Backend
ASPNETCORE_ENVIRONMENT=Production
JWT_SECRET_KEY=YourSuperSecretKeyThatIsAtLeast32CharactersLong!
JWT_ISSUER=RoverMitra
JWT_AUDIENCE=RoverMitra-Users

# Database (Railway will provide this automatically)
DATABASE_URL=postgresql://...

# Email Configuration
SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
SMTP_USERNAME=your-email@gmail.com
SMTP_PASSWORD=your-app-password

# Python API Integration
PythonApiUrl=https://your-python-app.railway.app
```

## **Step 7: Run Data Upload**

### **Upload Users:**
```bash
cd /data/abdul/RoverMitra/RoverMitraDataCreation
python UsersCreation.py \
    --users-json ../users/data/users_core.json \
    --base-url https://your-railway-app.railway.app \
    --verify-ssl true \
    --rate-limit 0.1
```

### **Upload Matchmaker Data:**
```bash
python upload_matchmaker_data.py
```

## **🎯 Final Result**

After completing these steps, you'll have:

1. ✅ **.NET Backend** deployed on Railway
2. ✅ **User data** uploaded to Railway database
3. ✅ **Matchmaker data** uploaded to Railway database
4. ✅ **Llama model** integrated via Python API
5. ✅ **Matching API** working with AI

Your complete system will be running on Railway with both user management and AI matching capabilities!
