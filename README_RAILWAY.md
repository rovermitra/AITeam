# RoverMitra AI Service - Railway Deployment

This guide will help you deploy the RoverMitra AI matching service on Railway platform.

## 🚀 Quick Deployment Steps

### 1. Prepare Your Repository
1. Ensure all files are committed to your GitHub repository
2. The AI service files should be in the `AITeam/` directory

### 2. Deploy on Railway

#### Option A: Deploy from GitHub (Recommended)
1. Go to [Railway.app](https://railway.app)
2. Click "New Project"
3. Select "Deploy from GitHub repo"
4. Choose your repository
5. Railway will auto-detect Python and deploy

#### Option B: Manual Deployment
1. Install Railway CLI: `npm install -g @railway/cli`
2. Login: `railway login`
3. Initialize: `railway init`
4. Deploy: `railway up`

### 3. Configure Environment Variables

In your Railway project dashboard, add these environment variables:

**Required:**
- `OPENAI_API_KEY` - Your OpenAI API key for AI matching
- `DATABASE_URL` - Railway provides this automatically

**Optional:**
- `AI_MODEL` - Default: "gpt-4o"
- `MAX_CANDIDATES` - Default: 1000
- `MIN_COMPATIBILITY_SCORE` - Default: 0.75

### 4. Database Setup

The AI service will automatically connect to your existing PostgreSQL database on Railway. Make sure your database has the required tables:

- `users_core` (from your data generation scripts)
- `matchmaker_profiles` (from your data generation scripts)

## 📡 API Endpoints

Once deployed, your AI service will be available at:
`https://your-project-name.railway.app`

### Available Endpoints:

#### Health Check
```
GET /health
```

#### Find Matches
```
POST /api/matches
Content-Type: application/json

{
  "name": "John Doe",
  "age": 28,
  "gender": "Male",
  "home_base": {"city": "Berlin", "country": "Germany"},
  "languages": ["en", "de"],
  "interests": ["museums", "hiking", "photography"],
  "travel_prefs": {"pace": "balanced"},
  "budget": {"amount": 150, "currency": "EUR"},
  "diet_health": {"diet": "vegetarian"},
  "comfort": {"smoking": "never", "alcohol": "moderate"},
  "bio": "Love exploring new cultures..."
}
```

#### Test AI System
```
GET /api/test
```

#### Get Candidates (Debug)
```
GET /api/candidates
```

## 🔧 Integration with .NET Backend

### Add AI Service Client to .NET

Create a new service in your .NET backend:

```csharp
public interface IAIMatchingService
{
    Task<MatchResult> FindMatchesAsync(UserProfile profile);
}

public class AIMatchingService : IAIMatchingService
{
    private readonly HttpClient _httpClient;
    private readonly string _aiServiceUrl;

    public AIMatchingService(HttpClient httpClient, IConfiguration config)
    {
        _httpClient = httpClient;
        _aiServiceUrl = config["AIService:BaseUrl"];
    }

    public async Task<MatchResult> FindMatchesAsync(UserProfile profile)
    {
        var response = await _httpClient.PostAsJsonAsync($"{_aiServiceUrl}/api/matches", profile);
        response.EnsureSuccessStatusCode();
        return await response.Content.ReadFromJsonAsync<MatchResult>();
    }
}
```

### Register in Program.cs

```csharp
builder.Services.AddHttpClient<IAIMatchingService, AIMatchingService>();
builder.Services.Configure<AIServiceOptions>(builder.Configuration.GetSection("AIService"));
```

### Add to appsettings.json

```json
{
  "AIService": {
    "BaseUrl": "https://your-ai-service.railway.app"
  }
}
```

## 🧪 Testing

### Test the Deployment

1. **Health Check:**
   ```bash
   curl https://your-project-name.railway.app/health
   ```

2. **Test AI System:**
   ```bash
   curl https://your-project-name.railway.app/api/test
   ```

3. **Find Matches:**
   ```bash
   curl -X POST https://your-project-name.railway.app/api/matches \
     -H "Content-Type: application/json" \
     -d '{
       "name": "Test User",
       "age": 28,
       "gender": "Other",
       "home_base": {"city": "Berlin", "country": "Germany"},
       "languages": ["en"],
       "interests": ["museums"],
       "travel_prefs": {"pace": "balanced"},
       "budget": {"amount": 150, "currency": "EUR"},
       "diet_health": {"diet": "none"},
       "comfort": {"smoking": "never", "alcohol": "moderate"}
     }'
   ```

## 🐛 Troubleshooting

### Common Issues:

1. **Import Errors:**
   - Ensure all dependencies are in `requirements.txt`
   - Check that `Updated_main.py` is in the same directory

2. **Database Connection:**
   - Verify `DATABASE_URL` is set correctly
   - Check that your PostgreSQL database is accessible

3. **OpenAI API:**
   - Verify `OPENAI_API_KEY` is set
   - Check API key has sufficient credits

4. **Memory Issues:**
   - Railway free tier has 512MB RAM limit
   - Consider reducing `MAX_CANDIDATES` if needed

### Logs:
View logs in Railway dashboard or via CLI:
```bash
railway logs
```

## 💰 Cost Considerations

**Railway Free Tier:**
- $5 credit monthly
- 512MB RAM
- 1GB disk space
- Usually sufficient for small AI services

**If you exceed free tier:**
- Pay-as-you-go pricing
- Very affordable for small services
- Easy to upgrade

## 🔄 Updates

To update your deployment:
1. Push changes to GitHub
2. Railway auto-deploys on push
3. Or manually trigger: `railway redeploy`

## 📞 Support

- Railway Documentation: https://docs.railway.app
- Railway Discord: https://discord.gg/railway
- This project: Check the main README.md
