# Railway Environment Variables Setup

## Required Environment Variables

Add these environment variables in your Railway dashboard:

### 1. PORT (Railway automatically sets this)
- **Variable**: `PORT`
- **Value**: Railway automatically provides this (usually 5000 or 8080)
- **Purpose**: Port for the web service to bind to

### 2. OpenAI API Key (Required for AI functionality)
- **Variable**: `OPENAI_API_KEY`
- **Value**: Your OpenAI API key (starts with `sk-`)
- **Purpose**: Enables GPT-4o integration for intelligent matching explanations

### 3. Railway Environment (Optional)
- **Variable**: `RAILWAY_ENVIRONMENT`
- **Value**: Railway automatically sets this (e.g., "production")
- **Purpose**: Identifies the deployment environment

## How to Add Environment Variables in Railway:

1. Go to your Railway dashboard
2. Select your AITeam service
3. Click on the "Variables" tab
4. Click "New Variable"
5. Add each variable:
   - Name: `OPENAI_API_KEY`
   - Value: `your_actual_openai_api_key_here`

## Optional Environment Variables:

### For Development/Debugging:
- **Variable**: `FLASK_ENV`
- **Value**: `development` or `production`
- **Purpose**: Controls Flask debug mode

### For AI Model Selection:
- **Variable**: `AI_MODEL`
- **Value**: `gpt-4o` (default) or `gpt-4o-mini`
- **Purpose**: Selects which OpenAI model to use

## Environment Variables Status Check:

Once deployed, you can check if environment variables are set correctly by calling:
```bash
curl https://your-service.railway.app/health
```

The response will show:
- `ai_mode`: "full" if OPENAI_API_KEY is set, "fallback" otherwise
- Environment information in the logs

## Troubleshooting:

### If PORT is not set:
- Railway should automatically set this
- Check Railway service settings
- Ensure your service is properly configured

### If OPENAI_API_KEY is not working:
- Verify the key is correct and active
- Check OpenAI account credits
- Ensure no extra spaces in the variable value

### If service won't start:
- Check Railway logs for environment variable errors
- Verify all required variables are set
- Check variable names for typos
