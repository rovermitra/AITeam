# RoverMitra Llama API - Complete Usage Guide

## 🎯 What You Get

A fully functional AI-powered travel companion matching API that:
1. **Finds compatible travel buddies** using semantic embeddings (BGE-M3)
2. **Ranks matches intelligently** using fine-tuned Llama 3.2 3B
3. **Generates personalized explanations** for each match
4. **Serves via public ngrok URL** - accessible from anywhere!

---

## 📡 API Endpoints

### 1. **Main Endpoint: Find Travel Companions**

**Endpoint:** `POST /api/matches-bulk`

This is your primary endpoint - it does everything automatically:
- AI prefiltering using embeddings (reduces 1000+ candidates to ~80-100)
- LLM ranking with compatibility scoring
- Personalized explanations for each match

---

## 📥 INPUT Format

### Request Structure

```json
{
  "query_user": {
    "email": "john.doe@example.com",
    "name": "John Doe",
    "age": 28,
    "gender": "Male",
    "home_base": {
      "city": "Berlin",
      "country": "Germany"
    },
    "languages": ["en", "de"],
    "interests": [
      "hiking",
      "photography",
      "street food",
      "museums",
      "live music"
    ],
    "values": [
      "sustainability",
      "cultural immersion",
      "adventure"
    ],
    "budget": {
      "amount": 150,
      "currency": "EUR"
    },
    "bio": "Adventure photographer seeking authentic experiences. Love exploring off-the-beaten-path locations and meeting locals.",
    "travel_prefs": {
      "pace": "balanced",
      "accommodation_types": ["hostel", "apartment"],
      "room_setup": "twin",
      "transport_allowed": ["train", "bus", "plane"],
      "must_haves": ["wifi", "local_sim"]
    },
    "diet_health": {
      "diet": "vegetarian",
      "allergies": ["peanuts"]
    },
    "comfort": {
      "smoking": "never",
      "alcohol": "social",
      "risk_tolerance": "high",
      "noise_tolerance": "high",
      "cleanliness_preference": "medium",
      "chronotype": "night_owl"
    },
    "work": {
      "remote_work_ok": true,
      "hours_online_needed": 4,
      "wifi_quality_needed": "good"
    },
    "companion_preferences": {
      "genders_ok": ["any"]
    },
    "faith": {
      "consider_in_matching": false,
      "religion": "",
      "policy": "open"
    },
    "match_intent": ["friendship", "shared_activities"]
  },
  "top_k": 5,
  "prefilter_percent": 0.02,
  "min_k": 80
}
```

### Parameters Explained

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `query_user` | object | **required** | Complete user profile to match |
| `top_k` | int | 5 | Number of top matches to return |
| `prefilter_percent` | float | 0.02 | % of pool to prefilter (2% = ~20-40 from 1000) |
| `min_k` | int | 80 | Minimum candidates to send to LLM |

---

## 📤 OUTPUT Format

### Successful Response (200 OK)

```json
{
  "matches": [
    {
      "user_id": "sarah.johnson@example.com",
      "name": "Sarah Johnson",
      "explanation": "For you, Sarah is an ideal match with shared vegetarian diet, similar budget (€140/day), overlapping interests in hiking and photography, and compatible balanced travel pace.",
      "compatibility_score": 0.92
    },
    {
      "user_id": "marco.rossi@example.com",
      "name": "Marco Rossi",
      "explanation": "For you, Marco shares your passion for photography and street food exploration, has a similar budget (€160/day), and prefers the same balanced travel style with night owl chronotype.",
      "compatibility_score": 0.87
    },
    {
      "user_id": "emma.clark@example.com",
      "name": "Emma Clark",
      "explanation": "For you, Emma aligns on sustainability values, enjoys museums and cultural immersion, has compatible dietary needs (also vegetarian), and matches your social alcohol preference.",
      "compatibility_score": 0.84
    },
    {
      "user_id": "alex.kim@example.com",
      "name": "Alex Kim",
      "explanation": "For you, Alex shares high risk tolerance for adventure, overlapping interest in live music and hiking, similar budget range (€145/day), and compatible remote work needs.",
      "compatibility_score": 0.79
    },
    {
      "user_id": "lisa.mueller@example.com",
      "name": "Lisa Mueller",
      "explanation": "For you, Lisa matches on German language, photography interest, balanced pace preference, and similar accommodation preferences (hostels/apartments).",
      "compatibility_score": 0.76
    }
  ]
}
```

### Response Fields

| Field | Type | Description |
|-------|------|-------------|
| `user_id` | string | Email/unique identifier of matched user |
| `name` | string | Full name of matched user |
| `explanation` | string | **AI-generated** personalized explanation (always starts with "For you,") |
| `compatibility_score` | float | 0.0-1.0 score (higher = better match) |

---

## 🔧 Example Usage

### Using cURL

```bash
curl -X POST "https://your-ngrok-url.ngrok-free.dev/api/matches-bulk" \
  -H "Content-Type: application/json" \
  -d '{
    "query_user": {
      "email": "test@example.com",
      "name": "Test User",
      "age": 28,
      "gender": "Female",
      "home_base": {"city": "Berlin", "country": "Germany"},
      "languages": ["en", "de"],
      "interests": ["hiking", "photography"],
      "values": ["adventure"],
      "budget": {"amount": 100, "currency": "EUR"},
      "bio": "Love outdoor adventures",
      "travel_prefs": {
        "pace": "balanced",
        "accommodation_types": ["hostel"],
        "transport_allowed": ["train"]
      },
      "diet_health": {"diet": "none", "allergies": []},
      "comfort": {
        "smoking": "never",
        "alcohol": "social"
      }
    },
    "top_k": 3
  }'
```

### Using Python

```python
import requests

url = "https://your-ngrok-url.ngrok-free.dev/api/matches-bulk"

query_user = {
    "email": "john@example.com",
    "name": "John Doe",
    "age": 30,
    "gender": "Male",
    "home_base": {"city": "Paris", "country": "France"},
    "languages": ["en", "fr"],
    "interests": ["food", "wine", "art"],
    "values": ["authenticity"],
    "budget": {"amount": 200, "currency": "EUR"},
    "bio": "French foodie seeking culinary adventures",
    "travel_prefs": {
        "pace": "relaxed",
        "accommodation_types": ["hotel", "bnb"]
    }
}

response = requests.post(url, json={
    "query_user": query_user,
    "top_k": 5
})

matches = response.json()["matches"]
for match in matches:
    print(f"✅ {match['name']} - Score: {match['compatibility_score']}")
    print(f"   {match['explanation']}\n")
```

### Using JavaScript/Fetch

```javascript
const url = 'https://your-ngrok-url.ngrok-free.dev/api/matches-bulk';

const queryUser = {
  email: 'alice@example.com',
  name: 'Alice Smith',
  age: 25,
  gender: 'Female',
  home_base: { city: 'London', country: 'UK' },
  languages: ['en'],
  interests: ['yoga', 'meditation', 'nature'],
  values: ['wellness', 'mindfulness'],
  budget: { amount: 80, currency: 'GBP' },
  bio: 'Wellness enthusiast seeking peaceful retreats'
};

fetch(url, {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    query_user: queryUser,
    top_k: 5
  })
})
.then(res => res.json())
.then(data => {
  data.matches.forEach(match => {
    console.log(`${match.name}: ${match.compatibility_score}`);
    console.log(match.explanation);
  });
});
```

---

## 🏥 Health Check Endpoint

**Endpoint:** `GET /health`

Check if models and data are loaded properly.

```json
{
  "status": "ok",
  "models_loaded": true,
  "data_loaded": true,
  "pool_size": 847
}
```

---

## 📊 Pool Data Endpoint

**Endpoint:** `GET /pool`

View all loaded candidate profiles.

```json
{
  "pool": [...],  // Array of all user profiles
  "size": 847
}
```

---

## 🎯 Lower-Level Endpoints

### Direct LLM Ranking

**Endpoint:** `POST /rank`

For custom prompts (advanced usage).

```json
{
  "prompt": "Your custom ranking prompt here",
  "max_new_tokens": 120,
  "temperature": 0.0,
  "top_p": 0.9
}
```

---

## ⚡ What Happens Behind the Scenes

### The Magic Pipeline

```
1. You send query_user 
   ↓
2. AI Prefilter (BGE-M3 Embeddings)
   - Builds semantic embedding from bio + interests + values
   - Compares with 1000+ cached candidate embeddings
   - Selects top 80-100 most semantically similar
   ↓
3. LLM Ranking (Llama 3.2 3B)
   - Receives shortlist of ~50 candidates (token limit)
   - Analyzes: interests, budget, pace, diet, values, languages
   - Generates compatibility scores + explanations
   ↓
4. You receive top 5 matches with scores + explanations
```

### Performance

- **Prefilter:** ~100ms (GPU) / ~300ms (CPU)
- **LLM Ranking:** ~2-5 seconds (GPU) / ~10-20 seconds (CPU)
- **Total:** ~3-6 seconds for complete matching

---

## 🎨 Real-World Example

**User Query:**
- Name: Emma
- Age: 29
- Interests: Surfing, yoga, vegan food
- Budget: $100/day
- Bio: "Digital nomad seeking beach towns with good wifi and healthy food scene"

**AI Response:**
```json
{
  "matches": [
    {
      "user_id": "lucas@example.com",
      "name": "Lucas Santos",
      "explanation": "For you, Lucas is a perfect match with shared passion for surfing and vegan lifestyle, similar budget ($95/day), remote work needs, and loves beach destinations with strong wifi infrastructure.",
      "compatibility_score": 0.94
    }
  ]
}
```

---

## 🚀 Integration Example

### Full Frontend Integration

```javascript
// React component example
async function findTravelBuddies(userProfile) {
  const API_URL = 'https://your-ngrok-url.ngrok-free.dev';
  
  try {
    const response = await fetch(`${API_URL}/api/matches-bulk`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        query_user: userProfile,
        top_k: 5
      })
    });
    
    const { matches } = await response.json();
    
    return matches.map(match => ({
      id: match.user_id,
      name: match.name,
      score: Math.round(match.compatibility_score * 100),
      reason: match.explanation
    }));
  } catch (error) {
    console.error('Matching failed:', error);
    return [];
  }
}
```

---

## 🎉 Summary

**Input:** User profile with travel preferences  
**Processing:** AI semantic prefilter → LLM intelligent ranking  
**Output:** Top 5 compatible matches with scores + personalized explanations  
**Speed:** 3-6 seconds per query  
**Access:** Public ngrok URL - use from anywhere!

**Perfect for:**
- Travel companion matching apps
- Trip planning platforms
- Social travel networks
- Adventure booking sites
