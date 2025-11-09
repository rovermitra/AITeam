from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import uvicorn
from pyngrok import ngrok
from datetime import datetime

# Import your logic
from main import load_pool, hard_prefilter, ai_prefilter, llm_rank

def find_matches(query_user: Dict[str, Any]) -> List[Dict[str, Any]]:
    """The main pipeline function to find matches for a given user using main.py functions."""
    print(f"\n🚀 Starting match process for: {query_user.get('name')}")
    
    # Load pool data
    pool = load_pool()
    if not pool:
        print("No candidates found.")
        return []
    
    # 1. Hard Prefilters
    hard_filtered = hard_prefilter(query_user, pool)
    print(f"✅ Hard prefilter: {len(pool)} -> {len(hard_filtered)} candidates")
    if not hard_filtered:
        return []

    # 2. AI Prefilter (BGE)
    shortlist = ai_prefilter(query_user, hard_filtered, percent=0.02, min_k=80)
    print(f"✅ AI prefilter: {len(hard_filtered)} -> {len(shortlist)} candidates")
    if not shortlist:
        return []

    # 3. Final Ranking (Llama)
    final_matches = llm_rank(query_user, shortlist, out_top=5)
    print(f"✅ Llama ranking: {len(shortlist)} -> {len(final_matches)} matches")

    # 4. Filter by quality score
    high_quality = [m for m in final_matches if m.get("compatibility_score", 0) >= 0.75]
    print(f"✅ Found {len(high_quality)} high-quality matches (score >= 75%).")
    
    return high_quality if high_quality else final_matches

# --- Global variables ---
ngrok_public_url: Optional[str] = None
match_history: List[Dict[str, Any]] = []

# --- Pydantic models ---
class HomeBase(BaseModel):
    city: str
    country: str

class Budget(BaseModel):
    amount: int
    currency: str = "EUR"

class Faith(BaseModel):
    consider_in_matching: bool
    religion: str
    policy: str

class CompanionPreferences(BaseModel):
    genders_ok: List[str]

class QueryUser(BaseModel):
    name: str
    age: int
    gender: str
    home_base: HomeBase
    languages: List[str]
    interests: List[str]
    values: List[str]
    bio: Optional[str] = ""
    travel_prefs: Dict[str, Any]
    budget: Budget
    diet_health: Dict[str, Any]
    comfort: Dict[str, Any]
    work: Dict[str, Any]
    companion_preferences: CompanionPreferences
    faith: Faith

class MatchResult(BaseModel):
    email: str
    name: str
    explanation: str
    compatibility_score: float = Field(..., ge=0.0, le=1.0)

class MatchResponse(BaseModel):
    matches: List[MatchResult]

# --- FastAPI app ---
app = FastAPI(
    title="RoverMitra Matchmaking API",
    description="Provides travel companion matches based on user profiles.",
    version="2.0.0"
)

@app.on_event("startup")
def startup_event():
    print("\n--- Server starting up ---")
    load_pool()
    print("--- Data loaded and ready ---\n")

@app.post("/api/matches-bulk", response_model=MatchResponse)
def find_companions(query_user: QueryUser):
    """
    Main endpoint to find travel companions.
    Call this endpoint via the ngrok public URL.
    """
    try:
        print(f"🚀 Starting match process for: {query_user.name}")
        user_dict = query_user.model_dump()
        
        # Call the matching logic
        results = find_matches(user_dict)
        response_data = {"matches": results}
        
        # Store in history with timestamp
        match_history.append({
            "timestamp": datetime.now().isoformat(),
            "user_name": query_user.name,
            "match_count": len(results),
            "data": response_data
        })
        
        print(f"✅ Found {len(results)} matches for {query_user.name}")
        print(f"📊 Total requests processed: {len(match_history)}")
        
        return response_data

    except Exception as e:
        print(f"❌ Error during matching: {e}")
        raise HTTPException(
            status_code=500, 
            detail=f"Internal server error during match processing: {str(e)}"
        )

@app.get("/api/find-companions")
def find_companions_info():
    """Info about the find-companions endpoint"""
    return {
        "message": "POST to this endpoint with a QueryUser object to find travel companions",
        "method": "POST",
        "public_url": ngrok_public_url,
        "example_endpoint": f"{ngrok_public_url}/api/find-companions" if ngrok_public_url else "Not available",
        "documentation": f"{ngrok_public_url}/docs" if ngrok_public_url else "Not available"
    }

@app.get("/api/match-history")
def get_match_history(limit: Optional[int] = None):
    """
    Retrieve all stored match results.
    Query parameter 'limit' to get only recent results.
    Example: /api/match-history?limit=10
    """
    history = match_history[-limit:] if limit else match_history
    return {
        "total_requests": len(match_history),
        "returned": len(history),
        "history": history
    }

@app.delete("/api/match-history")
def clear_match_history():
    """Clear all stored match history"""
    global match_history
    count = len(match_history)
    match_history = []
    return {
        "status": "cleared",
        "deleted_count": count
    }

@app.get("/")
def root():
    """Root endpoint with API information"""
    return {
        "service": "RoverMitra Matchmaking API",
        "version": "2.0.0",
        "public_url": ngrok_public_url,
        "endpoints": {
            "find_companions": "POST /api/matches-bulk",
            "match_history": "GET /api/match-history",
            "clear_history": "DELETE /api/match-history",
            "docs": "GET /docs",
            "health": "GET /health"
        },
        "status": "running"
    }

@app.get("/health")
def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "requests_processed": len(match_history),
        "public_url": ngrok_public_url
    }

def run_server():
    """Separate function to run the server (Windows multiprocessing compatible)"""
    global ngrok_public_url
    
    port = 5003
    
    # Create ngrok tunnel BEFORE starting the server
    print("🌐 Creating Ngrok tunnel...")
    tunnel = ngrok.connect(port)
    ngrok_public_url = tunnel.public_url
    
    # Display connection info
    print("\n" + "="*80)
    print("🚀 RoverMitra Matchmaking API is LIVE!")
    print("="*80)
    print(f"📍 Public URL:        {ngrok_public_url}")
    print(f"📖 API Documentation: {ngrok_public_url}/docs")
    print(f"🔍 Find Companions:   POST {ngrok_public_url}/api/matches-bulk")
    print(f"📊 Match History:     GET  {ngrok_public_url}/api/match-history")
    print(f"❤️  Health Check:      GET  {ngrok_public_url}/health")
    print("="*80)
    print("💡 Use the public URL above to access your API from anywhere!")
    print("="*80 + "\n")
    
    # Start the server
    uvicorn.run("api:app", host="0.0.0.0", port=port, reload=False)

if __name__ == "__main__":
    run_server()