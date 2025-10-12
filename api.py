from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import uvicorn

# Import the core logic and initialization functions
from matcher import find_matches, load_pool, initialize_models

# --- Pydantic Models for API data validation ---
# These models define the exact structure of your API's JSON input and output.

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
    user_id: str
    name: str
    explanation: str
    compatibility_score: float = Field(..., ge=0.0, le=1.0)

class MatchResponse(BaseModel):
    matches: List[MatchResult]

# --- FastAPI Application ---
app = FastAPI(
    title="RoverMitra Matchmaking API",
    description="Provides travel companion matches based on user profiles.",
    version="1.0.0"
)

@app.on_event("startup")
def startup_event():
    """Actions to perform on application startup."""
    print("--- Server starting up ---")
    load_pool()          # Load user data from JSONs
    initialize_models()  # Load ML models into memory/GPU
    print("--- Server ready to accept requests ---")

@app.post("/match", response_model=MatchResponse)
def create_matches(query_user: QueryUser):
    """
    Accepts a user profile and returns a list of top potential travel matches.
    """
    try:
        # Convert Pydantic model back to a dictionary for the matcher functions
        user_dict = query_user.model_dump()
        
        results = find_matches(user_dict)
        
        return {"matches": results}
    except Exception as e:
        print(f"An error occurred during matching: {e}")
        # Be specific about the error in the logs, but generic to the client
        raise HTTPException(status_code=500, detail="Internal server error during match processing.")

if __name__ == "__main__":
    # This allows you to run the API directly for testing
    # For production, use: uvicorn api:app --host 0.0.0.0 --port 8000
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
