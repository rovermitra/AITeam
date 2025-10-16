#!/usr/bin/env python3
"""
Upload matchmaker data to Railway .NET API
"""

import requests
import json
import os
import time
from pathlib import Path

def upload_matchmaker_data():
    """Upload matchmaker profiles to Railway .NET API"""
    
    # Configuration
    base_url = os.getenv("RAILWAY_API_URL", "https://api.rovermitra.com")
    matchmaker_file = Path("../MatchMaker/data/matchmaker_profiles.json")
    
    if not matchmaker_file.exists():
        print(f"❌ Matchmaker file not found: {matchmaker_file}")
        return
    
    # Load matchmaker data
    print(f"📁 Loading matchmaker data from {matchmaker_file}")
    with open(matchmaker_file, "r", encoding="utf-8") as f:
        matchmaker_data = json.load(f)
    
    print(f"📊 Found {len(matchmaker_data)} matchmaker profiles")
    
    # Upload each matchmaker profile
    success_count = 0
    for i, profile in enumerate(matchmaker_data, 1):
        try:
            print(f"📤 Uploading profile {i}/{len(matchmaker_data)}: {profile.get('email', 'unknown')}")
            
            # Try different endpoints for matchmaker data
            endpoints_to_try = [
                "/Matching/matchmaker-profile",
                "/User/matchmaker-profile", 
                "/Matching/profile",
                "/User/preferences"  # Fallback to preferences endpoint
            ]
            
            uploaded = False
            for endpoint in endpoints_to_try:
                try:
                    response = requests.post(
                        f"{base_url}{endpoint}",
                        json=profile,
                        headers={"Content-Type": "application/json"},
                        timeout=30
                    )
                    
                    if response.status_code in [200, 201]:
                        print(f"✅ Uploaded to {endpoint}: {profile.get('email', 'unknown')}")
                        uploaded = True
                        success_count += 1
                        break
                    else:
                        print(f"⚠️  {endpoint} returned {response.status_code}: {response.text[:100]}")
                        
                except requests.exceptions.RequestException as e:
                    print(f"❌ Error with {endpoint}: {e}")
                    continue
            
            if not uploaded:
                print(f"❌ Failed to upload {profile.get('email', 'unknown')}")
            
            # Rate limiting
            time.sleep(0.1)
            
        except Exception as e:
            print(f"❌ Error processing profile {i}: {e}")
    
    print(f"\n🎉 Upload complete!")
    print(f"✅ Successfully uploaded: {success_count}/{len(matchmaker_data)} profiles")
    print(f"🌐 API URL: {base_url}")

def test_api_connection():
    """Test connection to Railway API"""
    base_url = os.getenv("RAILWAY_API_URL", "https://api.rovermitra.com")
    
    try:
        print(f"🔍 Testing connection to {base_url}")
        response = requests.get(f"{base_url}/api/health", timeout=10)
        
        if response.status_code == 200:
            print("✅ API connection successful!")
            print(f"Response: {response.json()}")
            return True
        else:
            print(f"❌ API returned status {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Connection failed: {e}")
        return False

if __name__ == "__main__":
    print("🚀 RoverMitra Matchmaker Data Uploader")
    print("=" * 50)
    
    # Test connection first
    if test_api_connection():
        upload_matchmaker_data()
    else:
        print("\n❌ Cannot connect to Railway API")
        print("Please check:")
        print("1. Railway API URL is correct")
        print("2. .NET backend is deployed and running")
        print("3. Environment variable RAILWAY_API_URL is set")
        print("\nExample:")
        print("export RAILWAY_API_URL=https://your-app.railway.app")
        print("python upload_matchmaker_data.py")
