#!/usr/bin/env python3
"""
Upload matchmaker data directly to Railway Postgres database
"""

import psycopg2
import json
import os
from pathlib import Path

def upload_matchmaker_data():
    """Upload matchmaker data directly to Railway Postgres"""
    
    DATABASE_URL = "postgresql://postgres:YOUR_PASSWORD@YOUR_HOST:YOUR_PORT/YOUR_DATABASE"
    
    try:
        print("🔗 Connecting to Railway Postgres...")
        conn = psycopg2.connect(DATABASE_URL)
        cursor = conn.cursor()
        
        print("✅ Connected to Railway Postgres")
        
        # Load matchmaker data
        matchmaker_file = Path("MatchMaker/data/matchmaker_profiles.json")
        if not matchmaker_file.exists():
            print(f"❌ Matchmaker file not found: {matchmaker_file}")
            return False
        
        print(f"📁 Loading matchmaker data from {matchmaker_file}")
        with open(matchmaker_file, "r", encoding="utf-8") as f:
            matchmaker_data = json.load(f)
        
        print(f"📊 Found {len(matchmaker_data)} matchmaker profiles")
        
        # Create a table for matchmaker data if it doesn't exist
        print("🔧 Creating matchmaker_profiles table...")
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS matchmaker_profiles (
                id SERIAL PRIMARY KEY,
                match_profile_id VARCHAR(255) UNIQUE NOT NULL,
                email VARCHAR(255) NOT NULL,
                status VARCHAR(50),
                created_at TIMESTAMP WITH TIME ZONE,
                updated_at TIMESTAMP WITH TIME ZONE,
                visibility JSONB,
                preferences JSONB,
                compatibility_scores JSONB,
                raw_data JSONB
            )
        """)
        
        # Clear existing matchmaker data
        print("🧹 Clearing existing matchmaker data...")
        cursor.execute("DELETE FROM matchmaker_profiles")
        deleted_count = cursor.rowcount
        print(f"  - Deleted {deleted_count} existing profiles")
        
        # Upload matchmaker profiles
        print("📤 Uploading matchmaker profiles...")
        success_count = 0
        
        for i, profile in enumerate(matchmaker_data, 1):
            try:
                # Extract key fields
                match_profile_id = profile.get("match_profile_id", f"mm_{i}")
                email = profile.get("email", "")
                status = profile.get("status", "active")
                created_at = profile.get("created_at")
                updated_at = profile.get("updated_at")
                visibility = json.dumps(profile.get("visibility", {}))
                preferences = json.dumps(profile.get("preferences", {}))
                compatibility_scores = json.dumps(profile.get("compatibility_scores", {}))
                raw_data = json.dumps(profile)
                
                # Insert into database
                cursor.execute("""
                    INSERT INTO matchmaker_profiles 
                    (match_profile_id, email, status, created_at, updated_at, 
                     visibility, preferences, compatibility_scores, raw_data)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (match_profile_id) DO UPDATE SET
                        email = EXCLUDED.email,
                        status = EXCLUDED.status,
                        updated_at = EXCLUDED.updated_at,
                        visibility = EXCLUDED.visibility,
                        preferences = EXCLUDED.preferences,
                        compatibility_scores = EXCLUDED.compatibility_scores,
                        raw_data = EXCLUDED.raw_data
                """, (match_profile_id, email, status, created_at, updated_at,
                      visibility, preferences, compatibility_scores, raw_data))
                
                success_count += 1
                
                if i % 1000 == 0:
                    print(f"  - Uploaded {i}/{len(matchmaker_data)} profiles...")
                    conn.commit()  # Commit every 1000 records
                    
            except Exception as e:
                print(f"❌ Error uploading profile {i}: {e}")
                continue
        
        # Final commit
        conn.commit()
        
        print(f"✅ Successfully uploaded {success_count}/{len(matchmaker_data)} matchmaker profiles")
        
        # Verify upload
        cursor.execute("SELECT COUNT(*) FROM matchmaker_profiles")
        final_count = cursor.fetchone()[0]
        print(f"🎉 Final result: {final_count} matchmaker profiles in database")
        
        cursor.close()
        conn.close()
        
        print("✅ Matchmaker data upload completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    print("🚀 Matchmaker Data Upload to Railway Postgres")
    print("=" * 50)
    upload_matchmaker_data()
