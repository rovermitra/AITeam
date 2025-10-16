#!/usr/bin/env python3
"""
Railway Database Data Fetcher
============================

This script fetches both datasets from Railway Postgres database:
1. Users table - Basic user information
2. Matchmaker profiles table - Travel matching preferences

Usage:
    python fetch_railway_data.py [--output-dir OUTPUT_DIR] [--limit LIMIT]
"""

import os
import json
import psycopg2
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
import pandas as pd

# Railway Database Configuration
RAILWAY_DB_URL = "postgresql://postgres:RzBikKnKvwEeEUMDmGYFskiVJStCeOOH@hopper.proxy.rlwy.net:11809/railway"

class RailwayDataFetcher:
    """Fetches data from Railway Postgres database"""
    
    def __init__(self, db_url: str = RAILWAY_DB_URL):
        self.db_url = db_url
        self.conn = None
        self.cursor = None
    
    def connect(self):
        """Establish connection to Railway database"""
        try:
            print("🔗 Connecting to Railway Postgres...")
            self.conn = psycopg2.connect(self.db_url)
            self.cursor = self.conn.cursor()
            print("✅ Connected to Railway Postgres")
            return True
        except Exception as e:
            print(f"❌ Failed to connect to Railway database: {e}")
            return False
    
    def disconnect(self):
        """Close database connection"""
        if self.cursor:
            self.cursor.close()
        if self.conn:
            self.conn.close()
        print("🔌 Database connection closed")
    
    def get_table_info(self) -> Dict[str, Any]:
        """Get information about available tables"""
        if not self.cursor:
            return {}
        
        try:
            # Get all tables
            self.cursor.execute("""
                SELECT table_name, 
                       (SELECT COUNT(*) FROM information_schema.columns 
                        WHERE table_name = t.table_name AND table_schema = 'public') as column_count
                FROM information_schema.tables t
                WHERE table_schema = 'public'
                ORDER BY table_name;
            """)
            
            tables = self.cursor.fetchall()
            table_info = {}
            
            for table_name, col_count in tables:
                # Get record count
                self.cursor.execute(f'SELECT COUNT(*) FROM "{table_name}";')
                record_count = self.cursor.fetchone()[0]
                
                table_info[table_name] = {
                    'columns': col_count,
                    'records': record_count
                }
            
            return table_info
        except Exception as e:
            print(f"❌ Error getting table info: {e}")
            return {}
    
    def fetch_users_data(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Fetch users data from Railway database"""
        if not self.cursor:
            return []
        
        try:
            print("📊 Fetching Users data...")
            
            # Build query with optional limit
            query = """
                SELECT "Id", "Email", "FirstName", "LastName", "MiddleName", 
                       "DateOfBirth", "Address", "City", "State", "PostalCode", 
                       "Country", "CreatedAt", "UpdatedAt", "IsActive", 
                       "IsEmailVerified", "IsPhoneVerified", "ProfilePictureUrl",
                       "UserName", "PhoneNumber", "IsProfileComplete", 
                       "IsIdVerified", "IdVerifiedAt"
                FROM "Users"
            """
            
            if limit:
                query += f" LIMIT {limit}"
            
            self.cursor.execute(query)
            columns = [desc[0] for desc in self.cursor.description]
            rows = self.cursor.fetchall()
            
            users_data = []
            for row in rows:
                user_dict = {}
                for i, value in enumerate(row):
                    # Convert datetime objects to ISO format
                    if isinstance(value, datetime):
                        user_dict[columns[i]] = value.isoformat()
                    else:
                        user_dict[columns[i]] = value
                users_data.append(user_dict)
            
            print(f"✅ Fetched {len(users_data)} users")
            return users_data
            
        except Exception as e:
            print(f"❌ Error fetching users data: {e}")
            return []
    
    def fetch_matchmaker_profiles_data(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Fetch matchmaker profiles data from Railway database"""
        if not self.cursor:
            return []
        
        try:
            print("🎯 Fetching Matchmaker Profiles data...")
            
            # Build query with optional limit
            query = """
                SELECT id, match_profile_id, email, status, created_at, updated_at,
                       visibility, preferences, compatibility_scores, raw_data
                FROM matchmaker_profiles
            """
            
            if limit:
                query += f" LIMIT {limit}"
            
            self.cursor.execute(query)
            columns = [desc[0] for desc in self.cursor.description]
            rows = self.cursor.fetchall()
            
            profiles_data = []
            for row in rows:
                profile_dict = {}
                for i, value in enumerate(row):
                    column_name = columns[i]
                    
                    # Handle JSON fields
                    if column_name in ['visibility', 'preferences', 'compatibility_scores', 'raw_data']:
                        if value:
                            try:
                                if isinstance(value, str):
                                    profile_dict[column_name] = json.loads(value)
                                else:
                                    profile_dict[column_name] = value
                            except json.JSONDecodeError:
                                profile_dict[column_name] = str(value)
                        else:
                            profile_dict[column_name] = None
                    # Handle datetime fields
                    elif isinstance(value, datetime):
                        profile_dict[column_name] = value.isoformat()
                    else:
                        profile_dict[column_name] = value
                
                profiles_data.append(profile_dict)
            
            print(f"✅ Fetched {len(profiles_data)} matchmaker profiles")
            return profiles_data
            
        except Exception as e:
            print(f"❌ Error fetching matchmaker profiles data: {e}")
            return []
    
    def save_to_json(self, data: List[Dict[str, Any]], filename: str, output_dir: Path):
        """Save data to JSON file"""
        output_path = output_dir / filename
        
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False, default=str)
            print(f"💾 Saved {len(data)} records to {output_path}")
            return True
        except Exception as e:
            print(f"❌ Error saving to {output_path}: {e}")
            return False
    
    def save_to_csv(self, data: List[Dict[str, Any]], filename: str, output_dir: Path):
        """Save data to CSV file"""
        if not data:
            return False
        
        output_path = output_dir / filename
        
        try:
            df = pd.DataFrame(data)
            df.to_csv(output_path, index=False, encoding='utf-8')
            print(f"📊 Saved {len(data)} records to {output_path}")
            return True
        except Exception as e:
            print(f"❌ Error saving to {output_path}: {e}")
            return False
    
    def create_summary_report(self, users_data: List[Dict[str, Any]], 
                            profiles_data: List[Dict[str, Any]], 
                            output_dir: Path):
        """Create a summary report of the fetched data"""
        report_path = output_dir / "data_summary_report.txt"
        
        try:
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write("Railway Database Data Summary Report\n")
                f.write("=" * 50 + "\n\n")
                f.write(f"Generated: {datetime.now().isoformat()}\n\n")
                
                # Users summary
                f.write("USERS DATASET:\n")
                f.write("-" * 20 + "\n")
                f.write(f"Total Records: {len(users_data):,}\n")
                if users_data:
                    f.write(f"Sample Users:\n")
                    for i, user in enumerate(users_data[:3], 1):
                        f.write(f"  {i}. {user.get('FirstName', 'N/A')} {user.get('LastName', 'N/A')} ({user.get('Email', 'N/A')})\n")
                        f.write(f"     Location: {user.get('City', 'N/A')}, {user.get('Country', 'N/A')}\n")
                        f.write(f"     Created: {user.get('CreatedAt', 'N/A')}\n")
                
                f.write("\nMATCHMAKER PROFILES DATASET:\n")
                f.write("-" * 30 + "\n")
                f.write(f"Total Records: {len(profiles_data):,}\n")
                if profiles_data:
                    f.write(f"Sample Profiles:\n")
                    for i, profile in enumerate(profiles_data[:3], 1):
                        f.write(f"  {i}. {profile.get('email', 'N/A')}\n")
                        f.write(f"     Status: {profile.get('status', 'N/A')}\n")
                        f.write(f"     Profile ID: {profile.get('match_profile_id', 'N/A')}\n")
                        
                        # Extract some raw_data info
                        raw_data = profile.get('raw_data', {})
                        if raw_data:
                            f.write(f"     Faith: {raw_data.get('faith_preference', {}).get('religion', 'N/A')}\n")
                            f.write(f"     Match Intent: {raw_data.get('match_intent', [])}\n")
                
                f.write("\nDATA RELATIONSHIP:\n")
                f.write("-" * 20 + "\n")
                if users_data and profiles_data:
                    user_emails = {user.get('Email') for user in users_data}
                    profile_emails = {profile.get('email') for profile in profiles_data}
                    linked_emails = user_emails.intersection(profile_emails)
                    f.write(f"Linked Records: {len(linked_emails):,}\n")
                    f.write(f"Linkage Rate: {(len(linked_emails)/len(users_data))*100:.1f}%\n")
            
            print(f"📋 Summary report saved to {report_path}")
            return True
        except Exception as e:
            print(f"❌ Error creating summary report: {e}")
            return False

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Fetch data from Railway Postgres database")
    parser.add_argument("--output-dir", default="railway_data", 
                       help="Output directory for data files (default: railway_data)")
    parser.add_argument("--limit", type=int, 
                       help="Limit number of records to fetch (default: fetch all)")
    parser.add_argument("--format", choices=["json", "csv", "both"], default="both",
                       help="Output format (default: both)")
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    print(f"📁 Output directory: {output_dir.absolute()}")
    
    # Initialize fetcher
    fetcher = RailwayDataFetcher()
    
    try:
        # Connect to database
        if not fetcher.connect():
            return 1
        
        # Get table information
        print("\n📊 Database Table Information:")
        table_info = fetcher.get_table_info()
        for table_name, info in table_info.items():
            print(f"  {table_name}: {info['records']:,} records, {info['columns']} columns")
        
        # Fetch users data
        print(f"\n{'='*60}")
        users_data = fetcher.fetch_users_data(args.limit)
        
        # Fetch matchmaker profiles data
        print(f"\n{'='*60}")
        profiles_data = fetcher.fetch_matchmaker_profiles_data(args.limit)
        
        # Save data
        print(f"\n{'='*60}")
        print("💾 Saving data...")
        
        if args.format in ["json", "both"]:
            fetcher.save_to_json(users_data, "users_data.json", output_dir)
            fetcher.save_to_json(profiles_data, "matchmaker_profiles_data.json", output_dir)
        
        if args.format in ["csv", "both"]:
            fetcher.save_to_csv(users_data, "users_data.csv", output_dir)
            fetcher.save_to_csv(profiles_data, "matchmaker_profiles_data.csv", output_dir)
        
        # Create summary report
        fetcher.create_summary_report(users_data, profiles_data, output_dir)
        
        print(f"\n✅ Data extraction completed successfully!")
        print(f"📁 Files saved to: {output_dir.absolute()}")
        
        return 0
        
    except KeyboardInterrupt:
        print("\n⚠️ Operation cancelled by user")
        return 1
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        return 1
    finally:
        fetcher.disconnect()

if __name__ == "__main__":
    exit(main())
