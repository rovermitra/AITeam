#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RoverMitra AI API - Railway Deployment Ready
Flask API wrapper for the AI matching system
"""

import os
import json
import uuid
import math
from pathlib import Path
from typing import Any, Dict, List, Optional
from flask import Flask, request, jsonify
from flask_cors import CORS
import psycopg2
from psycopg2.extras import RealDictCursor

# Import the AI matching functions from Updated_main.py
from Updated_main import (
    interactive_new_user, load_pool, hard_prefilter, 
    ai_prefilter, llm_rank, summarize_user, query_text
)

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend integration

# ----------------------------
# Database Configuration
# ----------------------------
def get_db_connection():
    """Get PostgreSQL connection using Railway environment variables"""
    try:
        # Railway provides DATABASE_URL automatically
        database_url = os.getenv('DATABASE_URL')
        if not database_url:
            # Fallback for local development
            database_url = os.getenv('POSTGRES_URL', 'postgresql://localhost/rovermitra')
        
        conn = psycopg2.connect(database_url)
        return conn
    except Exception as e:
        print(f"Database connection error: {e}")
        return None

# ----------------------------
# API Routes
# ----------------------------
@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "service": "rovermitra-ai",
        "version": "1.0.0"
    })

@app.route('/api/matches', methods=['POST'])
def find_matches():
    """
    Find travel companion matches for a user
    Expected JSON payload:
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
    """
    try:
        # Get user data from request
        user_data = request.get_json()
        if not user_data:
            return jsonify({"error": "No JSON data provided"}), 400
        
        # Validate required fields
        required_fields = ['name', 'age', 'gender', 'home_base', 'languages', 'interests']
        for field in required_fields:
            if field not in user_data:
                return jsonify({"error": f"Missing required field: {field}"}), 400
        
        # Add user ID if not present
        if 'id' not in user_data:
            user_data['id'] = f"u_api_{uuid.uuid4().hex[:8]}"
        
        # Load candidate pool from database
        pool = load_pool()
        if not pool:
            return jsonify({"error": "No candidates found in database"}), 500
        
        # Apply hard prefilters
        hard_filtered = hard_prefilter(user_data, pool)
        if not hard_filtered:
            return jsonify({
                "matches": [],
                "message": "No candidates passed initial filters"
            })
        
        # Apply AI prefilter
        shortlist = ai_prefilter(user_data, hard_filtered, percent=0.02, min_k=80)
        
        # Final ranking
        final_matches = llm_rank(user_data, shortlist, out_top=5)
        
        # Format response
        response = {
            "user_id": user_data['id'],
            "matches": final_matches,
            "total_candidates": len(pool),
            "after_hard_filter": len(hard_filtered),
            "after_ai_filter": len(shortlist),
            "final_matches": len(final_matches)
        }
        
        return jsonify(response)
        
    except Exception as e:
        print(f"Error in find_matches: {e}")
        return jsonify({"error": "Internal server error"}), 500

@app.route('/api/profile', methods=['POST'])
def create_profile():
    """
    Create a new user profile and save to database
    """
    try:
        user_data = request.get_json()
        if not user_data:
            return jsonify({"error": "No JSON data provided"}), 400
        
        # Add user ID
        user_data['id'] = f"u_api_{uuid.uuid4().hex[:8]}"
        
        # Save to database (you'll need to implement this based on your schema)
        conn = get_db_connection()
        if conn:
            # TODO: Implement database save logic
            # This depends on your existing user table schema
            pass
        
        return jsonify({
            "user_id": user_data['id'],
            "message": "Profile created successfully",
            "profile": user_data
        })
        
    except Exception as e:
        print(f"Error in create_profile: {e}")
        return jsonify({"error": "Internal server error"}), 500

@app.route('/api/candidates', methods=['GET'])
def get_candidates():
    """
    Get all available candidates (for testing/debugging)
    """
    try:
        pool = load_pool()
        candidates = []
        
        for rec in pool[:10]:  # Limit to first 10 for response size
            user = rec['user']
            candidates.append({
                "user_id": user.get('user_id'),
                "name": user.get('name'),
                "age": user.get('age'),
                "location": user.get('home_base', {}).get('city'),
                "summary": summarize_user(user, rec.get('mm'))
            })
        
        return jsonify({
            "candidates": candidates,
            "total": len(pool)
        })
        
    except Exception as e:
        print(f"Error in get_candidates: {e}")
        return jsonify({"error": "Internal server error"}), 500

@app.route('/api/test', methods=['GET'])
def test_ai():
    """
    Test endpoint to verify AI functionality
    """
    try:
        # Create a test user
        test_user = {
            "id": "test_user",
            "name": "Test User",
            "age": 28,
            "gender": "Other",
            "home_base": {"city": "Berlin", "country": "Germany"},
            "languages": ["en", "de"],
            "interests": ["museums", "hiking"],
            "travel_prefs": {"pace": "balanced"},
            "budget": {"amount": 150, "currency": "EUR"},
            "diet_health": {"diet": "none"},
            "comfort": {"smoking": "never", "alcohol": "moderate"},
            "bio": "Test user for AI verification"
        }
        
        # Test the pipeline
        pool = load_pool()
        if not pool:
            return jsonify({"error": "No candidates in database"}), 500
        
        hard_filtered = hard_prefilter(test_user, pool)
        shortlist = ai_prefilter(test_user, hard_filtered, percent=0.1, min_k=5)
        matches = llm_rank(test_user, shortlist, out_top=3)
        
        return jsonify({
            "status": "AI system working",
            "test_results": {
                "total_candidates": len(pool),
                "after_hard_filter": len(hard_filtered),
                "after_ai_filter": len(shortlist),
                "final_matches": len(matches)
            },
            "sample_matches": matches
        })
        
    except Exception as e:
        print(f"Error in test_ai: {e}")
        return jsonify({"error": f"AI test failed: {str(e)}"}), 500

# ----------------------------
# Error Handlers
# ----------------------------
@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "Endpoint not found"}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({"error": "Internal server error"}), 500

# ----------------------------
# Main
# ----------------------------
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
