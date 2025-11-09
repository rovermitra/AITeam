#!/bin/bash

# RoverMitra API Test Script
# Usage: ./test_api.sh

API_URL="http://localhost:8003"

echo "🧪 RoverMitra API Test Suite"
echo "=============================="

# Test 1: Health Check
echo "1️⃣ Testing Health Check..."
curl -s "$API_URL/health" | python -m json.tool
echo ""

# Test 2: Test Alex Berlin
echo "2️⃣ Testing Alex Berlin (Male, Museums, Architecture)..."
curl -X POST "$API_URL/api/matches-bulk" \
  -H "Content-Type: application/json" \
  -d @test_user_alex.json | python -m json.tool
echo ""

# Test 3: Test Maria Munich
echo "3️⃣ Testing Maria Munich (Female, Food Tours, Christian)..."
curl -X POST "$API_URL/api/matches-bulk" \
  -H "Content-Type: application/json" \
  -d @test_user_maria.json | python -m json.tool
echo ""

# Test 4: Test Tom Karachi
echo "4️⃣ Testing Tom Karachi (Male, Adventure, Muslim)..."
curl -X POST "$API_URL/api/matches-bulk" \
  -H "Content-Type: application/json" \
  -d @test_user_tom.json | python -m json.tool
echo ""

# Test 5: Match History
echo "5️⃣ Testing Match History..."
curl -s "$API_URL/api/match-history" | python -m json.tool
echo ""

echo "✅ All tests completed!"
echo "📖 API Documentation: $API_URL/docs"
