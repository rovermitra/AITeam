#!/bin/bash
# Setup script for RoverMitra - Enhanced Travel Companion Matching

set -e  # Exit on any error

echo "🚀 Setting up RoverMitra - Enhanced Travel Companion Matching..."
echo "================================================================"

# Check Python version
echo "🐍 Checking Python version..."
python3 --version
if ! python3 -c "import sys; exit(0 if sys.version_info >= (3, 8) else 1)"; then
    echo "❌ Python 3.8+ is required. Please upgrade Python."
    exit 1
fi

# Create necessary directories
echo "📁 Creating project directories..."
mkdir -p artifacts
mkdir -p models
mkdir -p logs
mkdir -p users/data
mkdir -p MatchMaker/data
mkdir -p Activities/data
mkdir -p Events/data
mkdir -p Flight/data
mkdir -p Hotels/data
mkdir -p Rentals/data
mkdir -p Restaurants/data
mkdir -p test_output
mkdir -p tmp

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "🔧 Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "🔌 Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "⬆️  Upgrading pip..."
pip install --upgrade pip

# Install dependencies
echo "📦 Installing dependencies..."
pip install -r requirements.txt

# Make scripts executable
echo "🔧 Making scripts executable..."
chmod +x main.py
chmod +x Scripts/*.py
chmod +x finetune_llama.py

# Generate initial data if it doesn't exist
if [ ! -f "users/data/users_core.json" ]; then
    echo "👥 Generating initial user data..."
    python Scripts/user_data_generator.py
fi

if [ ! -f "MatchMaker/data/matchmaker_profiles.json" ]; then
    echo "🤝 Generating matchmaker profiles..."
    python Scripts/matchmaker_data_generator.py
fi

# Run basic tests
echo "🧪 Running basic tests..."
python -c "
import json
import os
print('✅ JSON module working')
print('✅ OS module working')

# Test data files exist
if os.path.exists('users/data/users_core.json'):
    with open('users/data/users_core.json', 'r') as f:
        users = json.load(f)
    print(f'✅ User data loaded: {len(users)} users')
else:
    print('⚠️  User data not found')

if os.path.exists('MatchMaker/data/matchmaker_profiles.json'):
    with open('MatchMaker/data/matchmaker_profiles.json', 'r') as f:
        profiles = json.load(f)
    print(f'✅ Matchmaker profiles loaded: {len(profiles)} profiles')
else:
    print('⚠️  Matchmaker profiles not found')
"

echo ""
echo "✅ Setup complete!"
echo "================================================================"
echo ""
echo "🎯 Next steps:"
echo "1. Generate training data: python Scripts/generate_llama_training_data.py --num-examples 1000"
echo "2. Fine-tune model: python finetune_llama.py"
echo "3. Run matching: python main.py"
echo "4. Run tests: python -m pytest"
echo ""
echo "📊 Data generation:"
echo "- Users: python Scripts/user_data_generator.py"
echo "- Matchmaker profiles: python Scripts/matchmaker_data_generator.py"
echo "- Training data: python Scripts/generate_llama_training_data.py"
echo ""
echo "🔧 Development:"
echo "- Format code: black ."
echo "- Lint code: flake8 ."
echo "- Run tests: pytest"
echo ""
echo "🚀 Happy coding with RoverMitra!"
