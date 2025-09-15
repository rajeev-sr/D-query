#!/bin/bash
# scripts/quick_start.sh
# Quick start script for AI Query Handler

echo " AI QUERY HANDLER - QUICK START"
echo "=================================="

# Check Python version
python_version=$(python3 --version 2>&1 | awk '{print $2}' | awk -F. '{print $1"."$2}')
echo " Python version: $python_version"

if [[ $(echo "$python_version < 3.8" | bc -l) -eq 1 ]]; then
    echo " Python 3.8+ required. Current version: $python_version"
    exit 1
fi

# Create virtual environment if not exists
if [ ! -d "venv" ]; then
    echo " Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo " Activating virtual environment..."
source venv/bin/activate

# Install dependencies
echo " Installing dependencies..."
pip install -q --upgrade pip
pip install -q -r requirements.txt

# Create necessary directories
echo " Creating directories..."
mkdir -p data/institute_docs
mkdir -p data/vector_db
mkdir -p config
mkdir -p logs
mkdir -p models

# Check if system is set up
echo " Checking system setup..."

if [ ! -f "models/fine_tuned/config.json" ]; then
    echo "   Fine-tuned model not found. Please run the 5-day setup first:"
    echo "   Day 1: python day1_integration_test.py"
    echo "   Day 2: python day2_integration_test.py" 
    echo "   Day 3: python train_model.py && python day3_integration_test.py"
    echo "   Day 4: python day4_integration_test.py"
    echo "   Day 5: python day5_integration_test.py"
    exit 1
fi

# Create config if not exists
if [ ! -f "config/automation_config.json" ]; then
    echo "⚙️ Creating configuration..."
    python config/create_config.py
fi

# Launch system
echo " Launching AI Query Handler System..."
python scripts/launch_system.py