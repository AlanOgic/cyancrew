#!/bin/bash
# Setup script for the Product Research and Sales Guide CrewAI Project
# For Unix-based systems (macOS/Linux)

# Print header
echo
echo "=================================================="
echo "Product Research and Sales Guide CrewAI Project Setup"
echo "=================================================="
echo

# Check Python version
echo "Checking Python version..."
python_version=$(python3 --version 2>&1)
if [[ $python_version =~ Python\ 3\.([0-9]+) ]]; then
    minor_version=${BASH_REMATCH[1]}
    if [ "$minor_version" -lt 8 ]; then
        echo "Error: Python 3.8+ is required."
        echo "Current Python version: $python_version"
        exit 1
    else
        echo "Python version check passed: $python_version"
    fi
else
    echo "Error: Python 3.8+ is required."
    echo "Current Python version: $python_version"
    exit 1
fi

# Create virtual environment
echo
echo "Creating virtual environment..."
if [ -d "venv" ]; then
    echo "Virtual environment already exists."
else
    python3 -m venv venv
    if [ $? -eq 0 ]; then
        echo "Virtual environment created successfully."
    else
        echo "Error creating virtual environment."
        exit 1
    fi
fi

# Activate virtual environment
echo
echo "Activating virtual environment..."
source venv/bin/activate
if [ $? -eq 0 ]; then
    echo "Virtual environment activated."
else
    echo "Error activating virtual environment."
    exit 1
fi

# Install dependencies
echo
echo "Installing dependencies..."
pip install -r requirements.txt
if [ $? -eq 0 ]; then
    echo "Dependencies installed successfully."
else
    echo "Error installing dependencies."
    exit 1
fi

# Print success message
echo
echo "=================================================="
echo "Setup completed successfully!"
echo "=================================================="
echo

# Print instructions
echo "To use the project:"
echo "1. Activate the virtual environment (if not already activated):"
echo "   source venv/bin/activate"
echo "2. Run the script:"
echo "   python crew-1.py"
echo "3. When finished, deactivate the virtual environment:"
echo "   deactivate"
echo

# Deactivate virtual environment
deactivate
