#!/bin/bash

echo "Setting up the environment..."

# Ensure pip is installed
if ! command -v pip &> /dev/null; then
    echo "pip not found. Please install Python and pip first."
    exit 1
fi

# Install dependencies from requirements.txt
echo "Installing dependencies..."
pip install -r requirements.txt

echo "Setup complete!"
