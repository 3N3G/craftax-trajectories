#!/bin/bash
# Installation script for craftax-trajectories on Apple Silicon (M1/M2/M3)

set -e  # Exit on error

echo "=== Installing craftax-trajectories on Apple Silicon ==="
echo ""

# Check architecture
if [[ $(uname -m) != "arm64" ]]; then
    echo "Warning: This script is for Apple Silicon (arm64). You're running on $(uname -m)."
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Step 1: Install basic dependencies
echo "Step 1/5: Installing basic dependencies..."
pip install -r requirements.txt

# Step 2: Install JAX
echo ""
echo "Step 2/5: Installing JAX..."
pip install jax

# Step 3: Install jaxlib from Google storage
echo ""
echo "Step 3/5: Installing jaxlib for Apple Silicon..."
pip install jaxlib --find-links https://storage.googleapis.com/jax-releases/jax_releases.html

# Step 4: Install Craftax
echo ""
echo "Step 4/5: Installing Craftax..."
if [ ! -d "Craftax" ]; then
    git clone https://github.com/MichaelTMatthews/Craftax.git
fi
cd Craftax
pip install -e .
cd ..

# Step 5: Install this package
echo ""
echo "Step 5/5: Installing craftax-trajectories..."
pip install -e .

echo ""
echo "=== Installation complete! ==="
echo ""
echo "Test the installation with:"
echo "  python -m recorder.play --help"
echo ""
