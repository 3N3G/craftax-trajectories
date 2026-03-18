#!/bin/bash
# Simple installation script for craftax-trajectories
# Works on Apple Silicon Macs

echo "=== Craftax Trajectories Installation ==="
echo ""

# Remove old environment if it exists
echo "Removing old craftax-recording environment (if exists)..."
~/anaconda3/bin/conda env remove -n craftax-recording -y 2>/dev/null || true

# Wait a moment for cleanup
sleep 1

# Create new environment with Python 3.9
echo ""
echo "Creating conda environment with Python 3.9..."
~/anaconda3/bin/conda env create -f environment.yml

# Activate and install remaining packages
echo ""
echo "Installing JAX..."
source ~/anaconda3/etc/profile.d/conda.sh
conda activate craftax-recording

pip install jax
pip install jaxlib --find-links https://storage.googleapis.com/jax-releases/jax_releases.html

# Install Craftax
echo ""
echo "Installing Craftax..."
if [ ! -d "Craftax" ]; then
    git clone https://github.com/MichaelTMatthews/Craftax.git
fi
cd Craftax
pip install -e .
cd ..

# Install this package
echo ""
echo "Installing craftax-trajectories..."
pip install -e .

echo ""
echo "=== Installation Complete! ==="
echo ""
echo "To use:"
echo "  conda activate craftax-recording"
echo "  python -m recorder.play --help"
echo ""
