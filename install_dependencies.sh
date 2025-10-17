#!/bin/bash

# JAX-FEM Installation Script with NumPy Compatibility Fix
# This script handles NumPy 1.x vs 2.x compatibility issues

echo "JAX-FEM Installation Script"
echo "============================"

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "❌ Python3 not found. Please install Python 3.8+ first."
    exit 1
fi

echo "✓ Python3 found: $(python3 --version)"

# Check if pip is available
if ! command -v pip3 &> /dev/null; then
    echo "❌ pip3 not found. Please install pip first."
    exit 1
fi

echo "✓ pip3 found"

# Create virtual environment if it doesn't exist
if [ ! -d "jax_fem_env" ]; then
    echo "Creating virtual environment..."
    python3 -m venv jax_fem_env
fi

# Activate virtual environment
echo "Activating virtual environment..."
source jax_fem_env/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Uninstall any existing problematic packages
echo "Cleaning up existing packages..."
pip uninstall -y numpy jax jaxlib jax-fem 2>/dev/null || true

# Install NumPy 1.x (compatible version)
echo "Installing NumPy 1.x..."
pip install "numpy>=1.21.0,<2.0.0"

# Install JAX CPU version
echo "Installing JAX CPU version..."
pip install jax jaxlib

# Try to install JAX with CUDA support
echo "Attempting to install JAX with CUDA support..."
pip install --upgrade "jax[cuda]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html || {
    echo "⚠️  CUDA installation failed, continuing with CPU version"
}

# Install JAX-FEM
echo "Installing JAX-FEM..."
pip install jax-fem

# Install other requirements
echo "Installing other requirements..."
pip install matplotlib scipy

# Test installation
echo "Testing installation..."
python test_installation.py

echo ""
echo "Installation completed!"
echo "To activate the environment in the future, run:"
echo "source jax_fem_env/bin/activate"
echo ""
echo "To run the benchmark:"
echo "python run_benchmark.py"

