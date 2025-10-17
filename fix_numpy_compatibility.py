#!/usr/bin/env python3
"""
Fix NumPy compatibility issues for JAX-FEM
"""

import subprocess
import sys
import os

def run_command(cmd, description):
    """Run a command and print the result"""
    print(f"\n{description}...")
    print(f"Running: {cmd}")
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✓ {description} completed successfully")
            if result.stdout:
                print(f"Output: {result.stdout}")
        else:
            print(f"❌ {description} failed")
            print(f"Error: {result.stderr}")
        return result.returncode == 0
    except Exception as e:
        print(f"❌ {description} failed with exception: {e}")
        return False

def check_numpy_version():
    """Check current NumPy version"""
    try:
        import numpy as np
        print(f"Current NumPy version: {np.__version__}")
        return np.__version__
    except ImportError:
        print("NumPy not installed")
        return None

def fix_numpy_compatibility():
    """Fix NumPy compatibility issues"""
    print("NumPy Compatibility Fix")
    print("=" * 40)
    
    # Check current NumPy version
    current_version = check_numpy_version()
    
    if current_version and current_version.startswith('2.'):
        print(f"\n⚠️  Detected NumPy 2.x ({current_version})")
        print("This can cause compatibility issues with JAX-FEM")
        print("Downgrading to NumPy 1.x...")
        
        # Uninstall current NumPy
        if not run_command("pip uninstall numpy -y", "Uninstalling current NumPy"):
            print("Failed to uninstall NumPy")
            return False
        
        # Install compatible NumPy version
        if not run_command("pip install 'numpy>=1.21.0,<2.0.0'", "Installing NumPy 1.x"):
            print("Failed to install compatible NumPy")
            return False
    
    # Reinstall JAX with compatible versions
    print("\nReinstalling JAX with compatible versions...")
    
    # Uninstall JAX packages
    run_command("pip uninstall jax jaxlib -y", "Uninstalling JAX packages")
    
    # Install JAX CPU version first
    if not run_command("pip install jax jaxlib", "Installing JAX CPU version"):
        print("Failed to install JAX CPU version")
        return False
    
    # Try to install JAX CUDA version
    print("\nAttempting to install JAX with CUDA support...")
    cuda_cmd = "pip install --upgrade 'jax[cuda]' -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html"
    if not run_command(cuda_cmd, "Installing JAX with CUDA"):
        print("⚠️  CUDA installation failed, continuing with CPU version")
    
    # Install JAX-FEM
    if not run_command("pip install jax-fem", "Installing JAX-FEM"):
        print("Failed to install JAX-FEM")
        return False
    
    # Install other requirements
    if not run_command("pip install matplotlib scipy", "Installing other requirements"):
        print("Failed to install other requirements")
        return False
    
    print("\n" + "=" * 40)
    print("Installation completed!")
    print("You can now run: python test_installation.py")

def create_conda_environment():
    """Create a conda environment with compatible versions"""
    print("\nAlternative: Creating conda environment...")
    
    conda_cmd = """
    conda create -n jax_fem python=3.9 numpy=1.24 matplotlib scipy -y
    conda activate jax_fem
    pip install jax jaxlib
    pip install jax-fem
    """
    
    print("Run these commands in your terminal:")
    print(conda_cmd)

def main():
    """Main function"""
    print("JAX-FEM NumPy Compatibility Fix")
    print("=" * 50)
    
    # Check if we're in a virtual environment
    if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
        print("✓ Virtual environment detected")
    else:
        print("⚠️  No virtual environment detected")
        print("Consider creating one: python -m venv jax_fem_env")
        print("Then activate it and run this script again")
        response = input("\nContinue anyway? (y/n): ")
        if response.lower() != 'y':
            return
    
    # Try to fix the compatibility
    if fix_numpy_compatibility():
        print("\n✓ Compatibility fix completed successfully!")
        print("Run 'python test_installation.py' to verify")
    else:
        print("\n❌ Compatibility fix failed")
        print("\nAlternative solutions:")
        print("1. Use conda environment (see create_conda_environment())")
        print("2. Use Docker container")
        print("3. Downgrade Python to 3.9 and use older package versions")

if __name__ == "__main__":
    main()

