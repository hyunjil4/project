#!/usr/bin/env python3
"""
Test script to verify JAX-FEM installation and JAX backend
"""

import sys

def test_imports():
    """Test if all required packages can be imported"""
    print("Testing package imports...")
    
    try:
        import jax
        print(f"✓ JAX version: {jax.__version__}")
        print(f"  Backend: {jax.default_backend()}")
        print(f"  Devices: {jax.devices()}")
    except ImportError as e:
        print(f"❌ JAX import failed: {e}")
        return False
    
    try:
        import jax.numpy as jnp
        print("✓ JAX NumPy imported")
    except ImportError as e:
        print(f"❌ JAX NumPy import failed: {e}")
        return False
    
    try:
        import numpy as np
        print(f"✓ NumPy version: {np.__version__}")
    except ImportError as e:
        print(f"❌ NumPy import failed: {e}")
        return False
    
    try:
        import matplotlib.pyplot as plt
        print("✓ Matplotlib imported")
    except ImportError as e:
        print(f"❌ Matplotlib import failed: {e}")
        return False
    
    try:
        import jax_fem
        print("✓ JAX-FEM imported")
    except ImportError as e:
        print(f"❌ JAX-FEM import failed: {e}")
        print("  Install with: pip install jax-fem")
        return False
    
    return True

def test_jax_functionality():
    """Test basic JAX functionality"""
    print("\nTesting JAX functionality...")
    
    try:
        import jax.numpy as jnp
        
        # Test basic operations
        x = jnp.array([1, 2, 3, 4])
        y = jnp.sum(x)
        print(f"✓ Basic JAX operation: sum([1,2,3,4]) = {y}")
        
        # Test JIT compilation
        @jax.jit
        def simple_func(x):
            return jnp.sum(x ** 2)
        
        result = simple_func(x)
        print(f"✓ JIT compilation: sum(x^2) = {result}")
        
        return True
        
    except Exception as e:
        print(f"❌ JAX functionality test failed: {e}")
        return False

def test_gpu_availability():
    """Test if GPU is available"""
    print("\nTesting GPU availability...")
    
    try:
        import jax
        
        devices = jax.devices()
        gpu_devices = [d for d in devices if 'gpu' in str(d).lower() or 'cuda' in str(d).lower()]
        
        if gpu_devices:
            print(f"✓ GPU devices found: {gpu_devices}")
            print(f"  Default backend: {jax.default_backend()}")
            return True
        else:
            print("⚠️  No GPU devices found. Will run on CPU.")
            print(f"  Available devices: {devices}")
            return False
            
    except Exception as e:
        print(f"❌ GPU test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("JAX-FEM Installation Test")
    print("=" * 40)
    
    # Test imports
    if not test_imports():
        print("\n❌ Import test failed. Please install missing packages.")
        sys.exit(1)
    
    # Test JAX functionality
    if not test_jax_functionality():
        print("\n❌ JAX functionality test failed.")
        sys.exit(1)
    
    # Test GPU availability
    gpu_available = test_gpu_availability()
    
    print("\n" + "=" * 40)
    if gpu_available:
        print("✓ All tests passed! GPU is available.")
        print("  You can run the full benchmark with: python run_benchmark.py")
    else:
        print("✓ All tests passed! Running on CPU.")
        print("  You can run the full benchmark with: python run_benchmark.py")
        print("  (Note: GPU acceleration will not be available)")

if __name__ == "__main__":
    main()

