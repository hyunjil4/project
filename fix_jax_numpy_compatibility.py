#!/usr/bin/env python3
"""
Fix JAX/NumPy compatibility issues
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

def check_versions():
    """Check current package versions"""
    print("Checking current package versions...")
    
    try:
        import numpy as np
        print(f"NumPy: {np.__version__}")
    except ImportError:
        print("NumPy: Not installed")
    
    try:
        import jax
        print(f"JAX: {jax.__version__}")
    except ImportError:
        print("JAX: Not installed")
    
    try:
        import jaxlib
        print(f"JAXlib: {jaxlib.__version__}")
    except ImportError:
        print("JAXlib: Not installed")

def fix_jax_numpy_compatibility():
    """Fix JAX/NumPy compatibility issues"""
    print("JAX/NumPy Compatibility Fix")
    print("=" * 40)
    
    # Check current versions
    check_versions()
    
    print("\nFixing compatibility issues...")
    
    # Uninstall problematic packages
    run_command("pip uninstall jax jaxlib -y", "Uninstalling JAX packages")
    
    # Install compatible NumPy version
    if not run_command("pip install 'numpy>=1.21.0,<2.0.0'", "Installing compatible NumPy"):
        print("Failed to install compatible NumPy")
        return False
    
    # Install JAX with specific versions that are known to work
    if not run_command("pip install 'jax>=0.4.0,<0.5.0' 'jaxlib>=0.4.0,<0.5.0'", "Installing compatible JAX"):
        print("Failed to install compatible JAX")
        return False
    
    # Test the installation
    print("\nTesting installation...")
    try:
        import jax
        import jax.numpy as jnp
        import numpy as np
        
        # Test basic operations
        x = np.linspace(0, 1, 5)
        x_jax = jnp.array(x)
        y = jnp.sum(x_jax)
        
        print(f"✓ Basic test passed: sum({x}) = {y}")
        
        # Test JIT compilation
        @jax.jit
        def test_func(x):
            return jnp.sum(x ** 2)
        
        result = test_func(x_jax)
        print(f"✓ JIT test passed: sum(x^2) = {result}")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

def create_minimal_working_example():
    """Create a minimal working example"""
    print("\nCreating minimal working example...")
    
    example_code = '''
import numpy as np
import jax.numpy as jnp
import jax

# Create mesh using numpy first
x = np.linspace(0, 1, 5)
y = np.linspace(0, 1, 5)
z = np.linspace(0, 1, 5)

# Convert to JAX arrays
x_jax = jnp.array(x)
y_jax = jnp.array(y)
z_jax = jnp.array(z)

# Create meshgrid
X, Y, Z = jnp.meshgrid(x_jax, y_jax, z_jax, indexing='ij')
points = jnp.stack([X.flatten(), Y.flatten(), Z.flatten()], axis=1)

print(f"Mesh created: {points.shape[0]} nodes")
print("✓ JAX/NumPy compatibility working!")
'''
    
    with open('minimal_example.py', 'w') as f:
        f.write(example_code)
    
    print("✓ Created minimal_example.py")

def main():
    """Main function"""
    print("JAX/NumPy Compatibility Fix")
    print("=" * 50)
    
    # Check if we're in a virtual environment
    if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
        print("✓ Virtual environment detected")
    else:
        print("⚠️  No virtual environment detected")
        print("Consider creating one: python -m venv jax_fem_env")
        response = input("\nContinue anyway? (y/n): ")
        if response.lower() != 'y':
            return
    
    # Try to fix the compatibility
    if fix_jax_numpy_compatibility():
        print("\n✓ Compatibility fix completed successfully!")
        create_minimal_working_example()
        print("\nYou can now run:")
        print("  python robust_demo.py")
        print("  python minimal_example.py")
    else:
        print("\n❌ Compatibility fix failed")
        print("\nTry running the robust demo instead:")
        print("  python robust_demo.py")

if __name__ == "__main__":
    main()

