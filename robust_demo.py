#!/usr/bin/env python3
"""
Robust 3D FEM demo that handles JAX/NumPy compatibility issues
"""

import time
import numpy as np

def robust_3d_fem_demo():
    """Robust 3D FEM demo with compatibility fixes"""
    print("Robust 3D FEM Demo")
    print("=" * 30)
    
    # Small mesh for quick demo
    nx, ny, nz = 3, 3, 3
    print(f"Mesh: {nx}×{ny}×{nz} elements")
    
    # Create simple mesh using regular numpy
    x = np.linspace(0, 1, nx + 1)
    y = np.linspace(0, 1, ny + 1)
    z = np.linspace(0, 1, nz + 1)
    
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    points = np.stack([X.flatten(), Y.flatten(), Z.flatten()], axis=1)
    
    num_nodes = len(points)
    num_dofs = 3 * num_nodes
    
    print(f"Nodes: {num_nodes}")
    print(f"DOFs: {num_dofs}")
    
    # Create simple stiffness matrix (CPU)
    print("\nCPU Assembly...")
    start_time = time.time()
    
    K_cpu = np.eye(num_dofs) * 1000.0  # Simplified diagonal matrix
    
    cpu_time = time.time() - start_time
    print(f"CPU time: {cpu_time:.6f} seconds")
    
    # Try JAX GPU version
    print("\nGPU Assembly...")
    start_time = time.time()
    
    try:
        import jax
        import jax.numpy as jnp
        
        # Convert to JAX arrays
        K_gpu = jnp.array(K_cpu)
        
        @jax.jit
        def create_k_gpu():
            return jnp.eye(num_dofs) * 1000.0
        
        K_gpu = create_k_gpu()
        
        gpu_time = time.time() - start_time
        print(f"GPU time: {gpu_time:.6f} seconds")
        
        # Create load vector
        f = jnp.zeros(num_dofs)
        f = f.at[0].set(1.0)  # Simple point load
        
        # Solve on CPU
        print("\nCPU Solve...")
        start_time = time.time()
        u_cpu = np.linalg.solve(K_cpu, f)
        cpu_solve_time = time.time() - start_time
        print(f"CPU solve time: {cpu_solve_time:.6f} seconds")
        
        # Solve on GPU
        print("\nGPU Solve...")
        start_time = time.time()
        
        @jax.jit
        def solve_gpu(K, f):
            return jnp.linalg.solve(K, f)
        
        u_gpu = solve_gpu(K_gpu, f)
        gpu_solve_time = time.time() - start_time
        print(f"GPU solve time: {gpu_solve_time:.6f} seconds")
        
        # Results
        print("\n" + "=" * 30)
        print("RESULTS")
        print("=" * 30)
        print(f"Assembly speedup: {cpu_time / gpu_time:.2f}x")
        print(f"Solve speedup: {cpu_solve_time / gpu_solve_time:.2f}x")
        print(f"Total CPU time: {cpu_time + cpu_solve_time:.6f} seconds")
        print(f"Total GPU time: {gpu_time + gpu_solve_time:.6f} seconds")
        print(f"Total speedup: {(cpu_time + cpu_solve_time) / (gpu_time + gpu_solve_time):.2f}x")
        
        # Check solution
        print(f"\nSolution check:")
        print(f"Max displacement (CPU): {np.max(np.abs(u_cpu)):.6f}")
        print(f"Max displacement (GPU): {np.max(np.abs(u_gpu)):.6f}")
        print(f"Solutions match: {np.allclose(u_cpu, u_gpu)}")
        
    except ImportError as e:
        print(f"JAX not available: {e}")
        print("Running CPU-only version...")
        
        # CPU-only version
        f = np.zeros(num_dofs)
        f[0] = 1.0
        
        print("\nCPU Solve...")
        start_time = time.time()
        u_cpu = np.linalg.solve(K_cpu, f)
        cpu_solve_time = time.time() - start_time
        print(f"CPU solve time: {cpu_solve_time:.6f} seconds")
        
        print(f"\nMax displacement: {np.max(np.abs(u_cpu)):.6f}")
        
    except Exception as e:
        print(f"Error during GPU computation: {e}")
        print("Falling back to CPU-only version...")
        
        # Fallback to CPU
        f = np.zeros(num_dofs)
        f[0] = 1.0
        
        print("\nCPU Solve...")
        start_time = time.time()
        u_cpu = np.linalg.solve(K_cpu, f)
        cpu_solve_time = time.time() - start_time
        print(f"CPU solve time: {cpu_solve_time:.6f} seconds")
        
        print(f"\nMax displacement: {np.max(np.abs(u_cpu)):.6f}")

def test_jax_installation():
    """Test JAX installation and compatibility"""
    print("Testing JAX Installation...")
    print("=" * 30)
    
    try:
        import jax
        print(f"✓ JAX version: {jax.__version__}")
        print(f"✓ Backend: {jax.default_backend()}")
        print(f"✓ Devices: {jax.devices()}")
        
        import jax.numpy as jnp
        print("✓ JAX NumPy imported")
        
        # Test basic operations
        x = jnp.array([1, 2, 3, 4])
        y = jnp.sum(x)
        print(f"✓ Basic operation: sum([1,2,3,4]) = {y}")
        
        # Test JIT compilation
        @jax.jit
        def test_func(x):
            return jnp.sum(x ** 2)
        
        result = test_func(x)
        print(f"✓ JIT compilation: sum(x^2) = {result}")
        
        return True
        
    except ImportError as e:
        print(f"❌ JAX import failed: {e}")
        return False
    except Exception as e:
        print(f"❌ JAX test failed: {e}")
        return False

if __name__ == "__main__":
    print("JAX-FEM Compatibility Test")
    print("=" * 40)
    
    # Test JAX installation first
    if test_jax_installation():
        print("\n✓ JAX is working correctly!")
        robust_3d_fem_demo()
    else:
        print("\n⚠️  JAX has issues, running CPU-only demo...")
        robust_3d_fem_demo()

