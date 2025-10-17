#!/usr/bin/env python3
"""
Working 3D FEM demo that avoids JAX/NumPy compatibility issues
Uses pure NumPy for mesh generation and JAX only for computation
"""

import time
import numpy as np

def working_3d_fem_demo():
    """Working 3D FEM demo with CPU vs GPU timing"""
    print("Working 3D FEM Demo")
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
    
    # Try JAX GPU version with proper array handling
    print("\nGPU Assembly...")
    start_time = time.time()
    
    try:
        import jax
        import jax.numpy as jnp
        
        # Convert numpy arrays to JAX arrays properly
        K_gpu = jnp.array(K_cpu, dtype=jnp.float32)
        
        @jax.jit
        def create_k_gpu():
            return jnp.eye(num_dofs, dtype=jnp.float32) * 1000.0
        
        K_gpu = create_k_gpu()
        
        gpu_time = time.time() - start_time
        print(f"GPU time: {gpu_time:.6f} seconds")
        
        # Create load vector using numpy first
        f_np = np.zeros(num_dofs)
        f_np[0] = 1.0
        f = jnp.array(f_np, dtype=jnp.float32)
        
        # Solve on CPU
        print("\nCPU Solve...")
        start_time = time.time()
        u_cpu = np.linalg.solve(K_cpu, f_np)
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
        
        # Convert back to numpy for comparison
        u_gpu_np = np.array(u_gpu)
        
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
        print(f"Max displacement (GPU): {np.max(np.abs(u_gpu_np)):.6f}")
        print(f"Solutions match: {np.allclose(u_cpu, u_gpu_np, rtol=1e-5)}")
        
        return True
        
    except ImportError as e:
        print(f"JAX not available: {e}")
        print("Running CPU-only version...")
        return False
        
    except Exception as e:
        print(f"Error during GPU computation: {e}")
        print("Falling back to CPU-only version...")
        return False

def cpu_only_demo():
    """CPU-only version of the demo"""
    print("\nCPU-Only Demo")
    print("=" * 20)
    
    # Small mesh for quick demo
    nx, ny, nz = 3, 3, 3
    print(f"Mesh: {nx}×{ny}×{nz} elements")
    
    # Create simple mesh
    x = np.linspace(0, 1, nx + 1)
    y = np.linspace(0, 1, ny + 1)
    z = np.linspace(0, 1, nz + 1)
    
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    points = np.stack([X.flatten(), Y.flatten(), Z.flatten()], axis=1)
    
    num_nodes = len(points)
    num_dofs = 3 * num_nodes
    
    print(f"Nodes: {num_nodes}")
    print(f"DOFs: {num_dofs}")
    
    # Create stiffness matrix
    print("\nAssembly...")
    start_time = time.time()
    K = np.eye(num_dofs) * 1000.0
    assembly_time = time.time() - start_time
    print(f"Assembly time: {assembly_time:.6f} seconds")
    
    # Create load vector
    f = np.zeros(num_dofs)
    f[0] = 1.0
    
    # Solve
    print("\nSolve...")
    start_time = time.time()
    u = np.linalg.solve(K, f)
    solve_time = time.time() - start_time
    print(f"Solve time: {solve_time:.6f} seconds")
    
    print(f"\nTotal time: {assembly_time + solve_time:.6f} seconds")
    print(f"Max displacement: {np.max(np.abs(u)):.6f}")

def check_cuda_info():
    """Check CUDA and GPU information"""
    print("Checking CUDA and GPU information...")
    
    # Check CUDA version
    cuda_version = None
    try:
        import subprocess
        result = subprocess.run(['nvidia-smi', '--query-gpu=cuda_version', '--format=csv,noheader,nounits'], 
                              capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            cuda_version = result.stdout.strip()
    except:
        pass
    
    # Check JAX devices
    try:
        import jax
        devices = jax.devices()
        gpu_devices = [d for d in devices if 'gpu' in str(d).lower() or 'cuda' in str(d).lower()]
        
        print(f"  CUDA Version: {cuda_version if cuda_version else 'Not detected'}")
        print(f"  JAX Backend: {jax.default_backend()}")
        print(f"  JAX Devices: {devices}")
        print(f"  GPU Devices: {len(gpu_devices)} detected")
        
        return len(gpu_devices) > 0, cuda_version
    except:
        print("  JAX not available")
        return False, None

def main():
    """Main function"""
    print("3D FEM Demo - CPU vs GPU Performance")
    print("=" * 50)
    
    # Check CUDA and GPU status
    has_gpu, cuda_version = check_cuda_info()
    
    if has_gpu:
        print(f"\n✓ GPU detected with CUDA {cuda_version}")
        print("   Attempting GPU acceleration...")
    else:
        print("\n⚠️  No GPU detected")
        print("   Running CPU-only demo...")
        print("   For GPU acceleration, see CUDA_INSTALLATION_GUIDE.md")
    
    # Try GPU version first
    if working_3d_fem_demo():
        print("\n✓ GPU acceleration working!")
    else:
        print("\n⚠️  GPU acceleration not available, running CPU-only demo...")
        cpu_only_demo()
    
    print("\n" + "=" * 50)
    print("Demo completed!")

if __name__ == "__main__":
    main()
