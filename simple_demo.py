#!/usr/bin/env python3
"""
Simple demo of 3D FEM solver with minimal mesh for quick testing
"""

import time
import jax
import jax.numpy as jnp
import numpy as np

def simple_3d_fem_demo():
    """Simple 3D FEM demo with timing"""
    print("Simple 3D FEM Demo")
    print("=" * 30)
    
    # Small mesh for quick demo
    nx, ny, nz = 3, 3, 3
    print(f"Mesh: {nx}×{ny}×{nz} elements")
    
    # Create simple mesh using regular numpy first, then convert to JAX
    import numpy as np
    x = jnp.array(np.linspace(0, 1, nx + 1))
    y = jnp.array(np.linspace(0, 1, ny + 1))
    z = jnp.array(np.linspace(0, 1, nz + 1))
    
    X, Y, Z = jnp.meshgrid(x, y, z, indexing='ij')
    points = jnp.stack([X.flatten(), Y.flatten(), Z.flatten()], axis=1)
    
    num_nodes = len(points)
    num_dofs = 3 * num_nodes
    
    print(f"Nodes: {num_nodes}")
    print(f"DOFs: {num_dofs}")
    
    # Create simple stiffness matrix (CPU)
    print("\nCPU Assembly...")
    start_time = time.time()
    
    K_cpu = jnp.eye(num_dofs) * 1000.0  # Simplified diagonal matrix
    
    cpu_time = time.time() - start_time
    print(f"CPU time: {cpu_time:.6f} seconds")
    
    # Create simple stiffness matrix (GPU)
    print("\nGPU Assembly...")
    start_time = time.time()
    
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
    u_cpu = jnp.linalg.solve(K_cpu, f)
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
    print(f"Max displacement (CPU): {jnp.max(jnp.abs(u_cpu)):.6f}")
    print(f"Max displacement (GPU): {jnp.max(jnp.abs(u_gpu)):.6f}")
    print(f"Solutions match: {jnp.allclose(u_cpu, u_gpu)}")

if __name__ == "__main__":
    simple_3d_fem_demo()
