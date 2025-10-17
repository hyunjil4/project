#!/usr/bin/env python3
"""
GPU Information Demo with CUDA version verification
"""

import time
import numpy as np

def get_cuda_info():
    """Get CUDA version and GPU information"""
    cuda_info = {}
    
    # Check CUDA version using nvidia-smi
    try:
        import subprocess
        result = subprocess.run(['nvidia-smi', '--query-gpu=driver_version,cuda_version', '--format=csv,noheader,nounits'], 
                              capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            lines = result.stdout.strip().split('\n')
            if lines and lines[0]:
                parts = lines[0].split(', ')
                if len(parts) >= 2:
                    cuda_info['driver_version'] = parts[0]
                    cuda_info['cuda_version'] = parts[1]
    except:
        pass
    
    # Check JAX devices
    try:
        import jax
        cuda_info['jax_version'] = jax.__version__
        cuda_info['jax_backend'] = jax.default_backend()
        cuda_info['jax_devices'] = [str(d) for d in jax.devices()]
        
        # Check for GPU devices
        gpu_devices = [d for d in jax.devices() if 'gpu' in str(d).lower() or 'cuda' in str(d).lower()]
        cuda_info['has_gpu'] = len(gpu_devices) > 0
        cuda_info['gpu_devices'] = [str(d) for d in gpu_devices]
    except:
        cuda_info['has_gpu'] = False
    
    return cuda_info

def gpu_info_demo():
    """Demo with GPU information display"""
    print("GPU Information Demo")
    print("=" * 40)
    
    # Get CUDA info
    cuda_info = get_cuda_info()
    
    # Display CUDA information
    print("\nCUDA Information:")
    if 'cuda_version' in cuda_info:
        print(f"  CUDA Version: {cuda_info['cuda_version']}")
    else:
        print("  CUDA Version: Not detected")
    
    if 'driver_version' in cuda_info:
        print(f"  Driver Version: {cuda_info['driver_version']}")
    else:
        print("  Driver Version: Not detected")
    
    # Display JAX information
    print("\nJAX Information:")
    if 'jax_version' in cuda_info:
        print(f"  JAX Version: {cuda_info['jax_version']}")
        print(f"  JAX Backend: {cuda_info['jax_backend']}")
        print(f"  JAX Devices: {cuda_info['jax_devices']}")
        
        if cuda_info['has_gpu']:
            print(f"  GPU Devices: {cuda_info['gpu_devices']}")
        else:
            print("  GPU Devices: None detected")
    else:
        print("  JAX: Not available")
    
    # Small mesh for demo
    nx, ny, nz = 3, 3, 3
    print(f"\nMesh: {nx}×{ny}×{nz} elements")
    
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
    
    # CPU computation
    print("\nCPU Assembly...")
    start_time = time.time()
    K_cpu = np.eye(num_dofs) * 1000.0
    cpu_time = time.time() - start_time
    print(f"CPU time: {cpu_time:.6f} seconds")
    
    # GPU computation (if available)
    if cuda_info.get('has_gpu', False):
        print("\nGPU Assembly...")
        start_time = time.time()
        
        try:
            import jax.numpy as jnp
            
            # Convert to JAX arrays
            K_gpu = jnp.array(K_cpu, dtype=jnp.float32)
            
            @jax.jit
            def create_k_gpu():
                return jnp.eye(num_dofs, dtype=jnp.float32) * 1000.0
            
            K_gpu = create_k_gpu()
            
            gpu_time = time.time() - start_time
            print(f"GPU time: {gpu_time:.6f} seconds")
            print(f"Speedup: {cpu_time / gpu_time:.2f}x")
            
            # Test computation device
            print(f"Computation device: {K_gpu.device()}")
            
        except Exception as e:
            print(f"GPU computation failed: {e}")
            print("Falling back to CPU-only mode")
    else:
        print("\n⚠️  No GPU available, using CPU-only mode")
    
    # Create load vector and solve
    f = np.zeros(num_dofs)
    f[0] = 1.0
    
    print("\nCPU Solve...")
    start_time = time.time()
    u = np.linalg.solve(K_cpu, f)
    solve_time = time.time() - start_time
    print(f"Solve time: {solve_time:.6f} seconds")
    
    print(f"\nMax displacement: {np.max(np.abs(u)):.6f}")
    
    # Summary
    print("\n" + "=" * 40)
    print("SUMMARY")
    print("=" * 40)
    
    if cuda_info.get('has_gpu', False):
        print("✓ GPU acceleration available")
        if 'cuda_version' in cuda_info:
            print(f"✓ CUDA version: {cuda_info['cuda_version']}")
        if 'gpu_devices' in cuda_info:
            print(f"✓ GPU devices: {len(cuda_info['gpu_devices'])}")
    else:
        print("⚠️  GPU acceleration not available")
        print("  - Check CUDA installation")
        print("  - Check JAX CUDA support")
        print("  - Run: python check_cuda_version.py")

if __name__ == "__main__":
    gpu_info_demo()

