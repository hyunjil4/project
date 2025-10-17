#!/usr/bin/env python3
"""
Check CUDA version and GPU availability for JAX
"""

import subprocess
import sys
import os

def check_cuda_version():
    """Check CUDA version using nvidia-smi"""
    print("Checking CUDA version...")
    try:
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            lines = result.stdout.split('\n')
            for line in lines:
                if 'CUDA Version' in line:
                    cuda_version = line.split('CUDA Version: ')[1].split()[0]
                    print(f"✓ CUDA Version: {cuda_version}")
                    return cuda_version
            print("⚠️  CUDA version not found in nvidia-smi output")
            return None
        else:
            print(f"❌ nvidia-smi failed: {result.stderr}")
            return None
    except FileNotFoundError:
        print("❌ nvidia-smi not found. CUDA may not be installed.")
        return None
    except subprocess.TimeoutExpired:
        print("❌ nvidia-smi timed out")
        return None
    except Exception as e:
        print(f"❌ Error checking CUDA: {e}")
        return None

def check_nvidia_driver():
    """Check NVIDIA driver version"""
    print("\nChecking NVIDIA driver...")
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=driver_version', '--format=csv,noheader,nounits'], 
                              capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            driver_version = result.stdout.strip()
            print(f"✓ NVIDIA Driver Version: {driver_version}")
            return driver_version
        else:
            print(f"❌ Failed to get driver version: {result.stderr}")
            return None
    except Exception as e:
        print(f"❌ Error checking driver: {e}")
        return None

def check_gpu_devices():
    """Check available GPU devices"""
    print("\nChecking GPU devices...")
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=index,name,memory.total', '--format=csv,noheader,nounits'], 
                              capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            lines = result.stdout.strip().split('\n')
            if lines and lines[0]:
                print("✓ Available GPU devices:")
                for i, line in enumerate(lines):
                    parts = line.split(', ')
                    if len(parts) >= 3:
                        gpu_id, name, memory = parts[0], parts[1], parts[2]
                        print(f"  GPU {gpu_id}: {name} ({memory} MB)")
                return True
            else:
                print("❌ No GPU devices found")
                return False
        else:
            print(f"❌ Failed to query GPU devices: {result.stderr}")
            return False
    except Exception as e:
        print(f"❌ Error checking GPU devices: {e}")
        return False

def check_jax_cuda():
    """Check JAX CUDA support"""
    print("\nChecking JAX CUDA support...")
    try:
        import jax
        print(f"✓ JAX version: {jax.__version__}")
        print(f"✓ JAX backend: {jax.default_backend()}")
        
        devices = jax.devices()
        print(f"✓ JAX devices: {devices}")
        
        # Check for GPU devices
        gpu_devices = [d for d in devices if 'gpu' in str(d).lower() or 'cuda' in str(d).lower()]
        if gpu_devices:
            print(f"✓ GPU devices detected: {gpu_devices}")
            return True
        else:
            print("⚠️  No GPU devices detected in JAX")
            return False
            
    except ImportError:
        print("❌ JAX not installed")
        return False
    except Exception as e:
        print(f"❌ Error checking JAX CUDA: {e}")
        return False

def check_jaxlib_cuda():
    """Check JAXlib CUDA support"""
    print("\nChecking JAXlib CUDA support...")
    try:
        import jaxlib
        print(f"✓ JAXlib version: {jaxlib.__version__}")
        
        # Check if JAXlib was compiled with CUDA
        try:
            import jaxlib.xla_extension as xla
            print(f"✓ JAXlib XLA extension available")
            
            # Try to get CUDA version from JAXlib
            try:
                cuda_version = xla.cuda_version()
                print(f"✓ JAXlib CUDA version: {cuda_version}")
                return True
            except AttributeError:
                print("⚠️  CUDA version not available in JAXlib")
                return False
                
        except ImportError:
            print("❌ JAXlib XLA extension not available")
            return False
            
    except ImportError:
        print("❌ JAXlib not installed")
        return False
    except Exception as e:
        print(f"❌ Error checking JAXlib CUDA: {e}")
        return False

def test_gpu_computation():
    """Test actual GPU computation"""
    print("\nTesting GPU computation...")
    try:
        import jax
        import jax.numpy as jnp
        
        # Create a simple array
        x = jnp.array([1.0, 2.0, 3.0, 4.0])
        
        # Test basic operation
        y = jnp.sum(x)
        print(f"✓ Basic operation: sum([1,2,3,4]) = {y}")
        
        # Test JIT compilation
        @jax.jit
        def test_func(x):
            return jnp.sum(x ** 2)
        
        result = test_func(x)
        print(f"✓ JIT compilation: sum(x^2) = {result}")
        
        # Check which device the computation ran on
        print(f"✓ Computation device: {result.device()}")
        
        return True
        
    except Exception as e:
        print(f"❌ GPU computation test failed: {e}")
        return False

def check_cuda_installation_paths():
    """Check common CUDA installation paths"""
    print("\nChecking CUDA installation paths...")
    
    cuda_paths = [
        '/usr/local/cuda',
        '/opt/cuda',
        '/usr/local/cuda-*',
        '/usr/cuda',
        'C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA',
        'C:\\Program Files\\NVIDIA Corporation\\NVIDIA GPU Computing Toolkit\\CUDA'
    ]
    
    found_paths = []
    for path in cuda_paths:
        if os.path.exists(path):
            found_paths.append(path)
            print(f"✓ Found CUDA at: {path}")
    
    if not found_paths:
        print("⚠️  No CUDA installation paths found")
        return False
    
    return True

def main():
    """Main function to check CUDA and GPU setup"""
    print("CUDA and GPU Verification")
    print("=" * 50)
    
    # Check CUDA version
    cuda_version = check_cuda_version()
    
    # Check NVIDIA driver
    driver_version = check_nvidia_driver()
    
    # Check GPU devices
    gpu_available = check_gpu_devices()
    
    # Check JAX CUDA support
    jax_cuda = check_jax_cuda()
    
    # Check JAXlib CUDA support
    jaxlib_cuda = check_jaxlib_cuda()
    
    # Test GPU computation
    gpu_test = test_gpu_computation()
    
    # Check CUDA installation paths
    cuda_paths = check_cuda_installation_paths()
    
    # Summary
    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)
    
    if cuda_version:
        print(f"✓ CUDA Version: {cuda_version}")
    else:
        print("❌ CUDA Version: Not detected")
    
    if driver_version:
        print(f"✓ NVIDIA Driver: {driver_version}")
    else:
        print("❌ NVIDIA Driver: Not detected")
    
    if gpu_available:
        print("✓ GPU Devices: Available")
    else:
        print("❌ GPU Devices: Not available")
    
    if jax_cuda:
        print("✓ JAX CUDA: Supported")
    else:
        print("❌ JAX CUDA: Not supported")
    
    if jaxlib_cuda:
        print("✓ JAXlib CUDA: Supported")
    else:
        print("❌ JAXlib CUDA: Not supported")
    
    if gpu_test:
        print("✓ GPU Computation: Working")
    else:
        print("❌ GPU Computation: Failed")
    
    # Recommendations
    print("\n" + "=" * 50)
    print("RECOMMENDATIONS")
    print("=" * 50)
    
    if not cuda_version:
        print("1. Install CUDA toolkit from: https://developer.nvidia.com/cuda-downloads")
    
    if not jax_cuda:
        print("2. Install JAX with CUDA support:")
        print("   pip install --upgrade 'jax[cuda]' -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html")
    
    if not gpu_test:
        print("3. Check JAX installation and CUDA compatibility")
        print("4. Verify CUDA version compatibility with JAX")
    
    if cuda_version and jax_cuda and gpu_test:
        print("✓ Everything looks good! GPU acceleration should work.")
    else:
        print("⚠️  Some issues detected. GPU acceleration may not work properly.")

if __name__ == "__main__":
    main()

