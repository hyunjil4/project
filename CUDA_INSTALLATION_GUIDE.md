# CUDA Installation Guide for JAX-FEM

## Current Status
Based on the verification, your system currently has:
- ❌ CUDA toolkit not installed
- ❌ NVIDIA driver not detected
- ❌ GPU devices not available
- ✅ JAX CPU version working (v0.8.0)

## Step-by-Step CUDA Installation

### 1. Check if you have an NVIDIA GPU
```bash
lspci | grep -i nvidia
```
If you see NVIDIA graphics cards listed, you can proceed with CUDA installation.

### 2. Install NVIDIA Driver
```bash
# Ubuntu/Debian
sudo apt update
sudo apt install nvidia-driver-525  # or latest version

# CentOS/RHEL
sudo yum install nvidia-driver

# macOS (if supported)
# Note: CUDA is not officially supported on macOS
```

### 3. Install CUDA Toolkit
```bash
# Download from NVIDIA website
wget https://developer.download.nvidia.com/compute/cuda/12.2.0/local_installers/cuda_12.2.0_535.54.03_linux.run
sudo sh cuda_12.2.0_535.54.03_linux.run

# Or use package manager
sudo apt install cuda-toolkit-12-2
```

### 4. Set up environment variables
Add to your `~/.bashrc` or `~/.zshrc`:
```bash
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
export CUDA_HOME=/usr/local/cuda
```

### 5. Install JAX with CUDA support
```bash
# Uninstall current JAX
pip uninstall jax jaxlib -y

# Install JAX with CUDA support
pip install --upgrade "jax[cuda]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

### 6. Verify installation
```bash
python check_cuda_version.py
```

## Alternative: Use Conda Environment

### 1. Create conda environment with CUDA
```bash
conda create -n jax_cuda python=3.9
conda activate jax_cuda
conda install cudatoolkit=11.8 -c conda-forge
pip install jax[cuda] -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

### 2. Install JAX-FEM
```bash
pip install jax-fem
pip install matplotlib scipy
```

## Docker Alternative (Recommended for Testing)

### 1. Use NVIDIA CUDA Docker image
```bash
# Pull CUDA image
docker pull nvidia/cuda:12.2-devel-ubuntu20.04

# Run container with GPU support
docker run --gpus all -it nvidia/cuda:12.2-devel-ubuntu20.04

# Inside container, install Python and JAX
apt update && apt install python3-pip
pip install jax[cuda] jax-fem matplotlib scipy
```

## Verification Commands

### Check CUDA installation
```bash
nvidia-smi
nvcc --version
```

### Check JAX CUDA support
```python
import jax
print(f"JAX version: {jax.__version__}")
print(f"JAX devices: {jax.devices()}")
print(f"JAX backend: {jax.default_backend()}")
```

### Test GPU computation
```python
import jax.numpy as jnp
x = jnp.array([1.0, 2.0, 3.0])
print(f"Device: {x.device()}")
```

## Troubleshooting

### Common Issues:

1. **"nvidia-smi not found"**
   - Install NVIDIA driver first
   - Reboot after driver installation

2. **"CUDA out of memory"**
   - Reduce batch size or mesh size
   - Use `jax.config.update('jax_platform_name', 'cpu')` to force CPU

3. **"JAX CUDA not working"**
   - Check CUDA version compatibility
   - Reinstall JAX with correct CUDA version
   - Verify environment variables

4. **"No GPU devices detected"**
   - Check if GPU is properly installed
   - Verify CUDA toolkit installation
   - Check JAX installation

## Performance Expectations

With proper CUDA setup, you should see:
- **Assembly speedup**: 2-10x faster on GPU
- **Solve speedup**: 1-5x faster on GPU
- **Total speedup**: 2-8x faster overall

## Current System Status
- **OS**: macOS (CUDA not officially supported)
- **JAX**: CPU-only version (v0.8.0)
- **GPU**: Not detected
- **Recommendation**: Use CPU version or consider Linux/Windows with NVIDIA GPU

