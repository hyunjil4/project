# Quick Start Guide

## 🚨 NumPy Compatibility Fix (IMPORTANT!)

If you get NumPy 2.x compatibility errors, run one of these:

### Option 1: Automatic Fix (Recommended)
```bash
# Linux/Mac
./install_dependencies.sh

# Windows
install_dependencies.bat
```

### Option 2: Manual Fix
```bash
python fix_numpy_compatibility.py
```

### Option 3: Manual Commands
```bash
pip uninstall numpy jax jaxlib jax-fem -y
pip install "numpy>=1.21.0,<2.0.0"
pip install jax jaxlib
pip install jax-fem
pip install matplotlib scipy
```

## 1. Test Installation
```bash
python test_installation.py
```

## 2. Run Complete Experiment (Recommended)
```bash
# For Linux/Mac
./run_experiment.sh

# For Windows
run_experiment.bat

# Or just test basic functionality
python working_demo.py
```

## 3. Run Robust Demo (Fallback)
```bash
python robust_demo.py
```

## 4. Run Simple Demo (May have issues)
```bash
python simple_demo.py
```

## 5. Run Full Benchmark
```bash
python run_benchmark.py
```

## 4. Install Requirements (if needed)
```bash
pip install -r requirements.txt
```

## Files Overview

### **Main Experiment Scripts**
- `run_experiment.sh` - **COMPLETE EXPERIMENT** - Linux/Mac (RECOMMENDED)
- `run_experiment.bat` - **COMPLETE EXPERIMENT** - Windows (RECOMMENDED)
- `EXECUTION_GUIDE.md` - Step-by-step execution guide

### **Individual Test Scripts**
- `test_installation.py` - Check if everything is installed correctly
- `working_demo.py` - **RECOMMENDED** - Working demo with fallback handling
- `robust_demo.py` - Robust demo with error handling
- `simple_demo.py` - Quick demo (may have JAX/NumPy issues)
- `run_benchmark.py` - Full benchmark with multiple mesh sizes
- `jax_fem_3d_solver.py` - Main solver implementation

### **Installation & Setup**
- `install_dependencies.sh` - Automatic installation (Linux/Mac)
- `install_dependencies.bat` - Automatic installation (Windows)
- `fix_numpy_compatibility.py` - Fix NumPy 1.x vs 2.x issues
- `fix_jax_numpy_compatibility.py` - Fix JAX/NumPy compatibility

### **CUDA & GPU Support**
- `check_cuda_version.py` - Comprehensive CUDA verification
- `gpu_info_demo.py` - GPU information with demo
- `CUDA_INSTALLATION_GUIDE.md` - CUDA setup instructions

## Expected Output

The benchmark will show timing comparisons like:
```
Assembly Time:
  CPU:  0.1234 seconds
  GPU:  0.0456 seconds
  Speedup: 2.71x

Solve Time:
  CPU:  0.0567 seconds
  GPU:  0.0234 seconds
  Speedup: 2.42x
```

## Troubleshooting

1. **JAX-FEM not found**: `pip install jax-fem`
2. **No GPU detected**: Install JAX with CUDA support
3. **Import errors**: Check all requirements are installed
