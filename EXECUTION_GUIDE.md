# 🚀 Complete Execution Guide

This guide shows you exactly how to run the 3D FEM solver experiments on a completely new computer.

## 📋 **Step-by-Step Execution Order**

### **For Linux/Mac Users:**

```bash
# 1. Make scripts executable
chmod +x *.sh

# 2. Run complete experiment (RECOMMENDED)
./run_experiment.sh

# 3. Or run specific parts
./install_dependencies.sh    # Install packages
python test_installation.py  # Verify installation
python working_demo.py       # Test basic functionality
python run_benchmark.py      # Run full benchmark
```

### **For Windows Users:**

```cmd
REM 1. Run complete experiment (RECOMMENDED)
run_experiment.bat

REM 2. Or run specific parts
install_dependencies.bat     REM Install packages
python test_installation.py REM Verify installation
python working_demo.py       REM Test basic functionality
python run_benchmark.py      REM Run full benchmark
```

## 🎯 **Quick Start (One Command)**

### **Linux/Mac:**
```bash
./run_experiment.sh
```

### **Windows:**
```cmd
run_experiment.bat
```

## 📁 **What Each File Does**

| File | Purpose | When to Run |
|------|---------|-------------|
| `run_experiment.sh` | **Complete experiment** | First time setup |
| `install_dependencies.sh` | Install packages | Before first run |
| `test_installation.py` | Verify installation | After installation |
| `working_demo.py` | **Basic test** | Quick functionality test |
| `run_benchmark.py` | **Full benchmark** | Performance testing |
| `check_cuda_version.py` | Check GPU | If you have NVIDIA GPU |

## 🔧 **Troubleshooting Order**

If something goes wrong, try in this order:

1. **Installation Issues:**
   ```bash
   python fix_numpy_compatibility.py
   python fix_jax_numpy_compatibility.py
   ```

2. **CUDA Issues:**
   ```bash
   python check_cuda_version.py
   # Follow CUDA_INSTALLATION_GUIDE.md
   ```

3. **Basic Functionality:**
   ```bash
   python working_demo.py
   ```

## 📊 **Expected Results**

After running `./run_experiment.sh`, you should see:

1. **Installation verification** ✅
2. **CUDA detection** (if GPU available) ✅
3. **Basic functionality test** ✅
4. **Performance benchmark** ✅
5. **Results saved** in `results_YYYYMMDD_HHMMSS/` folder

## 🎉 **Success Indicators**

- ✅ All tests pass
- ✅ Performance plot generated (`performance_comparison.png`)
- ✅ Results folder created with detailed outputs
- ✅ CPU vs GPU timing comparison shown

## ⚠️ **Common Issues & Solutions**

| Issue | Solution |
|-------|----------|
| NumPy 2.x error | Run `python fix_numpy_compatibility.py` |
| JAX/NumPy error | Run `python fix_jax_numpy_compatibility.py` |
| No GPU detected | Normal for CPU-only systems |
| Import errors | Run `python test_installation.py` |
| Permission denied | Run `chmod +x *.sh` (Linux/Mac) |

## 📈 **Performance Expectations**

- **CPU Only**: 1-5 seconds for small mesh
- **With GPU**: 0.1-1 seconds for small mesh
- **Speedup**: 2-10x faster on GPU (depends on hardware)

## 🎯 **Next Steps After Success**

1. Check `results_*/` folder for detailed outputs
2. Look at `performance_comparison.png` for visualization
3. Modify `jax_fem_3d_solver.py` for your specific needs
4. Follow `CUDA_INSTALLATION_GUIDE.md` for GPU optimization

---

**Remember**: The complete experiment takes 2-5 minutes and will create a results folder with all outputs!

