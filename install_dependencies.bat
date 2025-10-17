@echo off
REM JAX-FEM Installation Script for Windows with NumPy Compatibility Fix

echo JAX-FEM Installation Script
echo ============================

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python not found. Please install Python 3.8+ first.
    pause
    exit /b 1
)

echo ✓ Python found
python --version

REM Check if pip is available
pip --version >nul 2>&1
if errorlevel 1 (
    echo ❌ pip not found. Please install pip first.
    pause
    exit /b 1
)

echo ✓ pip found

REM Create virtual environment if it doesn't exist
if not exist "jax_fem_env" (
    echo Creating virtual environment...
    python -m venv jax_fem_env
)

REM Activate virtual environment
echo Activating virtual environment...
call jax_fem_env\Scripts\activate.bat

REM Upgrade pip
echo Upgrading pip...
python -m pip install --upgrade pip

REM Uninstall any existing problematic packages
echo Cleaning up existing packages...
pip uninstall -y numpy jax jaxlib jax-fem 2>nul

REM Install NumPy 1.x (compatible version)
echo Installing NumPy 1.x...
pip install "numpy>=1.21.0,<2.0.0"

REM Install JAX CPU version
echo Installing JAX CPU version...
pip install jax jaxlib

REM Try to install JAX with CUDA support
echo Attempting to install JAX with CUDA support...
pip install --upgrade "jax[cuda]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
if errorlevel 1 (
    echo ⚠️  CUDA installation failed, continuing with CPU version
)

REM Install JAX-FEM
echo Installing JAX-FEM...
pip install jax-fem

REM Install other requirements
echo Installing other requirements...
pip install matplotlib scipy

REM Test installation
echo Testing installation...
python test_installation.py

echo.
echo Installation completed!
echo To activate the environment in the future, run:
echo jax_fem_env\Scripts\activate.bat
echo.
echo To run the benchmark:
echo python run_benchmark.py
pause

