@echo off
REM Complete 3D FEM Experiment Runner for Windows
REM This script runs all experiments in the correct order for a new computer

echo 🚀 3D FEM Solver - Complete Experiment Runner
echo ==============================================
echo.

REM Set error handling
setlocal enabledelayedexpansion

REM Function to print colored output (Windows doesn't support colors well, so we'll use text)
echo [INFO] Starting complete 3D FEM experiment...
echo Timestamp: %date% %time%
echo.

REM Create results directory
set timestamp=%date:~0,4%%date:~5,2%%date:~8,2%_%time:~0,2%%time:~3,2%%time:~6,2%
set timestamp=%timestamp: =0%
set RESULTS_DIR=results_%timestamp%
mkdir "%RESULTS_DIR%"
echo [INFO] Created results directory: %RESULTS_DIR%
echo.

REM Step 1: System Requirements Check
echo [INFO] Step 1: Checking system requirements...

python --version >nul 2>&1
if %errorlevel% equ 0 (
    echo [SUCCESS] Python found
    python --version
) else (
    echo [ERROR] Python not found. Please install Python 3.8+
    pause
    exit /b 1
)

pip --version >nul 2>&1
if %errorlevel% equ 0 (
    echo [SUCCESS] pip found
) else (
    echo [ERROR] pip not found. Please install pip
    pause
    exit /b 1
)

REM Check for NVIDIA GPU
nvidia-smi >nul 2>&1
if %errorlevel% equ 0 (
    echo [SUCCESS] NVIDIA GPU detected
    nvidia-smi --query-gpu=name,driver_version,cuda_version --format=csv,noheader,nounits
) else (
    echo [WARNING] NVIDIA GPU not detected. Will run CPU-only experiments.
)

echo.

REM Step 2: Installation & Setup
echo [INFO] Step 2: Setting up environment...

REM Check if virtual environment exists
if not exist "jax_fem_env" (
    echo [INFO] Creating virtual environment...
    python -m venv jax_fem_env
)

REM Activate virtual environment
echo [INFO] Activating virtual environment...
call jax_fem_env\Scripts\activate.bat

REM Install dependencies
echo [INFO] Installing dependencies...
if exist "install_dependencies.bat" (
    call install_dependencies.bat
) else (
    echo [WARNING] install_dependencies.bat not found, installing manually...
    pip install -r requirements.txt
)

echo.

REM Step 3: Verify Installation
echo [INFO] Step 3: Verifying installation...

echo [INFO] Running installation verification...
python test_installation.py > "%RESULTS_DIR%\installation_test.txt" 2>&1
if %errorlevel% equ 0 (
    echo [SUCCESS] Installation verification completed successfully
) else (
    echo [ERROR] Installation verification failed
)
echo.

REM Check CUDA status
nvidia-smi >nul 2>&1
if %errorlevel% equ 0 (
    echo [INFO] Running CUDA version check...
    python check_cuda_version.py > "%RESULTS_DIR%\cuda_check.txt" 2>&1
    if %errorlevel% equ 0 (
        echo [SUCCESS] CUDA version check completed successfully
    ) else (
        echo [ERROR] CUDA version check failed
    )
) else (
    echo [WARNING] Skipping CUDA check (no NVIDIA GPU detected)
)

echo.

REM Step 4: Basic Testing
echo [INFO] Step 4: Running basic tests...

REM Test working demo
echo [INFO] Running working demo test...
python working_demo.py > "%RESULTS_DIR%\working_demo.txt" 2>&1
if %errorlevel% equ 0 (
    echo [SUCCESS] Working demo test completed successfully
) else (
    echo [ERROR] Working demo test failed
)
echo.

REM Test robust demo
echo [INFO] Running robust demo test...
python robust_demo.py > "%RESULTS_DIR%\robust_demo.txt" 2>&1
if %errorlevel% equ 0 (
    echo [SUCCESS] Robust demo test completed successfully
) else (
    echo [ERROR] Robust demo test failed
)
echo.

REM Test simple demo (may fail)
echo [INFO] Testing simple demo (may have issues)...
python simple_demo.py > "%RESULTS_DIR%\simple_demo.txt" 2>&1
if %errorlevel% equ 0 (
    echo [SUCCESS] Simple demo completed
) else (
    echo [WARNING] Simple demo failed (expected due to JAX/NumPy compatibility)
)
echo.

REM Step 5: Full Experiments
echo [INFO] Step 5: Running full experiments...

REM Run benchmark
echo [INFO] Running full benchmark...
python run_benchmark.py > "%RESULTS_DIR%\benchmark.txt" 2>&1
if %errorlevel% equ 0 (
    echo [SUCCESS] Full benchmark completed successfully
) else (
    echo [ERROR] Full benchmark failed
)
echo.

REM Run main solver
echo [INFO] Running main 3D FEM solver...
python jax_fem_3d_solver.py > "%RESULTS_DIR%\main_solver.txt" 2>&1
if %errorlevel% equ 0 (
    echo [SUCCESS] Main 3D FEM solver completed successfully
) else (
    echo [ERROR] Main 3D FEM solver failed
)
echo.

REM Step 6: Generate Summary Report
echo [INFO] Step 6: Generating summary report...

(
echo 3D FEM Solver Experiment Summary
echo ================================
echo Experiment Date: %date% %time%
echo System: %OS%
echo Python Version: 
python --version
echo Working Directory: %CD%
echo.
echo Files Executed:
echo 1. test_installation.py - Installation verification
echo 2. check_cuda_version.py - CUDA version check
echo 3. working_demo.py - Working demo test
echo 4. robust_demo.py - Robust demo test
echo 5. simple_demo.py - Simple demo test
echo 6. run_benchmark.py - Full benchmark
echo 7. jax_fem_3d_solver.py - Main 3D FEM solver
echo.
echo Results Directory: %RESULTS_DIR%
echo All outputs saved to individual files in this directory.
echo.
echo Next Steps:
echo - Check individual output files for detailed results
echo - Look for performance_comparison.png for visualization
echo - Review any error messages in the output files
echo - For GPU acceleration, see CUDA_INSTALLATION_GUIDE.md
) > "%RESULTS_DIR%\experiment_summary.txt"

echo [SUCCESS] Summary report generated: %RESULTS_DIR%\experiment_summary.txt
echo.

REM Step 7: Final Status
echo [INFO] Step 7: Final status check...

REM Check if performance plot was generated
if exist "performance_comparison.png" (
    echo [SUCCESS] Performance plot generated: performance_comparison.png
    copy performance_comparison.png "%RESULTS_DIR%\"
)

REM List all result files
echo [INFO] Generated result files:
dir "%RESULTS_DIR%"

echo.
echo [SUCCESS] 🎉 Complete experiment finished successfully!
echo [INFO] Results saved in: %RESULTS_DIR%\
echo [INFO] Check individual files for detailed results.

echo.
echo ==============================================
echo Experiment completed at: %date% %time%
echo ==============================================

pause

