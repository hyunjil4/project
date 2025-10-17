#!/bin/bash

# Complete 3D FEM Experiment Runner
# This script runs all experiments in the correct order for a new computer

set -e  # Exit on any error

echo "🚀 3D FEM Solver - Complete Experiment Runner"
echo "=============================================="
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to run a Python script with error handling
run_python_script() {
    local script_name="$1"
    local description="$2"
    
    print_status "Running $description..."
    
    if python3 "$script_name"; then
        print_success "$description completed successfully"
        echo ""
    else
        print_error "$description failed"
        echo ""
        return 1
    fi
}

# Function to create results directory
create_results_dir() {
    local timestamp=$(date +"%Y%m%d_%H%M%S")
    RESULTS_DIR="results_${timestamp}"
    mkdir -p "$RESULTS_DIR"
    print_status "Created results directory: $RESULTS_DIR"
}

# Function to save output to results
save_output() {
    local script_name="$1"
    local output_file="$RESULTS_DIR/${script_name%.py}_output.txt"
    print_status "Saving output to: $output_file"
}

# Main experiment function
run_experiment() {
    echo "Starting complete 3D FEM experiment..."
    echo "Timestamp: $(date)"
    echo ""
    
    # Create results directory
    create_results_dir
    
    # Step 1: System Requirements Check
    print_status "Step 1: Checking system requirements..."
    
    if command_exists python3; then
        python_version=$(python3 --version)
        print_success "Python found: $python_version"
    else
        print_error "Python3 not found. Please install Python 3.8+"
        exit 1
    fi
    
    if command_exists pip3; then
        print_success "pip3 found"
    else
        print_error "pip3 not found. Please install pip"
        exit 1
    fi
    
    # Check for NVIDIA GPU
    if command_exists nvidia-smi; then
        print_success "NVIDIA GPU detected"
        nvidia-smi --query-gpu=name,driver_version,cuda_version --format=csv,noheader,nounits | head -1
    else
        print_warning "NVIDIA GPU not detected. Will run CPU-only experiments."
    fi
    
    echo ""
    
    # Step 2: Installation & Setup
    print_status "Step 2: Setting up environment..."
    
    # Check if virtual environment exists
    if [ ! -d "jax_fem_env" ]; then
        print_status "Creating virtual environment..."
        python3 -m venv jax_fem_env
    fi
    
    # Activate virtual environment
    print_status "Activating virtual environment..."
    source jax_fem_env/bin/activate
    
    # Install dependencies
    print_status "Installing dependencies..."
    if [ -f "install_dependencies.sh" ]; then
        chmod +x install_dependencies.sh
        ./install_dependencies.sh
    else
        print_warning "install_dependencies.sh not found, installing manually..."
        pip install -r requirements.txt
    fi
    
    echo ""
    
    # Step 3: Verify Installation
    print_status "Step 3: Verifying installation..."
    
    run_python_script "test_installation.py" "Installation verification" 2>&1 | tee "$RESULTS_DIR/installation_test.txt"
    
    # Check CUDA status
    if command_exists nvidia-smi; then
        run_python_script "check_cuda_version.py" "CUDA version check" 2>&1 | tee "$RESULTS_DIR/cuda_check.txt"
    else
        print_warning "Skipping CUDA check (no NVIDIA GPU detected)"
    fi
    
    echo ""
    
    # Step 4: Basic Testing
    print_status "Step 4: Running basic tests..."
    
    # Test working demo
    run_python_script "working_demo.py" "Working demo test" 2>&1 | tee "$RESULTS_DIR/working_demo.txt"
    
    # Test robust demo
    run_python_script "robust_demo.py" "Robust demo test" 2>&1 | tee "$RESULTS_DIR/robust_demo.txt"
    
    # Test simple demo (may fail)
    print_status "Testing simple demo (may have issues)..."
    if python3 "simple_demo.py" 2>&1 | tee "$RESULTS_DIR/simple_demo.txt"; then
        print_success "Simple demo completed"
    else
        print_warning "Simple demo failed (expected due to JAX/NumPy compatibility)"
    fi
    
    echo ""
    
    # Step 5: Full Experiments
    print_status "Step 5: Running full experiments..."
    
    # Run benchmark
    run_python_script "run_benchmark.py" "Full benchmark" 2>&1 | tee "$RESULTS_DIR/benchmark.txt"
    
    # Run main solver
    run_python_script "jax_fem_3d_solver.py" "Main 3D FEM solver" 2>&1 | tee "$RESULTS_DIR/main_solver.txt"
    
    echo ""
    
    # Step 6: Generate Summary Report
    print_status "Step 6: Generating summary report..."
    
    cat > "$RESULTS_DIR/experiment_summary.txt" << EOF
3D FEM Solver Experiment Summary
================================
Experiment Date: $(date)
System: $(uname -a)
Python Version: $(python3 --version)
Working Directory: $(pwd)

Files Executed:
1. test_installation.py - Installation verification
2. check_cuda_version.py - CUDA version check
3. working_demo.py - Working demo test
4. robust_demo.py - Robust demo test
5. simple_demo.py - Simple demo test
6. run_benchmark.py - Full benchmark
7. jax_fem_3d_solver.py - Main 3D FEM solver

Results Directory: $RESULTS_DIR
All outputs saved to individual files in this directory.

Next Steps:
- Check individual output files for detailed results
- Look for performance_comparison.png for visualization
- Review any error messages in the output files
- For GPU acceleration, follow CUDA_INSTALLATION_GUIDE.md
EOF
    
    print_success "Summary report generated: $RESULTS_DIR/experiment_summary.txt"
    
    echo ""
    
    # Step 7: Final Status
    print_status "Step 7: Final status check..."
    
    # Check if performance plot was generated
    if [ -f "performance_comparison.png" ]; then
        print_success "Performance plot generated: performance_comparison.png"
        cp performance_comparison.png "$RESULTS_DIR/"
    fi
    
    # List all result files
    print_status "Generated result files:"
    ls -la "$RESULTS_DIR/"
    
    echo ""
    print_success "🎉 Complete experiment finished successfully!"
    print_status "Results saved in: $RESULTS_DIR/"
    print_status "Check individual files for detailed results."
    
    echo ""
    echo "=============================================="
    echo "Experiment completed at: $(date)"
    echo "=============================================="
}

# Function to show help
show_help() {
    echo "3D FEM Solver - Complete Experiment Runner"
    echo ""
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  -h, --help     Show this help message"
    echo "  -q, --quick    Run only basic tests (skip full experiments)"
    echo "  -c, --clean    Clean up before running (remove old results)"
    echo "  -v, --verbose  Verbose output"
    echo ""
    echo "Examples:"
    echo "  $0              # Run complete experiment"
    echo "  $0 --quick      # Run only basic tests"
    echo "  $0 --clean      # Clean up and run complete experiment"
    echo ""
}

# Function to run quick test
run_quick_test() {
    print_status "Running quick test mode..."
    
    create_results_dir
    
    # Basic tests only
    run_python_script "test_installation.py" "Installation verification" 2>&1 | tee "$RESULTS_DIR/installation_test.txt"
    run_python_script "working_demo.py" "Working demo test" 2>&1 | tee "$RESULTS_DIR/working_demo.txt"
    
    print_success "Quick test completed!"
}

# Function to clean up
cleanup() {
    print_status "Cleaning up old results..."
    rm -rf results_*
    rm -f performance_comparison.png
    print_success "Cleanup completed"
}

# Parse command line arguments
QUICK_MODE=false
CLEAN_MODE=false
VERBOSE=false

while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            show_help
            exit 0
            ;;
        -q|--quick)
            QUICK_MODE=true
            shift
            ;;
        -c|--clean)
            CLEAN_MODE=true
            shift
            ;;
        -v|--verbose)
            VERBOSE=true
            shift
            ;;
        *)
            print_error "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

# Main execution
echo "🚀 Starting 3D FEM Solver Experiment"
echo "====================================="

if [ "$CLEAN_MODE" = true ]; then
    cleanup
fi

if [ "$QUICK_MODE" = true ]; then
    run_quick_test
else
    run_experiment
fi

echo ""
print_success "All done! 🎉"

