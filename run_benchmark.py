#!/usr/bin/env python3
"""
Simple script to run the 3D FEM benchmark
"""

import sys
import os

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from jax_fem_3d_solver import main
    
    if __name__ == "__main__":
        print("Starting 3D FEM Benchmark...")
        print("Make sure you have installed the requirements:")
        print("pip install -r requirements.txt")
        print()
        
        main()
        
except ImportError as e:
    print(f"Import error: {e}")
    print("Please install the required packages:")
    print("pip install -r requirements.txt")
except Exception as e:
    print(f"Error running benchmark: {e}")
    import traceback
    traceback.print_exc()

