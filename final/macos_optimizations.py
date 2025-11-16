#!/usr/bin/env python3
"""
macOS-specific optimizations for Fish Speech TTS
Addresses MallocStackLogging warnings and other macOS-specific issues
"""

import os
import subprocess
import sys
from pathlib import Path

def disable_malloc_stack_logging():
    """
    Disable MallocStackLogging to reduce console noise on macOS
    This is safe and only affects debugging malloc operations
    """
    print("🍎 Applying macOS-specific optimizations...")
    
    # Set environment variable to disable malloc stack logging
    os.environ['MallocStackLogging'] = '0'
    os.environ['MallocStackLoggingNoCompact'] = '1'
    
    print("✅ MallocStackLogging disabled")

def optimize_macos_memory():
    """
    Apply macOS-specific memory optimizations
    """
    # Disable memory debugging features that can slow down performance
    os.environ['MallocScribble'] = '0'
    os.environ['MallocPreScribble'] = '0'
    os.environ['MallocGuardEdges'] = '0'
    
    print("✅ macOS memory debugging disabled for performance")

def set_macos_process_priority():
    """
    Set higher process priority for better performance on macOS
    """
    try:
        # Set nice priority (lower number = higher priority)
        os.nice(-5)  # Slightly higher priority
        print("✅ Process priority optimized")
    except PermissionError:
        print("⚠️  Could not set process priority (requires admin)")

def optimize_for_m1_air():
    """
    Apply M1 MacBook Air specific optimizations
    """
    # Set thread affinity for performance cores (0-3 on M1 Air)
    os.environ['OMP_NUM_THREADS'] = '4'
    os.environ['MKL_NUM_THREADS'] = '4'
    os.environ['OPENBLAS_NUM_THREADS'] = '4'
    
    # Optimize for Apple Silicon
    os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
    os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.0'
    
    print("✅ M1 Air optimizations applied")

def check_thermal_state():
    """
    Check macOS thermal state to warn about throttling
    """
    try:
        result = subprocess.run(['pmset', '-g', 'thermlog'], 
                              capture_output=True, text=True, timeout=5)
        if 'CPU_Speed_Limit' in result.stdout:
            print("⚠️  Thermal throttling detected - performance may be reduced")
        else:
            print("✅ No thermal throttling detected")
    except (subprocess.TimeoutExpired, FileNotFoundError):
        print("ℹ️  Could not check thermal state")

def apply_all_optimizations():
    """
    Apply all macOS optimizations
    """
    print("=" * 50)
    print("🍎 macOS Optimization Suite")
    print("=" * 50)
    
    disable_malloc_stack_logging()
    optimize_macos_memory()
    set_macos_process_priority()
    optimize_for_m1_air()
    check_thermal_state()
    
    print("=" * 50)
    print("✅ All macOS optimizations applied!")
    print("=" * 50)

if __name__ == "__main__":
    if sys.platform == "darwin":  # macOS
        apply_all_optimizations()
    else:
        print("❌ This script is for macOS only")
        sys.exit(1)
