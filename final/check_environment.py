"""
Quick environment check script
Verifies you're using the correct Python and packages
"""
import sys
import platform
from pathlib import Path

print("=" * 70)
print("ENVIRONMENT CHECK")
print("=" * 70)

# Python version and location
print(f"\n1. Python Version: {sys.version}")
print(f"   Python Executable: {sys.executable}")

# Check if we're in the right environment
python_path = Path(sys.executable)
if "venv312" in str(python_path):
    print("   ✅ Using venv312 (CORRECT)")
elif "anaconda" in str(python_path).lower() or "conda" in str(python_path).lower():
    print("   ❌ Using Anaconda/Conda (WRONG!)")
    print("   → Please deactivate conda and activate venv312")
else:
    print(f"   ⚠️  Unknown environment: {python_path}")

# Check NumPy
print("\n2. NumPy:")
try:
    import numpy as np
    numpy_path = Path(np.__file__)
    print(f"   Version: {np.__version__}")
    print(f"   Location: {numpy_path}")
    
    if "venv312" in str(numpy_path):
        print("   ✅ Using venv312 NumPy (CORRECT)")
    elif "anaconda" in str(numpy_path).lower():
        print("   ❌ Using Anaconda NumPy (WRONG!)")
        print("   → This will cause MINGW-W64 crashes!")
    else:
        print(f"   ⚠️  Unknown NumPy location")
except ImportError:
    print("   ❌ NumPy not installed!")

# Check PyTorch
print("\n3. PyTorch:")
try:
    import torch
    torch_path = Path(torch.__file__)
    print(f"   Version: {torch.__version__}")
    print(f"   Location: {torch_path}")
    print(f"   CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"   CUDA Version: {torch.version.cuda}")
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
    
    if "venv312" in str(torch_path):
        print("   ✅ Using venv312 PyTorch (CORRECT)")
    elif "anaconda" in str(torch_path).lower():
        print("   ❌ Using Anaconda PyTorch (WRONG!)")
    else:
        print(f"   ⚠️  Unknown PyTorch location")
except ImportError:
    print("   ❌ PyTorch not installed!")

# Check Fish Speech
print("\n4. Fish Speech:")
fish_speech_dir = Path(__file__).parent / "fish-speech"
if fish_speech_dir.exists():
    print(f"   ✅ Found at: {fish_speech_dir}")
    
    # Check if it's in sys.path
    if str(fish_speech_dir) in sys.path:
        print("   ✅ Already in sys.path")
    else:
        print("   ⚠️  Not in sys.path (will be added by engine)")
    
    # Try importing
    try:
        sys.path.insert(0, str(fish_speech_dir))
        import pyrootutils
        pyrootutils.setup_root(str(fish_speech_dir), indicator=".project-root", pythonpath=True)
        from fish_speech.models.dac.inference import load_model
        print("   ✅ Can import Fish Speech modules")
    except Exception as e:
        print(f"   ❌ Cannot import Fish Speech: {e}")
else:
    print(f"   ❌ Not found at: {fish_speech_dir}")

# System info
print("\n5. System:")
print(f"   OS: {platform.system()} {platform.release()}")
print(f"   Architecture: {platform.machine()}")
print(f"   Processor: {platform.processor()}")

print("\n" + "=" * 70)

# Final verdict
if "venv312" in str(sys.executable):
    print("✅ ENVIRONMENT OK - Ready to run!")
    print("\nYou can now:")
    print("  1. Run test: python test_v2_engine.py")
    print("  2. Start backend: python backend/app.py")
    print("  3. Or use: .\\run_all.bat")
else:
    print("❌ WRONG ENVIRONMENT!")
    print("\nFix it by running:")
    print("  conda deactivate")
    print("  .\\venv312\\Scripts\\activate")
    print("  python check_environment.py")

print("=" * 70)
