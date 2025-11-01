"""Test Fish Speech installation"""
import sys
from pathlib import Path

print("Testing Fish Speech installation...")
print()

# Test 1: Import fish_speech
try:
    print("1. Importing fish_speech module...")
    import fish_speech
    print("   [OK]")
except Exception as e:
    print(f"   [FAILED]: {e}")
    sys.exit(1)

# Test 2: Check inference script exists
try:
    print("2. Checking DAC inference script...")
    script_path = Path("fish-speech/fish_speech/models/dac/inference.py")
    if script_path.exists():
        print(f"   [OK] - Found at {script_path}")
    else:
        print(f"   [FAILED] - Not found at {script_path}")
        sys.exit(1)
except Exception as e:
    print(f"   [FAILED]: {e}")
    sys.exit(1)

# Test 3: Import dependencies
try:
    print("3. Testing Fish Speech dependencies...")
    import hydra
    import omegaconf
    import pyrootutils
    print("   [OK] - All dependencies available")
except Exception as e:
    print(f"   [FAILED]: {e}")
    sys.exit(1)

print()
print("=" * 50)
print("[OK] All tests passed!")
print("Fish Speech is properly installed and ready to use.")
print("=" * 50)
