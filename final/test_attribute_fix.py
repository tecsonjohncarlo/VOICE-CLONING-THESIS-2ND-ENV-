#!/usr/bin/env python3
"""
Quick test to verify OptimalConfig attribute names are correct
"""
import sys
import os

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))

# Set device to CPU
os.environ['DEVICE'] = 'cpu'

print("Testing OptimalConfig attribute names...")
print("=" * 60)

try:
    from backend.smart_backend import SmartAdaptiveBackend
    
    # Initialize backend
    print("1. Initializing backend...")
    backend = SmartAdaptiveBackend(model_path="checkpoints/openaudio-s1-mini")
    
    # Get config
    config = backend.config
    
    print("2. Checking OptimalConfig attributes...")
    print(f"   - useonnx: {config.useonnx}")
    print(f"   - usetorchcompile: {config.usetorchcompile}")
    print(f"   - device: {config.device}")
    print(f"   - precision: {config.precision}")
    print(f"   - quantization: {config.quantization}")
    
    print("\n3. Testing app.py health endpoint attributes...")
    health = backend.get_health()
    print(f"   - Status: {health['status']}")
    print(f"   - Device: {health['system_info']['device']}")
    
    print("\n✅ All attribute tests passed!")
    print("=" * 60)
    
    backend.cleanup()
    sys.exit(0)
    
except AttributeError as e:
    print(f"❌ AttributeError: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
