"""Test if all imports work"""
import sys
print("Testing imports...")

try:
    print("1. Testing FastAPI...")
    import fastapi
    print("   OK")
except Exception as e:
    print(f"   FAILED: {e}")
    sys.exit(1)

try:
    print("2. Testing Gradio...")
    import gradio
    print("   OK")
except Exception as e:
    print(f"   FAILED: {e}")
    sys.exit(1)

try:
    print("3. Testing backend.opt_engine...")
    sys.path.insert(0, 'backend')
    from opt_engine import OptimizedFishSpeech
    print("   OK")
except Exception as e:
    print(f"   FAILED: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\nAll imports successful!")
print("The issue might be elsewhere...")
