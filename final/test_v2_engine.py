"""
Quick test script for V2 engine
"""
import os
os.environ['FISH_SPEECH_DIR'] = r'C:\Users\VM02\Desktop\THESIS (SALAS)\SECOND PHASE ENV\final\fish-speech'

from backend.opt_engine_v2 import OptimizedFishSpeechV2

print("=" * 60)
print("Testing OptimizedFishSpeechV2")
print("=" * 60)

try:
    # Initialize engine
    print("\n1. Initializing engine...")
    engine = OptimizedFishSpeechV2(
        model_path="checkpoints/openaudio-s1-mini",
        device="cuda",
        enable_optimizations=True
    )
    
    print("\n2. Engine initialized successfully!")
    print(f"   Device: {engine.device}")
    print(f"   Precision: {engine.precision_mode}")
    
    # Test TTS
    print("\n3. Testing TTS (no reference audio)...")
    audio, sr, metrics = engine.tts(
        text="Hello world! This is a test of the optimized Fish Speech engine version two.",
        output_path="test_output_v2.wav"
    )
    
    print("\n4. TTS completed successfully!")
    print(f"   Audio shape: {audio.shape}")
    print(f"   Sample rate: {sr} Hz")
    print(f"   Duration: {metrics['audio_duration_s']:.2f}s")
    print(f"   Latency: {metrics['latency_ms']:.0f}ms")
    print(f"   RTF: {metrics['rtf']:.2f}x")
    print(f"   Peak VRAM: {metrics['peak_vram_mb']:.0f}MB")
    
    # Check health
    print("\n5. Checking health...")
    health = engine.get_health()
    print(f"   Status: {health['status']}")
    print(f"   Device: {health['device']}")
    
    print("\n" + "=" * 60)
    print("✅ ALL TESTS PASSED!")
    print("=" * 60)
    print("\nYou can now start the backend:")
    print("  python backend/app.py")
    
except Exception as e:
    print(f"\n❌ ERROR: {e}")
    import traceback
    traceback.print_exc()
