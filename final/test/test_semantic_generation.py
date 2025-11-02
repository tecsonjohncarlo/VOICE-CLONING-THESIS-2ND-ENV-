"""Test semantic generation directly"""
import os
import sys
import subprocess
from pathlib import Path

# Suppress NumPy warnings
os.environ['PYTHONWARNINGS'] = 'ignore'

# Configuration
FISH_SPEECH_DIR = r"C:\Users\VM02\Desktop\THESIS (SALAS)\SECOND PHASE ENV\final\fish-speech"

print("Testing semantic generation directly...")
print(f"Fish Speech directory: {FISH_SPEECH_DIR}")
print()

# Check if inference script exists
inference_script = Path(FISH_SPEECH_DIR) / "fish_speech" / "models" / "text2semantic" / "inference.py"
print(f"Checking inference script: {inference_script}")

if not inference_script.exists():
    print(f"ERROR: Inference script not found!")
    sys.exit(1)

print("Script exists. Testing execution...")
print()

# Build test command
cmd = [
    sys.executable,
    str(inference_script),
    "--text", "Hello world",
    "--device", "cpu",
    "--temperature", "0.7",
    "--top-p", "0.7",
    "--max-new-tokens", "50"
]

print(f"Command: {' '.join(cmd)}")
print(f"Working directory: {FISH_SPEECH_DIR}")
print()
print("Running command...")
print("=" * 60)

# Execute and show full output
try:
    result = subprocess.run(
        cmd, 
        cwd=FISH_SPEECH_DIR,
        capture_output=False,  # Show output directly
        text=True,
        timeout=120
    )
    print("=" * 60)
    print(f"Return code: {result.returncode}")
    
except subprocess.TimeoutExpired:
    print("ERROR: Command timed out after 120 seconds")
except Exception as e:
    print(f"ERROR: {e}")
