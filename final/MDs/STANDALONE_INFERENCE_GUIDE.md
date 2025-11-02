# Standalone Inference - Alternative Approach

## Problem

The current implementation relies on Fish Speech's inference scripts which have complex dependencies:
- `pyrootutils` for path setup
- `hydra` for configuration
- Fish Speech package structure
- Config files in specific locations

## Better Solution: Direct Model Loading

Instead of using subprocess calls to Fish Speech scripts, we can load and run the models directly in Python. This makes the project truly standalone.

## Implementation Plan

### Option 1: Keep Subprocess (Current - Simpler)
**Pros**:
- No need to understand Fish Speech internals
- Stable and isolated
- Easy to update when Fish Speech changes

**Cons**:
- Requires Fish Speech installation
- Subprocess overhead (~10-50ms)
- Path configuration needed

### Option 2: Direct Model Loading (Better - More Complex)
**Pros**:
- Truly standalone
- No subprocess overhead
- Faster inference
- Full control

**Cons**:
- Need to understand model architecture
- More code to maintain
- Harder to update

## Recommended: Hybrid Approach

Keep the current subprocess approach BUT make Fish Speech installation easier:

### Step 1: Include Fish Speech as Dependency

Add to `requirements.txt`:
```txt
fish-speech @ git+https://github.com/fishaudio/fish-speech.git
```

### Step 2: Auto-detect Fish Speech Location

Update `opt_engine.py` to find Fish Speech automatically:
```python
def _find_fish_speech_dir():
    # Try common locations
    locations = [
        Path(__file__).parent.parent / "fish-speech",
        Path.home() / ".fish-speech",
        # Check if installed as package
        Path(fish_speech.__file__).parent if 'fish_speech' in sys.modules else None
    ]
    for loc in locations:
        if loc and loc.exists():
            return loc
    raise FileNotFoundError("Fish Speech not found")
```

### Step 3: Provide Installation Script

Create `install_fish_speech.bat`:
```batch
@echo off
echo Installing Fish Speech...
git clone https://github.com/fishaudio/fish-speech.git
cd fish-speech
pip install -e .
echo Done!
```

## Current Workaround

For now, the simplest solution is to ensure `FISH_SPEECH_DIR` is correctly set in `.env`:

```bash
FISH_SPEECH_DIR=C:\Users\VM02\Desktop\THESIS (SALAS)\SECOND PHASE ENV\voice cloning (FINAL)\scripts\fish-speech
```

## Future Enhancement

Consider implementing direct model loading in a future version for true standalone operation.
