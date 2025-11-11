# Alternative Approach - Direct Model Loading

## Issue Identified

The Fish Speech inference scripts expect:
1. Specific model format (not OpenAudio S1-Mini format)
2. Configuration files in specific locations
3. Models trained with Fish Speech training pipeline

The OpenAudio S1-Mini model we downloaded is in HuggingFace format, not Fish Speech format.

## Solutions

### Option 1: Use Compatible Model (Recommended)

Download a Fish Speech compatible model:

```bash
# Remove current model
rmdir /s /q checkpoints\openaudio-s1-mini

# Download Fish Speech model
huggingface-cli download fishaudio/fish-speech-1.4 --local-dir checkpoints/fish-speech-1.4
```

### Option 2: Use Fish Speech Web UI

Instead of our custom backend, use Fish Speech's built-in web UI:

```bash
cd fish-speech
python -m tools.webui.app
```

### Option 3: Simplified Direct Loading

Create a simplified TTS engine that loads models directly without subprocess calls.

## Recommended: Try Option 1

Let's download the correct Fish Speech model format:

1. **Remove incompatible model:**
   ```bash
   rmdir /s /q checkpoints\openaudio-s1-mini
   ```

2. **Download Fish Speech model:**
   ```bash
   huggingface-cli download fishaudio/fish-speech-1.4 --local-dir checkpoints/fish-speech-1.4
   ```

3. **Update .env:**
   ```bash
   MODEL_DIR=checkpoints/fish-speech-1.4
   ```

4. **Restart backend**

This should resolve the compatibility issue between the model format and Fish Speech inference scripts.
