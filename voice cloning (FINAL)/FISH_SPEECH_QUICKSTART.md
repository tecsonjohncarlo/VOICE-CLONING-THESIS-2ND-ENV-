# Fish Speech Quick Start Guide

## Installation (5 minutes)

### Step 1: Install Fish Speech

```bash
# Clone the repository
git clone https://github.com/fishaudio/fish-speech.git
cd fish-speech

# Create conda environment
conda create -n fish-speech python=3.12
conda activate fish-speech

# Install dependencies (GPU with CUDA 12.9)
pip install -e .[cu129]

# Or for CPU only
pip install -e .[cpu]
```

### Step 2: Download Model Weights

```bash
# Install Hugging Face CLI
pip install huggingface_hub[cli]

# Download OpenAudio S1-mini (recommended for testing)
hf download fishaudio/openaudio-s1-mini --local-dir checkpoints/openaudio-s1-mini

# Or download full S1 model (better quality, larger)
# hf download fishaudio/openaudio-s1 --local-dir checkpoints/openaudio-s1
```

### Step 3: Copy Wrapper Files

Copy these files to your project:
- `fish_speech_wrapper.py` → Your scripts directory
- `fish_speech_integration_example.py` → Your scripts directory

---

## Basic Usage (2 minutes)

### Method 1: Using the Wrapper Class

```python
from fish_speech_wrapper import FishSpeechTTS

# Initialize
tts = FishSpeechTTS(model_path="checkpoints/openaudio-s1-mini")

# Basic synthesis
audio = tts.tts(
    text="Hello, this is a test of Fish Speech!",
    speaker_wav="reference_audio/sample.wav",
    output_path="output.wav"
)

# With emotions
audio = tts.tts(
    text="(excited) This is amazing! (laughing) I love it!",
    speaker_wav="reference_audio/sample.wav",
    output_path="output_emotional.wav"
)

# Cleanup
tts.cleanup()
```

### Method 2: Command Line (3-Stage Process)

```bash
# Stage 1: Extract VQ tokens from reference audio
python fish_speech/models/dac/inference.py \
  -i "reference.wav" \
  --checkpoint-path "checkpoints/openaudio-s1-mini/codec.pth"

# Stage 2: Generate semantic tokens from text
python fish_speech/models/text2semantic/inference.py \
  --text "Hello, how are you?" \
  --prompt-tokens "fake.npy" \
  --compile

# Stage 3: Synthesize audio
python fish_speech/models/dac/inference.py \
  -i "codes_0.npy" \
  --checkpoint-path "checkpoints/openaudio-s1-mini/codec.pth"

# Output: fake.wav
```

---

## Integration with Your Existing System

### Option 1: Add to VoiceCloner Class

```python
from fish_speech_wrapper import FishSpeechTTS

class VoiceCloner:
    def __init__(self):
        # Existing code...
        self.fish_speech_model = None
        self.model_type = None  # 'xtts', 'rvc', or 'fish_speech'
    
    def load_fish_speech(self):
        self.fish_speech_model = FishSpeechTTS()
        self.model_type = 'fish_speech'
        return True
    
    def synthesize(self, text, output_path, reference_audio_path=None, **kwargs):
        if self.model_type == 'fish_speech':
            return self.fish_speech_model.tts(
                text=text,
                speaker_wav=reference_audio_path,
                output_path=output_path
            )
        elif self.model_type == 'xtts':
            # Your existing XTTS code
            pass
```

### Option 2: Use Integration Example

```bash
# Run the integration example
cd "voice cloning (FINAL)/scripts"
python fish_speech_integration_example.py
```

---

## Testing (5 minutes)

### Test 1: Basic Synthesis

```python
from fish_speech_wrapper import FishSpeechTTS

tts = FishSpeechTTS()

# Use one of your existing reference audios
audio = tts.tts(
    text="This is a test of the Fish Speech system.",
    speaker_wav="reference_audio/sample1.wav",
    output_path="test_basic.wav"
)

print("Test 1 complete! Listen to test_basic.wav")
```

### Test 2: Emotional Speech

```python
audio = tts.tts(
    text="(excited) Hello everyone! (laughing) This is so cool!",
    speaker_wav="reference_audio/sample1.wav",
    output_path="test_emotional.wav"
)

print("Test 2 complete! Listen to test_emotional.wav")
```

### Test 3: Compare with XTTS

```python
from fish_speech_integration_example import EnhancedVoiceCloner

cloner = EnhancedVoiceCloner()
cloner.load_fish_speech()
cloner.load_xtts()

results = cloner.compare_models(
    text="The quick brown fox jumps over the lazy dog.",
    reference_audio="reference_audio/sample1.wav"
)

print("Comparison complete! Check the comparison/ directory")
```

---

## Emotion Markers

### Basic Emotions
```
(angry) (sad) (excited) (surprised) (happy) (scared) (worried)
(nervous) (frustrated) (depressed) (embarrassed) (proud) (relaxed)
(grateful) (confident) (interested) (curious) (confused) (joyful)
```

### Tone Markers
```
(whispering) (shouting) (screaming) (soft tone) (in a hurry tone)
```

### Special Effects
```
(laughing) (chuckling) (sobbing) (crying loudly) (sighing)
(panting) (groaning) (crowd laughing)
```

### Usage Examples
```python
# Combine multiple emotions
text = "(excited) Hello! (laughing) This is great! (happy)"

# Use tones
text = "(whispering) Can you hear me? (soft tone) I'm here."

# Dramatic speech
text = "(angry) Stop! (shouting) I said stop right now!"
```

---

## Troubleshooting

### Issue 1: "Model not found"
```bash
# Solution: Download the model
hf download fishaudio/openaudio-s1-mini --local-dir checkpoints/openaudio-s1-mini
```

### Issue 2: "CUDA out of memory"
```python
# Solution 1: Use S1-mini instead of S1
tts = FishSpeechTTS(model_path="checkpoints/openaudio-s1-mini")

# Solution 2: Use half precision
audio = tts.tts(text="...", half_precision=True)

# Solution 3: Clear cache between runs
import torch
torch.cuda.empty_cache()
```

### Issue 3: "fake.npy not found"
```python
# Solution: Check working directory
import os
print(f"Current dir: {os.getcwd()}")
print(f"Files: {os.listdir('.')}")

# Make sure you're in the fish-speech directory
os.chdir("path/to/fish-speech")
```

### Issue 4: Slow inference
```python
# Solution: Enable compilation (10x speedup)
audio = tts.tts(text="...", compile_mode=True)
```

### Issue 5: Windows compatibility
```bash
# Fish Speech requires Linux/WSL
# Solution: Use WSL2

# Install WSL2
wsl --install

# Then run Fish Speech in WSL
wsl
cd /mnt/c/Users/YourName/Desktop/...
conda activate fish-speech
python fish_speech_wrapper.py
```

---

## Performance Tips

### 1. Cache VQ Tokens
```python
# VQ tokens are automatically cached by default
tts = FishSpeechTTS()

# First call: extracts VQ tokens (slow)
audio1 = tts.tts(text="Text 1", speaker_wav="ref.wav")

# Second call: uses cached tokens (fast)
audio2 = tts.tts(text="Text 2", speaker_wav="ref.wav")

# Clear cache when done
tts.clear_cache()
```

### 2. Use Compilation
```python
# 10x speed improvement (15 → 150 tokens/sec)
audio = tts.tts(text="...", compile_mode=True)
```

### 3. Batch Processing
```python
# Extract VQ tokens once
vq_tokens = tts.extract_vq_tokens("reference.wav")

# Generate multiple texts with same voice
texts = ["Text 1", "Text 2", "Text 3"]
for i, text in enumerate(texts):
    semantic = tts.generate_semantic_tokens(text, vq_tokens)
    output = tts.synthesize_audio(semantic, f"output_{i}.wav")
```

---

## Next Steps

1. **Test with your reference audios**
   - Use audios from `reference_audio/` directory
   - Try different speakers
   - Compare quality with XTTS

2. **Experiment with emotions**
   - Try different emotion combinations
   - Test with your use cases
   - Find what works best

3. **Integrate into your workflow**
   - Add to your VoiceCloner class
   - Update your UI/interface
   - Test with real users

4. **Optimize performance**
   - Enable compilation
   - Cache VQ tokens
   - Monitor GPU memory

5. **Production deployment**
   - Set up proper error handling
   - Add logging
   - Monitor quality metrics
   - Collect user feedback

---

## Comparison: XTTS vs Fish Speech

| Aspect | XTTS | Fish Speech |
|--------|------|-------------|
| **Setup Time** | 5 min | 10 min |
| **Quality** | Good | Excellent |
| **Speed (no compile)** | ~15 tok/s | ~15 tok/s |
| **Speed (compiled)** | N/A | ~150 tok/s |
| **GPU Memory** | 4-6GB | 12GB |
| **Emotional Control** | Limited | 50+ emotions |
| **Voice Cloning** | Direct | VQ tokens |
| **Languages** | Many | EN/CN/JP |

---

## Support

- **Documentation**: https://speech.fish.audio/
- **GitHub**: https://github.com/fishaudio/fish-speech
- **Issues**: Report bugs on GitHub
- **Community**: Join Discord/discussions

---

## Summary

**Installation**: 5-10 minutes
**First synthesis**: 2 minutes
**Integration**: 10-20 minutes
**Total time**: ~30 minutes to get started

**Key advantages**:
- ✅ Better quality than XTTS
- ✅ 10x faster with compilation
- ✅ Rich emotional control
- ✅ Easy integration

**Considerations**:
- ⚠️ Requires 12GB GPU memory
- ⚠️ 3-stage pipeline (more complex)
- ⚠️ Linux/WSL only
- ⚠️ Fewer languages than XTTS

**Recommendation**: Start with parallel testing, compare quality, then decide on full migration.
