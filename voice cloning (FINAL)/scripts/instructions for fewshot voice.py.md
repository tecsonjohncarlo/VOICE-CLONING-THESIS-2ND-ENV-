# Simple Few-Shot Voice Cloning Usage Guide

This is exactly what you wanted - use your trained speaker encoder for single-audio voice cloning!

## Quick Setup

```bash
# Install CoquiTTS (if not already installed)
pip install coqui-tts

# Install audio processing
pip install noisereduce librosa soundfile
```

## Usage Examples

### 1. Basic Voice Cloning (30-60 second reference audio)

```bash
# Clone Addison Rae's voice (or any target speaker)
python few_shot_voice_cloner.py \
    --speaker_encoder models/best_speaker_encoder.pth \
    --reference_audio addison_rae_voice.wav \
    --text "Hello everyone, welcome to my channel! Today we're going to talk about voice cloning technology." \
    --output addison_cloned.wav \
    --language en
```

### 2. Interactive Python Usage

```python
from few_shot_voice_cloner import FewShotVoiceCloner

# Initialize with your trained model
cloner = FewShotVoiceCloner("models/best_speaker_encoder.pth")

# Clone voice from single reference
result = cloner.clone_voice(
    text="This is a test of voice cloning technology",
    reference_audio_path="target_speaker.wav",
    output_path="cloned_output.wav"
)

print(f"Cloned voice saved to: {result}")
```

### 3. Batch Generation (Same Voice, Multiple Texts)

```python
# Generate multiple samples with same voice
texts = [
    "Welcome to my podcast.",
    "Thanks for watching this video.",
    "Don't forget to like and subscribe!",
    "See you in the next episode."
]

results = cloner.clone_voice_batch(
    texts=texts,
    reference_audio_path="target_speaker.wav",
    output_dir="batch_outputs"
)

print(f"Generated {len(results)} audio files")
```

## Reference Audio Requirements

### Perfect Setup:
- **Duration**: 30-60 seconds
- **Format**: WAV, 22kHz, mono
- **Quality**: Clear speech, minimal background noise
- **Content**: Natural speaking (not reading robotically)
- **Variety**: Different sentences/emotions if possible

### Acceptable Setup:
- **Minimum duration**: 10-15 seconds
- **Any common format**: MP3, WAV, M4A (will be converted)
- **Sample rate**: Any (will be resampled to 22kHz)

## Quality Tips

### For Best Results:
1. **Use high-quality reference audio**
2. **Clean audio environment** (no echo, minimal noise)
3. **Natural speech patterns** (not monotone)
4. **Consistent voice characteristics** across reference

### Audio Preparation:
```python
# The system automatically:
# - Converts to 22kHz mono WAV
# - Applies noise reduction
# - Trims silence
# - Normalizes volume

# Manual preparation (if needed):
import librosa
import soundfile as sf

audio, sr = librosa.load("raw_audio.mp3", sr=22050)
sf.write("prepared_audio.wav", audio, sr)
```

## How It Works

Your approach is exactly right:

1. **Your Trained Speaker Encoder** extracts speaker identity from reference audio
2. **CoquiTTS** handles text-to-speech synthesis
3. **Voice Cloning** combines both to generate target speaker's voice

```
Reference Audio → Your Speaker Encoder → Speaker Embedding
                                              ↓
Text Input → CoquiTTS → Voice Cloning → Generated Speech
```

## Comparison with Traditional Approach

### Your Few-Shot Method:
- ✅ Single reference audio (30-60 seconds)
- ✅ No additional training needed
- ✅ Works with any target speaker
- ✅ Uses your existing speaker encoder
- ✅ Fast inference (seconds)

### Traditional Fine-Tuning:
- ❌ Requires 100+ training samples per speaker
- ❌ Hours of training per new speaker
- ❌ Expensive computation
- ❌ Overfitting risks

## Expected Performance

With your VCTK/RAVDESS trained encoder:

### Excellent Results:
- Speakers similar to VCTK/RAVDESS demographics
- Clear, expressive speech
- English language content

### Good Results:
- Any English speaker
- Clean reference audio
- Standard speech patterns

### Challenging Cases:
- Very different languages
- Heavily accented speech
- Noisy reference audio
- Extreme vocal characteristics

## Troubleshooting

### Issue: Generated audio sounds robotic
**Solution**: 
- Use longer reference audio (45-60 seconds)
- Ensure reference has natural intonation
- Try different text lengths

### Issue: Voice doesn't sound like target
**Solution**:
- Improve reference audio quality
- Use more expressive reference speech
- Try multiple reference segments

### Issue: CUDA out of memory
**Solution**:
```python
# Use CPU for inference (slower but works)
cloner = FewShotVoiceCloner(
    "models/best_speaker_encoder.pth",
    device="cpu"
)
```

## Advanced Features

### Speaker Comparison
```python
# Compare how similar two voices are
similarity = cloner.compare_speakers(
    "speaker1.wav", 
    "speaker2.wav"
)
print(f"Similarity: {similarity:.3f}")
```

### Custom Audio Cleaning
```python
# Disable auto-cleaning if you have perfect audio
cloner = FewShotVoiceCloner(
    "models/best_speaker_encoder.pth",
    enable_audio_cleaning=False
)
```

## Real-World Usage

```python
# Production-ready voice cloning function
def clone_celebrity_voice(celebrity_audio_path, script_text, output_path):
    """Clone celebrity voice for content creation"""
    
    cloner = FewShotVoiceCloner("models/best_speaker_encoder.pth")
    
    result = cloner.clone_voice(
        text=script_text,
        reference_audio_path=celebrity_audio_path,
        output_path=output_path
    )
    
    return result

# Example: Clone Addison Rae's voice
result = clone_celebrity_voice(
    celebrity_audio_path="addison_rae_reference.wav",
    script_text="Hey guys! Welcome back to my channel. Today's video is super exciting!",
    output_path="addison_rae_cloned.wav"
)
```
# run this command in the terminal
python fewshot_voice.py `
  --speaker_encoder "C:\Users\VM02\Desktop\THESIS (SALAS)\SECOND PHASE ENV\voice cloning (FINAL)\model\best_speaker_encoder.pth" `
  --reference_audio "C:\Users\VM02\Desktop\THESIS (SALAS)\SECOND PHASE ENV\voice cloning (FINAL)\reference_audio\addison_abithappy_60s.wav" `
  --text "Hello, this is a test of few-shot voice cloning." `
  --output "C:\Users\VM02\Desktop\THESIS (SALAS)\SECOND PHASE ENV\voice cloning (FINAL)\output\cloned.wav" `
  --language en `
  --speed 1.0 `
  --temperature 0.75 `
  --no_cleaning `
  --device cuda