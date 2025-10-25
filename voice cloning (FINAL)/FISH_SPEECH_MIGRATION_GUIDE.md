# Migration Guide: XTTS to Fish Speech (OpenAudio)

## Executive Summary

This guide provides a comprehensive roadmap for migrating your voice cloning system from **Coqui XTTS-v2** to **Fish Speech (OpenAudio)**, a state-of-the-art open-source TTS system that recently achieved #1 ranking on TTS-Arena2.

---

## Table of Contents

1. [Key Differences](#key-differences)
2. [Architecture Comparison](#architecture-comparison)
3. [Migration Strategy](#migration-strategy)
4. [Installation](#installation)
5. [Code Migration](#code-migration)
6. [Feature Mapping](#feature-mapping)
7. [Implementation Examples](#implementation-examples)
8. [Testing & Validation](#testing--validation)
9. [Performance Considerations](#performance-considerations)
10. [Troubleshooting](#troubleshooting)

---

## Key Differences

### XTTS-v2 vs Fish Speech

| Feature | XTTS-v2 | Fish Speech (OpenAudio S1) |
|---------|---------|----------------------------|
| **Architecture** | GPT-based with speaker embeddings | VQ-VAE + Transformer (3-stage pipeline) |
| **Voice Cloning** | Direct speaker embedding from audio | VQ tokens + prompt text/audio |
| **Quality** | Good | Excellent (0.008 WER, 0.004 CER) |
| **Speed** | ~15 tokens/sec | ~150 tokens/sec (with --compile) |
| **Emotional Control** | Limited | 50+ emotions + tone markers |
| **License** | Mozilla Public License 2.0 | Apache 2.0 (code) + CC-BY-NC-SA-4.0 (models) |
| **GPU Memory** | 4-6GB | 12GB (inference) |
| **Languages** | Multilingual | English, Chinese, Japanese (more coming) |
| **Model Size** | ~1.8GB | S1-mini available for smaller footprint |

---

## Architecture Comparison

### XTTS-v2 Architecture
```
Audio Input → Speaker Encoder → Speaker Embedding
                                       ↓
Text Input → GPT Model → Mel Spectrogram → Vocoder → Audio Output
```

### Fish Speech Architecture
```
Stage 1: Reference Audio → DAC Encoder → VQ Tokens (fake.npy)
                                              ↓
Stage 2: Text + VQ Tokens → Text2Semantic Model → Semantic Tokens (codes_N.npy)
                                                         ↓
Stage 3: Semantic Tokens → DAC Decoder → Audio Output
```

**Key Insight**: Fish Speech uses a **3-stage pipeline** instead of XTTS's single-pass approach. This provides better quality but requires more steps.

---

## Migration Strategy

### Phase 1: Parallel Implementation (Recommended)
- Keep XTTS running while implementing Fish Speech
- Test Fish Speech with subset of use cases
- Compare quality and performance
- Gradual rollout

### Phase 2: Direct Replacement
- Replace XTTS entirely
- Refactor all voice cloning code
- Update dependencies
- Full system testing

### Phase 3: Hybrid Approach
- Use Fish Speech for high-quality synthesis
- Keep XTTS for specific use cases (if needed)
- Implement fallback mechanism

---

## Installation

### Prerequisites
```bash
# System dependencies (Linux/WSL)
apt install portaudio19-dev libsox-dev ffmpeg

# For Windows, ensure you have WSL2 or use Docker
```

### Method 1: Conda (Recommended)
```bash
# Create environment
conda create -n fish-speech python=3.12
conda activate fish-speech

# GPU installation (CUDA 12.9)
pip install -e .[cu129]

# Or CPU-only
pip install -e .[cpu]
```

### Method 2: UV (Faster)
```bash
# GPU installation
uv sync --python 3.12 --extra cu129

# CPU-only
uv sync --python 3.12 --extra cpu
```

### Download Model Weights
```bash
# Install Hugging Face CLI
pip install huggingface_hub[cli]

# Download OpenAudio S1-mini (smaller, faster)
hf download fishaudio/openaudio-s1-mini --local-dir checkpoints/openaudio-s1-mini

# Or download full S1 model (better quality)
hf download fishaudio/openaudio-s1 --local-dir checkpoints/openaudio-s1
```

---

## Code Migration

### Current XTTS Implementation Structure

Your current system has:
1. **SpeakerEncoder** - Custom CNN-based speaker encoder
2. **VoiceCloner** - Main class handling XTTS synthesis
3. **Fine-tuning system** - For adding new speakers

### New Fish Speech Structure

You'll need:
1. **FishSpeechVoiceCloner** - New main class
2. **VQTokenExtractor** - Stage 1: Extract VQ tokens from reference audio
3. **SemanticGenerator** - Stage 2: Generate semantic tokens from text
4. **AudioSynthesizer** - Stage 3: Generate final audio

---

## Feature Mapping

### 1. Voice Cloning (Reference Audio)

**XTTS Approach:**
```python
# Single-step synthesis
wav = self.xtts_model.tts(
    text=text,
    speaker_wav=reference_audio_path,
    language=language
)
```

**Fish Speech Approach:**
```python
# Stage 1: Extract VQ tokens
vq_tokens = extract_vq_tokens(reference_audio_path)

# Stage 2: Generate semantic tokens
semantic_tokens = generate_semantic_tokens(
    text=text,
    prompt_tokens=vq_tokens,
    prompt_text=reference_text  # Optional transcript
)

# Stage 3: Synthesize audio
wav = synthesize_audio(semantic_tokens)
```

### 2. Speaker Embeddings

**XTTS:**
- Uses speaker embeddings directly
- Can save/reuse embeddings

**Fish Speech:**
- Uses VQ tokens (can be saved as `.npy` files)
- More flexible: can combine multiple reference audios
- Supports prompt text for better control

### 3. Language Support

**XTTS:**
```python
wav = model.tts(text=text, language="en")  # en, es, fr, de, etc.
```

**Fish Speech:**
```python
# Language auto-detected from text
# Currently supports: English, Chinese, Japanese
# More languages coming soon
```

### 4. Emotional Control (NEW in Fish Speech!)

**Fish Speech Only:**
```python
text = "(excited) Hello! (laughing) This is amazing!"
text = "(whispering) Can you hear me? (soft tone) I'm here."
text = "(angry) I can't believe this! (shouting) Stop!"
```

Available emotions:
- **Basic**: angry, sad, excited, surprised, happy, scared, worried, nervous, etc.
- **Advanced**: disdainful, anxious, hysterical, sarcastic, sincere, etc.
- **Tones**: whispering, shouting, screaming, soft tone, in a hurry tone
- **Effects**: laughing, chuckling, sobbing, crying, sighing, panting, groaning

---

## Implementation Examples

### Example 1: Basic Migration - Simple TTS

**Before (XTTS):**
```python
from TTS.api import TTS

tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2")
wav = tts.tts(
    text="Hello, how are you?",
    speaker_wav="reference.wav",
    language="en"
)
tts.save_wav(wav, "output.wav")
```

**After (Fish Speech):**
```python
import subprocess
import numpy as np

# Stage 1: Extract VQ tokens
subprocess.run([
    "python", "fish_speech/models/dac/inference.py",
    "-i", "reference.wav",
    "--checkpoint-path", "checkpoints/openaudio-s1-mini/codec.pth"
])

# Stage 2: Generate semantic tokens
subprocess.run([
    "python", "fish_speech/models/text2semantic/inference.py",
    "--text", "Hello, how are you?",
    "--prompt-tokens", "fake.npy",
    "--compile"
])

# Stage 3: Synthesize audio
subprocess.run([
    "python", "fish_speech/models/dac/inference.py",
    "-i", "codes_0.npy",
    "--checkpoint-path", "checkpoints/openaudio-s1-mini/codec.pth"
])
```

### Example 2: Creating a Fish Speech Wrapper Class

Create a new file: `fish_speech_wrapper.py`

```python
import subprocess
import numpy as np
import soundfile as sf
from pathlib import Path
import tempfile
import shutil

class FishSpeechTTS:
    """Wrapper for Fish Speech TTS to match XTTS-like interface"""
    
    def __init__(self, model_path="checkpoints/openaudio-s1-mini", device="cuda"):
        self.model_path = Path(model_path)
        self.device = device
        self.codec_path = self.model_path / "codec.pth"
        self.temp_dir = Path(tempfile.mkdtemp())
        
        if not self.codec_path.exists():
            raise FileNotFoundError(f"Model not found at {self.codec_path}")
        
        print(f"Fish Speech initialized with model: {model_path}")
    
    def extract_vq_tokens(self, audio_path, output_name="reference"):
        """Stage 1: Extract VQ tokens from reference audio"""
        audio_path = Path(audio_path)
        output_path = self.temp_dir / f"{output_name}.npy"
        
        cmd = [
            "python", "fish_speech/models/dac/inference.py",
            "-i", str(audio_path),
            "--checkpoint-path", str(self.codec_path)
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            raise RuntimeError(f"VQ extraction failed: {result.stderr}")
        
        # Move generated fake.npy to our temp directory
        if Path("fake.npy").exists():
            shutil.move("fake.npy", output_path)
        
        return output_path
    
    def generate_semantic_tokens(self, text, vq_tokens_path, 
                                 prompt_text=None, compile_mode=True):
        """Stage 2: Generate semantic tokens from text"""
        cmd = [
            "python", "fish_speech/models/text2semantic/inference.py",
            "--text", text,
            "--prompt-tokens", str(vq_tokens_path)
        ]
        
        if prompt_text:
            cmd.extend(["--prompt-text", prompt_text])
        
        if compile_mode:
            cmd.append("--compile")
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            raise RuntimeError(f"Semantic generation failed: {result.stderr}")
        
        # Find the generated codes file
        codes_files = list(Path(".").glob("codes_*.npy"))
        if not codes_files:
            raise RuntimeError("No semantic tokens generated")
        
        latest_codes = max(codes_files, key=lambda p: p.stat().st_mtime)
        return latest_codes
    
    def synthesize_audio(self, semantic_tokens_path, output_path="output.wav"):
        """Stage 3: Generate audio from semantic tokens"""
        cmd = [
            "python", "fish_speech/models/dac/inference.py",
            "-i", str(semantic_tokens_path),
            "--checkpoint-path", str(self.codec_path)
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            raise RuntimeError(f"Audio synthesis failed: {result.stderr}")
        
        # Move generated fake.wav to output path
        if Path("fake.wav").exists():
            shutil.move("fake.wav", output_path)
        
        return output_path
    
    def tts(self, text, speaker_wav=None, output_path="output.wav", 
            prompt_text=None, language="en", compile_mode=True):
        """
        XTTS-compatible interface for Fish Speech
        
        Args:
            text: Text to synthesize
            speaker_wav: Reference audio file path
            output_path: Output audio file path
            prompt_text: Optional transcript of reference audio
            language: Language (auto-detected, parameter kept for compatibility)
            compile_mode: Use compilation for faster inference
        
        Returns:
            numpy array of audio samples
        """
        try:
            # Stage 1: Extract VQ tokens from reference
            if speaker_wav:
                vq_tokens = self.extract_vq_tokens(speaker_wav)
            else:
                # Random voice selection
                vq_tokens = None
            
            # Stage 2: Generate semantic tokens
            semantic_tokens = self.generate_semantic_tokens(
                text=text,
                vq_tokens_path=vq_tokens,
                prompt_text=prompt_text,
                compile_mode=compile_mode
            )
            
            # Stage 3: Synthesize audio
            output_file = self.synthesize_audio(semantic_tokens, output_path)
            
            # Load and return audio
            audio, sr = sf.read(output_file)
            
            # Cleanup temporary files
            if vq_tokens and vq_tokens.exists():
                vq_tokens.unlink()
            if semantic_tokens.exists():
                semantic_tokens.unlink()
            
            return audio
            
        except Exception as e:
            print(f"TTS Error: {e}")
            raise
    
    def cleanup(self):
        """Clean up temporary directory"""
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
    
    def __del__(self):
        self.cleanup()


# Usage example
if __name__ == "__main__":
    tts = FishSpeechTTS(model_path="checkpoints/openaudio-s1-mini")
    
    # Basic synthesis
    audio = tts.tts(
        text="Hello, this is a test of Fish Speech!",
        speaker_wav="reference_audio/sample.wav",
        output_path="output.wav"
    )
    
    # With emotional control
    audio = tts.tts(
        text="(excited) This is amazing! (laughing) I love it!",
        speaker_wav="reference_audio/sample.wav",
        output_path="output_emotional.wav"
    )
    
    print("Synthesis complete!")
```

### Example 3: Integrating into Your VoiceCloner Class

Modify your existing `fewshot_voice.py`:

```python
class VoiceCloner:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_type = None  # 'xtts', 'rvc', or 'fish_speech'
        
        # Existing XTTS
        self.xtts_model = None
        
        # NEW: Fish Speech
        self.fish_speech_model = None
        
        # Existing RVC
        self.rvc_model = None
        
        print(f"Device: {self.device}")
    
    def load_fish_speech(self, model_path="checkpoints/openaudio-s1-mini"):
        """Load Fish Speech model"""
        try:
            from fish_speech_wrapper import FishSpeechTTS
            
            print("\nLoading Fish Speech model...")
            self.fish_speech_model = FishSpeechTTS(model_path=model_path)
            self.model_type = 'fish_speech'
            
            print("✓ Fish Speech loaded successfully!")
            return True
            
        except Exception as e:
            print(f"Error loading Fish Speech: {e}")
            return False
    
    def synthesize(self, text, output_path, 
                   reference_audio_path=None, speaker_embedding=None,
                   language="en", output_format="mp3", 
                   clean_reference=True, enable_preview=True):
        """Enhanced synthesize method with Fish Speech support"""
        
        if self.model_type == 'fish_speech':
            # Use Fish Speech
            print(f"\nSynthesizing with Fish Speech...")
            
            try:
                audio = self.fish_speech_model.tts(
                    text=text,
                    speaker_wav=reference_audio_path,
                    output_path=output_path,
                    compile_mode=True
                )
                
                print(f"✓ Synthesis complete: {output_path}")
                return str(output_path)
                
            except Exception as e:
                print(f"Fish Speech synthesis error: {e}")
                return None
        
        elif self.model_type == 'xtts':
            # Existing XTTS code
            # ... (keep your existing XTTS implementation)
            pass
        
        elif self.model_type == 'rvc':
            # Existing RVC code
            # ... (keep your existing RVC implementation)
            pass
```

---

## Testing & Validation

### Test Suite

Create `test_fish_speech_migration.py`:

```python
import unittest
from fish_speech_wrapper import FishSpeechTTS
from pathlib import Path

class TestFishSpeechMigration(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        cls.tts = FishSpeechTTS(model_path="checkpoints/openaudio-s1-mini")
        cls.test_audio = "reference_audio/test_sample.wav"
        cls.output_dir = Path("test_outputs")
        cls.output_dir.mkdir(exist_ok=True)
    
    def test_basic_synthesis(self):
        """Test basic text-to-speech"""
        output = self.tts.tts(
            text="This is a basic test.",
            speaker_wav=self.test_audio,
            output_path=self.output_dir / "basic.wav"
        )
        self.assertIsNotNone(output)
        self.assertTrue(Path(self.output_dir / "basic.wav").exists())
    
    def test_emotional_synthesis(self):
        """Test emotional control"""
        output = self.tts.tts(
            text="(excited) This is exciting! (laughing)",
            speaker_wav=self.test_audio,
            output_path=self.output_dir / "emotional.wav"
        )
        self.assertIsNotNone(output)
    
    def test_long_text(self):
        """Test long text synthesis"""
        long_text = "This is a longer text. " * 20
        output = self.tts.tts(
            text=long_text,
            speaker_wav=self.test_audio,
            output_path=self.output_dir / "long.wav"
        )
        self.assertIsNotNone(output)
    
    @classmethod
    def tearDownClass(cls):
        cls.tts.cleanup()

if __name__ == "__main__":
    unittest.main()
```

### Quality Comparison

```python
def compare_xtts_vs_fish_speech(text, reference_audio):
    """Compare XTTS and Fish Speech output quality"""
    
    # XTTS synthesis
    xtts_cloner = VoiceCloner()
    xtts_cloner.load_xtts()
    xtts_output = xtts_cloner.synthesize(
        text=text,
        reference_audio_path=reference_audio,
        output_path="comparison_xtts.wav"
    )
    
    # Fish Speech synthesis
    fish_cloner = VoiceCloner()
    fish_cloner.load_fish_speech()
    fish_output = fish_cloner.synthesize(
        text=text,
        reference_audio_path=reference_audio,
        output_path="comparison_fish.wav"
    )
    
    print("\nComparison complete!")
    print("Listen to both outputs and evaluate:")
    print("1. Naturalness")
    print("2. Voice similarity")
    print("3. Pronunciation accuracy")
    print("4. Audio quality")
```

---

## Performance Considerations

### Speed Comparison

| Model | Tokens/Second | Real-time Factor | GPU Memory |
|-------|---------------|------------------|------------|
| XTTS-v2 | ~15 | ~0.5x | 4-6GB |
| Fish Speech (no compile) | ~15 | ~0.5x | 12GB |
| Fish Speech (--compile) | ~150 | ~5x | 12GB |

### Optimization Tips

1. **Use --compile flag** for 10x speed improvement
2. **Cache VQ tokens** for repeated use of same reference audio
3. **Batch processing** for multiple texts with same speaker
4. **Use S1-mini** for faster inference with slightly lower quality

### Memory Management

```python
# Clear GPU cache between syntheses
import torch

def synthesize_with_cleanup(tts, text, reference):
    output = tts.tts(text=text, speaker_wav=reference)
    
    # Clear GPU memory
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return output
```

---

## Troubleshooting

### Common Issues

#### 1. "Model not found" Error
```bash
# Solution: Download model weights
hf download fishaudio/openaudio-s1-mini --local-dir checkpoints/openaudio-s1-mini
```

#### 2. CUDA Out of Memory
```python
# Solution: Use S1-mini or reduce batch size
# Or use --half parameter for FP16
subprocess.run([..., "--half"])
```

#### 3. "fake.npy not found"
```python
# Solution: Check working directory and file paths
# Ensure Stage 1 completes before Stage 2
import os
print(f"Current directory: {os.getcwd()}")
print(f"Files: {os.listdir('.')}")
```

#### 4. Poor Voice Quality
```python
# Solutions:
# 1. Use longer reference audio (10-30 seconds)
# 2. Provide prompt_text (transcript of reference)
# 3. Use higher quality reference audio (clean, clear)
# 4. Try full S1 model instead of S1-mini
```

#### 5. Windows Compatibility
```bash
# Fish Speech requires Linux/WSL
# Solution: Use WSL2 or Docker

# WSL2 setup:
wsl --install
wsl --set-default-version 2
```

---

## Migration Checklist

### Pre-Migration
- [ ] Review current XTTS implementation
- [ ] Identify all voice cloning use cases
- [ ] Document current performance metrics
- [ ] Set up test environment
- [ ] Install Fish Speech dependencies

### Migration Phase
- [ ] Install Fish Speech
- [ ] Download model weights
- [ ] Create wrapper class
- [ ] Integrate into existing codebase
- [ ] Update VoiceCloner class
- [ ] Test basic synthesis
- [ ] Test emotional control
- [ ] Test with multiple speakers

### Post-Migration
- [ ] Performance testing
- [ ] Quality comparison
- [ ] Update documentation
- [ ] Train team on new system
- [ ] Monitor production usage
- [ ] Collect user feedback

---

## Additional Resources

### Official Documentation
- **Fish Speech GitHub**: https://github.com/fishaudio/fish-speech
- **Installation Guide**: https://speech.fish.audio/install/
- **Inference Guide**: https://speech.fish.audio/inference/
- **Samples**: https://speech.fish.audio/examples

### Model Downloads
- **OpenAudio S1-mini**: https://huggingface.co/fishaudio/openaudio-s1-mini
- **OpenAudio S1**: https://huggingface.co/fishaudio/openaudio-s1

### Community
- **Discord**: Join Fish Audio community
- **Issues**: Report bugs on GitHub
- **Discussions**: Ask questions in GitHub Discussions

---

## Conclusion

Migrating from XTTS to Fish Speech offers significant improvements in:
- **Quality**: State-of-the-art TTS performance
- **Speed**: Up to 10x faster with compilation
- **Control**: Rich emotional and tonal control
- **Flexibility**: Better voice cloning with VQ tokens

The migration requires refactoring your synthesis pipeline from a single-step to a 3-stage process, but the wrapper class provided makes this transition smooth while maintaining compatibility with your existing code structure.

**Recommended Approach**: Start with the parallel implementation, test thoroughly, and gradually transition to Fish Speech for production use.

---

## Next Steps

1. **Install Fish Speech** following the installation section
2. **Download model weights** (S1-mini for testing)
3. **Create wrapper class** using the provided example
4. **Test with sample audio** from your reference_audio directory
5. **Compare quality** with your current XTTS output
6. **Integrate gradually** into your VoiceCloner class
7. **Monitor and optimize** based on your specific use cases

Good luck with your migration! 🚀
