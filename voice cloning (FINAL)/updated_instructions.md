# Voice Cloning System Setup Guide

Complete setup guide for training and using voice cloning models with your pre-trained speaker encoder.

## System Requirements

### Hardware Requirements
- **GPU**: NVIDIA V100 (32GB VRAM) - Recommended
- **CPU**: QEMU Virtual CPU 2.5GHz+ (16+ cores recommended)
- **RAM**: 32GB+ DDR4
- **Storage**: 100GB+ SSD space for datasets and models

### Software Requirements
- **OS**: Ubuntu 20.04+ or CentOS 7+
- **Python**: 3.8-3.10
- **CUDA**: 11.7+
- **cuDNN**: 8.5+

## Environment Setup

### 1. Create Conda Environment

```bash
# Create new conda environment
conda create -n voice_cloning python=3.9 -y
conda activate voice_cloning

# Install PyTorch with CUDA support (for V100)
conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia

# Verify CUDA installation
python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'CUDA Version: {torch.cuda.get_version.cuda}'); print(f'Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU\"}')"
```

### 2. Install Core Dependencies

```bash
# Audio processing libraries
pip install librosa==0.9.2
pip install soundfile==0.12.1
pip install torchaudio==0.13.1

# Machine learning and data processing
pip install numpy==1.24.3
pip install scipy==1.10.1
pip install scikit-learn==1.3.0
pip install pandas==2.0.3

# Training and monitoring
pip install tqdm==4.65.0
pip install tensorboard==2.13.0
pip install matplotlib==3.7.1

# Audio enhancement (optional but recommended)
pip install noisereduce==2.0.1
pip install pyworld==0.3.2

# Text processing
pip install phonemizer==3.2.1
pip install unidecode==1.3.6

# For better audio format support
pip install pydub==0.25.1
```

### 3. Install System Dependencies

```bash
# Ubuntu/Debian
sudo apt-get update
sudo apt-get install -y ffmpeg libsndfile1 espeak espeak-data libespeak1 libespeak-dev

# CentOS/RHEL
sudo yum update
sudo yum install -y ffmpeg libsndfile espeak espeak-devel
```

### 4. Verify Installation

```bash
python -c "
import torch
import librosa
import soundfile
import tensorboard
print('PyTorch version:', torch.__version__)
print('CUDA available:', torch.cuda.is_available())
print('Librosa version:', librosa.__version__)
print('All dependencies installed successfully!')
"
```

## Project Structure

Create the following directory structure:

```
voice_cloning_project/
├── models/
│   └── best_speaker_encoder.pth       # Your trained speaker encoder
├── data/
│   ├── training/                      # Training audio-text pairs
│   │   ├── speaker1_001.wav
│   │   ├── speaker1_001.txt
│   │   ├── speaker1_002.wav
│   │   ├── speaker1_002.txt
│   │   └── ...
│   └── reference/                     # Reference audio for cloning
│       └── target_voice.wav
├── output/
│   ├── models/                        # Trained voice cloning models
│   ├── generated/                     # Generated speech samples
│   └── logs/                          # Training logs
├── scripts/
│   ├── train_voice_cloning.py         # Training script
│   ├── voice_cloning_inference.py     # Inference script
│   └── utils/
├── configs/
│   └── training_config.json
└── requirements.txt
```

## Data Preparation

### 1. Training Data Format

For training the voice cloning model, you need paired audio-text files:

```
data/training/
├── sample_001.wav  (22050 Hz, mono, 2-10 seconds)
├── sample_001.txt  (corresponding text)
├── sample_002.wav
├── sample_002.txt
└── ...
```

**Audio Requirements:**
- **Format**: WAV
- **Sample Rate**: 22050 Hz
- **Channels**: Mono
- **Duration**: 2-10 seconds per sample
- **Quality**: Clean, minimal background noise

**Text Requirements:**
- **Encoding**: UTF-8
- **Content**: Exact transcription of audio
- **Length**: Match audio duration (natural speech pace)

### 2. Data Preprocessing Script

Create `scripts/prepare_data.py`:

```python
import os
import librosa
import soundfile as sf
from pathlib import Path

def preprocess_audio(input_dir, output_dir, target_sr=22050):
    """Convert all audio files to required format"""
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    for audio_file in input_path.glob("*.wav"):
        try:
            # Load audio
            audio, sr = librosa.load(str(audio_file), sr=target_sr, mono=True)
            
            # Trim silence
            audio, _ = librosa.effects.trim(audio, top_db=20)
            
            # Normalize
            audio = audio / (abs(audio).max() + 1e-8) * 0.9
            
            # Save
            output_file = output_path / audio_file.name
            sf.write(output_file, audio, target_sr)
            
            print(f"Processed: {audio_file.name}")
            
        except Exception as e:
            print(f"Error processing {audio_file.name}: {e}")

if __name__ == "__main__":
    preprocess_audio("data/raw_training", "data/training")
```

### 3. Validate Your Speaker Encoder

Before training, verify your speaker encoder works correctly:

```python
import torch
from scripts.train_voice_cloning import SpeakerEncoder

# Load your trained speaker encoder
checkpoint = torch.load("models/best_speaker_encoder.pth", map_location="cuda")
num_speakers = checkpoint['num_speakers']

speaker_encoder = SpeakerEncoder(
    input_dim=80,
    hidden_dim=256, 
    embedding_dim=128,
    num_speakers=num_speakers
)

speaker_encoder.load_state_dict(checkpoint['model_state_dict'])
speaker_encoder.eval()

print(f"Speaker encoder loaded successfully!")
print(f"Number of speakers: {num_speakers}")
print(f"Embedding dimension: {128}")

# Test with dummy input
test_input = torch.randn(1, 80, 100).cuda()
with torch.no_grad():
    embedding = speaker_encoder(test_input, return_embedding=True)
    print(f"Test embedding shape: {embedding.shape}")
```

## Training Configuration

### 1. Create Training Config

Create `configs/training_config.json`:

```json
{
    "model": {
        "speaker_embedding_dim": 128,
        "text_embedding_dim": 256,
        "hidden_dim": 512,
        "n_mels": 80,
        "max_seq_len": 1000
    },
    "training": {
        "batch_size": 16,
        "learning_rate": 1e-4,
        "num_epochs": 100,
        "gradient_clip": 1.0,
        "save_interval": 10,
        "validation_interval": 5
    },
    "data": {
        "training_dir": "data/training",
        "sample_rate": 22050,
        "max_text_len": 100
    },
    "optimization": {
        "optimizer": "AdamW",
        "weight_decay": 1e-6,
        "scheduler": "ReduceLROnPlateau",
        "patience": 5,
        "factor": 0.5
    }
}
```

## Training the Voice Cloning Model

### 1. Start Training

```bash
cd voice_cloning_project

# Activate environment
conda activate voice_cloning

# Start training with TensorBoard monitoring
python scripts/train_voice_cloning.py \
    --speaker_encoder models/best_speaker_encoder.pth \
    --data_dir data/training \
    --output_dir output/models \
    --batch_size 16 \
    --epochs 100 \
    --device cuda

# In another terminal, start TensorBoard
tensorboard --logdir output/logs --port 6006
```

### 2. Monitor Training

Open your browser and navigate to `http://localhost:6006` to monitor:
- Training loss curves
- Learning rate schedule
- Generated spectrograms
- Model weights histograms

### 3. Training Tips for V100

```bash
# Optimize for V100 (32GB VRAM)
python scripts/train_voice_cloning.py \
    --speaker_encoder models/best_speaker_encoder.pth \
    --data_dir data/training \
    --output_dir output/models \
    --batch_size 32 \
    --epochs 100 \
    --device cuda

# Monitor GPU usage
watch -n 1 nvidia-smi
```

## Voice Cloning Inference

### 1. Basic Usage

```bash
# Generate speech with cloned voice
python scripts/voice_cloning_inference.py \
    --speaker_encoder models/best_speaker_encoder.pth \
    --voice_model output/models/best_voice_cloning_model.pth \
    --reference_audio data/reference/target_voice.wav \
    --text "Hello, this is a test of voice cloning technology." \
    --output output/generated/test_speech.wav \
    --clean_audio \
    --device cuda
```

### 2. Advanced Usage Examples

```bash
# Batch generation with same voice
python -c "
from scripts.voice_cloning_inference import VoiceCloner

cloner = VoiceCloner(
    'models/best_speaker_encoder.pth',
    'output/models/best_voice_cloning_model.pth'
)

texts = [
    'Welcome to our voice cloning demonstration.',
    'This technology can reproduce speech patterns.',
    'The quality depends on the reference audio provided.'
]

generated_files = cloner.clone_voice_batch(
    texts, 
    'data/reference/target_voice.wav',
    'output/generated/batch_test'
)

print(f'Generated {len(generated_files)} audio files')
"
```

### 3. Quality Assessment

Create `scripts/evaluate_quality.py`:

```python
import librosa
import numpy as np
from pathlib import Path

def assess_audio_quality(audio_path):
    """Basic audio quality assessment"""
    audio, sr = librosa.load(audio_path, sr=22050)
    
    # Signal-to-noise ratio estimation
    sorted_audio = np.sort(np.abs(audio))
    noise_floor = np.mean(sorted_audio[:len(sorted_audio)//10])
    signal_level = np.mean(sorted_audio[-len(sorted_audio)//10:])
    snr = 20 * np.log10(signal_level / (noise_floor + 1e-8))
    
    # Dynamic range
    dynamic_range = 20 * np.log10(np.max(np.abs(audio)) / (np.mean(np.abs(audio)) + 1e-8))
    
    # Spectral characteristics
    spec = np.abs(librosa.stft(audio))
    spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=sr)[0]
    
    return {
        'duration': len(audio) / sr,
        'snr_estimate': snr,
        'dynamic_range': dynamic_range,
        'mean_spectral_centroid': np.mean(spectral_centroid),
        'peak_amplitude': np.max(np.abs(audio))
    }

# Usage
quality = assess_audio_quality('output/generated/test_speech.wav')
print("Audio Quality Assessment:")
for metric, value in quality.items():
    print(f"  {metric}: {value:.2f}")
```

## Troubleshooting

### Common Issues and Solutions

#### 1. CUDA Out of Memory

```bash
# Reduce batch size
python scripts/train_voice_cloning.py \
    --batch_size 8 \
    --epochs 100 \
    # ... other parameters

# Or enable gradient accumulation (modify training script)
```

#### 2. Audio Quality Issues

**Problem**: Generated audio sounds "muddy" or "robotic"
**Solutions**:
- Ensure reference audio is high quality (22050 Hz, clean)
- Use longer reference audio (30-60 seconds)
- Clean reference audio with noise reduction
- Verify mel spectrogram normalization matches training

**Problem**: Generated audio is too slow/fast
**Solutions**:
- Check hop_length and win_length in mel extraction
- Verify Griffin-Lim parameters match training
- Consider using vocoder instead of Griffin-Lim

#### 3. Model Loading Errors

```python
# Debug model loading
import torch

checkpoint = torch.load('models/best_speaker_encoder.pth', map_location='cpu')
print("Checkpoint keys:", list(checkpoint.keys()))
print("Number of speakers:", checkpoint.get('num_speakers', 'NOT FOUND'))

# Check model architecture compatibility
speaker_encoder = SpeakerEncoder(num_speakers=checkpoint['num_speakers'])
try:
    speaker_encoder.load_state_dict(checkpoint['model_state_dict'])
    print("Model loaded successfully")
except Exception as e:
    print(f"Model loading failed: {e}")
```

#### 4. Poor Voice Similarity

**Causes and Solutions**:

1. **Insufficient training data**: Need 50+ diverse samples per target voice
2. **Poor reference audio**: Use clean, expressive speech samples
3. **Domain mismatch**: Training data should match target speaker characteristics
4. **Model capacity**: May need larger model or longer training

### Performance Optimization for V100

#### 1. Memory Optimization

```python
# In training script, add these optimizations:

# Enable mixed precision training
from torch.cuda.amp import GradScaler, autocast
scaler = GradScaler()

# Training loop modification
for batch in dataloader:
    optimizer.zero_grad()
    
    with autocast():
        outputs = model(inputs)
        loss = criterion(outputs, targets)
    
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()

# Memory cleanup
if batch_idx % 100 == 0:
    torch.cuda.empty_cache()
```

#### 2. Data Loading Optimization

```python
# Optimize DataLoader for V100
train_loader = DataLoader(
    dataset,
    batch_size=32,  # Optimal for V100
    num_workers=8,  # Match CPU cores
    pin_memory=True,
    persistent_workers=True,
    prefetch_factor=2
)
```

## Production Deployment

### 1. Model Optimization

```python
# Convert to TorchScript for production
import torch

# Load trained model
model = VoiceCloner('speaker_encoder.pth', 'voice_cloning.pth')

# Convert to TorchScript
scripted_model = torch.jit.script(model.generator)
scripted_model.save('voice_cloning_optimized.pt')

# For inference
optimized_model = torch.jit.load('voice_cloning_optimized.pt')
```

### 2. API Server Example

Create `server/app.py`:

```python
from flask import Flask, request, jsonify, send_file
import tempfile
import os
from scripts.voice_cloning_inference import VoiceCloner

app = Flask(__name__)

# Initialize voice cloner
cloner = VoiceCloner(
    'models/best_speaker_encoder.pth',
    'models/best_voice_cloning_model.pth'
)

@app.route('/clone_voice', methods=['POST'])
def clone_voice():
    try:
        # Get parameters
        text = request.form['text']
        reference_audio = request.files['reference_audio']
        
        # Save reference audio temporarily
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as ref_file:
            reference_audio.save(ref_file.name)
            ref_path = ref_file.name
        
        # Generate speech
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as out_file:
            output_path = cloner.generate_speech(
                text=text,
                reference_audio_path=ref_path,
                output_path=out_file.name
            )
        
        # Clean up reference file
        os.unlink(ref_path)
        
        # Return generated audio
        return send_file(output_path, as_attachment=True)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
```

## Alternative Approach: Direct TTS Integration

You raised an excellent question about using existing TTS systems. Here's why the custom approach is better, but also how to implement the simpler method:

### Why Custom Training is Superior

1. **Quality**: Custom models learn voice-specific characteristics better
2. **Prosody**: Captures speaking style, not just voice timbre  
3. **Consistency**: More reliable across different texts
4. **Control**: Can fine-tune for specific use cases

### Simple TTS Integration Approach

If you want to try the simpler approach with existing TTS:

```python
# Install Coqui TTS
pip install TTS

# Simple voice cloning with existing TTS
from TTS.api import TTS
import torch

# Initialize TTS model capable of voice cloning
tts = TTS("tts_models/multilingual/multi-dataset/your_tts").to("cuda")

# Clone voice (this is what you originally wanted to try)
text = "Hello, this is a test of voice cloning."
reference_audio = "data/reference/target_voice.wav"

# Generate with voice cloning
wav = tts.tts_with_vc(
    text=text,
    speaker_wav=reference_audio,
    language="en"
)

# Save audio
import soundfile as sf
sf.write("simple_cloned_voice.wav", wav, 22050)
```

### Hybrid Approach

Combine your speaker encoder with existing TTS:

```python
# Use your speaker encoder for embedding extraction
# Then use TTS for synthesis with that embedding
class HybridVoiceCloner:
    def __init__(self, speaker_encoder_path):
        self.speaker_encoder = self.load_speaker_encoder(speaker_encoder_path)
        self.tts = TTS("tts_models/en/vctk/vits").to("cuda")
    
    def clone_voice_hybrid(self, text, reference_audio):
        # Extract embedding with your trained encoder
        embedding = self.extract_speaker_embedding(reference_audio)
        
        # Use TTS with speaker embedding
        # (This requires modifying TTS internals or using compatible models)
        wav = self.tts.tts(text=text, speaker_embedding=embedding)
        
        return wav
```

## Monitoring and Maintenance

### 1. Training Monitoring Script

```python
# scripts/monitor_training.py
import json
import matplotlib.pyplot as plt
from pathlib import Path

def plot_training_progress(log_dir):
    """Plot training metrics from TensorBoard logs"""
    # This would parse TensorBoard event files
    # For now, assume we save metrics to JSON
    
    metrics_file = Path(log_dir) / "training_metrics.json"
    if not metrics_file.exists():
        print("No metrics file found")
        return
    
    with open(metrics_file) as f:
        metrics = json.load(f)
    
    epochs = metrics['epochs']
    train_loss = metrics['train_loss']
    val_loss = metrics['val_loss']
    
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_loss, label='Training Loss')
    plt.plot(epochs, val_loss, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training Progress')
    
    plt.subplot(1, 2, 2)
    plt.plot(epochs[1:], np.diff(train_loss), label='Train Loss Change')
    plt.xlabel('Epoch')
    plt.ylabel('Loss Change')
    plt.legend()
    plt.title('Loss Convergence')
    
    plt.tight_layout()
    plt.savefig(f'{log_dir}/training_progress.png')
    plt.show()

if __name__ == "__main__":
    plot_training_progress("output/logs")
```

### 2. Model Validation Script

```python
# scripts/validate_model.py
import torch
import librosa
from scripts.voice_cloning_inference import VoiceCloner

def validate_voice_cloning_model(model_path, test_cases):
    """Validate trained voice cloning model"""
    
    cloner = VoiceCloner(
        "models/best_speaker_encoder.pth",
        model_path
    )
    
    results = []
    
    for i, (text, reference_audio, expected_duration) in enumerate(test_cases):
        try:
            output_path = f"validation_output_{i}.wav"
            
            # Generate speech
            result = cloner.generate_speech(
                text=text,
                reference_audio_path=reference_audio,
                output_path=output_path
            )
            
            if result:
                # Load and analyze generated audio
                audio, sr = librosa.load(result, sr=22050)
                actual_duration = len(audio) / sr
                
                results.append({
                    'test_case': i,
                    'success': True,
                    'expected_duration': expected_duration,
                    'actual_duration': actual_duration,
                    'duration_error': abs(actual_duration - expected_duration)
                })
            else:
                results.append({
                    'test_case': i,
                    'success': False,
                    'error': 'Generation failed'
                })
                
        except Exception as e:
            results.append({
                'test_case': i,
                'success': False,
                'error': str(e)
            })
    
    # Print validation results
    success_rate = sum(1 for r in results if r['success']) / len(results)
    print(f"Validation Results:")
    print(f"Success Rate: {success_rate:.2%}")
    
    for result in results:
        if result['success']:
            print(f"Test {result['test_case']}: ✓ Duration error: {result['duration_error']:.2f}s")
        else:
            print(f"Test {result['test_case']}: ✗ Error: {result['error']}")
    
    return results

# Example validation
test_cases = [
    ("Hello world", "data/reference/voice1.wav", 1.5),
    ("This is a longer sentence for testing.", "data/reference/voice1.wav", 3.0),
    ("Voice cloning technology is amazing!", "data/reference/voice2.wav", 2.5)
]

validate_voice_cloning_model("output/models/best_voice_cloning_model.pth", test_cases)
```

## Conclusion

This setup provides a complete voice cloning system that:

1. **Uses your trained speaker encoder** as a feature extractor
2. **Trains a dedicated voice cloning model** for high-quality synthesis
3. **Supports TensorBoard monitoring** for training visualization
4. **Handles 22kHz WAV files** with proper preprocessing
5. **Optimized for NVIDIA V100** with 32GB RAM

The key insight is that your original speaker encoder, while excellent for speaker identification, needs an additional synthesis component to produce natural-sounding speech. The custom voice cloning model bridges this gap by learning to generate mel-spectrograms conditioned on both text and speaker embeddings.

Your question about using existing TTS systems directly is valid - it's simpler but produces lower quality results. The hybrid approach I outlined gives you the best of both worlds: leveraging your trained speaker encoder while using proven TTS architectures for synthesis.




Deactivate and Remove the Old Environment: It's crucial to get rid of the broken environment completely.

bash
conda deactivate
conda env remove -n voice_fewshot -y
Create a New, Clean Environment: We'll create a new environment with Python 3.10, which is stable and compatible with all these packages.

bash
conda create -n voice_cloning_env python=3.10 -y
conda activate voice_cloning_env
Install PyTorch with CUDA: This must be done first via Conda to ensure proper GPU support.

bash
conda install pytorch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 pytorch-cuda=11.8 -c pytorch -c nvidia
Install All Other Dependencies with pip: Now, navigate to your scripts directory and use the new requirements.txt file. This will install everything else in one go.

bash
cd "C:\Users\VM02\Desktop\THESIS (SALAS)\SECOND PHASE ENV\voice cloning (FINAL)\scripts"
pip install -r requirements.txt
Install ffmpeg (Required for MP3/MP4): The script warned that ffmpeg was not found. You can install it easily with Conda.

bash
conda install ffmpeg -c conda-forge
After these steps, your environment will be clean, stable, and have all the correct packages to run the script with both XTTS and RVC capabilities.

Step 3: Run the Script
You should now be able to run the script without any import errors.

bash
python fewshot_voice.py
