# 🚀 Quick Start Guide

Get up and running with Optimized Fish Speech TTS in 5 minutes!

## ⚡ Fast Track

### 1. Install Dependencies (2 min)

```bash
# Create and activate virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

# Install packages
pip install -r requirements.txt
```

### 2. Download Model (3 min)

```bash
# Install Hugging Face CLI
pip install huggingface-hub

# Download model (~2GB)
huggingface-cli download fishaudio/openaudio-s1-mini --local-dir checkpoints/openaudio-s1-mini
```

### 3. Configure (30 sec)

```bash
# Copy example config
copy .env.example .env
# Default settings work great - no changes needed!
```

### 4. Launch (30 sec)

**Windows:**
```bash
# Terminal 1
start_backend.bat

# Terminal 2
start_gradio.bat
```

**Linux/Mac:**
```bash
# Terminal 1
python backend/app.py

# Terminal 2
python ui/gradio_app.py
```

### 5. Use! 🎉

Open browser: **http://localhost:7860**

1. Type text: `Hello! This is Fish Speech.`
2. Click "Generate Speech"
3. Listen to result!

## 🎤 Try Voice Cloning

1. Record 10-30 seconds of clear speech
2. Upload in "Voice Cloning" section
3. Enter text to synthesize
4. Click "Generate Speech"
5. Hear your cloned voice!

## 🎭 Try Emotions

Type with emotion markers:
```
(excited) Hello! This is amazing!
(whispering) Can you hear me?
(laughing) That's so funny!
```

## 📊 Check Performance

Look at the metrics after generation:
- **Latency**: How long it took
- **VRAM**: GPU memory used
- **GPU Util**: GPU usage percentage
- **RTF**: Real-time factor (lower is faster)

## ⚙️ Optimize for Your GPU

### For RTX 3060/4060 (6-8GB VRAM)
Default settings are perfect! ✅

### For RTX 3050 (4GB VRAM)
Edit `.env`:
```bash
QUANTIZATION=int8
```

### For GTX 1660 (6GB VRAM, no tensor cores)
Edit `.env`:
```bash
MIXED_PRECISION=fp16
QUANTIZATION=int8
```

### For CPU Only
Edit `.env`:
```bash
DEVICE=cpu
```
⚠️ Will be 3x slower but works!

## 🐛 Common Issues

### "Model directory not found"
```bash
# Download model first
huggingface-cli download fishaudio/openaudio-s1-mini --local-dir checkpoints/openaudio-s1-mini
```

### "Cannot connect to API"
```bash
# Make sure backend is running
python backend/app.py
```

### "CUDA out of memory"
```bash
# Enable memory optimization
# Edit .env:
QUANTIZATION=int8
```

### "Generation is slow"
First run is always slow (model loading). Subsequent runs are faster!

## 📚 Next Steps

- Read [README.md](README.md) for detailed documentation
- Check [FISH_SPEECH_ANALYSIS.md](FISH_SPEECH_ANALYSIS.md) for architecture details
- See [PROJECT_SUMMARY.md](PROJECT_SUMMARY.md) for complete feature list
- Explore API at http://localhost:8000/docs

## 🎯 Pro Tips

1. **Best reference audio**: 10-30 seconds, clear speech, no background noise
2. **Provide transcript**: Improves voice cloning quality significantly
3. **Use emotions**: Makes speech more expressive and natural
4. **Temperature 0.7**: Good default, lower for consistency, higher for variety
5. **Cache works**: Same reference audio = instant VQ extraction!

## 🔥 Advanced Usage

### API Call (Python)
```python
import requests

response = requests.post(
    "http://localhost:8000/tts",
    data={"text": "Hello world"},
    files={"speaker_file": open("reference.wav", "rb")}
)

with open("output.wav", "wb") as f:
    f.write(response.content)

print(f"Latency: {response.headers['X-Latency-Ms']}ms")
```

### API Call (curl)
```bash
curl -X POST "http://localhost:8000/tts" \
  -F "text=Hello world" \
  -F "speaker_file=@reference.wav" \
  --output output.wav
```

### Streamlit UI (Alternative)
```bash
streamlit run ui/streamlit_app.py
# Open http://localhost:8501
```

## 💡 Did You Know?

- Fish Speech ranked **#1 on TTS-Arena2** 🏆
- Supports **8+ languages** (EN, ZH, JA, KO, FR, DE, ES, AR)
- Has **60+ emotion markers** for expressive speech
- Achieves **0.011 WER** (Word Error Rate)
- Can clone voices with just **10 seconds** of audio

## 🎊 You're Ready!

Start creating amazing voice clones! 🎤✨

Need help? Check the full [README.md](README.md) or open an issue.

---

**Happy Synthesizing! 🐟**
