# Alternative: Use Fish Speech 1.5 (No Authentication)

If you don't want to deal with Hugging Face authentication, you can use **Fish Speech 1.5** which is publicly available without restrictions.

## Option 1: Download Fish Speech 1.5

```bash
# Download Fish Speech 1.5 (no authentication needed)
hf download fishaudio/fish-speech-1.5 --local-dir checkpoints/fish-speech-1.5
```

## Option 2: Clone from GitHub and Use Pretrained Models

```bash
# Clone the official repository
git clone https://github.com/fishaudio/fish-speech.git
cd fish-speech

# Download pretrained models (included in repo)
# The repository includes model download scripts
```

## Option 3: Use Fish Audio API (Easiest)

Instead of running locally, use their hosted API:

```python
import requests

def fish_audio_api_tts(text, reference_audio_path):
    """Use Fish Audio API instead of local model"""
    
    # Note: You'll need an API key from fish.audio
    api_key = "your_api_key_here"
    
    url = "https://api.fish.audio/v1/tts"
    
    with open(reference_audio_path, 'rb') as f:
        files = {'reference': f}
        data = {
            'text': text,
            'language': 'en'
        }
        headers = {'Authorization': f'Bearer {api_key}'}
        
        response = requests.post(url, files=files, data=data, headers=headers)
        
        if response.status_code == 200:
            with open('output.wav', 'wb') as out:
                out.write(response.content)
            return 'output.wav'
        else:
            print(f"API Error: {response.text}")
            return None

# Usage
output = fish_audio_api_tts(
    text="Hello, this is a test!",
    reference_audio_path="reference.wav"
)
```

## Comparison

| Method | Pros | Cons |
|--------|------|------|
| **OpenAudio S1-mini** | Best quality, latest model | Requires HF authentication |
| **Fish Speech 1.5** | No authentication, good quality | Slightly older model |
| **Fish Audio API** | No local setup, easy to use | Requires API key, costs money |
| **GitHub Clone** | Full control, all features | More complex setup |

## Recommended: Fish Speech 1.5

For your thesis project, I recommend **Fish Speech 1.5** as it provides:
- ✅ Good quality (close to S1-mini)
- ✅ No authentication hassles
- ✅ Publicly available
- ✅ Well documented
- ✅ Works with the same wrapper code

### Setup Fish Speech 1.5

```bash
# Download model
hf download fishaudio/fish-speech-1.5 --local-dir checkpoints/fish-speech-1.5

# Update wrapper to use 1.5
# In fish_speech_wrapper.py, change default path:
# model_path="checkpoints/fish-speech-1.5"
```

### Update Your Wrapper

```python
# In fish_speech_wrapper.py line 30:
def __init__(self, model_path="checkpoints/fish-speech-1.5", device="cuda"):
```

Then download:

```bash
hf download fishaudio/fish-speech-1.5 --local-dir checkpoints/fish-speech-1.5
```

This should work without any authentication!
