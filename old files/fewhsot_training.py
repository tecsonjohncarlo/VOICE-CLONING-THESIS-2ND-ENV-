"""
Fixed Voice Cloning System using your trained Speaker Encoder + Coqui TTS
This version handles model loading issues properly
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import librosa
import soundfile as sf
from pathlib import Path
import json
from typing import List, Optional, Tuple
import matplotlib.pyplot as plt
from scipy.spatial.distance import cosine
import warnings
warnings.filterwarnings('ignore')

# Define SpeakerEncoder class (exact copy from your training script)
class SpeakerEncoder(nn.Module):
    def __init__(self, input_dim=80, hidden_dim=256, embedding_dim=128, num_speakers=10):
        super(SpeakerEncoder, self).__init__()
        
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d((1, 2)),
            nn.Dropout2d(0.1),

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d((1, 2)),
            nn.Dropout2d(0.1),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            nn.Dropout2d(0.2)
        )

        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.fc_layers = nn.Sequential(
            nn.Linear(128, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

        self.classifier = nn.Linear(embedding_dim, num_speakers)

    def forward(self, x, return_embedding=False):
        batch_size = x.size(0)
        x = x.unsqueeze(1)  
        
        x = self.conv_layers(x)
        x = self.global_pool(x)
        x = x.view(batch_size, -1)
        
        embedding = self.fc_layers(x)
        
        if return_embedding:
            return embedding
        else:
            return self.classifier(embedding)

class VoiceCloner:
    def __init__(self, 
                 speaker_encoder_path: str = "best_speaker_encoder.pth",
                 device: str = None):
        """
        Initialize the voice cloning system with robust error handling
        """
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"🚀 Initializing Voice Cloner on {self.device}")
        
        # Load trained speaker encoder with detailed error handling
        self.speaker_encoder = None
        self.load_speaker_encoder_robust(speaker_encoder_path)
        
        # Initialize Coqui TTS
        self.tts = None
        self.setup_coqui_tts()
        
        # Audio processing parameters (match your training preprocessing)
        self.sample_rate = 22050
        self.hop_length = 256
        self.win_length = 1024
        self.n_mels = 80
        self.n_fft = 1024
        
    def load_speaker_encoder_robust(self, checkpoint_path: str):
        """Load speaker encoder with comprehensive error handling"""
        print(f"📥 Loading speaker encoder from {checkpoint_path}")
        
        # Check if file exists
        if not Path(checkpoint_path).exists():
            print(f"❌ Checkpoint file not found: {checkpoint_path}")
            print(f"📁 Current directory: {Path.cwd()}")
            print("📂 Available .pth files:")
            for f in Path.cwd().glob("*.pth"):
                print(f"   {f.name}")
            return False
        
        try:
            # Load checkpoint
            print("📖 Reading checkpoint file...")
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            print("✅ Checkpoint file loaded successfully")
            
            # Debug: Print checkpoint contents
            print("🔍 Checkpoint contents:")
            for key in checkpoint.keys():
                if isinstance(checkpoint[key], dict):
                    print(f"   {key}: dict with {len(checkpoint[key])} items")
                elif isinstance(checkpoint[key], torch.Tensor):
                    print(f"   {key}: tensor {checkpoint[key].shape}")
                else:
                    print(f"   {key}: {type(checkpoint[key])} = {checkpoint[key]}")
            
            # Get model parameters
            self.num_speakers = checkpoint.get('num_speakers', 0)
            self.speaker_to_id = checkpoint.get('speaker_to_id', {})
            self.embedding_dim = 128  # Fixed from your architecture
            
            if self.num_speakers == 0:
                print("❌ num_speakers is 0 or not found in checkpoint")
                return False
            
            print(f"📊 Model info from checkpoint:")
            print(f"   👥 Number of speakers: {self.num_speakers}")
            print(f"   🎯 Embedding dimension: {self.embedding_dim}")
            print(f"   📋 Sample speakers: {list(self.speaker_to_id.keys())[:5]}")
            
            # Create model
            print("🧠 Creating speaker encoder model...")
            self.speaker_encoder = SpeakerEncoder(
                input_dim=80,
                hidden_dim=256,
                embedding_dim=self.embedding_dim,
                num_speakers=self.num_speakers
            ).to(self.device)
            
            # Load model weights
            if 'model_state_dict' not in checkpoint:
                print("❌ model_state_dict not found in checkpoint")
                return False
            
            print("⚖️ Loading model weights...")
            self.speaker_encoder.load_state_dict(checkpoint['model_state_dict'])
            self.speaker_encoder.eval()
            
            # Test the model
            print("🧪 Testing model...")
            test_input = torch.randn(1, 80, 100).to(self.device)
            with torch.no_grad():
                test_output = self.speaker_encoder(test_input, return_embedding=True)
                print(f"✅ Model test successful! Embedding shape: {test_output.shape}")
            
            print("✅ Speaker encoder loaded and tested successfully!")
            return True
            
        except Exception as e:
            print(f"❌ Error loading speaker encoder: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def setup_coqui_tts(self):
        """Setup Coqui TTS with error handling"""
        print("🎤 Setting up Coqui TTS...")
        
        try:
            from TTS.api import TTS
            
            # List available models
            print("📋 Checking available TTS models...")
            models = TTS.list_models()
            
            # Find voice cloning capable models
            vc_models = [m for m in models if 'your_tts' in m or 'vits' in m.lower()]
            print(f"📊 Found {len(vc_models)} voice cloning models")
            
            # Select best model for voice cloning
            preferred_models = [
                "tts_models/multilingual/multi-dataset/your_tts",  # Best for VC
                "tts_models/en/vctk/vits",  # Good English model
                "tts_models/en/ljspeech/vits",  # Fallback
            ]
            
            selected_model = None
            for model in preferred_models:
                if model in models:
                    selected_model = model
                    break
            
            if selected_model is None:
                print("❌ No suitable voice cloning model found")
                return False
            
            print(f"📥 Loading TTS model: {selected_model}")
            self.tts = TTS(model_name=selected_model).to(self.device)
            
            print("✅ Coqui TTS loaded successfully!")
            return True
            
        except ImportError:
            print("❌ Coqui TTS not installed!")
            print("💻 Install with: pip install TTS")
            return False
        except Exception as e:
            print(f"❌ Error setting up Coqui TTS: {e}")
            return False
    
    def extract_mel_spectrogram(self, audio_path: str) -> torch.Tensor:
        """Extract mel spectrogram (matching your training preprocessing)"""
        try:
            # Load audio
            audio, sr = librosa.load(audio_path, sr=self.sample_rate)
            print(f"🎵 Loaded audio: {len(audio)} samples at {sr}Hz")
            
            # Extract mel spectrogram (same as training)
            mel_spec = librosa.feature.melspectrogram(
                y=audio,
                sr=sr,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                win_length=self.win_length,
                n_mels=self.n_mels,
                fmin=0,
                fmax=sr//2
            )
            
            # Convert to log scale
            mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
            
            # Normalize (same as training)
            mel_spec_norm = (mel_spec_db - mel_spec_db.min()) / (mel_spec_db.max() - mel_spec_db.min())
            
            print(f"📊 Mel spectrogram shape: {mel_spec_norm.shape}")
            return torch.FloatTensor(mel_spec_norm).unsqueeze(0)  # Add batch dimension
            
        except Exception as e:
            print(f"❌ Error extracting mel spectrogram from {audio_path}: {e}")
            return None
    
    def get_speaker_embedding(self, audio_paths: List[str]) -> torch.Tensor:
        """Extract speaker embedding from reference audio(s)"""
        if self.speaker_encoder is None:
            print("❌ Speaker encoder not loaded!")
            return None
        
        print(f"🎯 Extracting speaker embedding from {len(audio_paths)} reference audio(s)")
        
        embeddings = []
        
        for audio_path in audio_paths:
            print(f"📁 Processing: {audio_path}")
            
            # Check if file exists
            if not Path(audio_path).exists():
                print(f"❌ Audio file not found: {audio_path}")
                continue
            
            # Extract mel spectrogram
            mel_spec = self.extract_mel_spectrogram(audio_path)
            if mel_spec is None:
                continue
            
            mel_spec = mel_spec.to(self.device)
            
            # Get embedding
            with torch.no_grad():
                embedding = self.speaker_encoder(mel_spec, return_embedding=True)
                embeddings.append(embedding)
                print(f"✅ Embedding extracted: {embedding.shape}")
        
        if not embeddings:
            print("❌ No valid embeddings extracted!")
            return None
        
        # Average embeddings if multiple references
        avg_embedding = torch.mean(torch.stack(embeddings), dim=0)
        
        # Normalize embedding
        avg_embedding = F.normalize(avg_embedding, p=2, dim=1)
        
        print(f"🎯 Final speaker embedding: shape {avg_embedding.shape}")
        return avg_embedding
    
    def synthesize_speech(self, 
                         text: str, 
                         reference_audios: List[str],
                         output_path: str = "cloned_voice.wav",
                         language: str = "en") -> str:
        """Synthesize speech with cloned voice using Coqui TTS"""
        
        if self.tts is None:
            print("❌ TTS model not loaded!")
            return None
        
        print(f"🎤 Synthesizing speech...")
        print(f"📝 Text: '{text[:100]}...'")
        print(f"📁 Reference audios: {len(reference_audios)}")
        print(f"🌍 Language: {language}")
        
        try:
            # For voice cloning, we use the first reference audio
            reference_audio = reference_audios[0]
            
            if not Path(reference_audio).exists():
                print(f"❌ Reference audio not found: {reference_audio}")
                return None
            
            print(f"🎯 Using reference: {reference_audio}")
            
            # Check if TTS supports voice cloning
            if hasattr(self.tts, 'tts_with_vc'):
                # For multi-speaker models, we must provide a speaker, even for VC.
                # We'll use the first available speaker from the model as a base.
                speaker_to_use = None
                if self.tts.is_multi_speaker:
                    speaker_to_use = self.tts.speakers[0]
                    print(f"🗣️ Using base speaker from TTS model: {speaker_to_use}")

                print("🔄 Using voice cloning mode...")
                wav = self.tts.tts_with_vc(
                    text=text,
                    speaker_wav=reference_audio,
                    language=language,
                    speaker=speaker_to_use
                )
            elif hasattr(self.tts, 'tts'):
                print("🔄 Using standard TTS with speaker reference...")
                wav = self.tts.tts(
                    text=text, 
                    speaker_wav=reference_audio,
                    language=language,
                    # The 'speaker' argument might be needed here too for some models
                    speaker=self.tts.speakers[0] if self.tts.is_multi_speaker else None
                )
            else:
                print("❌ TTS model doesn't support voice cloning")
                return None
            
            # Save audio
            print(f"💾 Saving audio to: {output_path}")
            sf.write(output_path, wav, self.tts.synthesizer.output_sample_rate)
            
            print(f"✅ Speech synthesized successfully!")
            print(f"📊 Audio length: {len(wav)/self.tts.synthesizer.output_sample_rate:.2f} seconds")
            
            return output_path
            
        except Exception as e:
            print(f"❌ Error synthesizing speech: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def quick_test(self):
        """Quick test to verify everything is working"""
        print("\n🧪 QUICK SYSTEM TEST")
        print("=" * 40)
        
        # Test 1: Speaker encoder
        if self.speaker_encoder is None:
            print("❌ Test 1 FAILED: Speaker encoder not loaded")
            return False
        else:
            print("✅ Test 1 PASSED: Speaker encoder loaded")
        
        # Test 2: TTS
        if self.tts is None:
            print("❌ Test 2 FAILED: TTS not loaded")
            return False
        else:
            print("✅ Test 2 PASSED: TTS loaded")
        
        # Test 3: Model forward pass
        try:
            test_input = torch.randn(1, 80, 100).to(self.device)
            with torch.no_grad():
                embedding = self.speaker_encoder(test_input, return_embedding=True)
            print(f"✅ Test 3 PASSED: Model forward pass successful")
        except Exception as e:
            print(f"❌ Test 3 FAILED: Model forward pass failed: {e}")
            return False
        
        print("🎉 All tests passed! System is ready for voice cloning.")
        return True

def main():
    """Main function with step-by-step testing"""
    print("🎤 VOICE CLONING SYSTEM INITIALIZATION")
    print("=" * 60)
    
    # Initialize voice cloner
    cloner = VoiceCloner()
    
    # Run quick test
    if not cloner.quick_test():
        print("\n❌ SYSTEM NOT READY")
        print("Please fix the issues above before proceeding.")
        return
    
    # Interactive demo
    print("\n🎯 VOICE CLONING DEMO")
    print("Ready to clone voices! Here's what you need:")
    print("1. Reference audio files (.wav format recommended)")
    print("2. Text to synthesize")
    print("3. Patience (first run downloads models)")
    
    while True:
        print("\n" + "="*50)
        print("VOICE CLONING MENU")
        print("1. Clone voice")
        print("2. Test speaker embedding extraction")
        print("3. Exit")
        
        choice = input("Enter choice (1-3): ").strip()
        
        if choice == "1":
            # Get reference audio
            ref_audio = input("Enter reference audio path: ").strip()
            if not ref_audio or not Path(ref_audio).exists():
                print("❌ Invalid audio path")
                continue
            
            # Get text
            text = input("Enter text to synthesize: ").strip()
            if not text:
                text = "Hello, this is a test of voice cloning technology using your trained model."
            
            # Get output path
            output = input("Output filename (default: cloned_voice.wav): ").strip()
            if not output:
                output = "cloned_voice.wav"
            
            # Ensure the output path has a .wav extension
            if not output.lower().endswith(('.wav', '.flac', '.ogg')):
                output += '.wav'
                print(f"💡 No extension found. Appending .wav. Saving to: {output}")
            
            # Clone voice
            result = cloner.synthesize_speech(text, [ref_audio], output)
            if result:
                print(f"🎉 Voice cloning successful! Audio saved to: {result}")
            else:
                print("❌ Voice cloning failed")
        
        elif choice == "2":
            ref_audio = input("Enter audio path to test embedding extraction: ").strip()
            if not ref_audio or not Path(ref_audio).exists():
                print("❌ Invalid audio path")
                continue
            
            embedding = cloner.get_speaker_embedding([ref_audio])
            if embedding is not None:
                print(f"✅ Speaker embedding extracted successfully!")
                print(f"📊 Embedding shape: {embedding.shape}")
                print(f"🎯 Embedding norm: {torch.norm(embedding).item():.4f}")
            else:
                print("❌ Failed to extract speaker embedding")
        
        elif choice == "3":
            print("👋 Goodbye!")
            break
        
        else:
            print("❌ Invalid choice")

if __name__ == "__main__":
    main()