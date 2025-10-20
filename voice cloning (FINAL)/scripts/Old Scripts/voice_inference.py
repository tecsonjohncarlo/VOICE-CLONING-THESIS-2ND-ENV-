"""
Voice Cloning Inference Script
Uses trained voice cloning model to generate speech with custom voice
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import librosa
import soundfile as sf
from pathlib import Path
import json
import argparse
from typing import List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

# Try to import audio cleaning libraries
try:
    import noisereduce as nr
    NOISEREDUCE_AVAILABLE = True
except ImportError:
    NOISEREDUCE_AVAILABLE = False

try:
    from scipy.signal import butter, filtfilt
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

# Original Speaker Encoder (must match training)
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

# Voice Cloning Generator (must match training)
class VoiceCloningGenerator(nn.Module):
    def __init__(self, 
                 speaker_embedding_dim=128,
                 text_embedding_dim=256,
                 hidden_dim=512,
                 n_mels=80,
                 max_seq_len=1000):
        super(VoiceCloningGenerator, self).__init__()
        
        self.speaker_embedding_dim = speaker_embedding_dim
        self.text_embedding_dim = text_embedding_dim
        self.hidden_dim = hidden_dim
        self.n_mels = n_mels
        self.max_seq_len = max_seq_len
        
        # Text encoder
        self.text_encoder = nn.Sequential(
            nn.Embedding(256, text_embedding_dim),
            nn.LSTM(text_embedding_dim, hidden_dim//2, batch_first=True, bidirectional=True),
        )
        
        # Speaker conditioning
        self.speaker_proj = nn.Sequential(
            nn.Linear(speaker_embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Decoder
        self.decoder_rnn = nn.LSTM(
            input_size=hidden_dim + n_mels,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=0.1
        )
        
        # Attention
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=8,
            dropout=0.1
        )
        
        # Output projection
        self.mel_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, n_mels)
        )
        
        # Stop token prediction
        self.stop_proj = nn.Linear(hidden_dim, 1)
        
    def forward(self, text_input, speaker_embedding, target_mel=None, max_len=None):
        batch_size = text_input.size(0)
        
        # Encode text
        text_embedded = self.text_encoder.forward(text_input)
        if isinstance(text_embedded, tuple):
            text_encoded, _ = text_embedded
        else:
            text_encoded = text_embedded
            
        # Project speaker embedding
        speaker_context = self.speaker_proj(speaker_embedding)
        speaker_context = speaker_context.unsqueeze(1)
        
        # Repeat speaker context to match text length
        text_len = text_encoded.size(1)
        speaker_context = speaker_context.expand(-1, text_len, -1)
        
        # Combine text and speaker information
        encoder_output = text_encoded + speaker_context
        
        # Inference mode
        max_len = max_len or self.max_seq_len
        
        mel_outputs = []
        stop_outputs = []
        
        decoder_input = torch.zeros(batch_size, 1, self.n_mels, device=text_input.device)
        hidden = None
        
        for t in range(max_len):
            # Attention over encoder output
            query = encoder_output[:, [t % text_len]]
            
            # RNN step
            rnn_input = torch.cat([decoder_input, query], dim=-1)
            rnn_output, hidden = self.decoder_rnn(rnn_input, hidden)
            
            # Generate mel and stop token
            mel_output = self.mel_proj(rnn_output)
            stop_output = self.stop_proj(rnn_output)
            
            mel_outputs.append(mel_output)
            stop_outputs.append(stop_output)
            
            # Update decoder input for next step
            decoder_input = mel_output
            
            # Check stop condition
            if torch.sigmoid(stop_output).max() > 0.5:
                break
        
        mel_outputs = torch.cat(mel_outputs, dim=1)
        stop_outputs = torch.cat(stop_outputs, dim=1)
        
        return mel_outputs, stop_outputs

class AudioProcessor:
    """Audio processing utilities for voice cloning"""
    
    def __init__(self, sample_rate=22050):
        self.sample_rate = sample_rate
        
    def load_and_preprocess_audio(self, audio_path, clean_audio=True, target_duration=None):
        """Load and preprocess audio file"""
        print(f"Loading audio: {audio_path}")
        
        try:
            # Load audio
            audio, sr = librosa.load(audio_path, sr=self.sample_rate)
            print(f"Original: {len(audio)} samples, {len(audio)/sr:.2f}s duration")
            
            # Clean audio if requested
            if clean_audio:
                audio = self.clean_audio(audio)
            
            # Trim silence
            audio, _ = librosa.effects.trim(audio, top_db=20)
            print(f"After trim: {len(audio)} samples, {len(audio)/sr:.2f}s duration")
            
            # Adjust duration if specified
            if target_duration is not None:
                target_samples = int(target_duration * sr)
                if len(audio) > target_samples:
                    audio = audio[:target_samples]
                elif len(audio) < target_samples:
                    # Repeat audio to reach target duration
                    repeats = int(np.ceil(target_samples / len(audio)))
                    audio = np.tile(audio, repeats)[:target_samples]
            
            # Normalize
            audio = audio / (np.max(np.abs(audio)) + 1e-8)
            
            return audio
            
        except Exception as e:
            print(f"Error loading audio: {e}")
            return None
    
    def clean_audio(self, audio):
        """Clean audio using available methods"""
        if NOISEREDUCE_AVAILABLE:
            try:
                # Gentle noise reduction
                cleaned = nr.reduce_noise(
                    y=audio,
                    sr=self.sample_rate,
                    stationary=True,
                    prop_decrease=0.7
                )
                print("Applied noise reduction")
                return cleaned
            except:
                print("Noise reduction failed, using original")
        
        # Basic cleaning with scipy if available
        if SCIPY_AVAILABLE:
            try:
                # High-pass filter to remove low-frequency noise
                nyquist = self.sample_rate / 2
                low_cutoff = 80 / nyquist
                b, a = butter(4, low_cutoff, btype='high')
                filtered = filtfilt(b, a, audio)
                print("Applied high-pass filter")
                return filtered
            except:
                print("Filtering failed, using original")
        
        return audio
    
    def extract_mel_spectrogram(self, audio):
        """Extract mel spectrogram matching training preprocessing"""
        try:
            # Extract mel spectrogram (match training exactly)
            mel_spec = librosa.feature.melspectrogram(
                y=audio,
                sr=self.sample_rate,
                n_fft=1024,
                hop_length=256,
                win_length=1024,
                n_mels=80,
                fmin=0,
                fmax=self.sample_rate//2
            )
            
            # Convert to log scale and normalize (match training)
            mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
            mel_spec_norm = (mel_spec_db - mel_spec_db.min()) / (mel_spec_db.max() - mel_spec_db.min())
            
            print(f"Mel spectrogram shape: {mel_spec_norm.shape}")
            return torch.FloatTensor(mel_spec_norm)
            
        except Exception as e:
            print(f"Error extracting mel spectrogram: {e}")
            return None

class VoiceCloner:
    """Main voice cloning inference class"""
    
    def __init__(self, 
                 speaker_encoder_path,
                 voice_cloning_model_path,
                 device=None):
        
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Initializing Voice Cloner on {self.device}")
        
        # Initialize audio processor
        self.audio_processor = AudioProcessor()
        
        # Load models
        self.speaker_encoder = self._load_speaker_encoder(speaker_encoder_path)
        self.generator = self._load_generator(voice_cloning_model_path)
        
        if self.speaker_encoder is None or self.generator is None:
            raise ValueError("Failed to load required models")
        
        print("Voice Cloner initialized successfully!")
    
    def _load_speaker_encoder(self, checkpoint_path):
        """Load pre-trained speaker encoder"""
        try:
            print(f"Loading speaker encoder: {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            
            num_speakers = checkpoint.get('num_speakers', 0)
            if num_speakers == 0:
                print("Error: num_speakers not found in checkpoint")
                return None
            
            # Create and load speaker encoder
            speaker_encoder = SpeakerEncoder(
                input_dim=80,
                hidden_dim=256,
                embedding_dim=128,
                num_speakers=num_speakers
            ).to(self.device)
            
            speaker_encoder.load_state_dict(checkpoint['model_state_dict'])
            speaker_encoder.eval()
            
            print(f"Speaker encoder loaded successfully")
            return speaker_encoder
            
        except Exception as e:
            print(f"Error loading speaker encoder: {e}")
            return None
    
    def _load_generator(self, checkpoint_path):
        """Load trained voice cloning generator"""
        try:
            print(f"Loading voice cloning model: {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            
            speaker_embedding_dim = checkpoint.get('speaker_encoder_dim', 128)
            
            # Create generator
            generator = VoiceCloningGenerator(
                speaker_embedding_dim=speaker_embedding_dim,
                text_embedding_dim=256,
                hidden_dim=512,
                n_mels=80
            ).to(self.device)
            
            generator.load_state_dict(checkpoint['generator_state_dict'])
            generator.eval()
            
            print("Voice cloning model loaded successfully")
            return generator
            
        except Exception as e:
            print(f"Error loading voice cloning model: {e}")
            return None
    
    def extract_speaker_embedding(self, audio_path, clean_audio=True):
        """Extract speaker embedding from reference audio"""
        print(f"Extracting speaker embedding from: {audio_path}")
        
        # Load and preprocess audio
        audio = self.audio_processor.load_and_preprocess_audio(
            audio_path, clean_audio=clean_audio
        )
        
        if audio is None:
            return None
        
        # Extract mel spectrogram
        mel_spec = self.audio_processor.extract_mel_spectrogram(audio)
        if mel_spec is None:
            return None
        
        # Get speaker embedding
        with torch.no_grad():
            mel_spec = mel_spec.unsqueeze(0).to(self.device)  # Add batch dimension
            embedding = self.speaker_encoder(mel_spec, return_embedding=True)
            embedding = F.normalize(embedding, p=2, dim=1)  # Normalize
            
        print(f"Speaker embedding extracted: {embedding.shape}")
        return embedding
    
    def text_to_sequence(self, text, max_len=100):
        """Convert text to character sequence"""
        # Simple character-level encoding
        chars = list(text.lower())
        sequence = [ord(c) for c in chars if ord(c) < 256]
        
        # Pad or truncate
        if len(sequence) > max_len:
            sequence = sequence[:max_len]
        else:
            sequence.extend([0] * (max_len - len(sequence)))
        
        return torch.LongTensor(sequence).unsqueeze(0)  # Add batch dimension
    
    def generate_speech(self, text, reference_audio_path, output_path=None, clean_reference=True):
        """Generate speech with cloned voice"""
        print(f"Generating speech for text: '{text[:50]}...'")
        
        # Extract speaker embedding
        speaker_embedding = self.extract_speaker_embedding(
            reference_audio_path, clean_audio=clean_reference
        )
        
        if speaker_embedding is None:
            print("Failed to extract speaker embedding")
            return None
        
        # Convert text to sequence
        text_seq = self.text_to_sequence(text).to(self.device)
        speaker_embedding = speaker_embedding.to(self.device)
        
        # Generate mel spectrogram
        print("Generating mel spectrogram...")
        with torch.no_grad():
            pred_mel, pred_stop = self.generator(text_seq, speaker_embedding)
        
        # Convert mel spectrogram back to audio using Griffin-Lim
        print("Converting mel spectrogram to audio...")
        mel_spec = pred_mel.squeeze(0).cpu().numpy()  # Remove batch dimension
        
        # Denormalize mel spectrogram (reverse training normalization)
        # Note: This is an approximation - ideally you'd save normalization stats
        mel_spec = mel_spec * 80 - 40  # Approximate denormalization
        
        # Convert from dB back to linear
        mel_spec_linear = librosa.db_to_power(mel_spec)
        
        # Use Griffin-Lim to reconstruct audio
        audio = librosa.feature.inverse.mel_to_audio(
            mel_spec_linear,
            sr=self.audio_processor.sample_rate,
            n_fft=1024,
            hop_length=256,
            win_length=1024,
            fmin=0,
            fmax=self.audio_processor.sample_rate//2
        )
        
        # Normalize audio
        audio = audio / (np.max(np.abs(audio)) + 1e-8) * 0.9
        
        # Save audio
        if output_path is None:
            output_path = "generated_speech.wav"
        
        sf.write(output_path, audio, self.audio_processor.sample_rate)
        print(f"Generated speech saved to: {output_path}")
        print(f"Duration: {len(audio)/self.audio_processor.sample_rate:.2f} seconds")
        
        return output_path
    
    def clone_voice_batch(self, texts, reference_audio_path, output_dir="generated_speech"):
        """Generate multiple speech samples with same voice"""
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        print(f"Batch generating {len(texts)} speech samples...")
        
        # Extract speaker embedding once
        speaker_embedding = self.extract_speaker_embedding(reference_audio_path)
        if speaker_embedding is None:
            return []
        
        generated_files = []
        
        for i, text in enumerate(texts):
            output_path = output_dir / f"generated_{i+1:03d}.wav"
            
            try:
                # Convert text and generate
                text_seq = self.text_to_sequence(text).to(self.device)
                
                with torch.no_grad():
                    pred_mel, pred_stop = self.generator(text_seq, speaker_embedding)
                
                # Convert to audio (same process as single generation)
                mel_spec = pred_mel.squeeze(0).cpu().numpy()
                mel_spec = mel_spec * 80 - 40
                mel_spec_linear = librosa.db_to_power(mel_spec)
                
                audio = librosa.feature.inverse.mel_to_audio(
                    mel_spec_linear,
                    sr=self.audio_processor.sample_rate,
                    n_fft=1024,
                    hop_length=256,
                    win_length=1024,
                    fmin=0,
                    fmax=self.audio_processor.sample_rate//2
                )
                
                audio = audio / (np.max(np.abs(audio)) + 1e-8) * 0.9
                sf.write(output_path, audio, self.audio_processor.sample_rate)
                
                generated_files.append(str(output_path))
                print(f"Generated {i+1}/{len(texts)}: {output_path.name}")
                
            except Exception as e:
                print(f"Error generating sample {i+1}: {e}")
                continue
        
        print(f"Batch generation complete: {len(generated_files)}/{len(texts)} successful")
        return generated_files

def main():
    """Main inference script"""
    parser = argparse.ArgumentParser(description='Voice Cloning Inference')
    parser.add_argument('--speaker_encoder', type=str, required=True,
                      help='Path to trained speaker encoder (.pth)')
    parser.add_argument('--voice_model', type=str, required=True,
                      help='Path to trained voice cloning model (.pth)')
    parser.add_argument('--reference_audio', type=str, required=True,
                      help='Path to reference audio file (.wav)')
    parser.add_argument('--text', type=str, required=True,
                      help='Text to synthesize')
    parser.add_argument('--output', type=str, default='generated_speech.wav',
                      help='Output audio file path')
    parser.add_argument('--clean_audio', action='store_true',
                      help='Clean reference audio before processing')
    parser.add_argument('--device', type=str, default=None,
                      help='Device to use (cuda/cpu)')
    
    args = parser.parse_args()
    
    # Validate inputs
    if not Path(args.speaker_encoder).exists():
        print(f"Speaker encoder not found: {args.speaker_encoder}")
        return
    
    if not Path(args.voice_model).exists():
        print(f"Voice cloning model not found: {args.voice_model}")
        return
    
    if not Path(args.reference_audio).exists():
        print(f"Reference audio not found: {args.reference_audio}")
        return
    
    try:
        # Initialize voice cloner
        cloner = VoiceCloner(
            speaker_encoder_path=args.speaker_encoder,
            voice_cloning_model_path=args.voice_model,
            device=args.device
        )
        
        # Generate speech
        result = cloner.generate_speech(
            text=args.text,
            reference_audio_path=args.reference_audio,
            output_path=args.output,
            clean_reference=args.clean_audio
        )
        
        if result:
            print(f"Success! Generated speech saved to: {result}")
        else:
            print("Failed to generate speech")
            
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()