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

# Audio cleaning imports
try:
    import noisereduce as nr
    NOISEREDUCE_AVAILABLE = True
except ImportError:
    NOISEREDUCE_AVAILABLE = False
    print("⚠️  noisereduce not installed. Install with: pip install noisereduce")

try:
    from scipy.signal import butter, filtfilt, medfilt
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

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

class AudioConverter:
    """Universal audio format converter with robust error handling"""
    
    def __init__(self, target_sample_rate=22050, target_channels=1):
        self.target_sample_rate = target_sample_rate
        self.target_channels = target_channels
        
        # Check for additional libraries
        self.pydub_available = self._check_pydub()
        self.ffmpeg_available = self._check_ffmpeg()
        
    def _check_pydub(self):
        """Check if pydub is available"""
        try:
            import pydub
            return True
        except ImportError:
            return False
    
    def _check_ffmpeg(self):
        """Check if ffmpeg is available"""
        try:
            import subprocess
            result = subprocess.run(['ffmpeg', '-version'], 
                                 capture_output=True, text=True)
            return result.returncode == 0
        except (FileNotFoundError, subprocess.SubprocessError):
            return False
    
    def convert_audio_universal(self, 
                              input_path: str, 
                              output_path: str = None,
                              normalize_volume: bool = True) -> str:
        """
        Universal audio converter that handles all common formats
        Converts to target sample rate and channels with multiple fallback methods
        """
        input_path = Path(input_path)
        
        if not input_path.exists():
            print(f"❌ Input file not found: {input_path}")
            return None
        
        # Generate output path if not provided
        if output_path is None:
            output_path = input_path.parent / f"converted_{input_path.stem}.wav"
        else:
            output_path = Path(output_path)
        
        print(f"🔄 Converting: {input_path.name} -> {output_path.name}")
        print(f"🎯 Target: {self.target_sample_rate}Hz, {self.target_channels} channel(s)")
        
        # Try multiple conversion methods in order of preference
        methods = [
            self._convert_with_librosa,
            self._convert_with_pydub,
            self._convert_with_soundfile,
        ]
        
        for i, method in enumerate(methods, 1):
            try:
                print(f"🔄 Trying method {i}...")
                result = method(str(input_path), str(output_path), normalize_volume)
                if result:
                    self._validate_conversion(str(output_path))
                    print(f"✅ Conversion successful using method {i}")
                    return str(output_path)
            except Exception as e:
                print(f"⚠️  Method {i} failed: {e}")
                continue
        
        print("❌ All conversion methods failed")
        return None
    
    def _convert_with_librosa(self, input_path: str, output_path: str, normalize: bool) -> bool:
        """Convert using librosa (best quality, handles most formats)"""
        # Load with librosa (handles many formats)
        audio, sr = librosa.load(input_path, sr=None, mono=False)
        
        print(f"📊 Original: {sr}Hz, shape: {audio.shape}")
        
        # Handle mono/stereo conversion
        if audio.ndim > 1:
            if self.target_channels == 1:
                # Convert to mono
                audio = librosa.to_mono(audio)
                print("🔄 Converted to mono")
            else:
                # Keep stereo or take first channel
                if audio.shape[0] > 1:
                    audio = audio[0]  # Take first channel
        else:
            # Already mono
            if self.target_channels > 1:
                # Duplicate for stereo (rarely needed)
                audio = np.tile(audio, (self.target_channels, 1))
        
        # Resample if needed
        if sr != self.target_sample_rate:
            print(f"🔄 Resampling: {sr}Hz -> {self.target_sample_rate}Hz")
            audio = librosa.resample(audio, orig_sr=sr, target_sr=self.target_sample_rate)
        
        # Normalize volume
        if normalize:
            audio = self._normalize_audio(audio)
        
        # Save as WAV
        sf.write(output_path, audio, self.target_sample_rate)
        return True
    
    def _convert_with_pydub(self, input_path: str, output_path: str, normalize: bool) -> bool:
        """Convert using pydub (good for compressed formats like MP3)"""
        if not self.pydub_available:
            raise ImportError("pydub not available")
        
        from pydub import AudioSegment
        
        # Load audio file
        audio = AudioSegment.from_file(input_path)
        print(f"📊 Original: {audio.frame_rate}Hz, {audio.channels} channels, {len(audio)}ms")
        
        # Convert to mono if needed
        if audio.channels > 1 and self.target_channels == 1:
            audio = audio.set_channels(1)
            print("🔄 Converted to mono")
        
        # Resample if needed
        if audio.frame_rate != self.target_sample_rate:
            print(f"🔄 Resampling: {audio.frame_rate}Hz -> {self.target_sample_rate}Hz")
            audio = audio.set_frame_rate(self.target_sample_rate)
        
        # Normalize volume (pydub method)
        if normalize:
            # Normalize to -3dB to prevent clipping
            target_dBFS = -3.0
            change_in_dBFS = target_dBFS - audio.dBFS
            audio = audio.apply_gain(change_in_dBFS)
            print(f"🔊 Normalized volume: {audio.dBFS:.1f} dBFS")
        
        # Export as WAV
        audio.export(output_path, format="wav")
        return True
    
    def _convert_with_soundfile(self, input_path: str, output_path: str, normalize: bool) -> bool:
        """Convert using soundfile (fallback method)"""
        # Read with soundfile
        audio, sr = sf.read(input_path)
        
        print(f"📊 Original: {sr}Hz, shape: {audio.shape}")
        
        # Handle channels
        if audio.ndim > 1:
            if self.target_channels == 1:
                # Convert to mono (average channels)
                audio = np.mean(audio, axis=1)
                print("🔄 Converted to mono")
        
        # Simple resampling (basic linear interpolation)
        if sr != self.target_sample_rate:
            print(f"🔄 Basic resampling: {sr}Hz -> {self.target_sample_rate}Hz")
            # Calculate resampling ratio
            ratio = self.target_sample_rate / sr
            new_length = int(len(audio) * ratio)
            
            # Simple linear interpolation
            old_indices = np.linspace(0, len(audio) - 1, new_length)
            audio = np.interp(old_indices, np.arange(len(audio)), audio)
        
        # Normalize
        if normalize:
            audio = self._normalize_audio(audio)
        
        # Save
        sf.write(output_path, audio, self.target_sample_rate)
        return True
    
    def _normalize_audio(self, audio: np.ndarray, target_level: float = 0.707) -> np.ndarray:
        """Normalize audio to target level (default: -3dB)"""
        peak = np.max(np.abs(audio))
        if peak > 0:
            normalized = audio * (target_level / peak)
            print(f"🔊 Volume normalized: peak {peak:.3f} -> {target_level:.3f}")
            return normalized
        return audio
    
    def _validate_conversion(self, output_path: str):
        """Validate the converted audio file"""
        try:
            # Quick validation
            audio, sr = sf.read(output_path, frames=1000)  # Read first 1000 samples
            
            if sr != self.target_sample_rate:
                raise ValueError(f"Sample rate mismatch: got {sr}, expected {self.target_sample_rate}")
            
            if audio.ndim > 1 and self.target_channels == 1:
                raise ValueError("Expected mono audio but got multi-channel")
            
            print(f"✅ Validation passed: {sr}Hz, {audio.shape}")
            
        except Exception as e:
            raise ValueError(f"Converted file validation failed: {e}")
    
    def batch_convert(self, 
                     input_dir: str, 
                     output_dir: str = None,
                     file_extensions: List[str] = None) -> List[str]:
        """Batch convert multiple audio files"""
        
        input_dir = Path(input_dir)
        if not input_dir.exists():
            print(f"❌ Input directory not found: {input_dir}")
            return []
        
        if output_dir is None:
            output_dir = input_dir / "converted"
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        # Default extensions to convert
        if file_extensions is None:
            file_extensions = ['.mp3', '.m4a', '.aac', '.flac', '.ogg', '.wma', '.aiff']
        
        # Find files to convert
        files_to_convert = []
        for ext in file_extensions:
            files_to_convert.extend(input_dir.glob(f'*{ext}'))
            files_to_convert.extend(input_dir.glob(f'*{ext.upper()}'))
        
        if not files_to_convert:
            print(f"❌ No audio files found with extensions: {file_extensions}")
            return []
        
        print(f"📁 Found {len(files_to_convert)} files to convert")
        
        converted_files = []
        failed_files = []
        
        for file_path in files_to_convert:
            output_path = output_dir / f"{file_path.stem}.wav"
            result = self.convert_audio_universal(str(file_path), str(output_path))
            
            if result:
                converted_files.append(result)
            else:
                failed_files.append(str(file_path))
        
        print(f"📊 Batch conversion complete:")
        print(f"   ✅ Success: {len(converted_files)}")
        print(f"   ❌ Failed: {len(failed_files)}")
        
        if failed_files:
            print("❌ Failed files:")
            for f in failed_files:
                print(f"   {Path(f).name}")
        
        return converted_files
    
    def get_audio_info(self, audio_path: str) -> dict:
        """Get detailed information about an audio file"""
        try:
            # Try librosa first
            audio, sr = librosa.load(audio_path, sr=None)
            
            info = {
                'sample_rate': sr,
                'channels': 1 if audio.ndim == 1 else audio.shape[0],
                'duration': len(audio) / sr,
                'samples': len(audio),
                'format': Path(audio_path).suffix.lower(),
                'compatible': sr == self.target_sample_rate and (audio.ndim == 1 or audio.shape[0] == self.target_channels)
            }
            
            return info
            
        except Exception as e:
            return {'error': str(e)}

class AudioCleaner:
    """Advanced audio cleaning and preprocessing class"""
    
    def __init__(self, sample_rate=22050):
        self.sample_rate = sample_rate
        
    def detect_silence_segments(self, audio, threshold_db=-30):
        """Detect silence segments for noise profiling"""
        # Convert to dB
        audio_db = librosa.amplitude_to_db(np.abs(audio), ref=np.max)
        
        # Find silence segments
        silence_mask = audio_db < threshold_db
        silence_segments = []
        
        in_silence = False
        start_idx = 0
        
        for i, is_silent in enumerate(silence_mask):
            if is_silent and not in_silence:
                start_idx = i
                in_silence = True
            elif not is_silent and in_silence:
                if i - start_idx > int(0.1 * self.sample_rate):  # At least 100ms
                    silence_segments.append((start_idx, i))
                in_silence = False
        
        return silence_segments
    
    def advanced_noise_reduction(self, audio, method='noisereduce', preserve_prosody=True):
        """
        Advanced noise reduction with multiple methods
        """
        print(f"🧹 Applying noise reduction using: {method}")
        
        if method == 'noisereduce' and NOISEREDUCE_AVAILABLE:
            return self._noisereduce_method(audio, preserve_prosody)
        elif method == 'spectral_gating':
            return self._spectral_gating_method(audio)
        elif method == 'wiener':
            return self._wiener_filter_method(audio)
        else:
            print(f"⚠️  Method {method} not available, using basic filtering")
            return self._basic_noise_reduction(audio)
    
    def _noisereduce_method(self, audio, preserve_prosody=True):
        """Noise reduction using noisereduce library"""
        try:
            # Stationary noise reduction (gentle to preserve prosody)
            if preserve_prosody:
                cleaned_audio = nr.reduce_noise(
                    y=audio, 
                    sr=self.sample_rate,
                    stationary=True,
                    prop_decrease=0.8  # Reduce noise by 80%
                )
            else:
                # More aggressive cleaning
                cleaned_audio = nr.reduce_noise(
                    y=audio, 
                    sr=self.sample_rate,
                    stationary=False,
                    prop_decrease=0.9
                )
            
            return cleaned_audio
            
        except Exception as e:
            print(f"⚠️  Noisereduce failed: {e}")
            return self._basic_noise_reduction(audio)
    
    def _spectral_gating_method(self, audio):
        """Spectral gating noise reduction"""
        # Compute STFT
        D = librosa.stft(audio, hop_length=512, win_length=2048)
        magnitude, phase = librosa.magphase(D)
        
        # Estimate noise floor
        noise_floor = np.percentile(np.abs(magnitude), 10, axis=1, keepdims=True)
        
        # Create gating mask (preserve signals above noise floor + margin)
        gate_threshold = noise_floor * 3  # 3x noise floor
        gate_mask = np.abs(magnitude) > gate_threshold
        
        # Apply gating
        gated_magnitude = magnitude * gate_mask
        
        # Reconstruct audio
        cleaned_stft = gated_magnitude * phase
        cleaned_audio = librosa.istft(cleaned_stft, hop_length=512, win_length=2048)
        
        return cleaned_audio
    
    def _wiener_filter_method(self, audio):
        """Wiener filter for noise reduction"""
        if not SCIPY_AVAILABLE:
            return self._basic_noise_reduction(audio)
        
        # Simple Wiener filter implementation
        # Estimate noise from quiet segments
        silence_segments = self.detect_silence_segments(audio)
        
        if silence_segments:
            # Extract noise samples
            noise_samples = []
            for start, end in silence_segments[:3]:  # Use first 3 silence segments
                noise_samples.extend(audio[start:end])
            
            if noise_samples:
                noise_power = np.var(noise_samples)
                signal_power = np.var(audio)
                
                # Wiener filter coefficient
                alpha = signal_power / (signal_power + noise_power)
                
                # Apply filter in frequency domain
                D = librosa.stft(audio)
                magnitude, phase = librosa.magphase(D)
                
                # Apply Wiener filtering
                filtered_magnitude = magnitude * alpha
                filtered_stft = filtered_magnitude * phase
                
                cleaned_audio = librosa.istft(filtered_stft)
                return cleaned_audio
        
        return self._basic_noise_reduction(audio)
    
    def _basic_noise_reduction(self, audio):
        """Basic noise reduction using filtering"""
        if not SCIPY_AVAILABLE:
            return audio
        
        # High-pass filter to remove low-frequency noise
        nyquist = self.sample_rate / 2
        low_cutoff = 80 / nyquist  # 80 Hz high-pass
        
        b, a = butter(4, low_cutoff, btype='high')
        filtered_audio = filtfilt(b, a, audio)
        
        # Median filter for impulse noise
        filtered_audio = medfilt(filtered_audio, kernel_size=3)
        
        return filtered_audio
    
    def enhance_speech(self, audio):
        """Speech enhancement techniques"""
        print("🎙️ Enhancing speech characteristics...")
        
        # Normalize amplitude
        audio = audio / (np.max(np.abs(audio)) + 1e-8)
        
        # Dynamic range compression (gentle)
        compressed_audio = np.sign(audio) * np.power(np.abs(audio), 0.8)
        
        # Pre-emphasis filter (boost high frequencies)
        if SCIPY_AVAILABLE:
            pre_emphasis = 0.95
            emphasized_audio = np.append(compressed_audio[0], 
                                       compressed_audio[1:] - pre_emphasis * compressed_audio[:-1])
            return emphasized_audio
        
        return compressed_audio
    
    def clean_audio_comprehensive(self, 
                                audio_path: str, 
                                output_path: Optional[str] = None,
                                noise_reduction_method: str = 'noisereduce',
                                preserve_prosody: bool = True) -> str:
        """
        Comprehensive audio cleaning pipeline
        """
        print(f"🧼 Starting comprehensive audio cleaning: {Path(audio_path).name}")
        
        # Load audio
        try:
            audio, sr = librosa.load(audio_path, sr=self.sample_rate)
            print(f"📥 Loaded audio: {len(audio)} samples, {len(audio)/sr:.2f}s duration")
        except Exception as e:
            print(f"❌ Error loading audio: {e}")
            return None
        
        # Step 1: Trim silence from beginning and end
        audio_trimmed, _ = librosa.effects.trim(audio, top_db=20)
        print(f"✂️  Trimmed silence: {len(audio_trimmed)} samples remaining")
        
        # Step 2: Noise reduction
        audio_denoised = self.advanced_noise_reduction(
            audio_trimmed, 
            method=noise_reduction_method,
            preserve_prosody=preserve_prosody
        )
        
        # Step 3: Speech enhancement
        if preserve_prosody:
            audio_enhanced = self.enhance_speech(audio_denoised)
        else:
            audio_enhanced = audio_denoised
        
        # Step 4: Final normalization (preserve dynamics)
        peak_level = np.max(np.abs(audio_enhanced))
        if peak_level > 0:
            # Normalize to -3dB to prevent clipping but maintain dynamics
            target_level = 0.707  # -3dB
            audio_final = audio_enhanced * (target_level / peak_level)
        else:
            audio_final = audio_enhanced
        
        # Save cleaned audio
        if output_path is None:
            output_path = str(Path(audio_path).parent / f"cleaned_{Path(audio_path).name}")
        
        try:
            sf.write(output_path, audio_final, self.sample_rate)
            print(f"💾 Cleaned audio saved: {output_path}")
            
            # Quality metrics
            snr_estimate = self._estimate_snr(audio_final)
            print(f"📊 Estimated SNR: {snr_estimate:.1f} dB")
            
            return output_path
            
        except Exception as e:
            print(f"❌ Error saving cleaned audio: {e}")
            return None
    
    def _estimate_snr(self, audio):
        """Estimate Signal-to-Noise Ratio"""
        # Simple SNR estimation
        rms_signal = np.sqrt(np.mean(audio**2))
        
        # Estimate noise from quieter segments
        sorted_audio = np.sort(np.abs(audio))
        noise_threshold = int(len(sorted_audio) * 0.1)  # Bottom 10%
        rms_noise = np.sqrt(np.mean(sorted_audio[:noise_threshold]**2))
        
        if rms_noise > 0:
            snr = 20 * np.log10(rms_signal / rms_noise)
            return snr
        return float('inf')
    
    def batch_clean_audio(self, audio_paths: List[str], **kwargs) -> List[str]:
        """Clean multiple audio files"""
        cleaned_paths = []
        
        for audio_path in audio_paths:
            print(f"\n🔄 Processing {Path(audio_path).name}...")
            cleaned_path = self.clean_audio_comprehensive(audio_path, **kwargs)
            if cleaned_path:
                cleaned_paths.append(cleaned_path)
            else:
                print(f"⚠️  Skipping {audio_path} due to cleaning failure")
        
        return cleaned_paths

class VoiceCloner:
    def __init__(self, 
                 speaker_encoder_path: str = "best_speaker_encoder.pth",
                 device: str = None,
                 enable_audio_cleaning: bool = True,
                 enable_audio_conversion: bool = True):
        """
        Initialize the voice cloning system with audio cleaning and conversion capabilities
        """
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.enable_cleaning = enable_audio_cleaning
        self.enable_conversion = enable_audio_conversion
        
        print(f"🚀 Initializing Voice Cloner on {self.device}")
        print(f"🧹 Audio cleaning: {'Enabled' if self.enable_cleaning else 'Disabled'}")
        print(f"🔄 Audio conversion: {'Enabled' if self.enable_conversion else 'Disabled'}")
        
        # Initialize audio converter
        if self.enable_conversion:
            self.audio_converter = AudioConverter(target_sample_rate=22050, target_channels=1)
        
        # Initialize audio cleaner
        if self.enable_cleaning:
            self.audio_cleaner = AudioCleaner(sample_rate=22050)
        
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
    
    def prepare_audio(self, 
                     audio_path: str, 
                     convert: bool = None,
                     clean: bool = None) -> str:
        """
        Prepare audio file: convert format/sample rate and optionally clean
        Returns path to prepared audio file
        """
        # Use instance settings if not specified
        if convert is None:
            convert = self.enable_conversion
        if clean is None:
            clean = self.enable_cleaning
        
        original_path = audio_path
        current_path = audio_path
        
        print(f"🔧 Preparing audio: {Path(audio_path).name}")
        
        # Step 1: Check if conversion is needed
        if convert and self.enable_conversion:
            audio_info = self.audio_converter.get_audio_info(audio_path)
            
            if 'error' not in audio_info:
                print(f"📊 Audio info: {audio_info['sample_rate']}Hz, "
                      f"{audio_info['channels']} channels, {audio_info['duration']:.1f}s")
                
                if not audio_info['compatible']:
                    print("🔄 Audio format conversion needed...")
                    converted_path = str(Path(audio_path).parent / f"converted_{Path(audio_path).stem}.wav")
                    
                    result = self.audio_converter.convert_audio_universal(
                        audio_path, converted_path, normalize_volume=True
                    )
                    
                    if result:
                        current_path = result
                        print(f"✅ Conversion successful: {Path(current_path).name}")
                    else:
                        print("⚠️  Conversion failed, using original file")
                else:
                    print("✅ Audio format already compatible")
            else:
                print(f"⚠️  Could not read audio info: {audio_info['error']}")
        
        # Step 2: Clean audio if requested
        if clean and self.enable_cleaning:
            print("🧹 Cleaning audio...")
            cleaned_path = self.audio_cleaner.clean_audio_comprehensive(
                current_path,
                preserve_prosody=True,
                noise_reduction_method='noisereduce'
            )
            
            if cleaned_path:
                current_path = cleaned_path
                print(f"✅ Cleaning successful: {Path(current_path).name}")
            else:
                print("⚠️  Cleaning failed, using previous version")
        
        if current_path != original_path:
            print(f"🎯 Final prepared audio: {Path(current_path).name}")
        else:
            print("🎯 Using original audio file")
        
        return current_path
        
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
    
    def get_speaker_embedding(self, audio_paths: List[str], prepare_audio: bool = None) -> torch.Tensor:
        """Extract speaker embedding from reference audio(s) with automatic preparation"""
        if self.speaker_encoder is None:
            print("❌ Speaker encoder not loaded!")
            return None
        
        # Use instance setting if not specified
        if prepare_audio is None:
            prepare_audio = self.enable_conversion or self.enable_cleaning
        
        print(f"🎯 Extracting speaker embedding from {len(audio_paths)} reference audio(s)")
        if prepare_audio:
            print("🔧 Audio preparation enabled")
        
        # Prepare audio files if requested
        processed_paths = []
        if prepare_audio:
            for audio_path in audio_paths:
                if not Path(audio_path).exists():
                    print(f"❌ Audio file not found: {audio_path}")
                    continue
                
                prepared_path = self.prepare_audio(audio_path)
                processed_paths.append(prepared_path)
        else:
            processed_paths = [path for path in audio_paths if Path(path).exists()]
        
        if not processed_paths:
            print("❌ No valid audio files to process!")
            return None
        
        embeddings = []
        
        for audio_path in processed_paths:
            print(f"📁 Processing: {Path(audio_path).name}")
            
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
                         language: str = "en",
                         prepare_reference: bool = None) -> str:
        """Synthesize speech with cloned voice using Coqui TTS"""
        
        if self.tts is None:
            print("❌ TTS model not loaded!")
            return None
        
        # Use instance setting if not specified
        if prepare_reference is None:
            prepare_reference = self.enable_conversion or self.enable_cleaning
        
        print(f"🎤 Synthesizing speech...")
        print(f"📝 Text: '{text[:100]}...'")
        print(f"📁 Reference audios: {len(reference_audios)}")
        print(f"🌍 Language: {language}")
        print(f"🔧 Prepare reference: {prepare_reference}")
        
        try:
            # Prepare reference audio if requested
            reference_audio = reference_audios[0]
            
            if not Path(reference_audio).exists():
                print(f"❌ Reference audio not found: {reference_audio}")
                return None
            
            processed_reference = reference_audio
            if prepare_reference:
                print(f"🔧 Preparing reference audio...")
                processed_reference = self.prepare_audio(reference_audio)
            
            print(f"🎯 Using reference: {Path(processed_reference).name}")
            
            # Check if TTS supports voice cloning
            if hasattr(self.tts, 'tts_with_vc'):
                # For multi-speaker models, we must provide a speaker, even for VC.
                speaker_to_use = None
                if self.tts.is_multi_speaker:
                    speaker_to_use = self.tts.speakers[0]
                    print(f"🗣️ Using base speaker from TTS model: {speaker_to_use}")

                print("🔄 Using voice cloning mode...")
                wav = self.tts.tts_with_vc(
                    text=text,
                    speaker_wav=processed_reference,
                    language=language,
                    speaker=speaker_to_use
                )
            elif hasattr(self.tts, 'tts'):
                print("🔄 Using standard TTS with speaker reference...")
                wav = self.tts.tts(
                    text=text, 
                    speaker_wav=processed_reference,
                    language=language,
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
        
        # Test 3: Audio cleaner
        if self.enable_cleaning:
            if hasattr(self, 'audio_cleaner'):
                print("✅ Test 3 PASSED: Audio cleaner initialized")
            else:
                print("⚠️  Test 3 WARNING: Audio cleaner not initialized")
        else:
            print("ℹ️  Test 3 SKIPPED: Audio cleaning disabled")
        
        # Test 4: Model forward pass
        try:
            test_input = torch.randn(1, 80, 100).to(self.device)
            with torch.no_grad():
                embedding = self.speaker_encoder(test_input, return_embedding=True)
            print(f"✅ Test 4 PASSED: Model forward pass successful")
        except Exception as e:
            print(f"❌ Test 4 FAILED: Model forward pass failed: {e}")
            return False
        
        print("🎉 All tests passed! System is ready for voice cloning.")
        return True

def main():
    """Main function with step-by-step testing and audio cleaning options"""
    print("🎤 ENHANCED VOICE CLONING SYSTEM WITH AUDIO CLEANING")
    print("=" * 70)
    
    # Check dependencies
    print("🔍 Checking dependencies...")
    if not NOISEREDUCE_AVAILABLE:
        print("⚠️  For best results, install: pip install noisereduce")
    if not SCIPY_AVAILABLE:
        print("⚠️  For advanced filtering, install: pip install scipy")
    
    # Initialize voice cloner
    cloner = VoiceCloner(enable_audio_cleaning=True)
    
    # Run quick test
    if not cloner.quick_test():
        print("\n❌ SYSTEM NOT READY")
        print("Please fix the issues above before proceeding.")
        return
    
    # Interactive demo
    print("\n🎯 ENHANCED VOICE CLONING DEMO")
    print("Features:")
    print("✨ Advanced noise reduction")
    print("🎙️ Speech enhancement")
    print("🧹 Automatic audio cleaning")
    print("🔧 Prosody preservation")
    
    while True:
        print("\n" + "="*60)
        print("ENHANCED VOICE CLONING MENU")
        print("1. Clone voice (with auto-cleaning)")
        print("2. Clone voice (no cleaning)")
        print("3. Test speaker embedding extraction")
        print("4. Clean audio file only")
        print("5. Batch clean audio files")
        print("6. Exit")
        
        choice = input("Enter choice (1-6): ").strip()
        
        if choice == "1":
            # Clone voice with cleaning
            ref_audio = input("Enter reference audio path: ").strip()
            if not ref_audio or not Path(ref_audio).exists():
                print("❌ Invalid audio path")
                continue
            
            # Get text
            text = input("Enter text to synthesize: ").strip()
            if not text:
                text = "Hello, this is a test of voice cloning technology using your trained model with advanced audio cleaning."
            
            # Get output path
            output = input("Output filename (default: cloned_voice_cleaned.wav): ").strip()
            if not output:
                output = "cloned_voice_cleaned.wav"
            
            # Ensure the output path has a .wav extension
            if not output.lower().endswith(('.wav', '.flac', '.ogg')):
                output += '.wav'
                print(f"💡 No extension found. Appending .wav. Saving to: {output}")
            
            # Clone voice with cleaning
            result = cloner.synthesize_speech(text, [ref_audio], output, prepare_reference=True)
            if result:
                print(f"🎉 Voice cloning with cleaning successful! Audio saved to: {result}")
            else:
                print("❌ Voice cloning failed")
        
        elif choice == "2":
            # Clone voice without cleaning
            ref_audio = input("Enter reference audio path: ").strip()
            if not ref_audio or not Path(ref_audio).exists():
                print("❌ Invalid audio path")
                continue
            
            # Get text
            text = input("Enter text to synthesize: ").strip()
            if not text:
                text = "Hello, this is a test of voice cloning technology using your trained model."
            
            # Get output path
            output = input("Output filename (default: cloned_voice_raw.wav): ").strip()
            if not output:
                output = "cloned_voice_raw.wav"
            
            # Ensure the output path has a .wav extension
            if not output.lower().endswith(('.wav', '.flac', '.ogg')):
                output += '.wav'
                print(f"💡 No extension found. Appending .wav. Saving to: {output}")
            
            # Clone voice without cleaning
            result = cloner.synthesize_speech(text, [ref_audio], output, prepare_reference=False)
            if result:
                print(f"🎉 Voice cloning (raw) successful! Audio saved to: {result}")
            else:
                print("❌ Voice cloning failed")
        
        elif choice == "3":
            # Test speaker embedding extraction
            ref_audio = input("Enter audio path to test embedding extraction: ").strip()
            if not ref_audio or not Path(ref_audio).exists():
                print("❌ Invalid audio path")
                continue
            
            clean_choice = input("Clean audio before extraction? (y/n, default: y): ").strip().lower()
            clean_audio = clean_choice != 'n'
            
            embedding = cloner.get_speaker_embedding([ref_audio], prepare_audio=clean_audio)
            if embedding is not None:
                print(f"✅ Speaker embedding extracted successfully!")
                print(f"📊 Embedding shape: {embedding.shape}")
                print(f"🎯 Embedding norm: {torch.norm(embedding).item():.4f}")
                
                # Show embedding statistics
                embedding_np = embedding.cpu().numpy().flatten()
                print(f"📈 Embedding stats:")
                print(f"   Mean: {np.mean(embedding_np):.4f}")
                print(f"   Std: {np.std(embedding_np):.4f}")
                print(f"   Min: {np.min(embedding_np):.4f}")
                print(f"   Max: {np.max(embedding_np):.4f}")
            else:
                print("❌ Failed to extract speaker embedding")
        
        elif choice == "4":
            # Clean audio file only
            if not cloner.enable_cleaning:
                print("❌ Audio cleaning is disabled")
                continue
            
            input_audio = input("Enter audio path to clean: ").strip()
            if not input_audio or not Path(input_audio).exists():
                print("❌ Invalid audio path")
                continue
            
            output_audio = input(f"Output path (default: cleaned_{Path(input_audio).name}): ").strip()
            if not output_audio:
                output_audio = None
            
            # Cleaning options
            print("\nCleaning options:")
            print("1. Gentle (preserve prosody) - Recommended")
            print("2. Standard (balanced)")
            print("3. Aggressive (maximum cleaning)")
            
            clean_level = input("Choose cleaning level (1-3, default: 1): ").strip()
            
            if clean_level == "3":
                preserve_prosody = False
                method = 'noisereduce'
            elif clean_level == "2":
                preserve_prosody = True
                method = 'spectral_gating'
            else:
                preserve_prosody = True
                method = 'noisereduce'
            
            print(f"🧼 Cleaning audio with {method} method...")
            cleaned_path = cloner.audio_cleaner.clean_audio_comprehensive(
                input_audio,
                output_path=output_audio,
                noise_reduction_method=method,
                preserve_prosody=preserve_prosody
            )
            
            if cleaned_path:
                print(f"✅ Audio cleaned successfully: {cleaned_path}")
                
                # Offer to play comparison (if matplotlib available for visualization)
                try:
                    import matplotlib.pyplot as plt
                    compare = input("Show before/after comparison plot? (y/n): ").strip().lower()
                    if compare == 'y':
                        # Load both audios
                        original, sr = librosa.load(input_audio, sr=22050)
                        cleaned, _ = librosa.load(cleaned_path, sr=22050)
                        
                        # Create comparison plot
                        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
                        
                        # Time domain
                        time_orig = np.arange(len(original)) / sr
                        time_clean = np.arange(len(cleaned)) / sr
                        
                        axes[0,0].plot(time_orig[:sr*3], original[:sr*3])  # First 3 seconds
                        axes[0,0].set_title('Original Audio (first 3s)')
                        axes[0,0].set_xlabel('Time (s)')
                        axes[0,0].set_ylabel('Amplitude')
                        
                        axes[0,1].plot(time_clean[:sr*3], cleaned[:sr*3])
                        axes[0,1].set_title('Cleaned Audio (first 3s)')
                        axes[0,1].set_xlabel('Time (s)')
                        axes[0,1].set_ylabel('Amplitude')
                        
                        # Spectrograms
                        spec_orig = librosa.stft(original)
                        spec_clean = librosa.stft(cleaned)
                        
                        axes[1,0].imshow(librosa.amplitude_to_db(np.abs(spec_orig)), 
                                       aspect='auto', origin='lower')
                        axes[1,0].set_title('Original Spectrogram')
                        axes[1,0].set_xlabel('Time')
                        axes[1,0].set_ylabel('Frequency')
                        
                        axes[1,1].imshow(librosa.amplitude_to_db(np.abs(spec_clean)), 
                                       aspect='auto', origin='lower')
                        axes[1,1].set_title('Cleaned Spectrogram')
                        axes[1,1].set_xlabel('Time')
                        axes[1,1].set_ylabel('Frequency')
                        
                        plt.tight_layout()
                        plt.show()
                        
                except ImportError:
                    print("💡 Install matplotlib to see audio comparison plots")
                
            else:
                print("❌ Audio cleaning failed")
        
        elif choice == "5":
            # Batch clean audio files
            if not cloner.enable_cleaning:
                print("❌ Audio cleaning is disabled")
                continue
            
            input_dir = input("Enter directory path with audio files: ").strip()
            if not input_dir or not Path(input_dir).exists():
                print("❌ Invalid directory path")
                continue
            
            # Find audio files
            audio_extensions = {'.wav', '.mp3', '.flac', '.ogg', '.m4a', '.aac'}
            audio_files = []
            
            for ext in audio_extensions:
                audio_files.extend(Path(input_dir).glob(f'*{ext}'))
                audio_files.extend(Path(input_dir).glob(f'*{ext.upper()}'))
            
            if not audio_files:
                print("❌ No audio files found in directory")
                continue
            
            print(f"📁 Found {len(audio_files)} audio files:")
            for i, f in enumerate(audio_files[:10]):  # Show first 10
                print(f"   {i+1}. {f.name}")
            if len(audio_files) > 10:
                print(f"   ... and {len(audio_files) - 10} more")
            
            proceed = input(f"Process all {len(audio_files)} files? (y/n): ").strip().lower()
            if proceed != 'y':
                continue
            
            # Cleaning options
            print("\nBatch cleaning options:")
            print("1. Gentle (preserve prosody) - Recommended")
            print("2. Standard (balanced)")
            
            clean_level = input("Choose cleaning level (1-2, default: 1): ").strip()
            preserve_prosody = clean_level != "2"
            
            print(f"🔄 Starting batch cleaning of {len(audio_files)} files...")
            
            cleaned_paths = cloner.audio_cleaner.batch_clean_audio(
                [str(f) for f in audio_files],
                preserve_prosody=preserve_prosody,
                noise_reduction_method='noisereduce'
            )
            
            print(f"✅ Batch cleaning complete!")
            print(f"📊 Successfully cleaned: {len(cleaned_paths)}/{len(audio_files)} files")
            
            if cleaned_paths:
                print("📁 Cleaned files saved as 'cleaned_[original_name]'")
        
        elif choice == "6":
            print("👋 Goodbye!")
            break
        
        else:
            print("❌ Invalid choice")

def install_dependencies():
    """Helper function to install required dependencies"""
    print("📦 DEPENDENCY INSTALLER")
    print("=" * 40)
    
    dependencies = {
        'noisereduce': 'pip install noisereduce',
        'scipy': 'pip install scipy',
        'matplotlib': 'pip install matplotlib',
        'TTS': 'pip install TTS',
        'librosa': 'pip install librosa',
        'soundfile': 'pip install soundfile',
        'torch': 'pip install torch torchvision torchaudio',
    }
    
    print("Required packages for enhanced voice cloning:")
    for package, install_cmd in dependencies.items():
        try:
            __import__(package)
            print(f"✅ {package}: Installed")
        except ImportError:
            print(f"❌ {package}: Missing - Run: {install_cmd}")
    
    print("\n💡 For GPU support (recommended):")
    print("   Visit: https://pytorch.org/get-started/locally/")
    print("   And install the appropriate PyTorch version for your system")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == '--install-deps':
        install_dependencies()
    else:
        main()