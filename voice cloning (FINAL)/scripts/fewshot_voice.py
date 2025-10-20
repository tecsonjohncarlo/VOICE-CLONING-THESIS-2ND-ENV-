"""
Fixed XTTS Voice Cloner - Removes attention_mask access instead of adding it
"""

import torch
import torch.nn as nn
import numpy as np
import librosa
import soundfile as sf
from pathlib import Path
import json
import warnings
from pydub import AudioSegment
from pydub.utils import which

warnings.filterwarnings('ignore')

# Check for ffmpeg
FFMPEG_AVAILABLE = which("ffmpeg") is not None
if not FFMPEG_AVAILABLE:
    print("Warning: ffmpeg not found. MP3 export will not work.")

# Check for audio playback
try:
    import simpleaudio
    PLAYBACK_AVAILABLE = True
except ImportError:
    PLAYBACK_AVAILABLE = False
    print("Info: simpleaudio not available (optional for audio preview)")

# Audio processing
try:
    import noisereduce as nr
    NOISEREDUCE_AVAILABLE = True
except ImportError:
    NOISEREDUCE_AVAILABLE = False
    print("Info: noisereduce not available (optional for audio cleaning)")

# RVC support
try:
    from rvc_python.modules.vc.modules import VC
    from rvc_python.modules.vc.utils import load_hubert
    RVC_AVAILABLE = True
except ImportError:
    RVC_AVAILABLE = False
    print("Info: rvc-python not available. RVC models cannot be used. Install with: pip install rvc-python")

def patch_gpt_inference_forward():
    """
    Patch GPT2InferenceModel.forward to handle None attention_mask
    The model doesn't use attention_mask, but some code tries to access it
    """
    try:
        from TTS.tts.layers.xtts.gpt_inference import GPT2InferenceModel
        
        original_forward = GPT2InferenceModel.forward
        
        # This patch ensures the argument is always passed, even if None
        def patched_forward(self, input_ids, past_key_values=None, attention_mask=None, **kwargs):
            
            # Clean out other problematic kwargs that might be passed
            kwargs_clean = {k: v for k, v in kwargs.items() if k not in [
                'token_type_ids', 'position_ids', 'head_mask', 'inputs_embeds',
                'encoder_hidden_states', 'encoder_attention_mask', 'labels',
                'use_cache', 'output_attentions', 'output_hidden_states', 'return_dict'
            ]}
            
            # Call original, but explicitly pass attention_mask=None
            return original_forward(
                self, input_ids, past_key_values=past_key_values, attention_mask=attention_mask, **kwargs_clean
            )
        
        GPT2InferenceModel.forward = patched_forward
        print("✓ Patched GPT2InferenceModel.forward to ignore attention_mask")
        return True
        
    except Exception as e:
        print(f"Forward patch warning: {e}")
        return False

def patch_gpt_inference_model_logic():
    try:
        from TTS.tts.layers.xtts.gpt_inference import GPT2InferenceModel
        original_forward_logic = GPT2InferenceModel.forward

        def patched_forward_logic(self, input_ids, attention_mask=None, **kwargs):
            # If attention_mask is None, create a default one. This is the core fix.
            if attention_mask is None:
                attention_mask = torch.ones_like(input_ids)

            return original_forward_logic(self, input_ids=input_ids, attention_mask=attention_mask, **kwargs)

        GPT2InferenceModel.forward = patched_forward_logic
        print("✓ Applied direct logic patch to GPT2InferenceModel.forward")
        return True
    except Exception as e:
        print(f"Direct logic patch warning: {e}")
        return False

def patch_generate_kwargs():
    """Remove attention_mask from generate() calls"""
    try:
        from TTS.tts.layers.xtts.gpt import GPT
        
        if hasattr(GPT, 'generate'):
            original_generate = GPT.generate
            
            def patched_generate(self, *args, **kwargs):
                # Remove all problematic kwargs that the model doesn't support
                clean_kwargs = {k: v for k, v in kwargs.items()
                              if k not in ['bos_token_id', 'eos_token_id', 
                                          'pad_token_id', 'decoder_start_token_id',
                                          'attention_mask']}  # Remove attention_mask too
                
                return original_generate(self, *args, **clean_kwargs)
            
            GPT.generate = patched_generate
            print("✓ Patched GPT.generate to remove attention_mask")
        
        return True
    except Exception as e:
        print(f"Generate patch warning: {e}")
        return False


# Apply patches BEFORE importing TTS
print("Applying patches...")
patch_gpt_inference_model_logic() # Use the more direct and reliable patch
patch_generate_kwargs()

# Import TTS
try:
    from TTS.api import TTS
    TTS_AVAILABLE = True
except ImportError:
    TTS_AVAILABLE = False
    print("Error: TTS not installed. Install with: pip install TTS")


class SpeakerEncoder(nn.Module):
    """Speaker encoder architecture"""
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
    """User-friendly voice cloning system"""
    
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.current_model_path = None
        self.speaker_encoder = None
        self.rvc_model = None
        self.rvc_hubert_model = None
        self.model_type = None # 'xtts' or 'rvc'
        self.model_info = {}
        self.xtts_model = None
        self.reference_embeddings = {}
        
        print(f"Device: {self.device}")
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")
        if RVC_AVAILABLE:
            self.rvc_hubert_model = load_hubert(self.device)
    
    def list_available_models(self, model_dir="fine_tuned_models"):
        """List all available speaker encoder models"""
        model_dir = Path(model_dir)
        
        models = []
        search_paths = [model_dir, Path("custom_models"), Path(".")]
        
        for search_path in search_paths:
            if not search_path.exists():
                continue
            
            for pth_file in search_path.glob("*.pth"):
                # RVC model check: Look for a matching .index file
                index_file = pth_file.with_suffix(".index")
                if index_file.exists():
                    models.append({
                        'path': str(pth_file),
                        'name': pth_file.stem,
                        'size_mb': pth_file.stat().st_size / (1024 * 1024),
                        'type': 'rvc'
                    })
                    continue

                # XTTS Speaker Encoder check
                info_file = pth_file.parent / f"{pth_file.stem}_info.json"
                
                model_data = {
                    'path': str(pth_file),
                    'name': pth_file.stem,
                    'size_mb': pth_file.stat().st_size / (1024 * 1024)
                }
                model_data['type'] = 'xtts'

                if info_file.exists():
                    try:
                        with open(info_file, 'r') as f:
                            info = json.load(f)
                        model_data['info'] = info
                    except:
                        pass
                
                models.append(model_data)
        
        return models
    
    def _load_rvc_model(self, model_path):
        """Load an RVC model."""
        if not RVC_AVAILABLE:
            print("Error: rvc-python is not installed. Cannot load RVC model.")
            return False
        
        model_path = Path(model_path)
        index_path = model_path.with_suffix(".index")

        if not index_path.exists():
            print(f"Error: RVC model requires a .index file. '{index_path}' not found.")
            return False

        try:
            print(f"\nLoading RVC model: {model_path.name}")
            self.rvc_model = VC(self.device)
            self.rvc_model.get_vc(str(model_path))
            self.rvc_model.load_index(str(index_path))
            
            self.current_model_path = model_path
            self.model_type = 'rvc'
            self.speaker_encoder = None # Unload other model type
            self.model_info = {'name': model_path.stem}

            print("✓ RVC model loaded successfully!")
            return True
        except Exception as e:
            print(f"Error loading RVC model: {e}")
            self.rvc_model = None
            return False

    def load_model(self, model_path):
        """Load a speaker encoder model"""
        model_path = Path(model_path)
        
        if not model_path.exists():
            print(f"Model not found: {model_path}")
            return False

        # --- RVC Model Loading Logic ---
        index_path = model_path.with_suffix(".index")
        if index_path.exists():
            return self._load_rvc_model(model_path)

        # --- XTTS Speaker Encoder Loading Logic ---
        if self.rvc_model:
            print("Unloading RVC model to load XTTS speaker encoder.")
            self.rvc_model = None
            return False
        
        try:
            print(f"\nLoading model: {model_path.name}")
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)

            # Check if this is a full checkpoint or just a state_dict
            if 'model_state_dict' in checkpoint and 'num_speakers' in checkpoint:
                # This is a full checkpoint from your training script
                print("  Detected a full training checkpoint.")
                num_speakers = checkpoint.get('num_speakers')
                speaker_to_id = checkpoint.get('speaker_to_id', {})
                model_state_dict = checkpoint['model_state_dict']
            else:
                # This is likely just a state_dict from a downloaded model
                print("  Detected a raw state_dict. Loading as a generic speaker encoder.")
                num_speakers = 2 # Dummy value, classifier is not used for cloning
                speaker_to_id = {'generic_speaker': 0}
                model_state_dict = checkpoint

                # Check for a common mismatch: the final classifier layer size
                if 'classifier.weight' in model_state_dict:
                    num_speakers = model_state_dict['classifier.weight'].shape[0]
                    print(f"  Inferred number of speakers from classifier layer: {num_speakers}")
                else:
                    print("  Warning: Classifier layer not found. This is fine for cloning.")

            if not num_speakers:
                 print("Error: Could not determine number of speakers for the model.")
                 return False

            self.speaker_encoder = SpeakerEncoder(
                input_dim=80,
                hidden_dim=256,
                embedding_dim=128,
                num_speakers=num_speakers
            ).to(self.device)

            # Load the state dict
            self.speaker_encoder.load_state_dict(model_state_dict)
            self.speaker_encoder.eval()
            
            for param in self.speaker_encoder.parameters():
                param.requires_grad = False
            
            self.current_model_path = model_path
            self.model_info = {
                'num_speakers': num_speakers,
                'speaker_to_id': speaker_to_id,
                'embedding_dim': checkpoint.get('embedding_dim', 128)
            }
            self.model_type = 'xtts'
            self.reference_embeddings = checkpoint.get('reference_embeddings', {})
            
            print(f"✓ Model loaded successfully!")
            print(f"  Speakers: {num_speakers}")
            
            return True
            
        except Exception as e:
            print(f"Error loading model: {e}")
            return False
    
    def load_xtts(self):
        """Load XTTS model"""
        if self.xtts_model is not None:
            return True
        
        if not TTS_AVAILABLE:
            print("Error: TTS not available")
            return False
        
        try:
            print("\nLoading XTTS-v2 model...")
            print("(This may take a moment on first run)")
            
            # Register safe globals for PyTorch 2.6+
            try:
                from TTS.tts.configs.xtts_config import XttsConfig
                from TTS.tts.models.xtts import Xtts
                torch.serialization.add_safe_globals([XttsConfig, Xtts])
                print("✓ Registered TTS safe globals")
            except:
                pass
            
            model_name = "tts_models/multilingual/multi-dataset/xtts_v2"
            
            # Try loading
            try:
                self.xtts_model = TTS(model_name).to(self.device)
            except Exception as first_error:
                print("First load attempt failed, trying with weights_only=False...")
                
                original_load = torch.load
                
                def load_with_weights_only_false(*args, **kwargs):
                    kwargs['weights_only'] = False
                    return original_load(*args, **kwargs)
                
                torch.load = load_with_weights_only_false
                
                try:
                    self.xtts_model = TTS(model_name).to(self.device)
                finally:
                    torch.load = original_load
            
            # Apply instance-level patches
            self._apply_instance_patches()
            
            print("✓ XTTS-v2 loaded successfully!")
            return True
            
        except Exception as e:
            print(f"Error loading XTTS: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _apply_instance_patches(self):
        """Apply patches to the loaded model instance"""
        try:
            if not hasattr(self.xtts_model, 'synthesizer') or \
               not hasattr(self.xtts_model.synthesizer, 'tts_model'):
                return
            
            tts_model = self.xtts_model.synthesizer.tts_model
            
            if hasattr(tts_model, 'gpt') and hasattr(tts_model.gpt, 'generate'):
                original_generate = tts_model.gpt.generate
                
                def patched_instance_generate(*args, **kwargs):
                    # Remove attention_mask and other unsupported params
                    clean_kwargs = {k: v for k, v in kwargs.items()
                                  if k not in ['bos_token_id', 'eos_token_id', 
                                              'pad_token_id', 'decoder_start_token_id',
                                              'attention_mask']}
                    
                    return original_generate(*args, **clean_kwargs)
                
                tts_model.gpt.generate = patched_instance_generate
                print("✓ Applied instance-level patch")
            
        except Exception as e:
            print(f"Instance patch warning: {e}")
    
    def prepare_reference_audio(self, audio_path, clean_audio=True):
        """Prepare reference audio for XTTS"""
        try:
            audio, sr = librosa.load(audio_path, sr=22050, mono=True)
            
            if clean_audio and NOISEREDUCE_AVAILABLE:
                audio = nr.reduce_noise(y=audio, sr=sr, stationary=True, prop_decrease=0.7)
            
            audio, _ = librosa.effects.trim(audio, top_db=25)
            
            min_length, max_length = 6 * sr, 30 * sr
            
            if len(audio) < min_length:
                repeats = int(np.ceil(min_length / len(audio)))
                audio = np.tile(audio, repeats)[:min_length]
            elif len(audio) > max_length:
                hop_length = 512
                rms = librosa.feature.rms(y=audio, hop_length=hop_length)[0]
                segment_frames = max_length // hop_length
                
                best_start = 0
                best_energy = 0
                for i in range(len(rms) - segment_frames):
                    energy = np.mean(rms[i:i+segment_frames])
                    if energy > best_energy:
                        best_energy = energy
                        best_start = i
                
                start_sample = best_start * hop_length
                audio = audio[start_sample:start_sample + max_length]
            
            audio = audio / (np.max(np.abs(audio)) + 1e-8) * 0.9
            
            return audio, sr
            
        except Exception as e:
            print(f"Error preparing audio: {e}")
            return None, None
    
    def play_audio_preview(self, audio_path):
        """Play audio preview"""
        if not PLAYBACK_AVAILABLE:
            print("\nAudio preview not available (simpleaudio not installed)")
            return False
        
        try:
            print(f"\nPlaying preview: {Path(audio_path).name}")
            
            audio_data, sample_rate = sf.read(audio_path)
            audio_data = (audio_data * 32767).astype(np.int16)
            
            if audio_data.ndim == 1:
                audio_data = np.column_stack((audio_data, audio_data))
            
            play_obj = simpleaudio.play_buffer(
                audio_data.tobytes(),
                num_channels=2 if audio_data.ndim == 2 else 1,
                bytes_per_sample=2,
                sample_rate=sample_rate
            )
            
            print("Controls: [Press Ctrl+C to stop]")
            
            try:
                play_obj.wait_done()
                print("Playback finished")
            except KeyboardInterrupt:
                play_obj.stop()
                print("Playback stopped")
            
            return True
            
        except Exception as e:
            print(f"Error playing audio: {e}")
            return False
    
    def preview_and_save(self, temp_audio_path, final_output_path, output_format):
        """Preview audio and save"""
        print("\n" + "="*70)
        print("AUDIO PREVIEW")
        print("="*70)
        
        while True:
            print("\nOptions:")
            print("1. Play preview")
            print("2. Save audio")
            print("3. Regenerate")
            print("4. Cancel")
            
            choice = input("\nChoice (1-4): ").strip()
            
            if choice == "1":
                self.play_audio_preview(temp_audio_path)
            
            elif choice == "2":
                print(f"\nSaving to: {final_output_path}")
                
                try:
                    audio_data, sample_rate = sf.read(temp_audio_path)
                    output_path = Path(final_output_path)
                    
                    if output_format.lower() == "mp3":
                        if not FFMPEG_AVAILABLE:
                            print("Warning: ffmpeg not available, saving as WAV")
                            output_path = output_path.with_suffix('.wav')
                            sf.write(str(output_path), audio_data, sample_rate)
                        else:
                            output_path = output_path.with_suffix('.mp3')
                            temp_wav = "temp_conversion.wav"
                            sf.write(temp_wav, audio_data, sample_rate)
                            audio_segment = AudioSegment.from_wav(temp_wav)
                            audio_segment.export(str(output_path), format="mp3", bitrate="192k")
                            try:
                                Path(temp_wav).unlink()
                            except:
                                pass
                    else:
                        output_path = output_path.with_suffix(f'.{output_format}')
                        sf.write(str(output_path), audio_data, sample_rate)
                    
                    try:
                        Path(temp_audio_path).unlink()
                    except:
                        pass
                    
                    print(f"\n✓ Audio saved: {output_path}")
                    return str(output_path)
                    
                except Exception as e:
                    print(f"Error saving: {e}")
                    return None
            
            elif choice == "3":
                try:
                    Path(temp_audio_path).unlink()
                except:
                    pass
                return "regenerate"
            
            elif choice == "4":
                try:
                    Path(temp_audio_path).unlink()
                except:
                    pass
                return None
            
            else:
                print("Invalid choice")

    def synthesize(self, text, output_path, 
                   reference_audio_path=None, speaker_embedding=None,
                   language="en", output_format="mp3", 
                   clean_reference=True, enable_preview=True):
        """Synthesize speech"""
        speaker_wav_path = None # Initialize to None
        
        if self.model_type is None:
            print("Error: No model loaded")
            return None
        
        if not self.load_xtts():
            return None

        
        try:
            print(f"\nSynthesizing speech...")
            print(f"  Text: '{text[:50]}{'...' if len(text) > 50 else ''}'")

            if reference_audio_path:
                if not Path(reference_audio_path).exists():
                    print(f"Reference audio not found: {reference_audio_path}")
                    return None
                print(f"  Reference: {Path(reference_audio_path).name}")
                
                ref_audio, ref_sr = self.prepare_reference_audio(
                    reference_audio_path, clean_audio=clean_reference
                )
                
                if ref_audio is None:
                    return None
                
                speaker_wav_path = "temp_ref.wav"
                sf.write(speaker_wav_path, ref_audio, ref_sr)
                speaker_embedding = None # Ensure only one is used
            elif speaker_embedding is not None:
                print("  Using saved speaker embedding.")
            else:
                print("Error: Must provide either a reference audio path or a speaker embedding.")
                return None

            if self.model_type == 'xtts':
                print("  Generating audio with XTTS...")
                # Use EITHER speaker_wav OR speaker_embedding, never both
                if speaker_wav_path is not None:
                    wav = self.xtts_model.tts(
                        text=text,
                        speaker_wav=speaker_wav_path,
                        language=language
                    )
                elif speaker_embedding is not None:
                    wav = self.xtts_model.tts(
                        text=text,
                        language=language,
                        speaker_embedding=speaker_embedding
                    )
                else:
                    print("Error: No speaker reference provided for XTTS.")
                    return None
            elif self.model_type == 'rvc':
                print("  Generating audio with RVC (2-step process)...")
                print("  Step 1: Generating baseline audio with XTTS...")
                baseline_wav = self.xtts_model.tts(text=text, language=language)

                if baseline_wav is None or len(baseline_wav) == 0:
                    print("RVC Step 1 Failed: Could not generate baseline audio.")
                    return None

                baseline_path = "temp_rvc_baseline.wav"
                sf.write(baseline_path, np.array(baseline_wav), self.xtts_model.synthesizer.output_sample_rate)

                print("  Step 2: Converting voice with RVC model...")
                # RVC conversion parameters
                index_rate = 0.75
                f0_up_key = 0  # Transpose
                f0_method = "rmvpe"  # or "pm", "harvest"
                filter_radius = 3
                resample_sr = 0
                rms_mix_rate = 0.25
                protect = 0.33

                wav_opt = self.rvc_model.vc_single(0, baseline_path, f0_up_key, f0_method, index_rate,
                                                   filter_radius, resample_sr, rms_mix_rate, protect)

                rvc_sr, wav = wav_opt[1]
                Path(baseline_path).unlink()
            else:
                print(f"Error: Unknown model type '{self.model_type}'")
                return None

            if wav is None or len(wav) == 0:
                print("\nSynthesis failed: Model returned empty audio.")
                return None

            wav = np.array(wav)
            wav = wav / (np.max(np.abs(wav)) + 1e-8) * 0.95

            if self.model_type == 'rvc':
                sample_rate = rvc_sr
            else:
                sample_rate = getattr(self.xtts_model.synthesizer, 'output_sample_rate', 22050)
            
            temp_wav = "temp_output.wav"
            sf.write(temp_wav, wav, sample_rate)
            
            if enable_preview:
                return self.preview_and_save(temp_wav, output_path, output_format)
            else:
                output_path = Path(output_path)
                if output_format.lower() == "mp3" and FFMPEG_AVAILABLE:
                    output_path = output_path.with_suffix('.mp3')
                    audio_segment = AudioSegment.from_wav(temp_wav)
                    audio_segment.export(str(output_path), format="mp3", bitrate="192k")
                    Path(temp_wav).unlink()
                else:
                    output_path = output_path.with_suffix(f'.{output_format}')
                    Path(temp_wav).rename(output_path)
                
                print(f"\n✓ Output saved: {output_path}")
                return str(output_path)
            
        except Exception as e:
            print(f"\nSynthesis error: {e}")
            import traceback
            traceback.print_exc()
            return None

    def compute_and_save_reference_embedding(self, speaker_name, reference_audio_path):
        """Compute and save a reference embedding for a speaker."""
        if self.speaker_encoder is None:
            print("Error: Load a model first.")
            return False

        if speaker_name not in self.model_info.get('speaker_to_id', {}):
            print(f"Error: Speaker '{speaker_name}' not in the loaded model.")
            return False

        print(f"\nComputing reference embedding for '{speaker_name}'...")
        
        # Prepare audio and extract mel spectrogram
        ref_audio, ref_sr = self.prepare_reference_audio(reference_audio_path, clean_audio=True)
        if ref_audio is None:
            print("Failed to prepare audio.")
            return False

        mel_spec = self.extract_mel_spectrogram(ref_audio, ref_sr)
        if mel_spec is None:
            return False

        # Get speaker embedding
        with torch.no_grad():
            embedding = self.speaker_encoder(mel_spec.to(self.device), return_embedding=True)
            embedding = F.normalize(embedding, p=2, dim=1)

        # Store the embedding
        self.reference_embeddings[speaker_name] = embedding.cpu()
        print(f"✓ Embedding for '{speaker_name}' computed and stored.")

        # Save the updated embeddings back to the model file
        try:
            checkpoint = torch.load(self.current_model_path, map_location='cpu')
            
            # Convert tensor embeddings to a serializable format if needed
            serializable_embeddings = {name: emb.tolist() for name, emb in self.reference_embeddings.items()}
            checkpoint['reference_embeddings'] = serializable_embeddings
            
            torch.save(checkpoint, self.current_model_path)
            print(f"✓ Reference embeddings saved to '{self.current_model_path.name}'")
            return True
        except Exception as e:
            print(f"Error saving reference embeddings to model file: {e}")
            return False

    def extract_mel_spectrogram(self, audio, sr):
        """Helper to extract mel spectrogram from audio data."""
        mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=80, n_fft=1024, hop_length=256, win_length=1024)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        mel_spec_norm = (mel_spec_db - mel_spec_db.min()) / (mel_spec_db.max() - mel_spec_db.min())
        return torch.FloatTensor(mel_spec_norm).unsqueeze(0)


def main():
    """Interactive menu"""
    print("="*70)
    print("VOICE CLONING SYSTEM")
    print("="*70)
    
    cloner = VoiceCloner()
    
    while True:
        print("\n" + "="*70)
        print("MAIN MENU")
        print("="*70)
        print("1. List models")
        print("2. Load fine-tuned model")
        print("3. Clone voice (from audio file)")
        print("4. Synthesize with Saved Speaker")
        print("5. Create/Update Saved Speaker Embedding")
        print("6. Model info")
        print("7. Exit")
        
        choice = input("\nChoice (1-7): ").strip()
        
        if choice == "1":
            models = cloner.list_available_models()
            if not models:
                print("\nNo models found")
            else:
                print(f"\nFound {len(models)} models:")
                xtts_models = [m for m in models if m['type'] == 'xtts']
                rvc_models = [m for m in models if m['type'] == 'rvc']

                if xtts_models:
                    print("\n--- XTTS Speaker Encoder Models ---")
                    for m in xtts_models: print(f"  - {m['name']} ({m['size_mb']:.1f} MB)")
                if rvc_models:
                    print("\n--- RVC Models ---")
                    for m in rvc_models: print(f"  - {m['name']} ({m['size_mb']:.1f} MB)")
        
        elif choice == "2":
            model_path = input("\nModel path (.pth file): ").strip()
            cloner.load_model(model_path)
        
        elif choice == "3":
            if cloner.speaker_encoder is None:
                print("\nError: Load a model first")
                continue
            
            text = input("\nText to synthesize: ").strip()
            if not text:
                continue
            
            reference_path = input("Reference audio path: ").strip()
            if not Path(reference_path).exists():
                print("Invalid path")
                continue
            
            if Path(reference_path).is_dir():
                print("Error: Path is a directory")
                continue
            
            output_name = input("Output filename (e.g., 'cloned_voice.mp3'): ").strip()
            output_name = output_name if output_name else "cloned_voice"
            
            # Call synthesize with reference_audio_path
            result = cloner.synthesize(
                text=text,
                reference_audio_path=reference_path,
                output_path=output_name,
            )
            
            if result and result != "regenerate":
                print(f"\n✓ Success: {result}")
        
        elif choice == "4": # Synthesize with Saved Speaker
            if cloner.model_type != 'xtts' or not cloner.reference_embeddings:
                print("\nError: No saved speaker embeddings found. Use option 5 to create one.")
                continue

            print("\nAvailable saved speakers:")
            speaker_names = list(cloner.reference_embeddings.keys())
            for i, name in enumerate(speaker_names, 1):
                print(f"{i}. {name}")

            try:
                speaker_choice = int(input("Choose a speaker: ").strip()) - 1
                if not (0 <= speaker_choice < len(speaker_names)):
                    raise ValueError
                selected_speaker = speaker_names[speaker_choice]
                speaker_embedding = cloner.reference_embeddings[selected_speaker].to(cloner.device)
            except (ValueError, IndexError):
                print("Invalid choice.")
                continue

            text = input("\nText to synthesize: ").strip()
            if not text:
                continue

            output_name = input(f"Output filename (e.g., '{selected_speaker}_voice.mp3'): ").strip()
            output_name = output_name if output_name else f"{selected_speaker}_voice"

            # Call synthesize with speaker_embedding
            result = cloner.synthesize(
                text=text,
                speaker_embedding=speaker_embedding,
                reference_audio_path=None, # Explicitly set to None
                output_path=output_name,
            )

            if result and result != "regenerate":
                print(f"\n✓ Success: {result}")

        elif choice == "5": # Create/Update Saved Speaker Embedding
            if cloner.model_type != 'xtts' or cloner.speaker_encoder is None:
                print("\nError: Load a model first.")
                continue
            
            speaker_name = input("Enter speaker name to create/update embedding for: ").strip()
            if not speaker_name:
                continue
            
            reference_path = input(f"Reference audio path for '{speaker_name}': ").strip()
            if not Path(reference_path).exists():
                print("Invalid path.")
                continue

            cloner.compute_and_save_reference_embedding(speaker_name, reference_path)

        elif choice == "6":
            if cloner.model_type is None:
                print("\nNo model loaded")
            elif cloner.model_type == 'xtts':
                print(f"\nModel: {Path(cloner.current_model_path).name}")
                print(f"Type: XTTS Speaker Encoder")
                print(f"Speakers: {cloner.model_info['num_speakers']}")
            elif cloner.model_type == 'rvc':
                print(f"\nModel: {Path(cloner.current_model_path).name}")
                print(f"Type: RVC")
        
        elif choice == "7":
            print("\nGoodbye!")
            break
        
        else:
            print("Invalid choice")


if __name__ == "__main__":
    main()