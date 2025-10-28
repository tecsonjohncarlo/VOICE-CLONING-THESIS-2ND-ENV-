"""
Fish Speech TTS Wrapper
Provides an XTTS-compatible interface for Fish Speech (OpenAudio)
"""

import subprocess
import numpy as np
import soundfile as sf
from pathlib import Path
import tempfile
import shutil
import json
import sys
import time
from typing import Optional, Union, List
import warnings

warnings.filterwarnings('ignore')


class FishSpeechTTS:
    """
    Wrapper for Fish Speech TTS to provide XTTS-like interface
    
    Fish Speech uses a 3-stage pipeline:
    1. Extract VQ tokens from reference audio
    2. Generate semantic tokens from text
    3. Synthesize audio from semantic tokens
    """
    
    def __init__(self, model_path="checkpoints/openaudio-s1", device="cuda"):
        """
        Initialize Fish Speech TTS
        
        Args:
            model_path: Path to Fish Speech model directory
            device: Device to use ('cuda' or 'cpu')
        """
        self.model_path = Path(model_path)
        self.device = device
        self.codec_path = self.model_path / "codec.pth"
        self.temp_dir = Path(tempfile.mkdtemp(prefix="fish_speech_"))
        
        # Validate model exists
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Model directory not found: {self.model_path}\n"
                f"Download with: hf download fishaudio/openaudio-s1-mini --local-dir {self.model_path}"
            )
        
        if not self.codec_path.exists():
            raise FileNotFoundError(f"Codec not found at {self.codec_path}")
        
        # Cache for VQ tokens to avoid re-extraction
        self.vq_cache = {}
        
        print(f"Fish Speech TTS initialized")
        print(f"  Model: {model_path}")
        print(f"  Device: {device}")
        print(f"  Temp dir: {self.temp_dir}")
    
    def extract_vq_tokens(self, audio_path: Union[str, Path], 
                         output_name: str = "reference",
                         use_cache: bool = True) -> Path:
        """
        Stage 1: Extract VQ tokens from reference audio
        
        Args:
            audio_path: Path to reference audio file
            output_name: Name for output file
            use_cache: Use cached tokens if available
            
        Returns:
            Path to generated .npy file containing VQ tokens
        """
        audio_path = Path(audio_path)
        
        if not audio_path.exists():
            raise FileNotFoundError(f"Reference audio not found: {audio_path}")
        
        # Check cache
        cache_key = str(audio_path.absolute())
        if use_cache and cache_key in self.vq_cache:
            print(f"  Using cached VQ tokens for {audio_path.name}")
            return self.vq_cache[cache_key]
        
        output_path = self.temp_dir / f"{output_name}.npy"
        
        print(f"  Extracting VQ tokens from {audio_path.name}...")
        
        cmd = [sys.executable,
        "fish_speech/models/dac/inference.py", "-i", str(audio_path),  # ← CORRECT!
        "--checkpoint-path", str(self.codec_path)
        ]

        
        try:
            result = subprocess.run(
                cmd, 
                capture_output=True, 
                text=True,
                timeout=60
            )
            
            if result.returncode != 0:
                raise RuntimeError(f"VQ extraction failed: {result.stderr}")
            
            # Move generated fake.npy to our temp directory
            if Path("fake.npy").exists():
                shutil.move("fake.npy", output_path)
            else:
                raise RuntimeError("VQ tokens not generated (fake.npy not found)")
            
            # Clean up fake.wav if it exists
            if Path("fake.wav").exists():
                Path("fake.wav").unlink()
            
            # Cache the result
            if use_cache:
                self.vq_cache[cache_key] = output_path
            
            print(f"  ✓ VQ tokens extracted: {output_path.name}")
            return output_path
            
        except subprocess.TimeoutExpired:
            raise RuntimeError("VQ extraction timed out (>60s)")
        except Exception as e:
            raise RuntimeError(f"VQ extraction error: {e}")
    
    def generate_semantic_tokens(self, text: str, 
                                 vq_tokens_path: Optional[Path] = None,
                                 prompt_text: Optional[str] = None, 
                                 compile_mode: bool = True,
                                 half_precision: bool = False) -> Path:
        """
        Stage 2: Generate semantic tokens from text
        
        Args:
            text: Text to synthesize
            vq_tokens_path: Path to VQ tokens from reference audio (optional)
            prompt_text: Transcript of reference audio (optional, improves quality)
            compile_mode: Use compilation for faster inference (~10x speedup)
            half_precision: Use FP16 for GPUs without bf16 support
            
        Returns:
            Path to generated codes_N.npy file
        """
        print(f"  Generating semantic tokens...")
        print(f"    Text: '{text[:50]}{'...' if len(text) > 50 else ''}'")
        
        cmd = [sys.executable,
         "fish_speech/models/text2semantic/inference.py", "--text",
          text]
        
        if vq_tokens_path:
            cmd.extend(["--prompt-tokens", str(vq_tokens_path)])
        
        if prompt_text:
            cmd.extend(["--prompt-text", prompt_text])
        
        if compile_mode:
            cmd.append("--compile")
        
        if half_precision:
            cmd.append("--half")
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=120
            )
            
            if result.returncode != 0:
                raise RuntimeError(f"Semantic generation failed: {result.stderr}")
            
            # Find the generated codes file
            codes_files = list(Path(".").glob("codes_*.npy"))
            if not codes_files:
                raise RuntimeError("No semantic tokens generated (codes_*.npy not found)")
            
            # Get the most recent codes file
            latest_codes = max(codes_files, key=lambda p: p.stat().st_mtime)
            
            # Move to temp directory
            codes_path = self.temp_dir / latest_codes.name
            shutil.move(latest_codes, codes_path)
            
            print(f"  ✓ Semantic tokens generated: {codes_path.name}")
            return codes_path
            
        except subprocess.TimeoutExpired:
            raise RuntimeError("Semantic generation timed out (>120s)")
        except Exception as e:
            raise RuntimeError(f"Semantic generation error: {e}")
    
    def synthesize_audio(self, semantic_tokens_path: Path, 
                        output_path: Union[str, Path] = "output.wav") -> Path:
        """
        Stage 3: Generate audio from semantic tokens
        
        Args:
            semantic_tokens_path: Path to codes_N.npy file
            output_path: Output audio file path
            
        Returns:
            Path to generated audio file
        """
        output_path = Path(output_path)
        
        print(f"  Synthesizing audio...")
        
        cmd = [sys.executable, 
        "fish_speech/models/dac/inference.py", "-i", str(audio_path), 
        "--checkpoint-path", str(self.codec_path)
        ]
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=60
            )
            
            if result.returncode != 0:
                raise RuntimeError(f"Audio synthesis failed: {result.stderr}")
            
            # Move generated fake.wav to output path
            if Path("fake.wav").exists():
                shutil.move("fake.wav", output_path)
            else:
                raise RuntimeError("Audio not generated (fake.wav not found)")
            
            print(f"  ✓ Audio synthesized: {output_path.name}")
            return output_path
            
        except subprocess.TimeoutExpired:
            raise RuntimeError("Audio synthesis timed out (>60s)")
        except Exception as e:
            raise RuntimeError(f"Audio synthesis error: {e}")
    
    def tts(self, text: str, 
            speaker_wav: Optional[Union[str, Path]] = None,
            output_path: Union[str, Path] = "output.wav",
            prompt_text: Optional[str] = None,
            language: str = "en",
            compile_mode: bool = True,
            half_precision: bool = False,
            use_cache: bool = True) -> np.ndarray:
        """
        XTTS-compatible interface for Fish Speech
        
        Args:
            text: Text to synthesize (supports emotion markers like "(excited)")
            speaker_wav: Reference audio file path (optional, random voice if None)
            output_path: Output audio file path
            prompt_text: Transcript of reference audio (improves quality)
            language: Language code (kept for compatibility, auto-detected)
            compile_mode: Use compilation for faster inference
            half_precision: Use FP16 for GPUs without bf16 support
            use_cache: Cache VQ tokens for reuse
            
        Returns:
            numpy array of audio samples
            
        Example:
            >>> tts = FishSpeechTTS()
            >>> audio = tts.tts(
            ...     text="(excited) Hello! This is amazing!",
            ...     speaker_wav="reference.wav",
            ...     output_path="output.wav"
            ... )
        """
        start_time = time.time()
        output_path = Path(output_path)
        
        print(f"\n{'='*70}")
        print("FISH SPEECH SYNTHESIS")
        print(f"{'='*70}")
        
        try:
            vq_tokens = None
            
            # Stage 1: Extract VQ tokens from reference (if provided)
            if speaker_wav:
                print("\nStage 1/3: Extracting VQ tokens from reference audio")
                vq_tokens = self.extract_vq_tokens(
                    speaker_wav, 
                    use_cache=use_cache
                )
            else:
                print("\nStage 1/3: Skipped (using random voice)")
            
            # Stage 2: Generate semantic tokens
            print("\nStage 2/3: Generating semantic tokens from text")
            semantic_tokens = self.generate_semantic_tokens(
                text=text,
                vq_tokens_path=vq_tokens,
                prompt_text=prompt_text,
                compile_mode=compile_mode,
                half_precision=half_precision
            )
            
            # Stage 3: Synthesize audio
            print("\nStage 3/3: Synthesizing audio")
            output_file = self.synthesize_audio(semantic_tokens, output_path)
            
            # Load audio
            audio, sr = sf.read(output_file)
            
            # Cleanup temporary files (but keep cached VQ tokens)
            if semantic_tokens.exists():
                semantic_tokens.unlink()
            
            elapsed = time.time() - start_time
            duration = len(audio) / sr
            rtf = elapsed / duration  # Real-time factor
            
            print(f"\n{'='*70}")
            print("SYNTHESIS COMPLETE")
            print(f"{'='*70}")
            print(f"  Output: {output_path}")
            print(f"  Duration: {duration:.2f}s")
            print(f"  Time taken: {elapsed:.2f}s")
            print(f"  Real-time factor: {rtf:.2f}x")
            print(f"  Sample rate: {sr} Hz")
            
            return audio
            
        except Exception as e:
            print(f"\n{'='*70}")
            print("SYNTHESIS FAILED")
            print(f"{'='*70}")
            print(f"Error: {e}")
            raise
    
    def tts_with_emotions(self, text: str, emotions: List[str],
                         speaker_wav: Optional[Union[str, Path]] = None,
                         output_path: Union[str, Path] = "output.wav",
                         **kwargs) -> np.ndarray:
        """
        Convenience method for adding emotions to text
        
        Args:
            text: Base text without emotion markers
            emotions: List of emotions to apply (e.g., ['excited', 'laughing'])
            speaker_wav: Reference audio file path
            output_path: Output audio file path
            **kwargs: Additional arguments passed to tts()
            
        Returns:
            numpy array of audio samples
            
        Example:
            >>> audio = tts.tts_with_emotions(
            ...     text="Hello! This is amazing!",
            ...     emotions=['excited', 'laughing'],
            ...     speaker_wav="reference.wav"
            ... )
            # Generates: "(excited) Hello! This is amazing! (laughing)"
        """
        # Add emotion markers to text
        emotional_text = f"({emotions[0]}) {text}"
        if len(emotions) > 1:
            emotional_text += f" ({emotions[1]})"
        
        return self.tts(
            text=emotional_text,
            speaker_wav=speaker_wav,
            output_path=output_path,
            **kwargs
        )
    
    def clear_cache(self):
        """Clear VQ token cache"""
        self.vq_cache.clear()
        print("VQ token cache cleared")
    
    def cleanup(self):
        """Clean up temporary directory"""
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
            print(f"Cleaned up temp directory: {self.temp_dir}")
    
    def __del__(self):
        """Cleanup on deletion"""
        try:
            self.cleanup()
        except:
            pass
    
    def get_available_emotions(self) -> dict:
        """
        Get available emotion markers for Fish Speech
        
        Returns:
            Dictionary of emotion categories and their markers
        """
        return {
            'basic': [
                'angry', 'sad', 'excited', 'surprised', 'satisfied', 'delighted',
                'scared', 'worried', 'upset', 'nervous', 'frustrated', 'depressed',
                'empathetic', 'embarrassed', 'disgusted', 'moved', 'proud', 'relaxed',
                'grateful', 'confident', 'interested', 'curious', 'confused', 'joyful'
            ],
            'advanced': [
                'disdainful', 'unhappy', 'anxious', 'hysterical', 'indifferent',
                'impatient', 'guilty', 'scornful', 'panicked', 'furious', 'reluctant',
                'keen', 'disapproving', 'negative', 'denying', 'astonished', 'serious',
                'sarcastic', 'conciliative', 'comforting', 'sincere', 'sneering',
                'hesitating', 'yielding', 'painful', 'awkward', 'amused'
            ],
            'tones': [
                'in a hurry tone', 'shouting', 'screaming', 'whispering', 'soft tone'
            ],
            'effects': [
                'laughing', 'chuckling', 'sobbing', 'crying loudly', 'sighing',
                'panting', 'groaning', 'crowd laughing', 'background laughter',
                'audience laughing'
            ]
        }
    
    def print_emotion_guide(self):
        """Print guide for using emotions"""
        emotions = self.get_available_emotions()
        
        print("\n" + "="*70)
        print("FISH SPEECH EMOTION GUIDE")
        print("="*70)
        
        print("\nBasic Emotions:")
        for emotion in emotions['basic']:
            print(f"  ({emotion})")
        
        print("\nAdvanced Emotions:")
        for emotion in emotions['advanced']:
            print(f"  ({emotion})")
        
        print("\nTone Markers:")
        for tone in emotions['tones']:
            print(f"  ({tone})")
        
        print("\nSpecial Effects:")
        for effect in emotions['effects']:
            print(f"  ({effect})")
        
        print("\nUsage Examples:")
        print("  '(excited) Hello! (laughing) This is great!'")
        print("  '(whispering) Can you hear me? (soft tone) I am here.'")
        print("  '(angry) Stop! (shouting) I said stop!'")
        print("\nNote: Emotions work best with English, Chinese, and Japanese")
        print("="*70)


def main():
    """Interactive demo"""
    print("="*70)
    print("FISH SPEECH TTS WRAPPER - DEMO")
    print("="*70)
    
    # Initialize
    try:
        tts = FishSpeechTTS(model_path="checkpoints/openaudio-s1-mini")
    except FileNotFoundError as e:
        print(f"\nError: {e}")
        print("\nPlease download the model first:")
        print("  hf download fishaudio/openaudio-s1-mini --local-dir checkpoints/openaudio-s1-mini")
        return
    
    while True:
        print("\n" + "="*70)
        print("MENU:")
        print("1. Basic synthesis")
        print("2. Synthesis with emotions")
        print("3. View emotion guide")
        print("4. Clear cache")
        print("5. Exit")
        
        choice = input("\nChoice (1-5): ").strip()
        
        if choice == "1":
            text = input("\nEnter text: ").strip()
            if not text:
                print("Text cannot be empty!")
                continue
            
            ref_audio = input("Reference audio path (or press Enter for random voice): ").strip()
            ref_audio = ref_audio if ref_audio else None
            
            output = input("Output path (default: output.wav): ").strip()
            output = output if output else "output.wav"
            
            try:
                audio = tts.tts(
                    text=text,
                    speaker_wav=ref_audio,
                    output_path=output
                )
                print(f"\n✓ Success! Audio saved to: {output}")
            except Exception as e:
                print(f"\n✗ Error: {e}")
        
        elif choice == "2":
            text = input("\nEnter text: ").strip()
            if not text:
                print("Text cannot be empty!")
                continue
            
            print("\nEnter emotions (comma-separated, e.g., 'excited,laughing'):")
            emotions_str = input("Emotions: ").strip()
            emotions = [e.strip() for e in emotions_str.split(',')]
            
            ref_audio = input("Reference audio path (or press Enter for random voice): ").strip()
            ref_audio = ref_audio if ref_audio else None
            
            output = input("Output path (default: output_emotional.wav): ").strip()
            output = output if output else "output_emotional.wav"
            
            try:
                audio = tts.tts_with_emotions(
                    text=text,
                    emotions=emotions,
                    speaker_wav=ref_audio,
                    output_path=output
                )
                print(f"\n✓ Success! Audio saved to: {output}")
            except Exception as e:
                print(f"\n✗ Error: {e}")
        
        elif choice == "3":
            tts.print_emotion_guide()
        
        elif choice == "4":
            tts.clear_cache()
        
        elif choice == "5":
            print("\nCleaning up...")
            tts.cleanup()
            print("Goodbye!")
            break
        
        else:
            print("Invalid choice!")


if __name__ == "__main__":
    main()
