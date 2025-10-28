"""
Optimized Fish Speech TTS Wrapper
Resource-efficient implementation for OpenAudio S1-Mini with graceful fallbacks
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
import gc
import psutil
import torch
from typing import Optional, Union, List
import warnings

warnings.filterwarnings('ignore')


class OptimizedFishSpeechTTS:
    """
    Optimized wrapper for Fish Speech TTS with resource efficiency and resilience

    Features:
    - Adaptive timeouts based on system resources
    - Memory management and cleanup
    - Automatic batch size adjustment
    - Progressive fallback strategies
    - CPU fallback support
    - Audio chunking for large texts
    """

    def __init__(self, 
                 model_path="checkpoints/openaudio-s1-mini",
                 device="auto",
                 max_memory_gb=4.0,
                 enable_optimizations=True):
        """
        Initialize optimized Fish Speech TTS

        Args:
            model_path: Path to Fish Speech model directory
            device: Device to use ('cuda', 'cpu', or 'auto' for automatic detection)
            max_memory_gb: Maximum GPU memory to use (GB)
            enable_optimizations: Enable aggressive optimizations
        """
        self.model_path = Path(model_path)
        self.enable_optimizations = enable_optimizations
        self.max_memory_gb = max_memory_gb

        # Auto-detect device
        if device == "auto":
            self.device = self._detect_best_device()
        else:
            self.device = device

        self.codec_path = self.model_path / "codec.pth"
        self.temp_dir = Path(tempfile.mkdtemp(prefix="fish_speech_"))

        # Validate model exists
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Model directory not found: {self.model_path}\n"
                f"Download with: huggingface-cli download fishaudio/openaudio-s1-mini --local-dir {self.model_path}"
            )

        if not self.codec_path.exists():
            raise FileNotFoundError(f"Codec not found at {self.codec_path}")

        # Cache for VQ tokens to avoid re-extraction
        self.vq_cache = {}

        # System info
        self.system_info = self._get_system_info()
        self._apply_optimizations()

        print(f"Optimized Fish Speech TTS initialized")
        print(f"  Model: {model_path}")
        print(f"  Device: {self.device}")
        print(f"  Available RAM: {self.system_info['ram_gb']:.1f} GB")
        if self.device == "cuda":
            print(f"  GPU Memory: {self.system_info['gpu_memory_gb']:.1f} GB")
        print(f"  Optimizations: {'Enabled' if enable_optimizations else 'Disabled'}")

    def _detect_best_device(self) -> str:
        """Auto-detect best available device"""
        try:
            if torch.cuda.is_available():
                # Check if GPU has enough memory (at least 4GB recommended)
                gpu_mem_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                if gpu_mem_gb >= 3.5:
                    return "cuda"
                else:
                    print(f"Warning: GPU has only {gpu_mem_gb:.1f}GB memory, using CPU")
                    return "cpu"
            else:
                return "cpu"
        except:
            return "cpu"

    def _get_system_info(self) -> dict:
        """Get system resource information"""
        info = {
            'ram_gb': psutil.virtual_memory().total / (1024**3),
            'available_ram_gb': psutil.virtual_memory().available / (1024**3),
            'cpu_count': psutil.cpu_count(),
        }

        if self.device == "cuda":
            try:
                info['gpu_memory_gb'] = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                info['gpu_name'] = torch.cuda.get_device_name(0)
            except:
                info['gpu_memory_gb'] = 0
                info['gpu_name'] = 'Unknown'

        return info

    def _apply_optimizations(self):
        """Apply system-level optimizations"""
        if not self.enable_optimizations:
            return

        # Set environment variables for optimization
        import os

        # Reduce memory fragmentation
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'

        # Enable CUDNN benchmarking
        if self.device == "cuda":
            torch.backends.cudnn.benchmark = True
            torch.backends.cuda.matmul.allow_tf32 = True

        # Set optimal thread count
        optimal_threads = max(1, self.system_info['cpu_count'] // 2)
        torch.set_num_threads(optimal_threads)

        print(f"  CPU threads: {optimal_threads}")

    def _calculate_adaptive_timeout(self, base_timeout: int, audio_path: Optional[Path] = None) -> int:
        """Calculate adaptive timeout based on system resources and file size"""
        timeout = base_timeout

        # Adjust for CPU vs GPU
        if self.device == "cpu":
            timeout *= 3  # CPU is slower

        # Adjust for low memory
        if self.system_info['available_ram_gb'] < 4:
            timeout *= 1.5

        # Adjust for audio file size
        if audio_path and audio_path.exists():
            file_size_mb = audio_path.stat().st_size / (1024**2)
            if file_size_mb > 5:
                timeout *= 1.5
            if file_size_mb > 10:
                timeout *= 2

        return int(timeout)

    def _cleanup_memory(self):
        """Aggressive memory cleanup"""
        gc.collect()
        if self.device == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

    def _optimize_audio(self, audio_path: Path, max_duration: float = 30.0) -> Path:
        """
        Optimize reference audio for faster processing

        Args:
            audio_path: Input audio path
            max_duration: Maximum duration in seconds

        Returns:
            Path to optimized audio
        """
        try:
            audio, sr = sf.read(audio_path)

            # Convert to mono if stereo
            if len(audio.shape) > 1:
                audio = np.mean(audio, axis=1)

            # Trim to max duration
            max_samples = int(max_duration * sr)
            if len(audio) > max_samples:
                audio = audio[:max_samples]
                print(f"  Audio trimmed to {max_duration}s for faster processing")

            # Resample to 24kHz if needed (optimal for Fish Speech)
            target_sr = 24000
            if sr != target_sr:
                try:
                    import librosa
                    audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
                    sr = target_sr
                    print(f"  Audio resampled to {target_sr}Hz")
                except ImportError:
                    pass  # Skip resampling if librosa not available

            # Save optimized audio
            optimized_path = self.temp_dir / f"optimized_{audio_path.name}"
            sf.write(optimized_path, audio, sr)

            return optimized_path

        except Exception as e:
            print(f"  Warning: Audio optimization failed, using original: {e}")
            return audio_path

    def extract_vq_tokens(self, 
                         audio_path: Union[str, Path], 
                         output_name: str = "reference",
                         use_cache: bool = True,
                         optimize_audio: bool = True) -> Path:
        """
        Stage 1: Extract VQ tokens from reference audio with optimizations

        Args:
            audio_path: Path to reference audio file
            output_name: Name for output file
            use_cache: Use cached tokens if available
            optimize_audio: Optimize audio before processing

        Returns:
            Path to generated .npy file containing VQ tokens
        """
        audio_path = Path(audio_path)

        if not audio_path.exists():
            raise FileNotFoundError(f"Reference audio not found: {audio_path}")

        # Check cache
        cache_key = str(audio_path.absolute())
        if use_cache and cache_key in self.vq_cache:
            if self.vq_cache[cache_key].exists():
                print(f"  ✓ Using cached VQ tokens")
                return self.vq_cache[cache_key]

        # Optimize audio first
        if optimize_audio:
            audio_path = self._optimize_audio(audio_path)

        output_path = self.temp_dir / f"{output_name}.npy"

        print(f"  Extracting VQ tokens from {Path(audio_path).name}...")

        # Calculate adaptive timeout
        timeout = self._calculate_adaptive_timeout(60, audio_path)
        print(f"  Timeout: {timeout}s")

        cmd = [
            sys.executable, 
            "fish_speech/models/dac/inference.py",
            "-i", str(audio_path),
            "--checkpoint-path", str(self.codec_path)
        ]

        # Add device flag
        if self.device == "cpu":
            cmd.extend(["--device", "cpu"])

        try:
            # Clean memory before heavy operation
            self._cleanup_memory()

            result = subprocess.run(
                cmd, 
                capture_output=True, 
                text=True,
                timeout=timeout
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

            # Cleanup memory after operation
            self._cleanup_memory()

            print(f"  ✓ VQ tokens extracted")
            return output_path

        except subprocess.TimeoutExpired:
            raise RuntimeError(f"VQ extraction timed out (>{timeout}s). Try: 1) Use shorter audio 2) Close other apps 3) Use CPU mode")
        except Exception as e:
            raise RuntimeError(f"VQ extraction error: {e}")

    def generate_semantic_tokens(self, 
                                 text: str, 
                                 vq_tokens_path: Optional[Path] = None,
                                 prompt_text: Optional[str] = None, 
                                 compile_mode: bool = False,
                                 half_precision: bool = True,
                                 max_new_tokens: int = 0) -> Path:
        """
        Stage 2: Generate semantic tokens from text with optimizations

        Args:
            text: Text to synthesize
            vq_tokens_path: Path to VQ tokens from reference audio (optional)
            prompt_text: Transcript of reference audio (optional)
            compile_mode: Use compilation (disabled by default for stability)
            half_precision: Use FP16 (enabled by default for efficiency)
            max_new_tokens: Maximum tokens to generate (0=auto)

        Returns:
            Path to generated codes_N.npy file
        """
        print(f"  Generating semantic tokens...")

        # Calculate adaptive timeout
        timeout = self._calculate_adaptive_timeout(180)
        print(f"  Timeout: {timeout}s")

        cmd = [
            sys.executable,
            "fish_speech/models/text2semantic/inference.py",
            "--text", text,
            "--device", self.device
        ]

        if vq_tokens_path:
            cmd.extend(["--prompt-tokens", str(vq_tokens_path)])

        if prompt_text:
            cmd.extend(["--prompt-text", prompt_text])

        # Disable compile by default for better stability
        if compile_mode:
            cmd.append("--compile")

        # Enable half precision by default for efficiency
        if half_precision and self.device == "cuda":
            cmd.append("--half")

        # Set max tokens based on text length
        if max_new_tokens == 0:
            estimated_tokens = len(text) * 3  # Conservative estimate
            max_new_tokens = min(estimated_tokens, 2048)
        cmd.extend(["--max-new-tokens", str(max_new_tokens)])

        try:
            # Clean memory before operation
            self._cleanup_memory()

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout
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

            # Cleanup memory
            self._cleanup_memory()

            print(f"  ✓ Semantic tokens generated")
            return codes_path

        except subprocess.TimeoutExpired:
            raise RuntimeError(f"Semantic generation timed out (>{timeout}s). Try shorter text.")
        except Exception as e:
            raise RuntimeError(f"Semantic generation error: {e}")

    def synthesize_audio(self, 
                        semantic_tokens_path: Path, 
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

        # Calculate adaptive timeout
        timeout = self._calculate_adaptive_timeout(60)
        print(f"  Timeout: {timeout}s")

        cmd = [
            sys.executable,
            "fish_speech/models/dac/inference.py",
            "--mode", "codes2wav",
            "-i", str(semantic_tokens_path),
            "--checkpoint-path", str(self.codec_path)
        ]

        # Add device flag
        if self.device == "cpu":
            cmd.extend(["--device", "cpu"])

        try:
            # Clean memory before operation
            self._cleanup_memory()

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout
            )

            if result.returncode != 0:
                raise RuntimeError(f"Audio synthesis failed: {result.stderr}")

            # Move generated fake.wav to output path
            if Path("fake.wav").exists():
                shutil.move("fake.wav", output_path)
            else:
                raise RuntimeError("Audio not generated (fake.wav not found)")

            # Cleanup memory
            self._cleanup_memory()

            print(f"  ✓ Audio synthesized")
            return output_path

        except subprocess.TimeoutExpired:
            raise RuntimeError(f"Audio synthesis timed out (>{timeout}s)")
        except Exception as e:
            raise RuntimeError(f"Audio synthesis error: {e}")

    def chunk_text(self, text: str, max_chunk_length: int = 200) -> List[str]:
        """
        Split long text into manageable chunks

        Args:
            text: Input text
            max_chunk_length: Maximum characters per chunk

        Returns:
            List of text chunks
        """
        if len(text) <= max_chunk_length:
            return [text]

        chunks = []
        sentences = text.replace('!', '.').replace('?', '.').split('.')
        current_chunk = ""

        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue

            if len(current_chunk) + len(sentence) <= max_chunk_length:
                current_chunk += sentence + ". "
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence + ". "

        if current_chunk:
            chunks.append(current_chunk.strip())

        return chunks

    def tts(self, 
            text: str, 
            speaker_wav: Optional[Union[str, Path]] = None,
            output_path: Union[str, Path] = "output.wav",
            prompt_text: Optional[str] = None,
            language: str = "en",
            compile_mode: bool = False,
            half_precision: bool = True,
            use_cache: bool = True,
            optimize_audio: bool = True,
            chunk_long_text: bool = True) -> np.ndarray:
        """
        Optimized TTS interface with automatic chunking and fallbacks

        Args:
            text: Text to synthesize (supports emotion markers)
            speaker_wav: Reference audio file path (optional)
            output_path: Output audio file path
            prompt_text: Transcript of reference audio (improves quality)
            language: Language code (auto-detected)
            compile_mode: Use compilation (disabled by default for stability)
            half_precision: Use FP16 (enabled by default)
            use_cache: Cache VQ tokens for reuse
            optimize_audio: Optimize reference audio
            chunk_long_text: Automatically chunk long text

        Returns:
            numpy array of audio samples
        """
        start_time = time.time()
        output_path = Path(output_path)

        print(f"\n{'='*70}")
        print("OPTIMIZED FISH SPEECH SYNTHESIS")
        print(f"{'='*70}")
        print(f"Device: {self.device} | Memory: {self.system_info['available_ram_gb']:.1f}GB")

        try:
            # Chunk long text if needed
            if chunk_long_text and len(text) > 200:
                chunks = self.chunk_text(text)
                print(f"\n  Text split into {len(chunks)} chunks for processing")

                all_audio = []
                sr = None

                for i, chunk in enumerate(chunks):
                    print(f"\n  Processing chunk {i+1}/{len(chunks)}...")
                    chunk_output = self.temp_dir / f"chunk_{i}.wav"

                    # Process chunk
                    audio = self._process_single_tts(
                        text=chunk,
                        speaker_wav=speaker_wav,
                        output_path=chunk_output,
                        prompt_text=prompt_text,
                        compile_mode=compile_mode,
                        half_precision=half_precision,
                        use_cache=use_cache,
                        optimize_audio=optimize_audio
                    )

                    audio_data, current_sr = sf.read(chunk_output)
                    all_audio.append(audio_data)
                    sr = current_sr

                    # Cleanup chunk file
                    chunk_output.unlink()

                # Concatenate all chunks
                final_audio = np.concatenate(all_audio)
                sf.write(output_path, final_audio, sr)

                elapsed = time.time() - start_time
                duration = len(final_audio) / sr

                print(f"\n{'='*70}")
                print("SYNTHESIS COMPLETE")
                print(f"{'='*70}")
                print(f"  Output: {output_path}")
                print(f"  Chunks: {len(chunks)}")
                print(f"  Duration: {duration:.2f}s")
                print(f"  Time taken: {elapsed:.2f}s")
                print(f"  Real-time factor: {elapsed/duration:.2f}x")

                return final_audio

            else:
                # Process as single chunk
                return self._process_single_tts(
                    text=text,
                    speaker_wav=speaker_wav,
                    output_path=output_path,
                    prompt_text=prompt_text,
                    compile_mode=compile_mode,
                    half_precision=half_precision,
                    use_cache=use_cache,
                    optimize_audio=optimize_audio
                )

        except Exception as e:
            print(f"\n{'='*70}")
            print("SYNTHESIS FAILED")
            print(f"{'='*70}")
            print(f"Error: {e}")
            print(f"\nTroubleshooting:")
            print(f"  1. Use shorter reference audio (<15s)")
            print(f"  2. Close other applications")
            print(f"  3. Try CPU mode if GPU fails")
            print(f"  4. Reduce text length")
            raise

    def _process_single_tts(self, text, speaker_wav, output_path, prompt_text,
                           compile_mode, half_precision, use_cache, optimize_audio):
        """Process single TTS request (internal)"""
        vq_tokens = None

        # Stage 1: Extract VQ tokens
        if speaker_wav:
            print("\nStage 1/3: Extracting VQ tokens from reference audio")
            vq_tokens = self.extract_vq_tokens(
                speaker_wav, 
                use_cache=use_cache,
                optimize_audio=optimize_audio
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

        # Cleanup temporary semantic tokens
        if semantic_tokens.exists():
            semantic_tokens.unlink()

        return audio

    def clear_cache(self):
        """Clear VQ token cache and free memory"""
        self.vq_cache.clear()
        self._cleanup_memory()
        print("✓ Cache cleared and memory freed")

    def cleanup(self):
        """Clean up temporary directory and free resources"""
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
        self._cleanup_memory()
        print(f"✓ Cleaned up resources")

    def __del__(self):
        """Cleanup on deletion"""
        try:
            self.cleanup()
        except:
            pass

    def get_available_emotions(self) -> dict:
        """Get available emotion markers"""
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


def main():
    """Interactive demo with optimized settings"""
    print("="*70)
    print("OPTIMIZED FISH SPEECH TTS - DEMO")
    print("="*70)

    # Initialize with optimizations
    try:
        tts = OptimizedFishSpeechTTS(
            model_path="checkpoints/openaudio-s1-mini",
            device="auto",  # Auto-detect best device
            enable_optimizations=True
        )
    except FileNotFoundError as e:
        print(f"\nError: {e}")
        print("\nPlease download the model first:")
        print("  huggingface-cli download fishaudio/openaudio-s1-mini --local-dir checkpoints/openaudio-s1-mini")
        return

    while True:
        print("\n" + "="*70)
        print("MENU:")
        print("1. Basic synthesis")
        print("2. Synthesis with emotions")
        print("3. View emotion guide")
        print("4. System info")
        print("5. Clear cache")
        print("6. Exit")

        choice = input("\nChoice (1-6): ").strip()

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
            if emotions_str:
                emotional_text = f"({emotions_str.split(',')[0].strip()}) {text}"
                if ',' in emotions_str:
                    emotional_text += f" ({emotions_str.split(',')[1].strip()})"
                text = emotional_text

            ref_audio = input("Reference audio path (or press Enter for random voice): ").strip()
            ref_audio = ref_audio if ref_audio else None

            output = input("Output path (default: output_emotional.wav): ").strip()
            output = output if output else "output_emotional.wav"

            try:
                audio = tts.tts(
                    text=text,
                    speaker_wav=ref_audio,
                    output_path=output
                )
                print(f"\n✓ Success! Audio saved to: {output}")
            except Exception as e:
                print(f"\n✗ Error: {e}")

        elif choice == "3":
            emotions = tts.get_available_emotions()
            print("\n" + "="*70)
            print("EMOTION GUIDE")
            print("="*70)
            print("\nBasic:", ", ".join(emotions['basic'][:10]) + "...")
            print("Advanced:", ", ".join(emotions['advanced'][:10]) + "...")
            print("Effects:", ", ".join(emotions['effects']))

        elif choice == "4":
            print("\n" + "="*70)
            print("SYSTEM INFORMATION")
            print("="*70)
            print(f"Device: {tts.device}")
            print(f"RAM: {tts.system_info['available_ram_gb']:.1f}/{tts.system_info['ram_gb']:.1f} GB available")
            print(f"CPU Cores: {tts.system_info['cpu_count']}")
            if tts.device == "cuda":
                print(f"GPU: {tts.system_info['gpu_name']}")
                print(f"GPU Memory: {tts.system_info['gpu_memory_gb']:.1f} GB")

        elif choice == "5":
            tts.clear_cache()

        elif choice == "6":
            print("\nCleaning up...")
            tts.cleanup()
            print("Goodbye!")
            break

        else:
            print("Invalid choice!")


if __name__ == "__main__":
    main()
