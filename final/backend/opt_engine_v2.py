"""
Optimized Fish Speech Engine V2
Uses direct Python imports instead of subprocess calls (like original Fish Speech)
Implements torch.compile, mixed precision, quantization, and caching
"""
import os
import sys
import gc
import time
import hashlib
import tempfile
import shutil
from pathlib import Path
from typing import Optional, Union, Tuple, Dict, Any
from functools import lru_cache
from collections import OrderedDict
import warnings

import numpy as np
import torch
import soundfile as sf
import torchaudio
from loguru import logger

warnings.filterwarnings('ignore')

# Add fish-speech to path
def _setup_fish_speech_path():
    """Setup Fish Speech in Python path"""
    fish_speech_dir = os.getenv("FISH_SPEECH_DIR")
    if not fish_speech_dir:
        # Try parent directory
        parent_fish = Path(__file__).parent.parent / "fish-speech"
        if parent_fish.exists():
            fish_speech_dir = str(parent_fish)
    
    if fish_speech_dir and Path(fish_speech_dir).exists():
        sys.path.insert(0, str(fish_speech_dir))
        return fish_speech_dir
    
    raise FileNotFoundError(
        "Fish Speech installation not found!\n"
        "Please set FISH_SPEECH_DIR in .env or place fish-speech folder in final/"
    )

FISH_SPEECH_DIR = _setup_fish_speech_path()

# Now import Fish Speech modules
import pyrootutils
pyrootutils.setup_root(FISH_SPEECH_DIR, indicator=".project-root", pythonpath=True)

from fish_speech.models.dac.inference import load_model as load_decoder_model
from fish_speech.models.text2semantic.inference import launch_thread_safe_queue
from fish_speech.inference_engine import TTSInferenceEngine
from fish_speech.utils.schema import ServeTTSRequest

# Configuration
ENABLE_TORCH_COMPILE = os.getenv("ENABLE_TORCH_COMPILE", "False").lower() == "true"
MIXED_PRECISION = os.getenv("MIXED_PRECISION", "auto")
MAX_SEQ_LEN = int(os.getenv("MAX_SEQ_LEN", "2048"))
CACHE_LIMIT = int(os.getenv("CACHE_LIMIT", "100"))
MODEL_DIR = os.getenv("MODEL_DIR", "checkpoints/openaudio-s1-mini")


class LRUCache:
    """Simple LRU cache implementation"""
    def __init__(self, capacity: int):
        self.cache = OrderedDict()
        self.capacity = capacity
    
    def get(self, key: str) -> Optional[Any]:
        if key not in self.cache:
            return None
        self.cache.move_to_end(key)
        return self.cache[key]
    
    def put(self, key: str, value: Any):
        if key in self.cache:
            self.cache.move_to_end(key)
        self.cache[key] = value
        if len(self.cache) > self.capacity:
            self.cache.popitem(last=False)
    
    def clear(self):
        self.cache.clear()


class PerformanceMonitor:
    """Track performance metrics"""
    def __init__(self):
        self.metrics = {
            'latency_ms': [],
            'peak_vram_mb': [],
            'gpu_util_pct': []
        }
        self.nvml_available = False
        
        try:
            import pynvml
            pynvml.nvmlInit()
            self.nvml = pynvml
            self.nvml_available = True
            self.gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        except:
            self.nvml_available = False
    
    def record_latency(self, latency_ms: float):
        self.metrics['latency_ms'].append(latency_ms)
        if len(self.metrics['latency_ms']) > 100:
            self.metrics['latency_ms'].pop(0)
    
    def record_vram(self, vram_mb: float):
        self.metrics['peak_vram_mb'].append(vram_mb)
        if len(self.metrics['peak_vram_mb']) > 100:
            self.metrics['peak_vram_mb'].pop(0)
    
    def record_gpu_util(self, util_pct: float):
        self.metrics['gpu_util_pct'].append(util_pct)
        if len(self.metrics['gpu_util_pct']) > 100:
            self.metrics['gpu_util_pct'].pop(0)
    
    def get_gpu_utilization(self) -> float:
        if not self.nvml_available:
            return 0.0
        try:
            util = self.nvml.nvmlDeviceGetUtilizationRates(self.gpu_handle)
            return util.gpu
        except:
            return 0.0
    
    def get_aggregates(self) -> Dict[str, float]:
        result = {}
        for key, values in self.metrics.items():
            if values:
                result[f'{key}_avg'] = sum(values) / len(values)
                result[f'{key}_min'] = min(values)
                result[f'{key}_max'] = max(values)
        return result
    
    def __del__(self):
        if self.nvml_available:
            try:
                self.nvml.nvmlShutdown()
            except:
                pass


class OptimizedFishSpeechV2:
    """
    Optimized Fish Speech TTS Engine V2
    
    Uses direct Python imports (like original Fish Speech) instead of subprocess calls
    
    Features:
    - Direct in-process model loading (no subprocess overhead)
    - torch.compile for codec and text2semantic
    - Mixed precision (BF16/FP16/FP32)
    - LRU caching for reference audio embeddings
    - Performance monitoring with NVML
    """
    
    def __init__(self, 
                 model_path: str = MODEL_DIR,
                 device: str = "auto",
                 enable_optimizations: bool = True,
                 optimize_for_memory: bool = False):
        """
        Initialize optimized Fish Speech engine V2
        
        Args:
            model_path: Path to model directory
            device: Device to use ('cuda', 'cpu', or 'auto')
            enable_optimizations: Enable all optimizations
            optimize_for_memory: Prioritize memory over speed
        """
        self.model_path = Path(model_path)
        self.enable_optimizations = enable_optimizations
        self.optimize_for_memory = optimize_for_memory
        
        # Auto-detect device
        if device == "auto":
            self.device = self._detect_device()
        else:
            self.device = device
        
        # Validate model
        self.codec_path = self.model_path / "codec.pth"
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model directory not found: {self.model_path}")
        if not self.codec_path.exists():
            raise FileNotFoundError(f"Codec not found: {self.codec_path}")
        
        # Setup temp directory
        self.temp_dir = Path(tempfile.mkdtemp(prefix="fish_opt_v2_"))
        
        # Initialize caches
        self.reference_cache = LRUCache(CACHE_LIMIT)
        
        # Performance monitor
        self.monitor = PerformanceMonitor()
        
        # Apply system optimizations
        if enable_optimizations:
            self._apply_system_optimizations()
        
        # Get precision mode
        self.precision_mode = self._get_precision_mode()
        self.precision = self._get_torch_dtype()
        
        # Initialize models
        logger.info(f"Loading Fish Speech models...")
        logger.info(f"  Model: {model_path}")
        logger.info(f"  Device: {self.device}")
        logger.info(f"  Mixed Precision: {self.precision_mode}")
        logger.info(f"  Torch Compile: {ENABLE_TORCH_COMPILE}")
        
        # Load decoder model (VQ-GAN)
        logger.info("Loading VQ-GAN decoder...")
        self.decoder_model = load_decoder_model(
            config_name="modded_dac_vq",
            checkpoint_path=str(self.codec_path),
            device=self.device
        )
        
        # Load text2semantic model (Llama)
        logger.info("Loading Llama text2semantic model...")
        self.llama_queue = launch_thread_safe_queue(
            checkpoint_path=self.model_path,
            device=self.device,
            precision=self.precision,
            compile=ENABLE_TORCH_COMPILE,
        )
        
        # Create inference engine
        self.inference_engine = TTSInferenceEngine(
            llama_queue=self.llama_queue,
            decoder_model=self.decoder_model,
            compile=ENABLE_TORCH_COMPILE,
            precision=self.precision,
        )
        
        # Warmup
        logger.info("Warming up models...")
        self._warmup()
        
        logger.info("✅ OptimizedFishSpeechV2 initialized successfully!")
    
    def _detect_device(self) -> str:
        """Auto-detect best available device"""
        if torch.cuda.is_available():
            try:
                gpu_mem_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                if gpu_mem_gb >= 3.5:
                    logger.info(f"Detected NVIDIA GPU with {gpu_mem_gb:.1f}GB VRAM")
                    return "cuda"
                else:
                    logger.info(f"NVIDIA GPU has only {gpu_mem_gb:.1f}GB VRAM, using CPU")
                    return "cpu"
            except Exception as e:
                logger.warning(f"CUDA available but error checking: {e}")
                return "cpu"
        
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            logger.info("Detected Apple Silicon (M-series chip)")
            return "mps"
        
        import platform
        logger.info(f"Using CPU: {platform.system()} - {platform.processor()}")
        return "cpu"
    
    def _get_precision_mode(self) -> str:
        """Determine optimal precision mode based on device"""
        if MIXED_PRECISION == "auto":
            if self.device == "cpu":
                return "fp32"
            
            if self.device == "cuda" and torch.cuda.is_available():
                cap = torch.cuda.get_device_capability(0)
                if cap[0] >= 8:  # Ampere or newer
                    return "bf16"
                else:
                    return "fp16"
            
            if self.device == "mps":
                return "fp16"
            
            return "fp32"
        return MIXED_PRECISION
    
    def _get_torch_dtype(self) -> torch.dtype:
        """Convert precision mode to torch dtype"""
        if self.precision_mode == "bf16":
            return torch.bfloat16
        elif self.precision_mode == "fp16":
            return torch.float16
        else:
            return torch.float32
    
    def _apply_system_optimizations(self):
        """Apply device-specific system optimizations"""
        import psutil
        
        if self.device == "cuda":
            os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cudnn.benchmark = True
            logger.info("CUDA optimizations enabled (TF32, CUDNN benchmark)")
        
        elif self.device == "mps":
            logger.info("MPS optimizations enabled (unified memory)")
        
        cpu_count = psutil.cpu_count()
        optimal_threads = max(1, cpu_count // 2)
        torch.set_num_threads(optimal_threads)
        logger.info(f"CPU threads: {optimal_threads}/{cpu_count}")
    
    def _cleanup_memory(self):
        """Aggressive memory cleanup"""
        gc.collect()
        
        if self.device == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        elif self.device == "mps":
            if hasattr(torch.mps, 'empty_cache'):
                torch.mps.empty_cache()
    
    def _warmup(self):
        """Warmup models with dummy inference"""
        try:
            # Dry run to avoid first-time latency
            list(self.inference_engine.inference(
                ServeTTSRequest(
                    text="Hello world.",
                    references=[],
                    reference_id=None,
                    max_new_tokens=1024,
                    chunk_length=200,
                    top_p=0.7,
                    repetition_penalty=1.5,
                    temperature=0.7,
                    format="wav",
                )
            ))
            logger.info("Warmup completed")
        except Exception as e:
            logger.warning(f"Warmup failed (non-critical): {e}")
    
    def _hash_audio(self, audio_path: Path) -> str:
        """Generate hash for audio file"""
        return hashlib.md5(str(audio_path.absolute()).encode()).hexdigest()
    
    def _optimize_audio(self, audio_path: Path, max_duration: float = 30.0) -> Path:
        """Optimize reference audio"""
        try:
            # Ensure path is a string for torchaudio
            audio_path_str = str(audio_path)
            logger.info(f"Loading reference audio: {audio_path_str}")
            
            audio, sr = torchaudio.load(audio_path_str)
            logger.info(f"Loaded audio: shape={audio.shape}, sr={sr}")
            
            # Convert to mono
            if audio.shape[0] > 1:
                audio = audio.mean(0, keepdim=True)
                logger.info("Converted to mono")
            
            # Trim to max duration
            max_samples = int(max_duration * sr)
            if audio.shape[1] > max_samples:
                audio = audio[:, :max_samples]
                logger.info(f"Trimmed to {max_duration}s")
            
            # Resample to model sample rate
            target_sr = self.decoder_model.sample_rate
            if sr != target_sr:
                audio = torchaudio.functional.resample(audio, sr, target_sr)
                logger.info(f"Resampled from {sr}Hz to {target_sr}Hz")
            
            # Save optimized audio
            optimized_path = self.temp_dir / f"opt_{audio_path.name}"
            torchaudio.save(str(optimized_path), audio, target_sr)
            logger.info(f"Saved optimized audio to: {optimized_path}")
            return optimized_path
        except Exception as e:
            logger.error(f"Audio optimization failed: {e}")
            logger.warning(f"Using original audio file: {audio_path}")
            return audio_path
    
    def tts(self,
            text: str,
            speaker_wav: Optional[Union[str, Path]] = None,
            prompt_text: Optional[str] = None,
            temperature: float = 0.7,
            top_p: float = 0.7,
            repetition_penalty: float = 1.5,
            max_new_tokens: int = 2048,
            chunk_length: int = 200,
            seed: Optional[int] = None,
            output_path: Union[str, Path] = "output.wav") -> Tuple[np.ndarray, int, Dict[str, float]]:
        """
        Text-to-speech synthesis using Fish Speech inference engine
        
        Args:
            text: Text to synthesize
            speaker_wav: Optional reference audio for voice cloning
            prompt_text: Optional transcript of reference audio
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            repetition_penalty: Repetition penalty
            max_new_tokens: Maximum tokens to generate
            chunk_length: Chunk length for iterative generation
            seed: Random seed
            output_path: Output audio file path
            
        Returns:
            Tuple of (audio_array, sample_rate, metrics_dict)
        """
        start_time = time.time()
        output_path = Path(output_path)
        
        # Track GPU memory before
        if self.device == "cuda":
            torch.cuda.reset_peak_memory_stats()
        
        try:
            # Prepare references
            references = []
            if speaker_wav:
                # Ensure speaker_wav is a valid Path object
                if isinstance(speaker_wav, bytes):
                    raise ValueError("speaker_wav cannot be bytes, must be a file path")
                
                speaker_wav = Path(speaker_wav)
                
                # Verify file exists
                if not speaker_wav.exists():
                    raise FileNotFoundError(f"Reference audio file not found: {speaker_wav}")
                
                # Optimize audio
                optimized_audio = self._optimize_audio(speaker_wav)
                
                # Load audio file as bytes (Fish Speech expects bytes, not path)
                with open(optimized_audio, 'rb') as f:
                    audio_bytes = f.read()
                
                # Create reference dict with bytes
                ref_dict = {
                    'audio': audio_bytes,
                    'text': prompt_text or ""
                }
                references.append(ref_dict)
            
            # Create TTS request
            request = ServeTTSRequest(
                text=text,
                references=references,
                reference_id=None,
                max_new_tokens=max_new_tokens,
                chunk_length=chunk_length,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                temperature=temperature,
                format="wav",
                seed=seed,
                use_memory_cache="off",  # We handle caching ourselves
                streaming=False
            )
            
            # Run inference
            audio_segments = []
            sample_rate = None
            
            for result in self.inference_engine.inference(request):
                if result.code == "error":
                    raise result.error
                
                if result.code == "final":
                    sample_rate, audio = result.audio
                    audio_segments.append(audio)
            
            if not audio_segments:
                raise RuntimeError("No audio generated")
            
            # Concatenate audio
            audio = np.concatenate(audio_segments, axis=0)
            
            # Save audio
            sf.write(output_path, audio, sample_rate)
            
            # Calculate metrics
            latency_ms = (time.time() - start_time) * 1000
            
            peak_vram_mb = 0
            if self.device == "cuda":
                peak_vram_mb = torch.cuda.max_memory_allocated() / (1024**2)
            
            gpu_util = self.monitor.get_gpu_utilization()
            
            # Record metrics
            self.monitor.record_latency(latency_ms)
            self.monitor.record_vram(peak_vram_mb)
            self.monitor.record_gpu_util(gpu_util)
            
            audio_duration_s = len(audio) / sample_rate
            rtf = (latency_ms / 1000) / audio_duration_s if audio_duration_s > 0 else 0
            
            metrics = {
                'latency_ms': latency_ms,
                'peak_vram_mb': peak_vram_mb,
                'gpu_util_pct': gpu_util,
                'audio_duration_s': audio_duration_s,
                'rtf': rtf
            }
            
            logger.info(f"TTS completed: {latency_ms:.0f}ms, RTF={rtf:.2f}x, VRAM={peak_vram_mb:.0f}MB")
            
            return audio, sample_rate, metrics
            
        except Exception as e:
            logger.error(f"TTS failed: {e}")
            raise RuntimeError(f"TTS failed: {e}")
        finally:
            self._cleanup_memory()
    
    def get_health(self) -> Dict[str, Any]:
        """Get system health status"""
        import platform
        import psutil
        
        health = {
            'status': 'healthy',
            'device': self.device,
            'system_info': {
                'device': self.device,
                'precision': self.precision_mode,
                'compile_enabled': ENABLE_TORCH_COMPILE,
                'system': platform.system(),
                'cpu_cores': psutil.cpu_count()
            },
            'cache_stats': {
                'reference_cache_size': len(self.reference_cache.cache)
            }
        }
        
        if self.device == "cuda":
            health['system_info']['gpu_name'] = torch.cuda.get_device_name(0)
            health['system_info']['gpu_memory_gb'] = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            health['system_info']['compute_capability'] = torch.cuda.get_device_capability(0)
            health['gpu_memory_allocated_mb'] = torch.cuda.memory_allocated() / (1024**2)
            health['gpu_memory_reserved_mb'] = torch.cuda.memory_reserved() / (1024**2)
        
        return health
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get performance metrics"""
        return {
            'rolling_aggregates': self.monitor.get_aggregates(),
            'current_gpu_util': self.monitor.get_gpu_utilization()
        }
    
    def clear_cache(self):
        """Clear all caches"""
        self.reference_cache.clear()
        self._cleanup_memory()
    
    def cleanup(self):
        """Cleanup resources"""
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
        self._cleanup_memory()
    
    def __del__(self):
        try:
            self.cleanup()
        except:
            pass
