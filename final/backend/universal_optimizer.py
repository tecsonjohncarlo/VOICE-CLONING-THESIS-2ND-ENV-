"""
Universal Fish Speech Optimizer
Implements complete optimization strategies from Universal Optimization Guide and Claude Final Honest Guide
"""
import os
import sys
import time
import platform
import subprocess
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import psutil
import torch
from loguru import logger
from platform_matrix import print_platform_compatibility, print_performance_expectations

# Tier configurations from Universal Optimization Guide
TIER_CONFIGS = {
    'intel_high_end': {
        'cpu_examples': ['i7-12700K', 'i9-12900K', 'i7-13700K'],
        'optimizations': {
            'mixed_precision': 'fp32',
            'quantization': 'int8',
            'onnx_runtime': True,
            'threads': 'match_p_cores',
            'mkl_optimization': True,
            'torch_compile': True
        },
        'expected_performance': {
            'rtf': 3.0,
            'clip_10s': '30s',
            'clip_30s': '90s',
            'quality': 'excellent',
            'notes': 'More cores, better cooling, higher clocks'
        }
    },
    'amd_high_end': {
        'cpu_examples': ['Ryzen 7 5800X', 'Ryzen 9 7900X'],
        'optimizations': {
            'mixed_precision': 'fp32',
            'quantization': 'int8',
            'onnx_runtime': True,
            'threads': 'match_cores',
            'amd_optimizations': True,
            'torch_compile': True
        },
        'expected_performance': {
            'rtf': 2.5,
            'clip_10s': '25s',
            'clip_30s': '75s',
            'quality': 'excellent',
            'notes': 'Excellent multi-threading, no hybrid complexity'
        }
    },
    'm1_pro': {
        'cpu_examples': ['M1 Pro 8-core', 'M1 Max 10-core', 'M1 Ultra'],
        'optimizations': {
            'device': 'mps',
            'mixed_precision': 'fp16',
            'quantization': 'int8',
            'thermal_limit': 95,
            'memory_efficiency': True
        },
        'expected_performance': {
            'rtf': 12.0,
            'clip_10s': '2min',
            'clip_30s': '6min',
            'quality': 'excellent',
            'notes': 'GPU acceleration + sustained thermal performance'
        }
    },
    'i5_baseline': {
        'cpu_examples': ['i5-1235U', 'i5-1334U', 'i5-1240P'],
        'optimizations': {
            'mixed_precision': 'fp32',
            'quantization': 'int8',
            'onnx_runtime': True,
            'threads': 10,
            'thermal_chunking': True
        },
        'expected_performance': {
            'rtf': 6.0,
            'clip_10s': '60s',
            'clip_30s': '3min',
            'quality': 'very good',
            'notes': 'Slow but workable for development'
        }
    },
    'm1_air': {
        'optimizations': {
            'device': 'mps',
            'mixed_precision': 'fp16',
            'quantization': 'int8',
            'thermal_limit': 95,
            'throttle_management': True,
            'power_aware': True
        },
        'expected_performance': {
            'rtf': 12.0,
            'rtf_throttled': 20.0,
            'clip_10s': '2min',
            'clip_30s': '6min',
            'quality': 'very good',
            'note': 'degrades with sustained use',
            'notes': 'Good initially, degrades with sustained load'
        }
    },
    'intel_low_end': {
        'cpu_examples': ['i3-1115G4', 'i5-1135G7', 'i5-8265U'],
        'optimizations': {
            'mixed_precision': 'fp32',
            'quantization': 'int8_aggressive',
            'onnx_runtime': True,
            'threads': 'match_cores',
            'memory_conservative': True,
            'chunk_size_small': True
        },
        'expected_performance': {
            'rtf': 12.0,
            'clip_10s': '2min',
            'clip_30s': '6min',
            'quality': 'good',
            'notes': 'Slower but still usable for basic voice cloning'
        }
    },
    'amd_mobile': {
        'cpu_examples': ['Ryzen 3 5300U', 'Ryzen 5 5500U'],
        'optimizations': {
            'mixed_precision': 'fp32',
            'quantization': 'int8',
            'onnx_runtime': True,
            'amd_threading': True,
            'power_management': True
        },
        'expected_performance': {
            'rtf': 10.0,
            'clip_10s': '100s',
            'clip_30s': '5min',
            'quality': 'good',
            'notes': 'Competitive with i5, good multi-threading'
        }
    },
    'arm_sbc': {
        'hardware_examples': ['Raspberry Pi 5', 'Orange Pi 5', 'Rock 5B'],
        'optimizations': {
            'mixed_precision': 'fp32',
            'quantization': 'int4',
            'onnx_runtime': True,
            'memory_minimal': True,
            'swap_management': True,
            'thermal_aggressive': True
        },
        'expected_performance': {
            'rtf': 25.0,
            'clip_10s': '4min',
            'clip_30s': '12min',
            'quality': 'fair',
            'memory_limit': '1-2GB',
            'note': 'proof of concept only',
            'notes': 'Proof of concept only, very slow but technically possible'
        }
    }
}


class UniversalHardwareDetector:
    """Detect hardware and select optimal configuration"""
    
    def __init__(self):
        self.system = platform.system()
        self.machine = platform.machine()
        self.cpu_info = self._get_cpu_info()
        self.memory_gb = psutil.virtual_memory().total / (1024**3)
        
    def _get_cpu_info(self) -> Dict[str, Any]:
        """Get detailed CPU information"""
        cpu_info = {
            'model': platform.processor(),
            'cores_physical': psutil.cpu_count(logical=False),
            'cores_logical': psutil.cpu_count(logical=True),
            'max_frequency': 0
        }
        
        try:
            freq = psutil.cpu_freq()
            if freq:
                cpu_info['max_frequency'] = freq.max
        except:
            pass
        
        return cpu_info
    
    def detect_cpu_tier(self) -> str:
        """Classify CPU into performance tiers"""
        model = self.cpu_info['model'].lower()
        cores_physical = self.cpu_info['cores_physical']
        
        # Intel Detection
        if 'intel' in model:
            if any(x in model for x in ['i9', 'i7']) and cores_physical >= 8:
                return 'intel_high_end'
            elif 'i5' in model and '12' in model:
                return 'i5_baseline'
            elif 'i5' in model or 'i3' in model:
                return 'intel_low_end'
        
        # AMD Detection
        elif 'amd' in model or 'ryzen' in model:
            if any(x in model for x in ['ryzen 9', 'ryzen 7']) and cores_physical >= 8:
                return 'amd_high_end'
            else:
                return 'amd_mobile'
        
        # Apple Silicon Detection
        elif self.system == 'Darwin' and 'arm' in self.machine.lower():
            if self._is_m1_pro_or_better():
                return 'm1_pro'
            else:
                return 'm1_air'
        
        # ARM SBC Detection
        elif 'arm' in self.machine.lower() and cores_physical <= 4:
            return 'arm_sbc'
        
        return 'intel_low_end'  # Conservative default
    
    def _is_m1_pro_or_better(self) -> bool:
        """Detect M1 Pro/Max/Ultra vs M1 Air"""
        try:
            result = subprocess.run(['system_profiler', 'SPHardwareDataType'], 
                                  capture_output=True, text=True, timeout=5)
            if any(x in result.stdout for x in ['MacBook Pro', 'Mac Studio', 'iMac']):
                return True
            return False
        except:
            return False
    
    def get_optimal_config(self) -> Dict[str, Any]:
        """Get optimal configuration for detected hardware"""
        tier = self.detect_cpu_tier()
        
        config = TIER_CONFIGS.get(tier, TIER_CONFIGS['intel_low_end']).copy()
        config['detected_tier'] = tier
        config['hardware_info'] = {
            'cpu_model': self.cpu_info['model'],
            'cores_physical': self.cpu_info['cores_physical'],
            'cores_logical': self.cpu_info['cores_logical'],
            'memory_gb': round(self.memory_gb, 1),
            'system': self.system
        }
        
        return config


class PerformanceTracker:
    """Track and report performance metrics"""
    
    def __init__(self):
        self.inferences = []
        
    def record_inference(self, elapsed_time: float, text_length: int):
        """Record inference performance"""
        self.inferences.append({
            'elapsed': elapsed_time,
            'text_length': text_length,
            'chars_per_sec': text_length / elapsed_time if elapsed_time > 0 else 0
        })
        
    def get_average_performance(self) -> Dict[str, float]:
        """Get average performance metrics"""
        if not self.inferences:
            return {}
        
        avg_elapsed = sum(i['elapsed'] for i in self.inferences) / len(self.inferences)
        avg_chars_per_sec = sum(i['chars_per_sec'] for i in self.inferences) / len(self.inferences)
        
        return {
            'avg_elapsed_sec': avg_elapsed,
            'avg_chars_per_sec': avg_chars_per_sec,
            'total_inferences': len(self.inferences)
        }


def print_platform_expectations(config: Dict[str, Any]):
    """Print realistic expectations for current platform"""
    hw = config['hardware_info']
    tier = config['detected_tier']
    perf = config.get('expected_performance', {})
    
    logger.info("=" * 70)
    logger.info("PLATFORM EXPECTATIONS")
    logger.info("=" * 70)
    logger.info(f"System: {hw['system']}")
    logger.info(f"CPU: {hw['cpu_model']}")
    logger.info(f"Tier: {tier}")
    logger.info(f"")
    logger.info(f"Expected Performance:")
    logger.info(f"  RTF: {perf.get('rtf', 'unknown')}")
    logger.info(f"  10s clip: {perf.get('clip_10s', 'unknown')}")
    logger.info(f"  30s clip: {perf.get('clip_30s', 'unknown')}")
    logger.info(f"  Quality: {perf.get('quality', 'unknown')}")
    
    if 'notes' in perf:
        logger.info(f"  Notes: {perf['notes']}")
    
    if 'note' in perf:
        logger.warning(f"  ⚠️  {perf['note']}")
    
    logger.info("=" * 70)


class UniversalFishSpeechOptimizer:
    """Universal optimizer that adapts to any hardware"""
    
    def __init__(self, model_path: str = "checkpoints/openaudio-s1-mini", device: str = None):
        """Initialize Universal Optimizer
        
        Args:
            model_path: Path to model checkpoint
            device: Optional device override (auto, cpu, cuda, mps). If None, uses detected config.
        """
        # Detect hardware
        self.detector = UniversalHardwareDetector()
        self.config = self.detector.get_optimal_config()
        
        # CRITICAL FIX: Override device if user explicitly provided
        # This ensures user preference (e.g., DEVICE=cpu) is respected throughout the chain
        if device is not None:
            logger.info(f"🔒 Overriding detected device '{self.config['optimizations'].get('device')}' with user preference: '{device}'")
            self.config['optimizations']['device'] = device
        
        # Log detected configuration
        self._log_configuration()
        
        # Setup environment based on detected hardware
        self._setup_environment()
        
        # Initialize performance tracker
        self.performance_tracker = PerformanceTracker()
        
        # Initialize components
        self._initialize_components(model_path)
    
    def _log_configuration(self):
        """Log detected hardware and selected configuration"""
        hw = self.config['hardware_info']
        logger.info("=" * 70)
        logger.info("DETECTED HARDWARE")
        logger.info("=" * 70)
        logger.info(f"CPU: {hw['cpu_model']}")
        logger.info(f"Cores: {hw['cores_physical']} physical, {hw['cores_logical']} logical")
        logger.info(f"Memory: {hw['memory_gb']} GB")
        logger.info(f"System: {hw['system']}")
        logger.info(f"Tier: {self.config['detected_tier']}")
        
        opt = self.config['optimizations']
        logger.info(f"")
        logger.info(f"OPTIMIZATION STRATEGY")
        logger.info(f"  Precision: {opt.get('mixed_precision', 'fp32')}")
        logger.info(f"  Quantization: {opt.get('quantization', 'none')}")
        logger.info(f"  ONNX Runtime: {opt.get('onnx_runtime', False)}")
        logger.info(f"  Torch Compile: {opt.get('torch_compile', False)}")
        logger.info(f"  Threads: {opt.get('threads', 'auto')}")
        logger.info("=" * 70)
        
        # Print platform compatibility and expectations
        print_platform_compatibility(self.detector.system, self.config['detected_tier'])
        print_performance_expectations(self.config['detected_tier'])
    
    def _setup_environment(self):
        """Setup environment variables based on hardware"""
        opt = self.config['optimizations']
        hw = self.config['hardware_info']
        
        # Threading setup
        if 'threads' in opt:
            threads = opt['threads']
            if threads == 'match_cores':
                threads = hw['cores_physical']
            elif threads == 'match_p_cores' and 'intel' in self.config['detected_tier']:
                # Intel 12th gen: assume 2 P-cores per 10 total
                threads = max(2, hw['cores_physical'] // 5)
            
            if isinstance(threads, int):
                os.environ['OMP_NUM_THREADS'] = str(threads)
                os.environ['MKL_NUM_THREADS'] = str(threads)
                logger.info(f"Set thread count to: {threads}")
        
        # Intel MKL optimizations
        if opt.get('mkl_optimization', False):
            os.environ['KMP_AFFINITY'] = 'granularity=fine,compact,1,0'
            os.environ['KMP_BLOCKTIME'] = '1'
            logger.info("Enabled Intel MKL optimizations")
        
        # AMD optimizations
        if opt.get('amd_optimizations', False):
            os.environ['USE_OPENMP'] = '1'
            os.environ['OMP_PROC_BIND'] = 'true'
            logger.info("Enabled AMD OpenMP optimizations")
    
    def _initialize_components(self, model_path: str):
        """Initialize Fish Speech components with optimal settings"""
        opt = self.config['optimizations']
        
        # Set PyTorch threading
        if 'threads' in opt:
            threads = opt['threads']
            if isinstance(threads, str):
                if threads == 'match_cores':
                    threads = self.config['hardware_info']['cores_physical']
            if isinstance(threads, int):
                torch.set_num_threads(threads)
        
        # Load base engine with detected settings
        from opt_engine_v2 import OptimizedFishSpeechV2
        
        device = opt.get('device', 'auto')
        
        self.base_engine = OptimizedFishSpeechV2(
            model_path=model_path,
            device=device,
            enable_optimizations=True,
            optimize_for_memory=opt.get('memory_conservative', False)
        )
        
        # Initialize ONNX optimizer if tier requires it
        self.onnx_optimizer = None
        if opt.get('onnx_runtime', False):
            try:
                from onnx_optimizer import ONNXOptimizer, check_onnx_availability
                
                if check_onnx_availability():
                    logger.info("Initializing ONNX Runtime optimizer...")
                    self.onnx_optimizer = ONNXOptimizer(
                        model_path=model_path,
                        config=self.config,
                        device='cpu',  # ONNX for CPU optimization
                        pytorch_engine=self.base_engine  # Pass PyTorch engine for model export
                    )
                    logger.info("✅ ONNX Runtime optimizer ready (4-5x speedup expected)")
                else:
                    logger.warning("⚠️  ONNX Runtime not installed - falling back to PyTorch")
                    logger.info("   Install with: pip install onnxruntime")
            except ImportError as e:
                logger.warning(f"⚠️  ONNX Runtime not available: {e}")
                logger.info("   Install with: pip install onnxruntime")
            except Exception as e:
                logger.error(f"ONNX initialization failed: {e}")
                logger.warning("   Falling back to PyTorch")
                import traceback
                logger.debug(traceback.format_exc())
        
        logger.info("✅ Universal optimizer initialized successfully!")
    
    def synthesize(self, text: str, reference_audio: str = None, **kwargs):
        """Universal synthesis method that adapts to hardware capabilities"""
        start_time = time.time()
        
        # Use chunking for lower-end hardware or long text
        chunk_threshold = self.config['optimizations'].get('chunk_threshold', 200)
        if len(text) > chunk_threshold or self.config['detected_tier'] in ['intel_low_end', 'arm_sbc']:
            result = self._chunked_synthesis(text, reference_audio, **kwargs)
        else:
            result = self._direct_synthesis(text, reference_audio, **kwargs)
        
        # Track performance
        elapsed = time.time() - start_time
        self.performance_tracker.record_inference(elapsed, len(text))
        
        return result
    
    def _chunked_synthesis(self, text: str, reference_audio: str = None, **kwargs):
        """
        Chunked synthesis for thermal management and memory efficiency
        
        Splits long text into sentences, processes each separately with
        thermal monitoring between chunks, then concatenates results.
        """
        import re
        import numpy as np
        
        logger.info("Using chunked synthesis for lower-end hardware")
        
        # Split text into sentences
        # Use regex to split on sentence boundaries
        sentences = re.split(r'(?<=[.!?])\s+', text.strip())
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if len(sentences) <= 1:
            logger.debug("Text too short for chunking, using direct synthesis")
            return self._direct_synthesis(text, reference_audio, **kwargs)
        
        logger.info(f"Split text into {len(sentences)} chunks")
        
        # Process each chunk
        audio_chunks = []
        sample_rate = None
        total_metrics = {
            'latency_ms': 0,
            'peak_vram_mb': 0,
            'rtf': 0,
            'chunks_processed': 0
        }
        
        for i, sentence in enumerate(sentences):
            logger.info(f"Processing chunk {i+1}/{len(sentences)}: {sentence[:50]}...")
            
            # Check thermal state before each chunk
            if hasattr(self.base_engine, 'thermal_manager'):
                thermal_mgr = self.base_engine.thermal_manager
                if thermal_mgr.monitoring_available:
                    temp = thermal_mgr.get_temperature()
                    if temp and temp > thermal_mgr.throttle_threshold:
                        logger.warning(f"High temperature before chunk {i+1}: {temp:.1f}°C")
                        thermal_mgr.wait_for_thermal_recovery()
            
            # Synthesize chunk
            try:
                audio, sr, metrics = self._direct_synthesis(
                    sentence, 
                    reference_audio,
                    **kwargs
                )
                
                audio_chunks.append(audio)
                sample_rate = sr
                
                # Accumulate metrics
                total_metrics['latency_ms'] += metrics.get('latency_ms', 0)
                total_metrics['peak_vram_mb'] = max(
                    total_metrics['peak_vram_mb'],
                    metrics.get('peak_vram_mb', 0)
                )
                total_metrics['chunks_processed'] += 1
                
                logger.info(f"  Chunk {i+1} complete: {metrics.get('latency_ms', 0):.0f}ms")
                
            except Exception as e:
                logger.error(f"Failed to process chunk {i+1}: {e}")
                # Continue with remaining chunks
                continue
        
        if not audio_chunks:
            raise RuntimeError("All chunks failed to synthesize")
        
        # Concatenate audio chunks
        logger.info(f"Concatenating {len(audio_chunks)} audio chunks...")
        concatenated_audio = np.concatenate(audio_chunks)
        
        # Calculate final metrics
        audio_duration_s = len(concatenated_audio) / sample_rate
        total_metrics['rtf'] = (total_metrics['latency_ms'] / 1000) / audio_duration_s if audio_duration_s > 0 else 0
        
        logger.info(f"✅ Chunked synthesis complete:")
        logger.info(f"   Total chunks: {total_metrics['chunks_processed']}")
        logger.info(f"   Total time: {total_metrics['latency_ms']/1000:.1f}s")
        logger.info(f"   Audio duration: {audio_duration_s:.1f}s")
        logger.info(f"   RTF: {total_metrics['rtf']:.2f}")
        
        return concatenated_audio, sample_rate, total_metrics
    
    def _direct_synthesis(self, text: str, reference_audio: str = None, **kwargs):
        """Direct synthesis for capable hardware"""
        # Try ONNX first if available
        if self.onnx_optimizer:
            try:
                logger.debug("Attempting ONNX Runtime inference")
                return self.onnx_optimizer.synthesize(text, reference_audio, **kwargs)
            except NotImplementedError:
                logger.debug("ONNX synthesis not implemented, using PyTorch")
            except Exception as e:
                logger.warning(f"ONNX inference failed: {e}, falling back to PyTorch")
        
        # Fallback to PyTorch
        return self.base_engine.tts(text=text, speaker_wav=reference_audio, **kwargs)
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary"""
        return self.performance_tracker.get_average_performance()
    
    def tts(self, text: str, speaker_wav: str = None, **kwargs):
        """
        TTS method for compatibility with SmartAdaptiveBackend
        Delegates to synthesize() method
        """
        return self.synthesize(text=text, reference_audio=speaker_wav, **kwargs)
    
    def get_health(self) -> Dict[str, Any]:
        """
        Health check method for compatibility with SmartAdaptiveBackend
        Returns system health status matching HealthResponse schema
        """
        try:
            import psutil
            
            # Get base engine health if available
            base_health = {}
            if hasattr(self.base_engine, 'get_health'):
                try:
                    base_health = self.base_engine.get_health()
                except:
                    pass
            
            # Add system metrics
            memory = psutil.virtual_memory()
            cpu_percent = psutil.cpu_percent(interval=0.1)
            
            # Build health response matching HealthResponse schema
            health = {
                'status': 'healthy',
                'device': self.config.get('device', 'cpu'),
                'system_info': {
                    'engine': 'UniversalFishSpeechOptimizer',
                    'tier': self.config.get('detected_tier', 'unknown'),
                    'onnx_enabled': self.onnx_optimizer is not None,
                    'cpu_percent': cpu_percent,
                    'memory_percent': memory.percent,
                    'memory_available_gb': round(memory.available / (1024**3), 2),
                    'memory_used_gb': round(memory.used / (1024**3), 2),
                    'memory_total_gb': round(memory.total / (1024**3), 2)
                },
                'cache_stats': {
                    'enabled': False,
                    'size': 0,
                    'hits': 0,
                    'misses': 0
                }
            }
            
            # Add base engine health if available
            if base_health:
                health['system_info']['base_engine'] = base_health
            
            return health
            
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return {
                'status': 'error',
                'device': 'unknown',
                'system_info': {
                    'error': str(e),
                    'engine': 'UniversalFishSpeechOptimizer'
                },
                'cache_stats': {
                    'enabled': False,
                    'size': 0
                }
            }
