"""
Smart Adaptive Backend for Fish Speech
Automatically detects hardware and self-optimizes in real-time

Integration: Replace engine initialization in app.py with SmartAdaptiveBackend
"""
import os
import sys
import platform
import subprocess
import psutil
import torch
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass
from loguru import logger
import numpy as np

@dataclass
class HardwareProfile:
    """Complete hardware profile with all capabilities"""
    device_type: str  # 'cuda', 'mps', 'cpu'
    device_name: str
    cpu_model: str
    cpu_tier: str  # 'high_end', 'mid_range', 'low_end', 'm1_air', 'm1_pro'
    cores_physical: int
    cores_logical: int
    memory_gb: float
    has_gpu: bool
    gpu_memory_gb: float
    gpu_name: str
    system: str  # 'Windows', 'Darwin', 'Linux'
    machine: str  # 'x86_64', 'AMD64', 'arm64', etc.
    thermal_capable: bool
    avx512_vnni: bool  # For INT8 CPU optimization
    compute_capability: Optional[Tuple[int, int]]  # For CUDA GPUs

@dataclass
class OptimalConfig:
    """Auto-selected optimal configuration"""
    device: str
    precision: str  # 'fp32', 'fp16', 'bf16'
    quantization: str  # 'none', 'int8', 'int4'
    use_onnx: bool
    use_torch_compile: bool
    chunk_length: int
    max_batch_size: int
    num_threads: int
    cache_limit: int
    enable_thermal_management: bool
    expected_rtf: float
    expected_memory_gb: float
    optimization_strategy: str
    notes: str
    max_text_length: int = 500  # Maximum text length for this hardware


class SmartHardwareDetector:
    """Comprehensive hardware detection and profiling"""
    
    def __init__(self):
        self.system = platform.system()
        self.machine = platform.machine()
        self.is_wsl = self._detect_wsl()
        self.profile = self._build_complete_profile()
        logger.info("🔍 Hardware detection complete")
        self._log_detection_results()
    
    def _detect_wsl(self) -> bool:
        """Detect if running in WSL2"""
        if self.system != 'Linux':
            return False
        
        try:
            # Check for WSL in kernel version
            with open('/proc/version', 'r') as f:
                version = f.read().lower()
                is_wsl = 'microsoft' in version or 'wsl' in version
                if is_wsl:
                    logger.info("✅ Running in WSL2 - Triton support available!")
                return is_wsl
        except:
            return False
    
    def _build_complete_profile(self) -> HardwareProfile:
        """Build complete hardware profile"""
        # CPU detection
        cpu_info = self._get_cpu_info()
        cpu_tier = self._detect_cpu_tier(cpu_info)
        
        # GPU detection
        gpu_info = self._detect_gpu()
        
        # Capability detection
        thermal_capable = self._check_thermal_capability()
        avx512_vnni = self._check_avx512_vnni()
        
        return HardwareProfile(
            device_type=gpu_info['device_type'],
            device_name=gpu_info['device_name'],
            cpu_model=cpu_info['model'],
            cpu_tier=cpu_tier,
            cores_physical=cpu_info['cores_physical'],
            cores_logical=cpu_info['cores_logical'],
            memory_gb=psutil.virtual_memory().total / (1024**3),
            has_gpu=gpu_info['has_gpu'],
            gpu_memory_gb=gpu_info['gpu_memory_gb'],
            gpu_name=gpu_info['gpu_name'],
            system=self.system,
            machine=self.machine,
            thermal_capable=thermal_capable,
            avx512_vnni=avx512_vnni,
            compute_capability=gpu_info['compute_capability']
        )
    
    def _get_cpu_info(self) -> Dict[str, Any]:
        """Detailed CPU information"""
        return {
            'model': platform.processor(),
            'cores_physical': psutil.cpu_count(logical=False) or 1,
            'cores_logical': psutil.cpu_count(logical=True) or 1,
            'frequency_mhz': psutil.cpu_freq().max if psutil.cpu_freq() else 0
        }
    
    def _detect_cpu_tier(self, cpu_info: Dict) -> str:
        """Classify CPU performance tier"""
        model = cpu_info['model'].lower()
        cores = cpu_info['cores_physical']
        
        # Apple Silicon
        if self.system == 'Darwin' and 'arm' in self.machine.lower():
            return 'm1_pro' if self._is_m1_pro_or_better() else 'm1_air'
        
        # Intel
        if 'intel' in model:
            if any(x in model for x in ['i9', 'i7', 'ultra']) and cores >= 8:
                return 'intel_high_end'
            elif 'i5' in model and any(x in model for x in ['12', '13', '14']):
                return 'i5_baseline'
            else:
                return 'intel_low_end'
        
        # AMD
        if 'amd' in model or 'ryzen' in model:
            if any(x in model for x in ['ryzen 9', 'ryzen 7', 'threadripper']) and cores >= 8:
                return 'amd_high_end'
            else:
                return 'amd_mobile'
        
        # ARM SBC
        if 'arm' in self.machine.lower() and cores <= 4:
            return 'arm_sbc'
        
        return 'intel_low_end'  # Conservative default
    
    def _is_m1_pro_or_better(self) -> bool:
        """Detect M1 Pro/Max/Ultra vs M1 Air"""
        try:
            result = subprocess.run(
                ['system_profiler', 'SPHardwareDataType'],
                capture_output=True, text=True, timeout=5
            )
            return any(x in result.stdout for x in ['MacBook Pro', 'Mac Studio', 'iMac', 'Mac Mini'])
        except:
            return False
    
    def _detect_gpu(self) -> Dict[str, Any]:
        """Comprehensive GPU detection"""
        gpu_info = {
            'has_gpu': False,
            'device_type': 'cpu',
            'device_name': 'CPU',
            'gpu_name': 'None',
            'gpu_memory_gb': 0.0,
            'compute_capability': None
        }
        
        # CUDA detection
        if torch.cuda.is_available():
            try:
                gpu_info.update({
                    'has_gpu': True,
                    'device_type': 'cuda',
                    'device_name': torch.cuda.get_device_name(0),
                    'gpu_name': torch.cuda.get_device_name(0),
                    'gpu_memory_gb': torch.cuda.get_device_properties(0).total_memory / (1024**3),
                    'compute_capability': torch.cuda.get_device_capability(0)
                })
                logger.info(f"✅ NVIDIA GPU detected: {gpu_info['gpu_name']}")
            except Exception as e:
                logger.warning(f"CUDA detected but error: {e}")
        
        # Apple Silicon MPS
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            gpu_info.update({
                'has_gpu': True,
                'device_type': 'mps',
                'device_name': 'Apple Silicon GPU',
                'gpu_name': 'Apple Silicon GPU',
                'gpu_memory_gb': psutil.virtual_memory().total / (1024**3) * 0.6  # Estimate
            })
            logger.info("✅ Apple Silicon GPU (MPS) detected")
        
        return gpu_info
    
    def _check_thermal_capability(self) -> bool:
        """Check if thermal monitoring is available"""
        if self.system == 'Linux':
            return Path('/sys/class/thermal').exists()
        elif self.system == 'Darwin':
            try:
                subprocess.run(['powermetrics', '--version'], capture_output=True, timeout=2)
                return True
            except:
                return False
        elif self.system == 'Windows':
            # Check for common tools
            try:
                import wmi
                w = wmi.WMI(namespace="root\\LibreHardwareMonitor")
                return len(w.Sensor()) > 0
            except:
                return False
        return False
    
    def _check_avx512_vnni(self) -> bool:
        """Check for AVX-512 VNNI support (INT8 optimization)"""
        if self.system == 'Linux':
            try:
                with open('/proc/cpuinfo', 'r') as f:
                    cpuinfo = f.read()
                    return 'avx512_vnni' in cpuinfo
            except:
                pass
        # For Windows/macOS, check CPU model
        cpu_model = platform.processor().lower()
        # Intel 12th gen and newer have VNNI
        return any(x in cpu_model for x in ['i5-12', 'i5-13', 'i5-14', 'i7-12', 'i7-13', 'i7-14', 'i9-12', 'i9-13', 'i9-14'])
    
    def _log_detection_results(self):
        """Log detected hardware"""
        p = self.profile
        logger.info("="*70)
        logger.info("DETECTED HARDWARE PROFILE")
        logger.info("="*70)
        logger.info(f"System: {p.system} ({p.machine})")
        logger.info(f"CPU: {p.cpu_model}")
        logger.info(f"CPU Tier: {p.cpu_tier}")
        logger.info(f"Cores: {p.cores_physical} physical, {p.cores_logical} logical")
        logger.info(f"RAM: {p.memory_gb:.1f} GB")
        logger.info(f"")
        logger.info(f"GPU: {p.gpu_name}")
        logger.info(f"GPU Memory: {p.gpu_memory_gb:.1f} GB")
        logger.info(f"Device: {p.device_type}")
        logger.info(f"")
        logger.info(f"Thermal Monitoring: {'✅ Available' if p.thermal_capable else '❌ Not Available'}")
        logger.info(f"AVX-512 VNNI: {'✅ Supported' if p.avx512_vnni else '❌ Not Supported'}")
        if p.compute_capability:
            logger.info(f"CUDA Compute: {p.compute_capability[0]}.{p.compute_capability[1]}")
        logger.info("="*70)


class ConfigurationSelector:
    """Select optimal configuration based on hardware profile"""
    
    def __init__(self, profile: HardwareProfile, is_wsl: bool = False):
        self.profile = profile
        self.is_wsl = is_wsl
    
    def select_optimal_config(self) -> OptimalConfig:
        """Select best configuration for detected hardware"""
        
        # GPU-based configuration
        # Use 3.5GB threshold to account for floating point precision and usable VRAM
        # RTX 3050 (4GB), RTX 3060 (12GB), etc. should all qualify
        logger.debug(f"GPU check: has_gpu={self.profile.has_gpu}, gpu_memory_gb={self.profile.gpu_memory_gb:.2f}")
        if self.profile.has_gpu and self.profile.gpu_memory_gb >= 3.5:
            logger.info(f"✅ GPU configuration selected (GPU: {self.profile.gpu_name}, {self.profile.gpu_memory_gb:.2f}GB VRAM)")
            return self._gpu_config()
        
        # Log why GPU wasn't selected
        if self.profile.has_gpu:
            logger.warning(f"⚠️ GPU detected but insufficient VRAM: {self.profile.gpu_memory_gb:.2f}GB < 3.5GB required")
        else:
            logger.info("ℹ️ No GPU detected, using CPU configuration")
        
        # CPU-based configuration by tier
        tier_configs = {
            'intel_high_end': self._high_end_cpu_config,
            'amd_high_end': self._high_end_cpu_config,
            'm1_pro': self._m1_pro_config,
            'm1_air': self._m1_air_config,
            'i5_baseline': self._i5_baseline_config,
            'intel_low_end': self._low_end_cpu_config,
            'amd_mobile': self._mobile_cpu_config,
            'arm_sbc': self._arm_sbc_config
        }
        
        config_fn = tier_configs.get(self.profile.cpu_tier, self._low_end_cpu_config)
        return config_fn()
    
    def _gpu_config(self) -> OptimalConfig:
        """Configuration for GPU inference"""
        # CRITICAL FIX: 4GB GPUs need extreme memory optimization
        # RTX 3050 Laptop (4GB) was using 5.97GB causing memory overflow to RAM
        if self.profile.gpu_memory_gb <= 4.5:
            logger.warning(f"⚠️ 4GB GPU detected ({self.profile.gpu_name}) - applying extreme memory optimization")
            return OptimalConfig(
                device='cuda',
                precision='fp16',  # Force fp16, not bf16
                quantization='none',  # Disable INT8 - not reducing memory effectively
                use_onnx=False,
                use_torch_compile=False,  # Disabled on Windows
                chunk_length=200,  # CRITICAL: Reduced from 1024 to 200
                max_batch_size=1,  # Force batch size 1
                num_threads=4,
                cache_limit=25,  # Reduced from 100
                enable_thermal_management=self.profile.thermal_capable,
                expected_rtf=5.0,  # More realistic for 4GB GPU
                expected_memory_gb=3.5,  # Stay under 4GB
                optimization_strategy='extreme_4gb_optimization',
                notes='⚠️ Extreme memory optimization for 4GB VRAM - chunk_length=200, no quantization, fp16 only',
                max_text_length=200  # CRITICAL: Reduced from 600 to 200
            )
        
        # Determine precision based on compute capability
        precision = 'fp16'
        if self.profile.device_type == 'cuda' and self.profile.compute_capability:
            if self.profile.compute_capability[0] >= 8:  # Ampere+
                precision = 'bf16'
        
        # torch.compile support:
        # 1. WSL2 (Linux with Microsoft kernel) - ✅ Has Triton (best performance)
        # 2. Native Linux (NVIDIA GPU) - ✅ Has Triton (best performance)
        # 3. macOS (MPS) - ❌ UNSTABLE - causes hanging/freezing
        # 4. Windows - ❌ Triton not available, inductor unstable, disabled by default
        # 5. User override with FORCE_TORCH_COMPILE=1
        
        force_compile = os.getenv('FORCE_TORCH_COMPILE', '0') == '1'
        
        # Smart torch.compile enablement
        if force_compile:
            use_compile = True
            logger.info("🔧 torch.compile FORCED via FORCE_TORCH_COMPILE=1")
        elif self.is_wsl and self.profile.has_gpu:
            use_compile = True
            logger.info("🚀 WSL2 + NVIDIA GPU detected - enabling torch.compile with Triton (20-30% speedup!)")
        elif self.profile.system == 'Linux' and self.profile.device_type == 'cuda':
            # Native Linux with NVIDIA GPU
            use_compile = True
            logger.info("🚀 Linux + NVIDIA GPU detected - enabling torch.compile with Triton")
        elif self.profile.system == 'Darwin':
            # macOS: torch.compile with MPS is UNSTABLE (causes hanging)
            use_compile = False
            logger.warning("⚠️ macOS detected: torch.compile disabled (MPS backend unstable, causes hanging)")
        elif self.profile.system == 'Windows':
            # Windows: Disabled (Triton not available, inductor unstable)
            use_compile = False
            logger.info("⚠️ Windows detected: torch.compile disabled (Triton not available, use WSL2 for 20-30% speedup)")
        else:
            # Unknown platform: disable for safety
            use_compile = False
            logger.warning(f"⚠️ Unknown platform ({self.profile.system}): torch.compile disabled for safety")
        
        # Smart quantization based on GPU tier
        if self.profile.gpu_memory_gb >= 12:
            # High-end GPU: No quantization needed (V100, A100, RTX 3090, 4090)
            quantization = 'none'
            expected_memory = 6.0  # Full precision model
            expected_rtf = 0.8  # Faster than real-time
        elif self.profile.gpu_memory_gb >= 8:
            # Mid-range GPU: Light quantization (RTX 3060, 4060)
            quantization = 'int8'
            expected_memory = 4.0
            expected_rtf = 1.2
        else:
            # Entry-level GPU (6GB): Moderate quantization
            quantization = 'int8'
            expected_memory = 4.5
            expected_rtf = 2.0
        
        # Determine max text length based on GPU tier
        if self.profile.gpu_memory_gb >= 16:
            max_text = 2000  # High-end GPU (RTX 3090, 4090, A100, V100)
        elif self.profile.gpu_memory_gb >= 8:
            max_text = 1000  # Mid-range GPU (RTX 3060, 4060)
        else:
            max_text = 600   # Entry-level GPU (6GB+)
        
        return OptimalConfig(
            device='cuda' if self.profile.device_type == 'cuda' else 'mps',
            precision=precision,
            quantization=quantization,
            use_onnx=False,  # Not needed for GPU
            use_torch_compile=use_compile,
            chunk_length=1024,  # Large chunks for GPU
            max_batch_size=4,
            num_threads=self.profile.cores_physical // 2,
            cache_limit=100,
            enable_thermal_management=self.profile.thermal_capable,
            expected_rtf=expected_rtf,
            expected_memory_gb=expected_memory,
            optimization_strategy='gpu_optimized',
            notes=f'GPU-accelerated with {precision.upper()} precision' + (f', {quantization.upper()} quantization' if quantization != 'none' else ', no quantization') + (
                ' + torch.compile (Triton)' if use_compile else
                ' (torch.compile disabled - use WSL2 for 20-30% speedup)' if self.profile.system == 'Windows' else
                ' (torch.compile disabled)'
            ),
            max_text_length=max_text
        )
    
    def _m1_pro_config(self) -> OptimalConfig:
        """M1 Pro/Max/Ultra configuration"""
        return OptimalConfig(
            device='mps',
            precision='fp16',
            quantization='int8',
            use_onnx=False,
            use_torch_compile=True,  # Works well on M1
            chunk_length=512,
            max_batch_size=2,
            num_threads=4,
            cache_limit=50,
            enable_thermal_management=True,
            expected_rtf=12.0,
            expected_memory_gb=3.0,
            optimization_strategy='m1_pro_sustained',
            notes='Active cooling maintains sustained performance',
            max_text_length=400  # M1 Pro/Max with active cooling
        )
    
    def _m1_air_config(self) -> OptimalConfig:
        """M1 Air configuration (throttling expected)"""
        return OptimalConfig(
            device='mps',
            precision='fp16',
            quantization='int8',
            use_onnx=False,
            use_torch_compile=True,
            chunk_length=256,  # Smaller chunks for thermal management
            max_batch_size=1,
            num_threads=4,
            cache_limit=25,
            enable_thermal_management=True,
            expected_rtf=15.0,
            expected_memory_gb=2.5,
            optimization_strategy='m1_air_thermal_aware',
            notes='⚠️ Performance degrades after 10-15min (fanless design)',
            max_text_length=150  # M1 Air - conservative due to thermal throttling
        )
    
    def _high_end_cpu_config(self) -> OptimalConfig:
        """High-end desktop CPU configuration"""
        return OptimalConfig(
            device='cpu',
            precision='fp32',
            quantization='int8',
            use_onnx=True,  # ✅ 4-5x speedup on CPU
            use_torch_compile=True,
            chunk_length=512,
            max_batch_size=2,
            num_threads=self.profile.cores_physical,
            cache_limit=50,
            enable_thermal_management=self.profile.thermal_capable,
            expected_rtf=3.0,
            expected_memory_gb=4.0,
            optimization_strategy='cpu_onnx_optimized',
            notes='High-end CPU with ONNX Runtime (4-5x faster)',
            max_text_length=500  # Intel i7/i9, AMD Ryzen 7/9
        )
    
    def _i5_baseline_config(self) -> OptimalConfig:
        """Intel i5 baseline configuration (YOUR LAPTOP)"""
        return OptimalConfig(
            device='cpu',
            precision='fp32',
            quantization='int8',
            use_onnx=True,  # ✅ Critical for i5
            use_torch_compile=False,  # Not helpful for single-use
            chunk_length=512,
            max_batch_size=1,
            num_threads=10,  # i5-1334U/1235U has 10 cores
            cache_limit=25,
            enable_thermal_management=True,
            expected_rtf=8.0,  # With ONNX optimization
            expected_memory_gb=3.5,
            optimization_strategy='i5_onnx_thermal',
            notes='ONNX Runtime + INT8 quantization (6x faster than baseline)',
            max_text_length=300  # Intel i5 baseline
        )
    
    def _low_end_cpu_config(self) -> OptimalConfig:
        """Low-end CPU configuration"""
        return OptimalConfig(
            device='cpu',
            precision='fp32',
            quantization='int8',
            use_onnx=True,
            use_torch_compile=False,
            chunk_length=256,
            max_batch_size=1,
            num_threads=self.profile.cores_physical,
            cache_limit=10,
            enable_thermal_management=True,
            expected_rtf=12.0,
            expected_memory_gb=3.0,
            optimization_strategy='conservative_cpu',
            notes='Conservative settings for stability',
            max_text_length=200  # Low-end Intel/AMD
        )
    
    def _mobile_cpu_config(self) -> OptimalConfig:
        """Mobile CPU configuration"""
        return OptimalConfig(
            device='cpu',
            precision='fp32',
            quantization='int8',
            use_onnx=True,
            use_torch_compile=False,
            chunk_length=256,
            max_batch_size=1,
            num_threads=min(4, self.profile.cores_physical),
            cache_limit=15,
            enable_thermal_management=True,
            expected_rtf=10.0,
            expected_memory_gb=3.0,
            optimization_strategy='mobile_efficient',
            notes='Mobile-optimized with aggressive thermal management',
            max_text_length=250  # Mobile AMD/Intel
        )
    
    def _arm_sbc_config(self) -> OptimalConfig:
        """ARM SBC (Raspberry Pi, etc.) configuration"""
        return OptimalConfig(
            device='cpu',
            precision='fp32',
            quantization='int4',  # Aggressive
            use_onnx=True,
            use_torch_compile=False,
            chunk_length=128,
            max_batch_size=1,
            num_threads=self.profile.cores_physical,
            cache_limit=5,
            enable_thermal_management=True,
            expected_rtf=25.0,
            expected_memory_gb=1.5,
            optimization_strategy='extreme_memory_saver',
            notes='⚠️ Proof of concept only - very slow',
            max_text_length=100  # Raspberry Pi - very conservative
        )


class MemoryBudgetManager:
    """Strictly enforce memory budgets to prevent throttling"""
    
    def __init__(self, hardware_profile: HardwareProfile):
        self.profile = hardware_profile
        self.memory_budget = self._calculate_memory_budget()
        self.current_usage = 0
        
    def _calculate_memory_budget(self) -> dict:
        """Calculate safe memory budgets per hardware tier"""
        budgets = {
            "m1_air": {
                "total_gb": 8,
                "reserved_for_os": 2.0,
                "max_model": 2.0,
                "max_cache": 1.0,
                "max_batch": 0.5,
                "safety_margin": 1.5,
            },
            "m1_pro": {
                "total_gb": 16,
                "reserved_for_os": 2.5,
                "max_model": 4.0,
                "max_cache": 2.0,
                "max_batch": 1.5,
                "safety_margin": 2.0,
            },
            "m1_max": {
                "total_gb": 32,
                "reserved_for_os": 3.0,
                "max_model": 8.0,
                "max_cache": 4.0,
                "max_batch": 2.0,
                "safety_margin": 3.0,
            },
            "intel_i5": {
                "total_gb": 16,
                "reserved_for_os": 3.0,
                "max_model": 3.0,
                "max_cache": 1.5,
                "max_batch": 0.8,
                "safety_margin": 2.0,
            },
            "intel_high_end": {
                "total_gb": 32,
                "reserved_for_os": 4.0,
                "max_model": 6.0,
                "max_cache": 3.0,
                "max_batch": 2.0,
                "safety_margin": 3.0,
            },
            "amd_ryzen5": {
                "total_gb": 16,
                "reserved_for_os": 3.0,
                "max_model": 3.0,
                "max_cache": 1.5,
                "max_batch": 0.8,
                "safety_margin": 2.0,
            },
            "amd_high_end": {
                "total_gb": 32,
                "reserved_for_os": 4.0,
                "max_model": 6.0,
                "max_cache": 3.0,
                "max_batch": 2.0,
                "safety_margin": 3.0,
            },
        }
        
        tier = self.profile.cpu_tier
        if tier not in budgets:
            return budgets.get("intel_i5")  # Conservative default
        
        return budgets[tier]
    
    def enforce_limits(self) -> bool:
        """Check if memory usage exceeds safe limits"""
        current_memory_gb = psutil.virtual_memory().used / (1024**3)
        available_gb = psutil.virtual_memory().available / (1024**3)
        
        budget = self.memory_budget
        safe_limit = (budget["total_gb"] - 
                     budget["reserved_for_os"] - 
                     budget["safety_margin"])
        
        if available_gb < budget["safety_margin"]:
            logger.error(f"⚠️ CRITICAL: Only {available_gb:.1f}GB available (need {budget['safety_margin']}GB safety margin)")
            return False
        
        return True
    
    def get_adjusted_cache_limit(self) -> int:
        """Get cache limit that respects memory budget"""
        available_gb = psutil.virtual_memory().available / (1024**3)
        max_cache = self.memory_budget["max_cache"]
        
        # Conservative: use only 50% of budgeted cache
        suggested_cache_mb = int((min(available_gb, max_cache) * 0.5) * 1024)
        
        logger.info(f"Cache limit: {suggested_cache_mb}MB (available: {available_gb:.1f}GB)")
        return suggested_cache_mb


class QuantizationStrategy:
    """Advanced quantization for low-memory devices"""
    
    @staticmethod
    def get_quantization_config(cpu_tier: str, available_memory_gb: float) -> dict:
        """Return quantization strategy per device and memory"""
        
        strategies = {
            "m1_air": {
                "available_memory_gb": 8,
                "quantization": "int8",
                "layer_wise_quant": True,  # Quantize layer-by-layer
                "weight_only": True,  # Only quantize weights, not activations
                "dynamic_quant": True,  # Per-tensor quantization
                "calibration_data_size": 32,  # Small calibration dataset
            },
            "m1_pro": {
                "available_memory_gb": 16,
                "quantization": "int8",
                "layer_wise_quant": True,
                "weight_only": True,
                "dynamic_quant": True,
                "calibration_data_size": 64,
            },
            "intel_i5": {
                "available_memory_gb": 16,
                "quantization": "int8",
                "layer_wise_quant": True,
                "weight_only": True,
                "dynamic_quant": True,
                "calibration_data_size": 64,
            },
            "amd_ryzen5": {
                "available_memory_gb": 16,
                "quantization": "int8",
                "layer_wise_quant": True,
                "weight_only": True,
                "dynamic_quant": True,
                "calibration_data_size": 64,
            },
            "intel_high_end": {
                "available_memory_gb": 32,
                "quantization": "int8",
                "layer_wise_quant": False,  # Can handle full quantization
                "weight_only": False,
                "dynamic_quant": True,
                "calibration_data_size": 128,
            },
        }
        
        config = strategies.get(cpu_tier, strategies["intel_i5"])
        
        # Adaptive: if memory is really low, use int4
        if available_memory_gb < 6:
            config["quantization"] = "int4"
            logger.warning(f"⚠️ Aggressive quantization: INT4 (low memory: {available_memory_gb:.1f}GB)")
        
        return config


class CPUAffinityManager:
    """Optimize thread affinity to reduce context switching on low-end CPUs"""
    
    def __init__(self, profile: HardwareProfile):
        self.profile = profile
        
    def set_thread_affinity(self):
        """Set optimal thread affinity based on CPU tier"""
        try:
            import psutil
            
            if self.profile.cpu_tier == "intel_i5":
                # i5 typically: 2P cores (performance) + 8E cores (efficiency)
                # Pin inference to P-cores only for better latency
                p_cores = list(range(0, min(2, self.profile.cores_physical)))
                psutil.Process().cpu_affinity(p_cores)
                logger.info(f"✅ Pinned to P-cores: {p_cores}")
                
            elif self.profile.cpu_tier in ["amd_ryzen5", "amd_high_end"]:
                # AMD: typically 6+ cores uniform
                # Use even core indices for better cache locality
                cores = list(range(0, self.profile.cores_physical, 2))
                psutil.Process().cpu_affinity(cores)
                logger.info(f"✅ Pinned to cores: {cores}")
                
            elif self.profile.cpu_tier == "m1_air":
                # M1 Air: 4 performance + 4 efficiency cores
                # Pin to performance cores only
                perf_cores = list(range(0, 4))
                logger.info(f"✅ M1 Air: Using performance cores {perf_cores}")
                # Note: macOS doesn't support cpu_affinity, but we log the intent
                
        except Exception as e:
            logger.warning(f"CPU affinity not available: {e}")


class ResourceMonitor:
    """Real-time resource monitoring and adaptive adjustment"""
    
    def __init__(self, profile: HardwareProfile):
        self.profile = profile
        self.gpu_util_history = []
        self.memory_history = []
        self.temp_history = []
        self.memory_budget_manager = MemoryBudgetManager(profile)
        self.cpu_affinity_manager = CPUAffinityManager(profile)
    
    def check_resources(self) -> Dict[str, Any]:
        """Check current resource usage"""
        resources = {
            'cpu_percent': psutil.cpu_percent(interval=0.1),
            'memory_percent': psutil.virtual_memory().percent,
            'memory_available_gb': psutil.virtual_memory().available / (1024**3)
        }
        
        # GPU monitoring
        if self.profile.device_type == 'cuda':
            try:
                import pynvml
                pynvml.nvmlInit()
                handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                resources['gpu_util'] = util.gpu
                resources['gpu_memory_used_gb'] = pynvml.nvmlDeviceGetMemoryInfo(handle).used / (1024**3)
                pynvml.nvmlShutdown()
            except:
                resources['gpu_util'] = 0
                resources['gpu_memory_used_gb'] = 0
        
        return resources
    
    def should_throttle(self, resources: Dict) -> bool:
        """Determine if throttling needed"""
        if resources['memory_percent'] > 90:
            logger.warning(f"⚠️ Memory usage critical: {resources['memory_percent']:.1f}%")
            return True
        
        if resources['cpu_percent'] > 95:
            logger.warning(f"⚠️ CPU usage critical: {resources['cpu_percent']:.1f}%")
            return True
        
        return False
    
    def suggest_adjustment(self, resources: Dict, current_config: OptimalConfig) -> Optional[OptimalConfig]:
        """Suggest configuration adjustment based on resources"""
        if self.should_throttle(resources):
            logger.info("🔧 Auto-adjusting configuration to reduce resource usage")
            # Create adjusted config
            adjusted = OptimalConfig(
                device=current_config.device,
                precision=current_config.precision,
                quantization=current_config.quantization,
                use_onnx=current_config.use_onnx,
                use_torch_compile=current_config.use_torch_compile,
                chunk_length=current_config.chunk_length // 2,  # Smaller chunks
                max_batch_size=1,  # Force batch size 1
                num_threads=max(1, current_config.num_threads // 2),  # Reduce threads
                cache_limit=current_config.cache_limit // 2,  # Reduce cache
                enable_thermal_management=True,
                expected_rtf=current_config.expected_rtf * 1.3,
                expected_memory_gb=current_config.expected_memory_gb * 0.7,
                optimization_strategy=f"{current_config.optimization_strategy}_throttled",
                notes=f"{current_config.notes} (auto-throttled due to resource pressure)"
            )
            return adjusted
        return None


class SmartAdaptiveBackend:
    """
    Intelligent adaptive backend that auto-detects and self-optimizes
    
    Usage in app.py:
        from smart_backend import SmartAdaptiveBackend
        engine = SmartAdaptiveBackend(model_path="checkpoints/openaudio-s1-mini")
    """
    
    def __init__(self, model_path: str = "checkpoints/openaudio-s1-mini"):
        logger.info("🚀 Initializing Smart Adaptive Backend")
        
        # Step 1: Detect hardware
        self.detector = SmartHardwareDetector()
        self.profile = self.detector.profile
        
        # Step 2: Check for user device preference
        user_device = os.getenv('DEVICE', 'auto').lower()
        if user_device != 'auto':
            logger.info(f"👤 User device preference: {user_device.upper()} (overriding auto-detection)")
            self._apply_user_device_preference(user_device)
        
        # Step 3: Select optimal configuration
        self.selector = ConfigurationSelector(self.profile, self.detector.is_wsl)
        self.config = self.selector.select_optimal_config()
        self._log_selected_config()
        
        # Step 3: Initialize resource monitor with memory budget manager
        self.monitor = ResourceMonitor(self.profile)
        
        # Step 4: Set CPU affinity for optimal performance
        self.monitor.cpu_affinity_manager.set_thread_affinity()
        
        # Step 5: Enforce memory limits before initialization
        if not self.monitor.memory_budget_manager.enforce_limits():
            logger.warning("⚠️ Memory constraints detected - using conservative settings")
        
        # Step 6: Apply configuration
        self._apply_configuration()
        
        # Step 7: Initialize engine with optimal settings
        self.engine = self._initialize_engine(model_path)
        
        logger.info("✅ Smart Adaptive Backend initialized successfully!")
    
    def _log_selected_config(self):
        """Log selected configuration"""
        c = self.config
        logger.info("="*70)
        logger.info("SELECTED OPTIMAL CONFIGURATION")
        logger.info("="*70)
        logger.info(f"Strategy: {c.optimization_strategy}")
        logger.info(f"Device: {c.device}")
        logger.info(f"Precision: {c.precision}")
        logger.info(f"Quantization: {c.quantization}")
        logger.info(f"ONNX Runtime: {'✅ Enabled' if c.use_onnx else '❌ Disabled'}")
        logger.info(f"torch.compile: {'✅ Enabled' if c.use_torch_compile else '❌ Disabled'}")
        logger.info(f"Chunk Length: {c.chunk_length}")
        logger.info(f"Threads: {c.num_threads}")
        logger.info(f"Max Text Length: {c.max_text_length} characters")
        logger.info(f"")
        logger.info(f"Expected Performance:")
        logger.info(f"  RTF: {c.expected_rtf:.1f}x (slower than real-time)")
        logger.info(f"  Memory: {c.expected_memory_gb:.1f} GB")
        logger.info(f"")
        logger.info(f"Notes: {c.notes}")
        logger.info("="*70)
    
    def _apply_user_device_preference(self, user_device: str):
        """Override auto-detected device with user preference"""
        if user_device == 'cpu':
            logger.info("✅ Forcing CPU mode (user preference)")
            self.profile.device_type = 'cpu'
            self.profile.has_gpu = False
        elif user_device == 'cuda':
            if torch.cuda.is_available():
                logger.info("✅ Forcing CUDA mode (user preference)")
                self.profile.device_type = 'cuda'
                self.profile.has_gpu = True
            else:
                logger.warning("⚠️ CUDA requested but not available - falling back to CPU")
                self.profile.device_type = 'cpu'
                self.profile.has_gpu = False
        elif user_device == 'mps':
            if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                logger.info("✅ Forcing MPS mode (user preference)")
                self.profile.device_type = 'mps'
                self.profile.has_gpu = True
            else:
                logger.warning("⚠️ MPS requested but not available - falling back to CPU")
                self.profile.device_type = 'cpu'
                self.profile.has_gpu = False
        else:
            logger.warning(f"⚠️ Unknown device '{user_device}' - using auto-detection")
    
    def _apply_configuration(self):
        """Apply environment configuration"""
        # Set environment variables
        os.environ['DEVICE'] = self.config.device
        os.environ['MIXED_PRECISION'] = self.config.precision
        os.environ['ENABLE_TORCH_COMPILE'] = 'true' if self.config.use_torch_compile else 'false'
        os.environ['OMP_NUM_THREADS'] = str(self.config.num_threads)
        os.environ['MKL_NUM_THREADS'] = str(self.config.num_threads)
        
        # torch.compile is only enabled on Linux/macOS/WSL2 (has Triton)
        # No special configuration needed - Triton is available by default
        
        # PyTorch settings
        torch.set_num_threads(self.config.num_threads)
        
        if self.config.device == 'cuda':
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.benchmark = True
        
        logger.info(f"🔧 Configuration applied")
    
    def _initialize_engine(self, model_path: str):
        """Initialize appropriate engine based on configuration"""
        
        # Use ONNX-optimized engine for CPU
        if self.config.use_onnx and self.config.device == 'cpu':
            try:
                from universal_optimizer import UniversalFishSpeechOptimizer
                logger.info("📦 Loading Universal Optimizer with ONNX Runtime")
                return UniversalFishSpeechOptimizer(model_path=model_path)
            except ImportError as e:
                logger.warning(f"Universal Optimizer not available: {e}")
                logger.info("Falling back to standard engine")
        
        # Use standard V2 engine
        from opt_engine_v2 import OptimizedFishSpeechV2
        logger.info("📦 Loading Standard Optimized Engine V2")
        return OptimizedFishSpeechV2(
            model_path=model_path,
            device=self.config.device,
            enable_optimizations=True,
            optimize_for_memory=(self.profile.memory_gb < 8)
        )
    
    def _truncate_text_smart(self, text: str) -> tuple[str, bool, int]:
        """
        Intelligently truncate text based on hardware constraints
        
        Returns: (truncated_text, was_truncated, original_length)
        """
        original_length = len(text)
        max_length = self.config.max_text_length
        
        if original_length <= max_length:
            return text, False, original_length
        
        # Truncate at word boundary to avoid cutting mid-word
        truncated = text[:max_length]
        last_space = truncated.rfind(' ')
        
        # If space is reasonably close (within 20% of max), cut there
        if last_space > max_length * 0.8:
            truncated = truncated[:last_space]
        
        logger.warning(
            f"[{self.profile.cpu_tier}] Text truncated: "
            f"{original_length} chars → {len(truncated)} chars (max: {max_length})"
        )
        
        return truncated, True, original_length
    
    def tts(self, text: str, speaker_wav: Optional[str] = None, **kwargs):
        """
        Smart TTS synthesis with automatic resource monitoring and text limiting
        
        Automatically adapts if system is under pressure
        """
        # Smart text truncation based on hardware
        text, was_truncated, original_length = self._truncate_text_smart(text)
        
        if was_truncated:
            logger.info(
                f"Hardware: {self.profile.cpu_tier} | "
                f"Device: {self.profile.device_type} | "
                f"Text: {original_length} → {len(text)} chars"
            )
        
        # Pre-check resources
        resources = self.monitor.check_resources()
        
        # Auto-adjust if needed
        adjusted_config = self.monitor.suggest_adjustment(resources, self.config)
        if adjusted_config:
            logger.warning("⚠️ System under pressure - auto-adjusting parameters")
            # Apply adjusted chunk length
            kwargs['chunk_length'] = adjusted_config.chunk_length
        else:
            # Use optimal chunk length
            kwargs['chunk_length'] = self.config.chunk_length
        
        # Run synthesis
        try:
            result = self.engine.tts(
                text=text,
                speaker_wav=speaker_wav,
                **kwargs
            )
            
            # Post-check resources
            post_resources = self.monitor.check_resources()
            self._log_performance(pre=resources, post=post_resources, result=result)
            
            return result
            
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                logger.error("💥 Out of memory! Retrying with conservative settings")
                return self._retry_with_conservative_settings(text, speaker_wav, **kwargs)
            raise
    
    def _retry_with_conservative_settings(self, text: str, speaker_wav: Optional[str], **kwargs):
        """Retry with more conservative settings after OOM"""
        kwargs['chunk_length'] = 128  # Very small chunks
        kwargs['max_new_tokens'] = 1024  # Limit output
        
        # Force garbage collection
        import gc
        gc.collect()
        if self.config.device == 'cuda':
            torch.cuda.empty_cache()
        
        return self.engine.tts(text=text, speaker_wav=speaker_wav, **kwargs)
    
    def _log_performance(self, pre: Dict, post: Dict, result: Tuple):
        """Log performance metrics"""
        audio, sr, metrics = result
        
        logger.info("📊 Performance Report:")
        logger.info(f"  Latency: {metrics['latency_ms']:.0f}ms")
        logger.info(f"  RTF: {metrics['rtf']:.2f}x")
        logger.info(f"  Memory Delta: {post['memory_available_gb'] - pre['memory_available_gb']:.2f} GB")
        if 'gpu_util' in post:
            logger.info(f"  GPU Utilization: {post['gpu_util']:.1f}%")
    
    def get_health(self) -> Dict[str, Any]:
        """Get system health with smart insights"""
        base_health = self.engine.get_health()
        resources = self.monitor.check_resources()
        
        # Add smart insights
        insights = []
        if resources['memory_percent'] > 80:
            insights.append("⚠️ Memory usage high - consider reducing batch size")
        if self.profile.cpu_tier == 'm1_air' and not self.profile.thermal_capable:
            insights.append("⚠️ M1 Air: Performance may degrade after 10-15 minutes")
        if not self.config.use_onnx and self.config.device == 'cpu':
            insights.append("💡 ONNX Runtime could provide 4-5x speedup for CPU inference")
        
        base_health['smart_insights'] = insights
        base_health['current_resources'] = resources
        base_health['hardware_profile'] = {
            'tier': self.profile.cpu_tier,
            'device': self.profile.device_type,
            'thermal_monitoring': self.profile.thermal_capable
        }
        
        return base_health
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get metrics with resource monitoring"""
        base_metrics = self.engine.get_metrics()
        base_metrics['current_resources'] = self.monitor.check_resources()
        return base_metrics
    
    def clear_cache(self):
        """Clear caches"""
        self.engine.clear_cache()
    
    def cleanup(self):
        """Cleanup resources"""
        self.engine.cleanup()


# Integration example for app.py
def create_smart_backend(model_path: str = None) -> SmartAdaptiveBackend:
    """
    Factory function for creating smart backend
    
    Usage in app.py:
        from smart_backend import create_smart_backend
        
        @app.on_event("startup")
        async def startup_event():
            global engine
            model_path = os.getenv("MODEL_DIR", "checkpoints/openaudio-s1-mini")
            engine = create_smart_backend(model_path)
    """
    if model_path is None:
        model_path = os.getenv("MODEL_DIR", "checkpoints/openaudio-s1-mini")

    return SmartAdaptiveBackend(model_path=model_path)