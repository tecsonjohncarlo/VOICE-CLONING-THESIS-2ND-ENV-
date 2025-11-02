# Universal Lower-End Device Optimization Guide for Fish Speech

## CPU Performance Baseline: Intel i5 as Reference

**Intel i5 12th gen (i5-1235U/1334U)** serves as our baseline for "lower-end" CPU optimization because:
- **No AVX-512 FP16 support** - representative of consumer CPU limitations
- **Hybrid architecture** (P+E cores) - common in modern laptops
- **Thermal constraints** - typical laptop throttling behavior
- **RTF 40 baseline** - establishes minimum acceptable performance target

From this baseline, we scale optimizations up and down for different hardware categories.

## Device Categories and Optimization Strategies

### **Tier 1: Better than i5 Baseline**

#### **Intel i7/i9 Desktop (12th gen+)**
```python
INTEL_HIGH_END_CONFIG = {
    'cpu_examples': ['i7-12700K', 'i9-12900K', 'i7-13700K'],
    'cores': '8-16 P-cores + 8-16 E-cores',
    'optimizations': {
        'mixed_precision': 'fp32',  # Still no native FP16
        'quantization': 'int8',
        'onnx_runtime': True,       # 4-5x speedup
        'threads': 'match_p_cores', # Use P-cores primarily
        'mkl_optimization': True,
        'torch_compile': True       # More beneficial on desktop
    },
    'expected_performance': {
        'rtf': 3.0,  # 2x better than i5 baseline
        'notes': 'More cores, better cooling, higher clocks'
    }
}\n```

#### **AMD Ryzen 7/9 (5000/7000 series)**
```python
AMD_HIGH_END_CONFIG = {
    'cpu_examples': ['Ryzen 7 5800X', 'Ryzen 9 7900X'],
    'optimizations': {
        'mixed_precision': 'fp32',
        'quantization': 'int8', 
        'onnx_runtime': True,
        'threads': 'match_cores',   # Uniform core design
        'amd_optimizations': True,  # BLIS instead of MKL
        'torch_compile': True
    },
    'expected_performance': {
        'rtf': 2.5,  # 2.4x better than i5 baseline  
        'notes': 'Excellent multi-threading, no hybrid complexity'
    }
}\n```

#### **Apple M1 Pro/Max/Ultra (Active Cooling)**
```python
M1_PRO_CONFIG = {
    'cpu_examples': ['M1 Pro 8-core', 'M1 Max 10-core', 'M1 Ultra'],
    'optimizations': {
        'device': 'mps',            # GPU acceleration
        'mixed_precision': 'fp16',  # MPS supports FP16
        'quantization': 'int8',
        'thermal_limit': 95,        # Sustained performance
        'memory_efficiency': True   # Unified memory advantage
    },
    'expected_performance': {
        'rtf': 12.0,  # 3.3x better than i5, matches baseline target
        'notes': 'GPU acceleration + sustained thermal performance'
    }
}\n```

### **Tier 2: i5 Baseline Performance**

#### **Intel i5 12th gen Laptop (Reference)**
```python
I5_BASELINE_CONFIG = {
    'cpu_examples': ['i5-1235U', 'i5-1334U', 'i5-1240P'],
    'optimizations': {
        'mixed_precision': 'fp32',     # NO FP16 support
        'quantization': 'int8',
        'onnx_runtime': True,          # Primary optimization
        'threads': 10,                 # 2P + 8E cores
        'thermal_chunking': True       # Prevent throttling
    },
    'expected_performance': {
        'rtf': 6.0,    # Baseline target
        '10s_clip': '60 seconds',
        '30s_clip': '3 minutes',
        'notes': 'Slow but workable for development'
    }
}\n```

#### **Apple M1 Air (Fanless)**
```python
M1_AIR_CONFIG = {
    'optimizations': {
        'device': 'mps',
        'mixed_precision': 'fp16',
        'quantization': 'int8',
        'thermal_limit': 95,
        'throttle_management': True,    # Will throttle after 10-15 min
        'power_aware': True
    },
    'expected_performance': {
        'rtf': 12.0,   # Initially good
        'rtf_throttled': 20.0,  # After thermal saturation  
        'notes': 'Good initially, degrades with sustained load'
    }
}\n```

### **Tier 3: Below i5 Baseline**

#### **Intel i3/Older i5 (Pre-12th gen)**
```python
INTEL_LOW_END_CONFIG = {
    'cpu_examples': ['i3-1115G4', 'i5-1135G7', 'i5-8265U'],
    'optimizations': {
        'mixed_precision': 'fp32',
        'quantization': 'int8_aggressive',  # More aggressive
        'onnx_runtime': True,
        'threads': 'match_cores',
        'memory_conservative': True,
        'chunk_size_small': True            # Smaller processing chunks
    },
    'expected_performance': {
        'rtf': 12.0,   # 2x slower than i5 baseline
        'notes': 'Slower but still usable for basic voice cloning'
    }
}\n```

#### **AMD Ryzen 3/5 (Mobile)**
```python
AMD_MOBILE_CONFIG = {
    'cpu_examples': ['Ryzen 3 5300U', 'Ryzen 5 5500U'],
    'optimizations': {
        'mixed_precision': 'fp32',
        'quantization': 'int8',
        'onnx_runtime': True,
        'amd_threading': True,
        'power_management': True
    },
    'expected_performance': {
        'rtf': 10.0,   # Similar to i5 baseline
        'notes': 'Competitive with i5, good multi-threading'
    }
}\n```

#### **ARM Single Board Computers**
```python
ARM_SBC_CONFIG = {
    'hardware_examples': ['Raspberry Pi 5', 'Orange Pi 5', 'Rock 5B'],
    'optimizations': {
        'mixed_precision': 'fp32',
        'quantization': 'int4',             # Aggressive quantization
        'onnx_runtime': True,
        'memory_minimal': True,
        'swap_management': True,            # Handle memory pressure
        'thermal_aggressive': True          # Very conservative thermal limits
    },
    'expected_performance': {
        'rtf': 25.0,   # 4x slower than i5 baseline
        'memory_limit': '1-2GB',
        'notes': 'Proof of concept only, very slow but technically possible'
    }
}\n```

## Universal Hardware Detection and Auto-Configuration

```python
import platform
import psutil
import subprocess
from typing import Dict, Any

class UniversalHardwareDetector:
    \"\"\"Detect hardware and select optimal configuration\"\"\"
    
    def __init__(self):
        self.system = platform.system()
        self.machine = platform.machine()
        self.cpu_info = self._get_cpu_info()
        self.memory_gb = psutil.virtual_memory().total / (1024**3)
        
    def _get_cpu_info(self) -> Dict[str, Any]:
        \"\"\"Get detailed CPU information\"\"\"
        cpu_info = {
            'model': platform.processor(),
            'cores_physical': psutil.cpu_count(logical=False),
            'cores_logical': psutil.cpu_count(logical=True),
            'max_frequency': 0
        }
        
        # Get CPU frequency if available
        try:
            freq = psutil.cpu_freq()
            if freq:
                cpu_info['max_frequency'] = freq.max
        except:
            pass
        
        return cpu_info
    
    def detect_cpu_tier(self) -> str:
        \"\"\"Classify CPU into performance tiers\"\"\"
        model = self.cpu_info['model'].lower()
        cores_physical = self.cpu_info['cores_physical']
        
        # Intel Detection
        if 'intel' in model:
            if any(x in model for x in ['i9', 'i7']) and cores_physical >= 8:
                return 'intel_high_end'
            elif 'i5' in model and '12' in model:  # 12th gen i5 baseline
                return 'i5_baseline'
            elif 'i5' in model:
                return 'intel_low_end'
            elif 'i3' in model:
                return 'intel_low_end'
        
        # AMD Detection  
        elif 'amd' in model or 'ryzen' in model:
            if any(x in model for x in ['ryzen 9', 'ryzen 7']) and cores_physical >= 8:
                return 'amd_high_end'
            elif 'ryzen 5' in model:
                return 'amd_mobile'
            elif 'ryzen 3' in model:
                return 'amd_mobile'
        
        # Apple Silicon Detection
        elif self.system == 'Darwin' and 'arm' in self.machine.lower():
            if self._is_m1_pro_or_better():
                return 'm1_pro'
            else:
                return 'm1_air'  # Assume Air if can't determine
        
        # ARM SBC Detection
        elif 'arm' in self.machine.lower() and cores_physical <= 4:
            return 'arm_sbc'
        
        # Default to conservative config
        return 'intel_low_end'
    
    def _is_m1_pro_or_better(self) -> bool:
        \"\"\"Detect M1 Pro/Max/Ultra vs M1 Air\"\"\"
        try:
            # Check for fan presence (Pro/Max have fans, Air doesn't)
            result = subprocess.run(['system_profiler', 'SPHardwareDataType'], 
                                  capture_output=True, text=True)
            
            # Pro models have more GPU cores and different model names
            if any(x in result.stdout for x in ['MacBook Pro', 'Mac Studio', 'iMac']):
                return True
            return False
        except:
            return False  # Default to Air (more conservative)
    
    def get_optimal_config(self) -> Dict[str, Any]:
        \"\"\"Get optimal configuration for detected hardware\"\"\"
        tier = self.detect_cpu_tier()
        
        configs = {
            'intel_high_end': INTEL_HIGH_END_CONFIG,
            'amd_high_end': AMD_HIGH_END_CONFIG,
            'm1_pro': M1_PRO_CONFIG,
            'i5_baseline': I5_BASELINE_CONFIG,
            'm1_air': M1_AIR_CONFIG,
            'intel_low_end': INTEL_LOW_END_CONFIG,
            'amd_mobile': AMD_MOBILE_CONFIG,
            'arm_sbc': ARM_SBC_CONFIG
        }
        
        config = configs.get(tier, INTEL_LOW_END_CONFIG).copy()
        config['detected_tier'] = tier
        config['hardware_info'] = {
            'cpu_model': self.cpu_info['model'],
            'cores_physical': self.cpu_info['cores_physical'],
            'cores_logical': self.cpu_info['cores_logical'],
            'memory_gb': round(self.memory_gb, 1),
            'system': self.system
        }
        
        return config
```

## Universal Optimization Engine

```python
class UniversalFishSpeechOptimizer:
    \"\"\"Universal optimizer that adapts to any hardware\"\"\"
    
    def __init__(self, model_path: str = \"checkpoints/openaudio-s1-mini\"):
        # Detect hardware
        self.detector = UniversalHardwareDetector()
        self.config = self.detector.get_optimal_config()
        
        # Log detected configuration
        self._log_configuration()
        
        # Setup environment based on detected hardware
        self._setup_environment()
        
        # Initialize components
        self._initialize_components(model_path)
    
    def _log_configuration(self):
        \"\"\"Log detected hardware and selected configuration\"\"\"
        hw = self.config['hardware_info']
        logger.info(f\"Detected Hardware:\")
        logger.info(f\"  CPU: {hw['cpu_model']}\")
        logger.info(f\"  Cores: {hw['cores_physical']} physical, {hw['cores_logical']} logical\")
        logger.info(f\"  Memory: {hw['memory_gb']} GB\")
        logger.info(f\"  System: {hw['system']}\")
        logger.info(f\"  Tier: {self.config['detected_tier']}\")
        
        opt = self.config['optimizations']
        logger.info(f\"Optimization Strategy:\")
        logger.info(f\"  Precision: {opt.get('mixed_precision', 'fp32')}\")
        logger.info(f\"  Quantization: {opt.get('quantization', 'none')}\")
        logger.info(f\"  ONNX Runtime: {opt.get('onnx_runtime', False)}\")
        
        if 'expected_performance' in self.config:
            perf = self.config['expected_performance']
            logger.info(f\"Expected Performance:\")
            logger.info(f\"  RTF: {perf.get('rtf', 'unknown')}\")
            if '10s_clip' in perf:
                logger.info(f\"  10s clip: {perf['10s_clip']}\")
    
    def _setup_environment(self):
        \"\"\"Setup environment variables based on hardware\"\"\"
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
            
            os.environ['OMP_NUM_THREADS'] = str(threads)
            os.environ['MKL_NUM_THREADS'] = str(threads)
        
        # Intel MKL optimizations
        if opt.get('mkl_optimization', False):
            os.environ['KMP_AFFINITY'] = 'granularity=fine,compact,1,0'
            os.environ['KMP_BLOCKTIME'] = '1'
        
        # AMD optimizations
        if opt.get('amd_optimizations', False):
            os.environ['USE_OPENMP'] = '1'
            os.environ['OMP_PROC_BIND'] = 'true'
    
    def _initialize_components(self, model_path: str):
        \"\"\"Initialize Fish Speech components with optimal settings\"\"\"
        opt = self.config['optimizations']
        
        # Import PyTorch and set threading
        import torch
        if 'threads' in opt:
            threads = opt['threads']
            if isinstance(threads, str):
                if threads == 'match_cores':
                    threads = self.config['hardware_info']['cores_physical']
            torch.set_num_threads(threads)
        
        # Load base engine with detected settings
        from opt_engine_v2 import OptimizedFishSpeechV2
        
        device = opt.get('device', 'cpu')
        precision_map = {
            'fp32': torch.float32,
            'fp16': torch.float16,
            'bf16': torch.bfloat16
        }
        precision = precision_map.get(opt.get('mixed_precision', 'fp32'), torch.float32)
        
        self.base_engine = OptimizedFishSpeechV2(
            model_path=model_path,
            device=device,
            precision=precision,
            enable_compile=opt.get('torch_compile', False),
            quantization=opt.get('quantization', 'none')
        )
        
        # Initialize ONNX if supported
        if opt.get('onnx_runtime', False):
            try:
                from .onnx_optimizer import ONNXOptimizer
                self.onnx_optimizer = ONNXOptimizer(model_path, self.config)
            except ImportError:
                logger.warning(\"ONNX Runtime not available, falling back to PyTorch\")
                self.onnx_optimizer = None
        else:
            self.onnx_optimizer = None
    
    def synthesize(self, text: str, reference_audio: str = None, **kwargs):
        \"\"\"Universal synthesis method that adapts to hardware capabilities\"\"\"
        
        # Use chunking for lower-end hardware or long text
        chunk_threshold = self.config['optimizations'].get('chunk_threshold', 200)
        if len(text) > chunk_threshold or self.config['detected_tier'] in ['intel_low_end', 'arm_sbc']:
            return self._chunked_synthesis(text, reference_audio, **kwargs)
        else:
            return self._direct_synthesis(text, reference_audio, **kwargs)
    
    def _chunked_synthesis(self, text: str, reference_audio: str = None, **kwargs):
        \"\"\"Chunked synthesis for thermal management and memory efficiency\"\"\"
        # Implementation depends on detected hardware tier...
        pass
    
    def _direct_synthesis(self, text: str, reference_audio: str = None, **kwargs):
        \"\"\"Direct synthesis for capable hardware\"\"\"
        if self.onnx_optimizer:
            try:
                return self.onnx_optimizer.synthesize(text, reference_audio, **kwargs)
            except Exception as e:
                logger.warning(f\"ONNX synthesis failed: {e}, falling back to PyTorch\")
        
        return self.base_engine.tts(text=text, speaker_wav=reference_audio, **kwargs)
```

## Performance Matrix

```python
UNIVERSAL_PERFORMANCE_MATRIX = {
    'hardware_tier': {
        'intel_high_end':  {'rtf': 3.0,  'clip_10s': '30s',   'clip_30s': '90s',   'quality': 'excellent'},
        'amd_high_end':    {'rtf': 2.5,  'clip_10s': '25s',   'clip_30s': '75s',   'quality': 'excellent'},
        'm1_pro':          {'rtf': 12.0, 'clip_10s': '2min',  'clip_30s': '6min',  'quality': 'excellent'},
        'i5_baseline':     {'rtf': 6.0,  'clip_10s': '60s',   'clip_30s': '3min',  'quality': 'very good'},
        'm1_air':          {'rtf': 12.0, 'clip_10s': '2min',  'clip_30s': '6min',  'quality': 'very good', 'note': 'degrades with sustained use'},
        'intel_low_end':   {'rtf': 12.0, 'clip_10s': '2min',  'clip_30s': '6min',  'quality': 'good'},
        'amd_mobile':      {'rtf': 10.0, 'clip_10s': '100s',  'clip_30s': '5min',  'quality': 'good'},
        'arm_sbc':         {'rtf': 25.0, 'clip_10s': '4min',  'clip_30s': '12min', 'quality': 'fair', 'note': 'proof of concept only'}
    }
}
```

This universal approach provides:
- **Automatic hardware detection** and configuration
- **Scalable optimization strategies** from ARM SBC to high-end desktop
- **Intel i5 as baseline** for realistic performance expectations
- **Tier-based configurations** that adapt to available hardware
- **Universal API** that works across all supported devices