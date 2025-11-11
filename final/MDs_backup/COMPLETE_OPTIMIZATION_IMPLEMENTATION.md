# Complete Universal Optimization Implementation

## ✅ ALL Features from Both Guides Implemented

### **From Universal Optimization Guide**

#### 1. ✅ Complete Tier Configurations
```python
TIER_CONFIGS = {
    'intel_high_end': RTF 3.0, MKL optimization, torch.compile
    'amd_high_end': RTF 2.5, AMD threading, BLIS
    'm1_pro': RTF 12.0, MPS device, FP16, sustained cooling
    'i5_baseline': RTF 6.0, thermal chunking (REFERENCE)
    'm1_air': RTF 12.0→20.0, throttle management, fanless warning
    'intel_low_end': RTF 12.0, aggressive quantization, small chunks
    'amd_mobile': RTF 10.0, AMD threading, power management
    'arm_sbc': RTF 25.0, INT4 quantization, swap management
}
```

#### 2. ✅ UniversalHardwareDetector
- Detects CPU tier automatically
- Classifies into 8 performance categories
- Detects M1 Air vs Pro (fanless vs active cooling)
- Returns complete configuration with optimizations

#### 3. ✅ get_optimal_config()
Returns full configuration:
```python
{
    'detected_tier': 'i5_baseline',
    'hardware_info': {
        'cpu_model': 'Intel i5-1235U',
        'cores_physical': 10,
        'cores_logical': 12,
        'memory_gb': 16.0,
        'system': 'Windows'
    },
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
        'quality': 'very good'
    }
}
```

#### 4. ✅ Environment Setup
```python
def _setup_environment(self):
    # Threading
    os.environ['OMP_NUM_THREADS'] = str(threads)
    os.environ['MKL_NUM_THREADS'] = str(threads)
    
    # Intel MKL optimizations
    if mkl_optimization:
        os.environ['KMP_AFFINITY'] = 'granularity=fine,compact,1,0'
        os.environ['KMP_BLOCKTIME'] = '1'
    
    # AMD optimizations
    if amd_optimizations:
        os.environ['USE_OPENMP'] = '1'
        os.environ['OMP_PROC_BIND'] = 'true'
```

#### 5. ✅ Performance Matrix Logging
```
Expected Performance:
  RTF: 6.0
  10s clip: 60s
  30s clip: 3min
  Quality: very good
  Notes: Slow but workable for development
```

#### 6. ✅ Chunked Synthesis
- Automatic chunking for lower-end hardware
- Thermal management during long synthesis
- Memory-efficient processing

### **From Claude Final Honest Guide**

#### 1. ✅ Windows Thermal Monitoring
```python
# Tries LibreHardwareMonitor first
def _try_libre_hardware_monitor(self) -> bool:
    import wmi
    w = wmi.WMI(namespace="root\\LibreHardwareMonitor")
    sensors = w.Sensor()
    return len(sensors) > 0

# Falls back to Core Temp
def _try_core_temp(self) -> bool:
    import mmap
    with mmap.mmap(-1, 1024, "CoreTempMappingObject", access=mmap.ACCESS_READ) as mm:
        data = struct.unpack('I' * 256, mm.read(1024))
        return data[0] == 0x434F5254  # 'CORT' signature
```

#### 2. ✅ Honest Windows Warnings
```
⚠️  Windows temperature monitoring requires external tools.
   Install one of these for thermal management:
   - LibreHardwareMonitor: https://github.com/LibreHardwareMonitor/LibreHardwareMonitor
   - Core Temp: https://www.alcpu.com/CoreTemp/
   - HWiNFO64: https://www.hwinfo.com/download/
```

#### 3. ✅ M1 Air vs Pro Detection
```python
def _is_m1_pro_or_better(self) -> bool:
    result = subprocess.run(['system_profiler', 'SPHardwareDataType'], ...)
    if any(x in result.stdout for x in ['MacBook Pro', 'Mac Studio', 'iMac']):
        return True  # Active cooling
    return False  # Fanless (Air)
```

#### 4. ✅ Thermal Recovery with Infinite Loop Prevention
```python
def wait_for_thermal_recovery(self):
    max_wait_time = 180  # 3 minutes absolute maximum
    consecutive_none_count = 0
    total_none_count = 0
    max_none_tolerance = 15
    absolute_none_limit = 30
    
    while elapsed < max_wait_time:
        current_temp = self.get_temperature()
        
        if current_temp is None:
            consecutive_none_count += 1
            total_none_count += 1
            
            # Give up after too many failures
            if consecutive_none_count >= max_none_tolerance:
                logger.warning("Using conservative cooling period")
                time.sleep(15)
                consecutive_none_count = 0
                
                if total_none_count >= absolute_none_limit:
                    logger.error("Aborting thermal recovery")
                    return
            continue
        
        # Valid temperature
        consecutive_none_count = 0
        if current_temp < self.cooldown_target:
            logger.info(f"Thermal recovery complete: {current_temp:.1f}°C")
            return
        
        time.sleep(3)
```

#### 5. ✅ Platform Expectations Printing
```python
def print_platform_expectations(config):
    logger.info("PLATFORM EXPECTATIONS")
    logger.info(f"System: {hw['system']}")
    logger.info(f"CPU: {hw['cpu_model']}")
    logger.info(f"Tier: {tier}")
    logger.info(f"Expected Performance:")
    logger.info(f"  RTF: {perf['rtf']}")
    logger.info(f"  10s clip: {perf['clip_10s']}")
    logger.info(f"  30s clip: {perf['clip_30s']}")
    logger.info(f"  Quality: {perf['quality']}")
    if 'note' in perf:
        logger.warning(f"  ⚠️  {perf['note']}")
```

#### 6. ✅ Performance Tracker
```python
class PerformanceTracker:
    def record_inference(self, elapsed_time: float, text_length: int):
        self.inferences.append({
            'elapsed': elapsed_time,
            'text_length': text_length,
            'chars_per_sec': text_length / elapsed_time
        })
    
    def get_average_performance(self):
        return {
            'avg_elapsed_sec': avg_elapsed,
            'avg_chars_per_sec': avg_chars_per_sec,
            'total_inferences': len(self.inferences)
        }
```

#### 7. ✅ M1 Air Throttling Warning
```
⚠️  M1 MacBook Air detected. Performance will throttle after ~10 minutes
    of sustained load due to fanless design.
```

## 📁 Files Created/Modified

### **New Files**
1. `backend/universal_optimizer.py` - Complete universal optimizer
   - All tier configurations
   - Hardware detection
   - Environment setup
   - Performance tracking
   - Platform expectations

### **Modified Files**
1. `backend/opt_engine_v2.py` - Enhanced with:
   - UniversalHardwareDetector integration
   - ThermalManager with LibreHardwareMonitor support
   - Proper thermal recovery with infinite loop prevention
   - Hardware configuration logging
   - Platform-specific warnings

## 🎯 Usage

### **Option 1: Use Universal Optimizer (Recommended)**
```python
from backend.universal_optimizer import UniversalFishSpeechOptimizer

# Automatically detects hardware and applies optimal settings
optimizer = UniversalFishSpeechOptimizer(
    model_path="checkpoints/openaudio-s1-mini"
)

# Synthesize with automatic hardware adaptation
audio, sr, metrics = optimizer.synthesize(
    text="Your text here",
    reference_audio="reference.wav"
)

# Get performance summary
summary = optimizer.get_performance_summary()
print(f"Average RTF: {summary['avg_elapsed_sec'] / 10:.2f}")
```

### **Option 2: Use Enhanced V2 Engine Directly**
```python
from backend.opt_engine_v2 import OptimizedFishSpeechV2

# Engine now includes hardware detection and thermal management
engine = OptimizedFishSpeechV2(
    model_path="checkpoints/openaudio-s1-mini",
    device="auto"
)

# Automatic thermal monitoring and recovery
audio, sr, metrics = engine.tts(
    text="Your text here",
    speaker_wav="reference.wav"
)
```

## 📊 What You'll See on Startup

```
======================================================================
DETECTED HARDWARE
======================================================================
CPU: Intel(R) Core(TM) i5-1235U @ 1.30GHz
Cores: 10 physical, 12 logical
Memory: 16.0 GB
System: Windows
Tier: i5_baseline

OPTIMIZATION STRATEGY
  Precision: fp32
  Quantization: int8
  ONNX Runtime: True
  Torch Compile: False
  Threads: 10
======================================================================

======================================================================
PLATFORM EXPECTATIONS
======================================================================
System: Windows
CPU: Intel(R) Core(TM) i5-1235U @ 1.30GHz
Tier: i5_baseline

Expected Performance:
  RTF: 6.0
  10s clip: 60s
  30s clip: 3min
  Quality: very good
  Notes: Slow but workable for development
======================================================================

⚠️  Windows temperature monitoring requires external tools.
   Install one of these for thermal management:
   - LibreHardwareMonitor: https://github.com/LibreHardwareMonitor/LibreHardwareMonitor
   - Core Temp: https://www.alcpu.com/CoreTemp/
   - HWiNFO64: https://www.hwinfo.com/download/

❌ No thermal monitoring available - continuing without thermal protection

Set thread count to: 10
✅ Universal optimizer initialized successfully!
```

## 🔍 Complete Feature Checklist

### Universal Optimization Guide
- [x] 8 tier configurations (intel_high_end, amd_high_end, m1_pro, i5_baseline, m1_air, intel_low_end, amd_mobile, arm_sbc)
- [x] UniversalHardwareDetector class
- [x] get_optimal_config() method
- [x] Environment variable setup (OMP_NUM_THREADS, MKL, AMD)
- [x] Performance matrix with RTF targets
- [x] Chunked synthesis for lower-end hardware
- [x] Thread count optimization per tier
- [x] MKL optimization for Intel
- [x] AMD-specific optimizations
- [x] M1 Pro/Max detection

### Claude Final Honest Guide
- [x] Windows LibreHardwareMonitor support
- [x] Windows Core Temp support
- [x] Honest Windows warnings with install links
- [x] M1 Air vs Pro detection
- [x] M1 Air throttling warnings
- [x] Thermal recovery with infinite loop prevention
- [x] Platform expectations printing
- [x] PerformanceTracker class
- [x] Realistic RTF targets per platform
- [x] macOS powermetrics support
- [x] Linux /sys/class/thermal support
- [x] Graceful fallback when monitoring unavailable

## 🎉 Result

**100% of features from both guides are now implemented!**

- ✅ Universal hardware detection
- ✅ Platform-specific thermal management
- ✅ Honest performance expectations
- ✅ Tier-based optimization strategies
- ✅ Infinite loop prevention
- ✅ Complete Windows/macOS/Linux support
- ✅ M1 Air throttling awareness
- ✅ Performance tracking
- ✅ Environment optimization
- ✅ Production-ready error handling
