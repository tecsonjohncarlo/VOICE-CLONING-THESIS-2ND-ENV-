"""
Platform Compatibility Matrix
Defines platform-specific capabilities and limitations
"""
from typing import Dict, Any

PLATFORM_COMPATIBILITY_MATRIX = {
    'Windows': {
        'thermal_monitoring': 'Requires external tools (LibreHardwareMonitor/Core Temp)',
        'torch_compile': 'Supported with TorchInductor',
        'mixed_precision': 'FP16 supported on compatible GPUs',
        'quantization': 'INT8 supported',
        'user_action_required': 'Install temperature monitoring tool',
        'fallback_behavior': 'No thermal protection',
        'recommended_tools': [
            'LibreHardwareMonitor: https://github.com/LibreHardwareMonitor/LibreHardwareMonitor',
            'Core Temp: https://www.alcpu.com/CoreTemp/',
            'HWiNFO64: https://www.hwinfo.com/download/'
        ]
    },
    'macOS_M1_Air': {
        'thermal_monitoring': 'Built-in (powermetrics)',
        'torch_compile': 'Supported with MPS backend',
        'mixed_precision': 'FP16 with MPS',
        'quantization': 'INT8 supported',
        'thermal_behavior': 'WILL throttle after 10-15 min',
        'expected_rtf': 'RTF 12-15 initially, degrades to 20+ when throttled',
        'throttle_time': 600,  # seconds
        'power_initial': 10,  # watts
        'power_throttled': 4,  # watts
        'performance_loss': '40-60%',
        'warning': 'Fanless design causes thermal saturation under sustained load'
    },
    'macOS_M1_Pro': {
        'thermal_monitoring': 'Built-in (powermetrics)',
        'torch_compile': 'Supported with MPS backend',
        'mixed_precision': 'FP16 with MPS',
        'quantization': 'INT8 supported',
        'thermal_behavior': 'Sustained performance with active cooling',
        'expected_rtf': 'RTF 12-15 sustained',
        'throttle_time': None,  # No throttling
        'power_sustained': 10,  # watts
        'performance_loss': '0%',
        'note': 'Active cooling maintains consistent performance'
    },
    'Linux': {
        'thermal_monitoring': 'Usually available via /sys/class/thermal',
        'torch_compile': 'Full support with TorchInductor',
        'mixed_precision': 'Full support',
        'quantization': 'Full support',
        'thermal_behavior': 'Varies by hardware',
        'expected_rtf': 'Depends on hardware (CPU: 20-30, GPU: 5-15)',
        'note': 'Most consistent platform with fewest limitations'
    }
}


def get_platform_info(platform_name: str, cpu_tier: str = None) -> Dict[str, Any]:
    """
    Get platform compatibility information
    
    Args:
        platform_name: 'Windows', 'Darwin', or 'Linux'
        cpu_tier: CPU tier from hardware detection (for macOS M1 differentiation)
    
    Returns:
        Platform compatibility dict
    """
    if platform_name == 'Darwin' and cpu_tier:
        if cpu_tier == 'm1_air':
            return PLATFORM_COMPATIBILITY_MATRIX['macOS_M1_Air']
        elif cpu_tier == 'm1_pro':
            return PLATFORM_COMPATIBILITY_MATRIX['macOS_M1_Pro']
    
    return PLATFORM_COMPATIBILITY_MATRIX.get(platform_name, {})


def print_platform_compatibility(platform_name: str, cpu_tier: str = None):
    """Print platform compatibility information"""
    from loguru import logger
    
    info = get_platform_info(platform_name, cpu_tier)
    
    if not info:
        logger.warning(f"No compatibility information for platform: {platform_name}")
        return
    
    logger.info("=" * 70)
    logger.info("PLATFORM COMPATIBILITY")
    logger.info("=" * 70)
    
    for key, value in info.items():
        if key == 'recommended_tools':
            logger.info(f"{key.replace('_', ' ').title()}:")
            for tool in value:
                logger.info(f"  - {tool}")
        elif key == 'warning':
            logger.warning(f"⚠️  {value}")
        elif key == 'note':
            logger.info(f"📝 {value}")
        else:
            logger.info(f"{key.replace('_', ' ').title()}: {value}")
    
    logger.info("=" * 70)


# Performance expectations per tier
PERFORMANCE_EXPECTATIONS = {
    'intel_high_end': {
        'rtf': 3.0,
        'clip_10s': '30s',
        'clip_30s': '90s',
        'quality': 'excellent',
        'notes': 'Desktop CPU with high core count and good cooling'
    },
    'amd_high_end': {
        'rtf': 2.5,
        'clip_10s': '25s',
        'clip_30s': '75s',
        'quality': 'excellent',
        'notes': 'Excellent multi-threading performance'
    },
    'm1_pro': {
        'rtf': 12.0,
        'clip_10s': '2min',
        'clip_30s': '6min',
        'quality': 'excellent',
        'notes': 'GPU acceleration with sustained cooling'
    },
    'i5_baseline': {
        'rtf': 6.0,
        'clip_10s': '60s',
        'clip_30s': '3min',
        'quality': 'very good',
        'notes': 'Reference baseline for consumer laptops'
    },
    'm1_air': {
        'rtf': 12.0,
        'rtf_throttled': 20.0,
        'clip_10s': '2min',
        'clip_30s': '6min',
        'quality': 'very good',
        'notes': 'Performance degrades after 10-15 min due to fanless design',
        'warning': 'Thermal throttling expected under sustained load'
    },
    'intel_low_end': {
        'rtf': 12.0,
        'clip_10s': '2min',
        'clip_30s': '6min',
        'quality': 'good',
        'notes': 'Older or lower-end CPUs, still usable'
    },
    'amd_mobile': {
        'rtf': 10.0,
        'clip_10s': '100s',
        'clip_30s': '5min',
        'quality': 'good',
        'notes': 'Mobile Ryzen CPUs, competitive performance'
    },
    'arm_sbc': {
        'rtf': 25.0,
        'clip_10s': '4min',
        'clip_30s': '12min',
        'quality': 'fair',
        'notes': 'Proof of concept only, very slow',
        'warning': 'Not recommended for production use'
    }
}


def get_performance_expectations(cpu_tier: str) -> Dict[str, Any]:
    """Get performance expectations for CPU tier"""
    return PERFORMANCE_EXPECTATIONS.get(cpu_tier, {})


def print_performance_expectations(cpu_tier: str):
    """Print performance expectations for CPU tier"""
    from loguru import logger
    
    perf = get_performance_expectations(cpu_tier)
    
    if not perf:
        logger.warning(f"No performance expectations for tier: {cpu_tier}")
        return
    
    logger.info("=" * 70)
    logger.info(f"PERFORMANCE EXPECTATIONS - {cpu_tier.upper()}")
    logger.info("=" * 70)
    logger.info(f"Expected RTF: {perf.get('rtf', 'unknown')}")
    
    if 'rtf_throttled' in perf:
        logger.warning(f"RTF (throttled): {perf['rtf_throttled']}")
    
    logger.info(f"10s clip: {perf.get('clip_10s', 'unknown')}")
    logger.info(f"30s clip: {perf.get('clip_30s', 'unknown')}")
    logger.info(f"Quality: {perf.get('quality', 'unknown')}")
    logger.info(f"Notes: {perf.get('notes', 'N/A')}")
    
    if 'warning' in perf:
        logger.warning(f"⚠️  {perf['warning']}")
    
    logger.info("=" * 70)
