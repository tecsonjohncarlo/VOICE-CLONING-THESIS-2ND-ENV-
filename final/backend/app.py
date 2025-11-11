"""
FastAPI Backend for Optimized Fish Speech TTS
WITH SMART ADAPTIVE BACKEND - Auto-detects hardware and self-optimizes

CHANGES FROM ORIGINAL:
1. Import SmartAdaptiveBackend instead of OptimizedFishSpeechV2
2. Remove device parameter (auto-detected)
3. Engine automatically optimizes based on hardware
"""

import os
import io
import tempfile
from pathlib import Path
from typing import Optional, List, Any
import asyncio
from concurrent.futures import ThreadPoolExecutor

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import soundfile as sf
import numpy as np
import psutil  # For memory estimation

# ========================================
# CHANGED: Use Smart Adaptive Backend
# ========================================
from smart_backend import SmartAdaptiveBackend as OptimizedFishSpeech
print("[INFO] Using SmartAdaptiveBackend (auto-optimizing)")

# Initialize FastAPI app
app = FastAPI(
    title="Optimized Fish Speech TTS API with Smart Backend",
    description="Self-optimizing TTS API that auto-detects hardware and adapts configuration",
    version="2.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global engine instance
engine: Optional[OptimizedFishSpeech] = None
executor = ThreadPoolExecutor(max_workers=4)

# Global performance monitor
monitor: Optional[Any] = None


# Pydantic models (unchanged)
class TTSRequest(BaseModel):
    """TTS request parameters"""
    text: str = Field(..., description="Text to synthesize")
    language: str = Field("en", description="Language code (auto-detected)")
    temperature: float = Field(0.7, ge=0.1, le=2.0, description="Sampling temperature")
    top_p: float = Field(0.7, ge=0.1, le=1.0, description="Nucleus sampling parameter")
    speed: float = Field(1.0, ge=0.5, le=2.0, description="Speech speed (not implemented)")
    seed: Optional[int] = Field(None, description="Random seed (not implemented)")
    optimize_for_memory: bool = Field(False, description="Prioritize memory over speed")
    prompt_text: Optional[str] = Field(None, description="Transcript of reference audio")


class HealthResponse(BaseModel):
    """Health check response with smart insights"""
    status: str
    device: str
    system_info: dict
    cache_stats: dict
    smart_insights: Optional[List[str]] = None
    current_resources: Optional[dict] = None
    hardware_profile: Optional[dict] = None


class MetricsResponse(BaseModel):
    """Metrics response with resource monitoring"""
    rolling_aggregates: dict
    current_gpu_util: float
    current_resources: Optional[dict] = None


class VoiceInfo(BaseModel):
    """Voice/speaker information"""
    id: str
    name: str
    cached: bool


@app.on_event("startup")
async def startup_event():
    """
    Initialize smart adaptive engine on startup
    
    CHANGED: Removed device parameter - auto-detected by smart backend
    """
    global engine, monitor
    
    model_path = os.getenv("MODEL_DIR", "checkpoints/openaudio-s1-mini")
    
    try:
        # ========================================
        # CHANGED: Just pass model_path
        # Smart backend auto-detects everything!
        # ========================================
        engine = OptimizedFishSpeech(model_path=model_path)
        
        print("[OK] Smart Adaptive Engine initialized")
        print(f"[INFO] Detected device: {engine.profile.device_type}")
        print(f"[INFO] CPU tier: {engine.profile.cpu_tier}")
        print(f"[INFO] Optimization strategy: {engine.config.optimization_strategy}")
        print(f"[INFO] Expected RTF: {engine.config.expected_rtf:.1f}x")
        
        # Initialize performance monitor
        from monitoring import PerformanceMonitor
        monitor = PerformanceMonitor(engine.profile)
        print(f"[OK] Performance monitoring initialized")
        
    except Exception as e:
        print(f"[ERROR] Failed to initialize engine: {e}")
        raise


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    global engine
    if engine:
        engine.cleanup()
        print("[OK] Engine cleaned up")


def run_tts_sync(text: str, 
                 speaker_file: Optional[Path],
                 prompt_text: Optional[str],
                 temperature: float,
                 top_p: float,
                 speed: float,
                 seed: Optional[int],
                 output_path: Path) -> dict:
    """
    Run TTS synchronously in thread pool
    
    UNCHANGED: Smart backend has same interface as OptimizedFishSpeechV2
    """
    audio, sr, metrics = engine.tts(
        text=text,
        speaker_wav=speaker_file,
        prompt_text=prompt_text,
        temperature=temperature,
        top_p=top_p,
        seed=seed,
        output_path=output_path
    )
    return {'audio': audio, 'sr': sr, 'metrics': metrics}


@app.post("/tts")
async def text_to_speech(
    text: str = Form(..., description="Text to synthesize"),
    speaker_file: Optional[UploadFile] = File(None, description="Reference audio file"),
    prompt_text: Optional[str] = Form(None, description="Transcript of reference audio"),
    language: str = Form("en", description="Language code"),
    temperature: float = Form(0.7, description="Sampling temperature"),
    top_p: float = Form(0.7, description="Nucleus sampling parameter"),
    speed: float = Form(1.0, description="Speech speed"),
    seed: Optional[int] = Form(None, description="Random seed"),
    optimize_for_memory: bool = Form(False, description="Memory optimization"),
    force_cpu: bool = Form(False, description="Force CPU mode (disable GPU)")
):
    """
    Generate speech from text with optional voice cloning
    
    NOW WITH: Automatic resource monitoring and adaptive optimization
    ADDED: force_cpu parameter to disable GPU at runtime
    
    Returns audio/wav with metrics in response headers
    """
    if not engine:
        raise HTTPException(status_code=503, detail="Engine not initialized")
    
    # Intelligent device selection (if DEVICE=auto)
    # Note: force_cpu is handled by temporarily overriding device in TTS call
    # Moving models at runtime causes tensor device mismatch errors
    
    if not force_cpu and not engine.device_locked:
        # Smart device selection enabled (DEVICE=auto)
        try:
            decision = engine.check_and_optimize_device()
            print(f"[INFO] Smart device decision: {decision['device']} - {decision['reason']}")
        except Exception as e:
            print(f"[WARNING] Smart device selection failed: {e}")
    
    if force_cpu:
        print(f"[INFO] Force CPU requested via UI - will use CPU for this request")
        print(f"[WARNING] Note: Force CPU is experimental and may not work properly")
        print(f"[TIP] For reliable CPU mode, set DEVICE=cpu in .env and restart backend")
    
    if not text or not text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")
    
    # Save uploaded speaker file if provided
    speaker_path = None
    if speaker_file:
        try:
            suffix = Path(speaker_file.filename).suffix if speaker_file.filename else ".wav"
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                content = await speaker_file.read()
                tmp.write(content)
                speaker_path = Path(tmp.name)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed to process speaker file: {e}")
    
    # Create temp output file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        output_path = Path(tmp.name)
    
    # Generate request ID for tracking
    import uuid
    request_id = str(uuid.uuid4())[:8]
    
    try:
        # Start monitoring (logs to CSV)
        if monitor:
            ref_audio_duration = 0.0
            if speaker_path and speaker_path.exists():
                try:
                    import soundfile as sf
                    audio_data, sr = sf.read(speaker_path)
                    ref_audio_duration = len(audio_data) / sr
                except:
                    pass
            
            monitor.start_synthesis(
                text=text,
                text_tokens=len(text.split()),  # Rough estimate
                ref_audio_s=ref_audio_duration,
                request_id=request_id,
                config=engine.config
            )
            # Start background monitoring loop
            monitor_task = asyncio.create_task(monitor.monitor_loop())
        
        # Run TTS in thread pool (smart backend handles resource monitoring)
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            executor,
            run_tts_sync,
            text,
            speaker_path,
            prompt_text,
            temperature,
            top_p,
            speed,
            seed,
            output_path
        )
        
        metrics = result['metrics']
        
        # End monitoring and save to CSV
        if monitor:
            # Update synthesis metrics with actual values from TTS engine
            if monitor.current_synthesis:
                monitor.current_synthesis.audio_duration_s = metrics.get('audio_duration_s', 0.0)
                monitor.current_synthesis.peak_gpu_util_pct = metrics.get('gpu_util_pct', 0.0)
                
                # Fish Speech specific metrics
                monitor.current_synthesis.generated_tokens = metrics.get('fish_tokens_generated', 0)
                monitor.current_synthesis.fish_tokens_per_sec = metrics.get('fish_tokens_per_sec', 0.0)
                monitor.current_synthesis.fish_bandwidth_gb_s = metrics.get('fish_bandwidth_gb_s', 0.0)
                monitor.current_synthesis.fish_gpu_memory_gb = metrics.get('fish_gpu_memory_gb', 0.0)
                monitor.current_synthesis.fish_generation_time_s = metrics.get('fish_generation_time_s', 0.0)
                monitor.current_synthesis.vq_features_shape = metrics.get('vq_features_shape', '')
            
            # Stop monitoring loop
            monitor.monitoring_active = False
            # Wait for monitoring task to finish
            try:
                await asyncio.wait_for(monitor_task, timeout=2.0)
            except:
                pass
            # Finalize and save metrics
            monitor.end_synthesis(success=True)
        
        # Read generated audio
        audio_bytes = output_path.read_bytes()
        
        # Create streaming response with enhanced metrics including text truncation info
        headers = {
            "Content-Disposition": "attachment; filename=output.wav",
            "X-Latency-Ms": str(int(metrics['latency_ms'])),
            "X-Peak-VRAM-Mb": str(int(metrics.get('peak_vram_mb', 0))),
            "X-GPU-Util-Pct": str(int(metrics.get('gpu_util_pct', 0))),
            "X-Audio-Duration-S": str(round(metrics['audio_duration_s'], 2)),
            "X-RTF": str(round(metrics['rtf'], 2)),
            "X-Optimization-Strategy": engine.config.optimization_strategy,
            "X-Hardware-Tier": engine.profile.cpu_tier,
            "X-Max-Text-Length": str(engine.config.max_text_length)
        }
        
        # Add truncation info if text was truncated
        if hasattr(metrics, 'get') and metrics.get('text_truncated'):
            headers["X-Text-Truncated"] = "true"
            headers["X-Original-Text-Length"] = str(metrics.get('original_text_length', 0))
            headers["X-Truncated-Text-Length"] = str(metrics.get('truncated_text_length', 0))
        
        response = StreamingResponse(
            io.BytesIO(audio_bytes),
            media_type="audio/wav",
            headers=headers
        )
        
        return response
        
    except Exception as e:
        # End monitoring with error
        if monitor and 'monitor_task' in locals():
            monitor.monitoring_active = False
            try:
                await asyncio.wait_for(monitor_task, timeout=1.0)
            except:
                pass
            monitor.end_synthesis(success=False, error=str(e))
        raise HTTPException(status_code=500, detail=f"TTS generation failed: {str(e)}")
    
    finally:
        # Cleanup temp files
        if speaker_path and speaker_path.exists():
            try:
                speaker_path.unlink()
            except:
                pass
        if output_path.exists():
            try:
                output_path.unlink()
            except:
                pass


@app.get("/voices", response_model=List[VoiceInfo])
async def list_voices():
    """
    List cached speakers and reference embeddings
    
    UNCHANGED
    """
    if not engine:
        raise HTTPException(status_code=503, detail="Engine not initialized")
    
    voices = []
    
    # Get cached references
    if hasattr(engine.engine, 'reference_cache'):
        cache = engine.engine.reference_cache
        for i, (key, value) in enumerate(cache.cache.items()):
            voices.append(VoiceInfo(
                id=f"voice_{i}",
                name=Path(key).name if isinstance(key, (str, Path)) else f"cached_{i}",
                cached=True
            ))
    
    return voices


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Get system health status with smart insights
    
    ENHANCED: Now includes intelligent insights and resource monitoring
    """
    if not engine:
        raise HTTPException(status_code=503, detail="Engine not initialized")
    
    # Get enhanced health from smart backend
    health = engine.get_health()
    return HealthResponse(**health)


@app.get("/metrics", response_model=MetricsResponse)
async def get_metrics():
    """
    Get performance metrics with resource monitoring
    
    ENHANCED: Now includes real-time resource usage
    """
    if not engine:
        raise HTTPException(status_code=503, detail="Engine not initialized")
    
    metrics = engine.get_metrics()
    return MetricsResponse(**metrics)


@app.post("/cache/clear")
async def clear_cache():
    """Clear all caches"""
    if not engine:
        raise HTTPException(status_code=503, detail="Engine not initialized")
    
    engine.clear_cache()
    return {"status": "success", "message": "Cache cleared"}


@app.get("/emotions")
async def list_emotions():
    """
    Get emotion and prosody guidance
    
    ⚠️ IMPORTANT: Fish Speech is a VOICE CLONING model, not emotion-controlled TTS.
    Emotions come from your REFERENCE AUDIO, not from text tags.
    
    These are suggestions for text formatting and reference audio selection.
    """
    return {
        'note': '⚠️ Emotions come from your REFERENCE AUDIO, not text tags!',
        'how_to_add_emotion': {
            'primary': 'Use an emotionally expressive reference audio file',
            'secondary': 'Use punctuation for prosody (!, ?, ..., —)',
            'advanced': 'Use multiple reference audios for different emotions'
        },
        'prosody_markers': {
            'description': 'Use punctuation to hint at prosody',
            'markers': {
                '...': 'Pause, uncertainty, trailing off',
                '!': 'Excitement, emphasis, surprise',
                '?': 'Question, uncertainty',
                '?!': 'Shocked question',
                '—': 'Interruption, sudden stop',
                'CAPS': 'Emphasis (use sparingly)'
            }
        },
        'reference_audio_tips': {
            'excited': 'Use reference with high energy, varied pitch',
            'nervous': 'Use reference with hesitation, softer tone',
            'angry': 'Use reference with strong emphasis, louder volume',
            'sad': 'Use reference with lower pitch, slower pace',
            'confident': 'Use reference with steady, clear delivery'
        },
        'sound_effects': {
            'description': 'These MAY work if in reference audio',
            'effects': [
                'laughing', 'chuckling', 'sobbing', 'crying', 'sighing',
                'panting', 'groaning', 'whispering'
            ]
        }
    }


# ========================================
# NEW ENDPOINT: Hardware Profile
# ========================================
@app.get("/hardware")
async def get_hardware_profile():
    """
    Get detected hardware profile and selected configuration
    
    NEW: Shows what the smart backend detected and why it chose this config
    """
    if not engine:
        raise HTTPException(status_code=503, detail="Engine not initialized")
    
    profile = engine.profile
    config = engine.config
    
    return {
        "hardware_profile": {
            "system": profile.system,
            "cpu_model": profile.cpu_model,
            "cpu_tier": profile.cpu_tier,
            "cores_physical": profile.cores_physical,
            "cores_logical": profile.cores_logical,
            "memory_gb": profile.memory_gb,
            "device_type": profile.device_type,
            "gpu_name": profile.gpu_name,
            "gpu_memory_gb": profile.gpu_memory_gb,
            "thermal_capable": profile.thermal_capable,
            "avx512_vnni": profile.avx512_vnni
        },
        "selected_configuration": {
            "device": config.device,
            "precision": config.precision,
            "quantization": config.quantization,
            "use_onnx": config.use_onnx,
            "use_torch_compile": config.use_torch_compile,
            "chunk_length": config.chunk_length,
            "num_threads": config.num_threads,
            "max_text_length": config.max_text_length,
            "expected_rtf": config.expected_rtf,
            "expected_memory_gb": config.expected_memory_gb,
            "optimization_strategy": config.optimization_strategy,
            "notes": config.notes
        }
    }


async def estimate_memory_for_synthesis(text: str, hardware_tier: str) -> dict:
    """Estimate memory usage before synthesis starts"""
    
    # Rough estimates based on text length
    text_tokens = len(text.split()) * 1.5  # Word → token conversion
    
    estimates = {
        "m1_air": {
            "base_model_mb": 2000,
            "per_100_tokens_mb": 50,
            "cache_overhead_mb": 200,
        },
        "m1_pro": {
            "base_model_mb": 3000,
            "per_100_tokens_mb": 40,
            "cache_overhead_mb": 300,
        },
        "intel_i5": {
            "base_model_mb": 2500,
            "per_100_tokens_mb": 60,
            "cache_overhead_mb": 250,
        },
        "amd_ryzen5": {
            "base_model_mb": 2500,
            "per_100_tokens_mb": 60,
            "cache_overhead_mb": 250,
        },
        "intel_high_end": {
            "base_model_mb": 3000,
            "per_100_tokens_mb": 50,
            "cache_overhead_mb": 300,
        },
    }
    
    est = estimates.get(hardware_tier, estimates["intel_i5"])
    total_mb = (est["base_model_mb"] + 
               (text_tokens / 100) * est["per_100_tokens_mb"] +
               est["cache_overhead_mb"])
    
    available_mb = psutil.virtual_memory().available / (1024**2)
    
    return {
        "estimated_mb": total_mb,
        "available_mb": available_mb,
        "safe": total_mb < (available_mb * 0.8),  # Leave 20% buffer
        "text_tokens": text_tokens,
        "text_length": len(text)
    }


@app.post("/estimate-memory")
async def estimate_memory(text: str = Form(...)):
    """Estimate memory before synthesis"""
    if not engine:
        raise HTTPException(status_code=503, detail="Engine not initialized")
    
    estimate = await estimate_memory_for_synthesis(text, engine.profile.cpu_tier)
    
    if not estimate["safe"]:
        max_words = int(estimate['available_mb'] * 0.6 / 50)
        return {
            "status": "warning",
            "estimate": estimate,
            "recommendation": f"Text may be too long. Recommended max: {max_words} words ({engine.config.max_text_length} characters)"
        }
    
    return {"status": "ok", "estimate": estimate}


@app.get("/system-status")
async def system_status():
    """Real-time system status with throttling detection"""
    if not engine:
        raise HTTPException(status_code=503, detail="Engine not initialized")
    
    resources = engine.monitor.check_resources()
    
    throttling_risk = {
        "memory_critical": resources['memory_percent'] > 85,
        "cpu_maxed": resources['cpu_percent'] > 90,
        "thermal_likely": engine.profile.cpu_tier == "m1_air" and resources['cpu_percent'] > 70,
    }
    
    # Check memory budget
    memory_safe = engine.monitor.memory_budget_manager.enforce_limits()
    
    return {
        "resources": resources,
        "throttling_risk": throttling_risk,
        "memory_budget_safe": memory_safe,
        "recommended_action": (
            "Reduce text length or clear cache" if throttling_risk['memory_critical'] 
            else "Wait for system to cool" if throttling_risk['thermal_likely']
            else "Reduce CPU load" if throttling_risk['cpu_maxed']
            else "OK"
        )
    }


@app.post("/optimize-for-hardware")
async def optimize_for_hardware():
    """Force re-optimization of current hardware settings"""
    if not engine:
        raise HTTPException(status_code=503, detail="Engine not initialized")
    
    # Clear caches
    engine.clear_cache()
    
    # Re-check resources
    resources = engine.monitor.check_resources()
    
    # Suggest adjusted config if needed
    adjusted = engine.monitor.suggest_adjustment(resources, engine.config)
    
    if adjusted:
        return {
            "status": "adjusted",
            "previous_config": {
                "chunk_length": engine.config.chunk_length,
                "num_threads": engine.config.num_threads,
                "cache_limit": engine.config.cache_limit,
                "max_text_length": engine.config.max_text_length,
            },
            "new_config": {
                "chunk_length": adjusted.chunk_length,
                "num_threads": adjusted.num_threads,
                "cache_limit": adjusted.cache_limit,
                "max_text_length": adjusted.max_text_length,
            }
        }
    
    return {"status": "optimal", "message": "No adjustments needed"}


@app.get("/")
async def root():
    """Root endpoint with enhanced information"""
    return {
        "name": "Optimized Fish Speech TTS API with Smart Adaptive Backend",
        "version": "2.1.0",
        "status": "running",
        "features": [
            "Auto hardware detection",
            "Self-optimizing configuration",
            "Real-time resource monitoring",
            "Adaptive performance tuning",
            "Memory budget management",
            "CPU affinity optimization",
            "Pre-synthesis memory estimation"
        ],
        "endpoints": {
            "tts": "/tts (POST)",
            "voices": "/voices (GET)",
            "health": "/health (GET) - with smart insights",
            "metrics": "/metrics (GET) - with resource monitoring",
            "hardware": "/hardware (GET) - hardware profile",
            "emotions": "/emotions (GET)",
            "clear_cache": "/cache/clear (POST)",
            "estimate_memory": "/estimate-memory (POST) - NEW: pre-synthesis check",
            "system_status": "/system-status (GET) - NEW: real-time status",
            "optimize": "/optimize-for-hardware (POST) - NEW: force re-optimization"
        }
    }


if __name__ == "__main__":
    import uvicorn
    
    port = int(os.getenv("PORT", "8000"))
    host = os.getenv("HOST", "0.0.0.0")
    
    print("="*70)
    print("Starting Fish Speech TTS API with Smart Adaptive Backend")
    print("="*70)
    print("Features:")
    print("  ✅ Automatic hardware detection")
    print("  ✅ Self-optimizing configuration")
    print("  ✅ Real-time resource monitoring")
    print("  ✅ Adaptive performance tuning")
    print("="*70)
    
    uvicorn.run(
        "app:app",
        host=host,
        port=port,
        reload=False,
        log_level="info"
    )