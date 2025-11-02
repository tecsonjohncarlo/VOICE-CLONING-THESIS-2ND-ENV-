"""
FastAPI Backend for Optimized Fish Speech TTS
Provides REST API endpoints for TTS synthesis with performance monitoring
"""

import os
import io
import tempfile
from pathlib import Path
from typing import Optional, List
import asyncio
from concurrent.futures import ThreadPoolExecutor

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import soundfile as sf
import numpy as np

# Try V2 engine first (uses direct imports), fallback to V1 (subprocess)
try:
    from opt_engine_v2 import OptimizedFishSpeechV2 as OptimizedFishSpeech
    print("[INFO] Using OptimizedFishSpeechV2 (direct imports)")
except ImportError as e:
    print(f"[WARNING] V2 engine not available ({e}), falling back to V1")
    from opt_engine import OptimizedFishSpeech

# Initialize FastAPI app
app = FastAPI(
    title="Optimized Fish Speech TTS API",
    description="High-performance TTS API with voice cloning using OpenAudio S1-Mini",
    version="1.0.0"
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


# Pydantic models
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
    """Health check response"""
    status: str
    device: str
    system_info: dict
    cache_stats: dict


class MetricsResponse(BaseModel):
    """Metrics response"""
    rolling_aggregates: dict
    current_gpu_util: float


class VoiceInfo(BaseModel):
    """Voice/speaker information"""
    id: str
    name: str
    cached: bool


@app.on_event("startup")
async def startup_event():
    """Initialize engine on startup"""
    global engine
    
    model_path = os.getenv("MODEL_DIR", "checkpoints/openaudio-s1-mini")
    device = os.getenv("DEVICE", "auto")
    
    try:
        engine = OptimizedFishSpeech(
            model_path=model_path,
            device=device,
            enable_optimizations=True
        )
        print("[OK] Engine initialized successfully")
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
    """Run TTS synchronously in thread pool"""
    # Note: speed parameter not implemented in Fish Speech, ignored
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
    optimize_for_memory: bool = Form(False, description="Memory optimization")
):
    """
    Generate speech from text with optional voice cloning
    
    Returns audio/wav with metrics in response headers
    """
    if not engine:
        raise HTTPException(status_code=503, detail="Engine not initialized")
    
    if not text or not text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")
    
    # Save uploaded speaker file if provided
    speaker_path = None
    if speaker_file:
        try:
            # Create temp file for speaker audio
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
    
    try:
        # Run TTS in thread pool to avoid blocking
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
        
        # Read generated audio
        audio_bytes = output_path.read_bytes()
        
        # Create streaming response
        response = StreamingResponse(
            io.BytesIO(audio_bytes),
            media_type="audio/wav",
            headers={
                "Content-Disposition": "attachment; filename=output.wav",
                "X-Latency-Ms": str(int(metrics['latency_ms'])),
                "X-Peak-VRAM-Mb": str(int(metrics['peak_vram_mb'])),
                "X-GPU-Util-Pct": str(int(metrics['gpu_util_pct'])),
                "X-Audio-Duration-S": str(round(metrics['audio_duration_s'], 2)),
                "X-RTF": str(round(metrics['rtf'], 2))
            }
        )
        
        return response
        
    except Exception as e:
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
    """
    if not engine:
        raise HTTPException(status_code=503, detail="Engine not initialized")
    
    voices = []
    
    # Get cached references (V2) or VQ tokens (V1)
    cache = getattr(engine, 'reference_cache', None) or getattr(engine, 'vq_cache', None)
    if cache:
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
    Get system health status
    
    Returns device info, compute capability, dtype, and optimization flags
    """
    if not engine:
        raise HTTPException(status_code=503, detail="Engine not initialized")
    
    health = engine.get_health()
    return HealthResponse(**health)


@app.get("/metrics", response_model=MetricsResponse)
async def get_metrics():
    """
    Get performance metrics
    
    Returns rolling latency, peak VRAM, and GPU utilization
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
    """Get available emotion markers"""
    emotions = {
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
    return emotions


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "name": "Optimized Fish Speech TTS API",
        "version": "1.0.0",
        "status": "running",
        "endpoints": {
            "tts": "/tts (POST)",
            "voices": "/voices (GET)",
            "health": "/health (GET)",
            "metrics": "/metrics (GET)",
            "emotions": "/emotions (GET)",
            "clear_cache": "/cache/clear (POST)"
        }
    }


if __name__ == "__main__":
    import uvicorn
    
    port = int(os.getenv("PORT", "8000"))
    host = os.getenv("HOST", "0.0.0.0")
    
    uvicorn.run(
        "app:app",
        host=host,
        port=port,
        reload=False,
        log_level="info"
    )
