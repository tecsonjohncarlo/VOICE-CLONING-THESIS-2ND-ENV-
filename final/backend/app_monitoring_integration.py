"""
Integration code for monitoring in app.py

Add these endpoints to app.py after the existing endpoints
"""

import uuid
import asyncio

# Helper function to estimate tokens
def estimate_tokens(text: str) -> int:
    """Rough token estimation"""
    return int(len(text.split()) * 1.5)


# Add this to the /tts endpoint (replace existing implementation)
@app.post("/tts")
async def text_to_speech_with_monitoring(
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
    Generate speech from text with comprehensive monitoring
    
    NOW WITH: Real-time performance tracking and metrics export
    """
    if not engine or not monitor:
        raise HTTPException(status_code=503, detail="Engine not initialized")
    
    if not text or not text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")
    
    # Generate request ID
    request_id = str(uuid.uuid4())[:8]
    text_tokens = estimate_tokens(text)
    ref_audio_s = 0.0  # TODO: Calculate from speaker_file if provided
    
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
    
    # Start monitoring
    monitor.start_synthesis(text, text_tokens, ref_audio_s, request_id, engine.config)
    
    # Start background monitoring loop
    monitor_task = asyncio.create_task(monitor.monitor_loop())
    
    try:
        # Run TTS in thread pool
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
        
        # End monitoring (success)
        monitor.end_synthesis(success=True)
        
        # Read generated audio
        audio_bytes = output_path.read_bytes()
        
        # Create streaming response with enhanced metrics
        headers = {
            "Content-Disposition": "attachment; filename=output.wav",
            "X-Request-ID": request_id,
            "X-Latency-Ms": str(int(metrics['latency_ms'])),
            "X-Peak-VRAM-Mb": str(int(metrics.get('peak_vram_mb', 0))),
            "X-GPU-Util-Pct": str(int(metrics.get('gpu_util_pct', 0))),
            "X-Audio-Duration-S": str(round(metrics['audio_duration_s'], 2)),
            "X-RTF": str(round(metrics['rtf'], 2)),
            "X-Optimization-Strategy": engine.config.optimization_strategy,
            "X-Hardware-Tier": engine.profile.cpu_tier,
            "X-Max-Text-Length": str(engine.config.max_text_length)
        }
        
        response = StreamingResponse(
            io.BytesIO(audio_bytes),
            media_type="audio/wav",
            headers=headers
        )
        
        return response
        
    except Exception as e:
        # End monitoring (failure)
        monitor.end_synthesis(success=False, error=str(e))
        raise HTTPException(status_code=500, detail=f"TTS generation failed: {str(e)}")
    
    finally:
        # Cancel monitoring task
        monitor_task.cancel()
        try:
            await monitor_task
        except asyncio.CancelledError:
            pass
        
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


# Add this new endpoint
@app.get("/metrics/export")
async def export_metrics():
    """Export all collected metrics for analysis"""
    if not monitor:
        raise HTTPException(status_code=503, detail="Monitor not initialized")
    
    analysis = monitor.export_analysis()
    
    return {
        "csv_files": {
            "synthesis": str(monitor.synthesis_log_path),
            "realtime": str(monitor.realtime_log_path),
        },
        "analysis": analysis,
        "total_syntheses": len(monitor.all_syntheses),
        "hardware_tier": monitor.profile.cpu_tier,
    }


# Add this new endpoint
@app.get("/metrics/summary")
async def metrics_summary():
    """Get quick summary of monitoring data"""
    if not monitor:
        raise HTTPException(status_code=503, detail="Monitor not initialized")
    
    if not monitor.all_syntheses:
        return {
            "status": "no_data",
            "message": "No synthesis requests monitored yet"
        }
    
    successful = [s for s in monitor.all_syntheses if s.success]
    
    return {
        "total_requests": len(monitor.all_syntheses),
        "successful": len(successful),
        "failed": len(monitor.all_syntheses) - len(successful),
        "success_rate": len(successful) / len(monitor.all_syntheses) if monitor.all_syntheses else 0,
        "avg_duration_s": sum(s.total_duration_s for s in successful) / len(successful) if successful else 0,
        "avg_rtf": sum(s.rtf for s in successful) / len(successful) if successful else 0,
        "avg_peak_memory_mb": sum(s.peak_memory_mb for s in successful) / len(successful) if successful else 0,
        "hardware_tier": monitor.profile.cpu_tier,
        "device_type": monitor.profile.device_type,
    }


# Update the root endpoint to include new monitoring endpoints
@app.get("/")
async def root():
    """Root endpoint with enhanced information"""
    return {
        "name": "Optimized Fish Speech TTS API with Smart Adaptive Backend",
        "version": "2.2.0",  # Updated version
        "status": "running",
        "features": [
            "Auto hardware detection",
            "Self-optimizing configuration",
            "Real-time resource monitoring",
            "Adaptive performance tuning",
            "Memory budget management",
            "CPU affinity optimization",
            "Pre-synthesis memory estimation",
            "Comprehensive performance monitoring",  # NEW
            "CSV metrics export for analysis",  # NEW
        ],
        "endpoints": {
            "tts": "/tts (POST)",
            "voices": "/voices (GET)",
            "health": "/health (GET) - with smart insights",
            "metrics": "/metrics (GET) - with resource monitoring",
            "hardware": "/hardware (GET) - hardware profile",
            "emotions": "/emotions (GET)",
            "clear_cache": "/cache/clear (POST)",
            "estimate_memory": "/estimate-memory (POST) - pre-synthesis check",
            "system_status": "/system-status (GET) - real-time status",
            "optimize": "/optimize-for-hardware (POST) - force re-optimization",
            "metrics_export": "/metrics/export (GET) - NEW: export all metrics",  # NEW
            "metrics_summary": "/metrics/summary (GET) - NEW: quick summary",  # NEW
        }
    }
