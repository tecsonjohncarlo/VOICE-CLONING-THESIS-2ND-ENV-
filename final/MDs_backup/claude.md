Build optimized web UI
Create a production-ready Python web application with a clean, minimal UI for zero-shot TTS voice cloning using OpenAudio S1‑Mini (Fish Audio), integrating the optimization pipeline defined above (quantization, mixed precision, torch.compile, chunking, CUDA streams, memory pooling). The local model directory with model.pth is already present; read it thoroughly and construct a robust runtime around it.

Targets

Reduce GPU utilization to ~8–15% on mid-tier GPUs and cut latency by 50–70% against the baseline, using the optimizations specified earlier.

Provide both a Gradio single-page UI and an optional Streamlit UI, backed by a FastAPI service for programmatic access and caching.

Use the local OpenAudio S1‑Mini model folder (default: checkpoints/openaudio-s1-mini), configurable via env.

Architecture

Backend (FastAPI):

POST /tts: accepts multipart form (text, reference audio, language, temperature, top_p, speed, seed, optimize_for_memory) and returns audio/wav bytes via StreamingResponse.

GET /voices: lists cached speakers and reference embeddings.

GET /health: reports device, compute capability, dtype, compile/quantization flags.

GET /metrics: returns rolling latency, peak VRAM, and GPU utilization (NVML).

Frontend A (default, Gradio):

Inputs: text, reference file upload, language selector, sliders for temperature/top_p/speed, seed, toggle for optimize_for_memory.

Outputs: audio player and compact metrics bar (latency, peak VRAM, GPU%).

Minimalist single-column layout with dark/light toggle.

Frontend B (optional, Streamlit):

Same controls and metrics; calls FastAPI endpoints.

Model layer:

Wrap the existing Fish/OpenAudio pipeline (VQ extraction → semantic generation → synthesis) with OptimizedFishSpeech:

torch.compile: codec mode="reduce-overhead", AR/decoder mode="max-autotune", fullgraph=True.

Mixed precision: bf16 on capability ≥ 8.0; else fp16. Enable TF32 (matmul + cudnn).

Quantization: dynamic INT8 for Linear and MultiheadAttention; optional 4‑bit (nf4) via bitsandbytes with graceful fallback.

Chunking: process VQ encode/decode and synthesis in chunks to cap peak VRAM; move intermediates to CPU immediately when feasible.

CUDA streams: overlap VQ, semantic generation, and synthesis across 3 streams with proper synchronization.

Memory: preallocation/pooling; pinned-memory DataLoader for reference audio; KV cache enabled for AR/decoder.

Caching: LRU caches for reference embeddings and semantic tokens keyed by (speaker_id, text hash).

Performance monitor:

Track per-request latency, peak GPU memory (torch.cuda.max_memory_allocated), and GPU utilization (pynvml).

Expose rolling aggregates via /metrics.

Deliverables

backend/app.py — FastAPI service with the endpoints above (Pydantic models, CORS).

backend/opt_engine.py — loads OpenAudio S1‑Mini and applies: torch.compile, mixed precision, quantization (INT8/4‑bit), chunking, CUDA streams, memory pooling, KV cache, caching, and performance telemetry.

ui/gradio_app.py — default Gradio UI wired to /tts and /metrics.

ui/streamlit_app.py — optional Streamlit UI using the same endpoints.

public/assets — favicon and small CSS theme for neat UI.

requirements.txt — see below.

README.md — run instructions (FastAPI + UI), tuning toggles, benchmarks guidance, and troubleshooting.

.env.example — MODEL_DIR, precision/quantization flags, server ports.

Configuration knobs (top of backend/opt_engine.py)

ENABLE_TORCH_COMPILE: True | False

MIXED_PRECISION: "bf16" | "fp16" | "fp32"

QUANTIZATION: "none" | "int8" | "4bit"

MAX_SEQ_LEN: int (default 1024)

CHUNK_SIZE: int (e.g., 8192 samples for VQ/audio chunks)

NUM_STREAMS: 3

CACHE_LIMIT: int (entries for embeddings/semantics)

Requirements (requirements.txt)

fastapi, uvicorn[standard]

gradio>=4.0 and streamlit (optional)

torch>=2.1, torchaudio

soundfile, librosa, numpy, pydub

bitsandbytes (optional; 4‑bit quantization)

pynvml (metrics)

python-dotenv

pydantic>=2

Key implementation details

Autocast dtype selection:

Use torch.bfloat16 on compute capability ≥ 8.0; else torch.float16.

Enable TF32: torch.backends.cuda.matmul.allow_tf32 = True; torch.backends.cudnn.allow_tf32 = True; cudnn.benchmark = True.

Quantization:

Default INT8 dynamic for Linear and MultiheadAttention; clamp-sensitive layers (output heads / vocoder) remain fp16/bf16.

4‑bit via bitsandbytes for Linear with nf4 compute_dtype=fp16; fall back to INT8 if kernels unavailable.

torch.compile:

codec = torch.compile(codec, mode="reduce-overhead", fullgraph=True)

text2sem / decoder = torch.compile(model, mode="max-autotune", fullgraph=True)

Chunking:

Split reference/audio into CHUNK_SIZE frames; move per-chunk outputs to CPU to reduce VRAM spikes.

CUDA streams:

Create vq_stream, semantic_stream, synthesis_stream; overlap stages and synchronize at boundaries.

Dynamic batching:

Compute optimal batch size from available VRAM and activation estimates; cap at a small number for stability; ensure power‑of‑2 where possible.

Caching:

LRU for embeddings and semantic tokens; invalidate on parameter changes (language, temperature, top_p, speed).

API responses:

/tts returns WAV audio stream plus x-metrics headers (latency-ms, peak-vram-mb, gpu-util-%).

Example run

Start backend: uvicorn backend.app:app --host 0.0.0.0 --port 8000

Start Gradio UI: python ui/gradio_app.py (default port 7860)

Optional Streamlit: streamlit run ui/streamlit_app.py

Example curl:

curl -F "text=Hello world" -F "speaker_file=@ref.wav" http://localhost:8000/tts --output out.wav

Acceptance criteria

On RTX 3060/4060, latency improves by 50–70% vs baseline; GPU utilization stabilizes below ~15% during synthesis.

Audio quality comparable to baseline under INT8; UI clearly labels 4‑bit as experimental.

Streaming responses; responsive UI; non-blocking main loop; graceful CPU fallback.

Notes

Use the existing local OpenAudio S1‑Mini folder; no external downloads required.

Keep UI minimal and visually clean with a compact metrics bar and clear controls.

Implement robust error handling and informative messages when features are unavailable on the current hardware.