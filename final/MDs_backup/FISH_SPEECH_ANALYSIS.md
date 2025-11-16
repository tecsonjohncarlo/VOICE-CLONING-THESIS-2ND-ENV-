# Fish Speech (OpenAudio S1-Mini) Architecture Analysis

## Overview
Fish Speech is a state-of-the-art zero-shot TTS system that achieved #1 ranking on TTS-Arena2. The OpenAudio S1-Mini variant is a distilled 0.5B parameter model optimized for local deployment.

## Architecture Components

### 1. Three-Stage Pipeline

#### Stage 1: VQ Token Extraction (Codec)
- **Module**: `fish_speech/models/dac/inference.py`
- **Purpose**: Extract Vector Quantized (VQ) tokens from reference audio
- **Input**: Reference audio file (WAV format)
- **Output**: `fake.npy` - VQ token embeddings
- **Model**: `codec.pth` - DAC (Descript Audio Codec) model
- **Key Features**:
  - Converts audio to discrete token representations
  - Enables voice cloning by capturing speaker characteristics
  - Cached for reuse with same reference audio

#### Stage 2: Semantic Token Generation (Text2Semantic)
- **Module**: `fish_speech/models/text2semantic/inference.py`
- **Purpose**: Generate semantic tokens from text using LLM
- **Input**: 
  - Text to synthesize
  - Optional: VQ tokens from reference (for voice cloning)
  - Optional: Prompt text (transcript of reference)
- **Output**: `codes_*.npy` - Semantic token sequence
- **Model**: Based on Qwen3 LLM architecture (0.5B parameters)
- **Key Features**:
  - Multilingual support (English, Chinese, Japanese, Korean, etc.)
  - Emotion markers support: `(excited)`, `(laughing)`, etc.
  - No phoneme dependency
  - Supports torch.compile for 10x speedup
  - Half-precision (FP16/BF16) support

#### Stage 3: Audio Synthesis (Codes2WAV)
- **Module**: `fish_speech/models/dac/inference.py` (mode: codes2wav)
- **Purpose**: Convert semantic tokens to audio waveform
- **Input**: Semantic tokens (`codes_*.npy`)
- **Output**: `fake.wav` - Synthesized audio
- **Model**: Same DAC codec used in reverse
- **Key Features**:
  - High-quality audio synthesis
  - Fast inference with GPU acceleration

### 2. Existing Wrapper Implementations

#### Basic Wrapper (`fish_speech_wrapper.py`)
- Simple subprocess-based interface
- XTTS-compatible API
- Basic VQ token caching
- No optimization features
- Issues found:
  - Line 215-217: Incorrect command (uses audio_path instead of semantic_tokens_path)
  - Fixed in optimized version

#### Optimized Wrapper (`(Optimized)Fish Speech Wrapper.py`)
- **Key Optimizations**:
  - Adaptive timeouts based on system resources
  - Memory management (gc.collect, torch.cuda.empty_cache)
  - Auto device detection (CUDA/CPU)
  - Audio preprocessing (mono conversion, resampling, trimming)
  - Text chunking for long inputs
  - Progressive fallback strategies
  - Resource monitoring (psutil)
  
- **Performance Features**:
  - CUDNN benchmarking enabled
  - TF32 matmul acceleration
  - Optimal thread count configuration
  - Memory fragmentation reduction
  - Compile mode support (disabled by default for stability)
  - Half-precision by default

## Model Specifications

### OpenAudio S1-Mini
- **Parameters**: 0.5B (distilled from 4B S1 model)
- **Architecture**: Qwen3-based transformer
- **Performance Metrics**:
  - WER: 0.011 (Word Error Rate)
  - CER: 0.005 (Character Error Rate)
  - Speaker Distance: 0.380
  - Real-time Factor: ~1:7 on RTX 4090
  
### Model Files
- `codec.pth` - DAC codec for VQ extraction and synthesis
- `model.pth` - Main text2semantic transformer model
- Additional config files in model directory

## Emotion & Control Markers

### Basic Emotions (24)
angry, sad, excited, surprised, satisfied, delighted, scared, worried, upset, nervous, frustrated, depressed, empathetic, embarrassed, disgusted, moved, proud, relaxed, grateful, confident, interested, curious, confused, joyful

### Advanced Emotions (26)
disdainful, unhappy, anxious, hysterical, indifferent, impatient, guilty, scornful, panicked, furious, reluctant, keen, disapproving, negative, denying, astonished, serious, sarcastic, conciliative, comforting, sincere, sneering, hesitating, yielding, painful, awkward, amused

### Tone Markers (5)
in a hurry tone, shouting, screaming, whispering, soft tone

### Special Effects (10)
laughing, chuckling, sobbing, crying loudly, sighing, panting, groaning, crowd laughing, background laughter, audience laughing

## Optimization Opportunities

### 1. Model-Level Optimizations
- **torch.compile**: 
  - Codec: `mode="reduce-overhead"`, `fullgraph=True`
  - Text2Semantic: `mode="max-autotune"`, `fullgraph=True`
  - Expected speedup: 2-3x
  
- **Mixed Precision**:
  - BF16 on compute capability ≥ 8.0 (Ampere+)
  - FP16 fallback for older GPUs
  - TF32 for matmul operations
  - Expected VRAM reduction: 40-50%

- **Quantization**:
  - INT8 dynamic quantization for Linear/Attention layers
  - 4-bit (NF4) via bitsandbytes (experimental)
  - Keep output heads in FP16/BF16 for quality
  - Expected VRAM reduction: 60-70% (INT8), 75% (4-bit)

### 2. Memory Optimizations
- **Chunking**: Process audio in chunks to cap peak VRAM
- **CUDA Streams**: Overlap VQ extraction, semantic generation, synthesis
- **Memory Pooling**: Preallocate buffers, use pinned memory
- **KV Cache**: Enable for transformer decoder
- **Gradient Checkpointing**: For training/fine-tuning

### 3. Caching Strategies
- **VQ Token Cache**: LRU cache keyed by reference audio path
- **Semantic Token Cache**: Cache by (speaker_id, text_hash)
- **Embedding Cache**: Cache speaker embeddings

### 4. System-Level Optimizations
- **CUDNN Benchmarking**: Auto-tune convolution algorithms
- **Thread Optimization**: Set optimal CPU thread count
- **Memory Fragmentation**: Configure PYTORCH_CUDA_ALLOC_CONF
- **Async I/O**: Non-blocking file operations

## Performance Targets (from claude.md)

### GPU Utilization
- Target: 8-15% on mid-tier GPUs (RTX 3060/4060)
- Baseline: ~30-40% without optimizations
- Reduction: 50-70%

### Latency
- Target: 50-70% reduction vs baseline
- Baseline RTF: ~1:7 on RTX 4090
- Expected: ~1:3-4 with optimizations

### VRAM Usage
- Baseline: ~4-6GB
- With INT8: ~2-3GB
- With 4-bit: ~1.5-2GB

## Implementation Strategy

### Backend Architecture (FastAPI)
```
POST /tts
├── Parse multipart form (text, audio, params)
├── Load OptimizedFishSpeech engine
├── Stage 1: Extract VQ tokens (cached)
├── Stage 2: Generate semantic tokens
├── Stage 3: Synthesize audio
└── Return StreamingResponse with metrics headers

GET /voices - List cached speakers
GET /health - System status
GET /metrics - Performance telemetry
```

### Frontend Architecture (Gradio)
```
Single-page UI
├── Text input (with emotion markers)
├── Reference audio upload
├── Parameter controls (temperature, top_p, speed, seed)
├── Optimization toggles (memory mode, compile, quantization)
├── Audio player output
└── Metrics bar (latency, VRAM, GPU%)
```

### Optimization Engine (opt_engine.py)
```python
class OptimizedFishSpeech:
    - Load model with quantization
    - Apply torch.compile selectively
    - Configure mixed precision
    - Setup CUDA streams
    - Initialize memory pools
    - Setup caching layers
    - Performance monitoring (NVML)
```

## Key Insights

1. **Subprocess-based approach**: Current wrappers use subprocess to call inference scripts
   - Pros: Simple, isolated, stable
   - Cons: Overhead, no direct model access, harder to optimize
   - Alternative: Direct model loading for better control

2. **Optimization trade-offs**:
   - Compile mode: Faster but less stable, longer warmup
   - Quantization: Lower VRAM but slight quality loss
   - Chunking: Lower peak VRAM but more overhead

3. **Critical paths**:
   - VQ extraction: I/O bound, benefits from audio optimization
   - Semantic generation: Compute bound, benefits from compile/quantization
   - Synthesis: Memory bound, benefits from chunking

4. **Caching effectiveness**:
   - VQ tokens: High hit rate for same speaker
   - Semantic tokens: Lower hit rate due to text variation
   - Best strategy: Cache VQ tokens aggressively

## Recommendations

### For Production Deployment
1. Use optimized wrapper as base
2. Implement direct model loading (avoid subprocess overhead)
3. Enable INT8 quantization by default
4. Use half-precision (FP16/BF16)
5. Disable compile mode initially (enable after testing)
6. Implement robust error handling and fallbacks
7. Add request queuing for concurrent requests
8. Monitor GPU memory and implement backpressure

### For Development
1. Start with subprocess-based approach (stable)
2. Add comprehensive logging and metrics
3. Implement A/B testing for optimizations
4. Profile each stage separately
5. Test on target hardware (RTX 3060/4060)
6. Validate audio quality at each optimization level

### For UI/UX
1. Show real-time progress for each stage
2. Display estimated time based on text length
3. Provide optimization presets (Quality/Balanced/Fast)
4. Allow emotion marker insertion via UI
5. Show VRAM usage warnings
6. Implement audio preview before full synthesis

## References
- [Fish Speech GitHub](https://github.com/fishaudio/fish-speech)
- [OpenAudio S1 Blog](https://openaudio.com/blogs/s1)
- [Hugging Face Model](https://huggingface.co/fishaudio/openaudio-s1-mini)
- [TTS-Arena2 Leaderboard](https://arena.speechcolab.org/)
