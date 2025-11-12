"""
Gradio Web UI for Optimized Fish Speech TTS
Clean, minimal interface with real-time metrics
"""

import os
import sys
import requests
import gradio as gr
from pathlib import Path
import json

# API configuration
API_URL = os.getenv("API_URL", "http://localhost:8000")


def get_health():
    """Get system health from API"""
    try:
        response = requests.get(f"{API_URL}/health", timeout=5)
        if response.status_code == 200:
            return response.json()
        return None
    except:
        return None


def get_emotions():
    """Get available emotions from API"""
    try:
        response = requests.get(f"{API_URL}/emotions", timeout=5)
        if response.status_code == 200:
            return response.json()
        return {}
    except:
        return {}


def synthesize_speech(
    text,
    speaker_file,
    prompt_text,
    language,
    temperature,
    top_p,
    speed,
    seed,
    optimize_for_memory,
    force_cpu
):
    """
    Call TTS API and return audio with metrics
    """
    if not text or not text.strip():
        return None, "❌ Error: Text cannot be empty", ""
    
    try:
        # Prepare form data
        files = {}
        data = {
            'text': text,
            'language': language,
            'temperature': temperature,
            'top_p': top_p,
            'speed': speed,
            'optimize_for_memory': optimize_for_memory,
            'force_cpu': force_cpu
        }
        
        if prompt_text:
            data['prompt_text'] = prompt_text
        
        if seed is not None and seed > 0:
            data['seed'] = seed
        
        # Add speaker file if provided
        if speaker_file is not None:
            files['speaker_file'] = open(speaker_file, 'rb')
        
        # Make request
        response = requests.post(
            f"{API_URL}/tts",
            data=data,
            files=files,
            timeout=300
        )
        
        # Close file if opened
        if files:
            files['speaker_file'].close()
        
        if response.status_code != 200:
            error_detail = response.json().get('detail', 'Unknown error')
            return None, f"❌ Error: {error_detail}", ""
        
        # Get metrics from headers
        latency = response.headers.get('X-Latency-Ms', 'N/A')
        vram = response.headers.get('X-Peak-VRAM-Mb', 'N/A')
        gpu_util = response.headers.get('X-GPU-Util-Pct', 'N/A')
        duration = response.headers.get('X-Audio-Duration-S', 'N/A')
        rtf = response.headers.get('X-RTF', 'N/A')
        
        # Format metrics
        metrics_text = f"""
### 📊 Performance Metrics

- **Latency**: {latency} ms
- **Peak VRAM**: {vram} MB
- **GPU Utilization**: {gpu_util}%
- **Audio Duration**: {duration}s
- **Real-Time Factor**: {rtf}x
"""
        
        # Save audio to temp file
        import tempfile
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp:
            tmp.write(response.content)
            audio_path = tmp.name
        
        status_text = "✅ Synthesis completed successfully!"
        
        return audio_path, status_text, metrics_text
        
    except requests.exceptions.Timeout:
        return None, "❌ Error: Request timed out (>5 minutes)", ""
    except Exception as e:
        return None, f"❌ Error: {str(e)}", ""


def clear_cache():
    """Clear API cache"""
    try:
        response = requests.post(f"{API_URL}/cache/clear", timeout=5)
        if response.status_code == 200:
            return "✅ Cache cleared successfully!"
        return "❌ Failed to clear cache"
    except Exception as e:
        return f"❌ Error: {str(e)}"


def format_emotion_guide():
    """Format emotion guide for display"""
    emotions = get_emotions()
    
    if not emotions:
        return "Unable to load emotions. Check API connection."
    
    guide = "# 🎭 Emotion & Control Markers Guide\n\n"
    guide += "Add emotion markers to your text using parentheses: `(emotion) text`\n\n"
    
    guide += "## Basic Emotions\n"
    guide += ", ".join(f"`({e})`" for e in emotions.get('basic', [])[:12])
    guide += "\n\n"
    
    guide += "## Advanced Emotions\n"
    guide += ", ".join(f"`({e})`" for e in emotions.get('advanced', [])[:12])
    guide += "\n\n"
    
    guide += "## Tone Markers\n"
    guide += ", ".join(f"`({t})`" for t in emotions.get('tones', []))
    guide += "\n\n"
    
    guide += "## Special Effects\n"
    guide += ", ".join(f"`({e})`" for e in emotions.get('effects', []))
    guide += "\n\n"
    
    guide += "### Examples:\n"
    guide += "- `(excited) Hello! This is amazing!`\n"
    guide += "- `(whispering) Can you hear me?`\n"
    guide += "- `(laughing) That's so funny!`\n"
    
    return guide


def get_hardware_info():
    """Get detailed hardware information for display"""
    health = get_health()
    if not health:
        return "🔴 API not available. Please start the backend server.", None
    
    device = health['device']
    sys_info = health['system_info']
    
    # Build hardware info string
    if device == 'cuda':
        gpu_name = sys_info.get('gpu_name', 'Unknown GPU')
        gpu_mem = sys_info.get('gpu_memory_gb', 0)
        compute_cap = sys_info.get('compute_capability', 'N/A')
        device_info = f"🟢 **GPU Detected**: {gpu_name} ({gpu_mem:.1f}GB VRAM, Compute {compute_cap})"
    elif device == 'mps':
        device_info = f"🟢 **Apple Silicon GPU**: {sys_info.get('gpu_name', 'Apple M-series')}"
    else:
        cpu_info = sys_info.get('cpu_model', 'Unknown CPU')
        device_info = f"🟡 **CPU Mode**: {cpu_info}"
    
    return device_info, health


def create_ui():
    """Create Gradio interface"""
    
    # Check API health
    device_info, health = get_hardware_info()
    
    # Custom CSS for dark theme and styling
    custom_css = """
    .gradio-container {
        max-width: 1200px !important;
        margin: auto !important;
    }
    .metrics-box {
        background: #1f2937;
        border-radius: 8px;
        padding: 16px;
        margin-top: 16px;
    }
    .status-box {
        padding: 12px;
        border-radius: 6px;
        margin: 8px 0;
    }
    """
    
    with gr.Blocks(
        theme=gr.themes.Soft(primary_hue="blue"),
        css=custom_css,
        title="Fish Speech TTS"
    ) as demo:
        
        gr.Markdown(
            f"""
            # 🐟 Optimized Fish Speech TTS
            ### Zero-Shot Voice Cloning with OpenAudio S1-Mini
            
            {device_info}
            """
        )
        
        with gr.Tabs():
            # Main TTS Tab
            with gr.Tab("🎙️ Synthesize"):
                with gr.Row():
                    with gr.Column(scale=2):
                        text_input = gr.Textbox(
                            label="Text to Synthesize",
                            placeholder="Enter text here...",
                            lines=5,
                            max_lines=10
                        )
                        
                        with gr.Accordion("📁 Voice Cloning (Optional)", open=False):
                            speaker_file = gr.Audio(
                                label="Reference Audio (10-30 seconds recommended)",
                                type="filepath",
                                sources=["upload", "microphone"]
                            )
                            prompt_text = gr.Textbox(
                                label="Reference Transcript (Optional, improves quality)",
                                placeholder="Transcript of the reference audio...",
                                lines=2
                            )
                        
                        with gr.Accordion("⚙️ Advanced Settings", open=False):
                            with gr.Row():
                                temperature = gr.Slider(
                                    minimum=0.1,
                                    maximum=2.0,
                                    value=0.7,
                                    step=0.1,
                                    label="Temperature",
                                    info="Higher = more random"
                                )
                                top_p = gr.Slider(
                                    minimum=0.1,
                                    maximum=1.0,
                                    value=0.7,
                                    step=0.05,
                                    label="Top P",
                                    info="Nucleus sampling"
                                )
                            
                            with gr.Row():
                                speed = gr.Slider(
                                    minimum=0.5,
                                    maximum=2.0,
                                    value=1.0,
                                    step=0.1,
                                    label="Speed",
                                    info="Not implemented yet"
                                )
                                seed = gr.Number(
                                    label="Seed (Optional)",
                                    value=None,
                                    precision=0,
                                    info="For reproducibility"
                                )
                            
                            language = gr.Dropdown(
                                choices=[
                                    ("English", "en"),
                                    ("中文 (Chinese)", "zh"),
                                    ("日本語 (Japanese)", "ja"),
                                    ("한국어 (Korean)", "ko"),
                                    ("Français (French)", "fr"),
                                    ("Deutsch (German)", "de"),
                                    ("Español (Spanish)", "es"),
                                    ("Polski (Polish)", "pl"),
                                    ("Русский (Russian)", "ru"),
                                    ("Italiano (Italian)", "it"),
                                    ("Português (Portuguese)", "pt"),
                                    ("العربية (Arabic)", "ar"),
                                    ("हिन्दी (Hindi)", "hi"),
                                    ("Türkçe (Turkish)", "tr")
                                ],
                                value="en",
                                label="Language",
                                info="Select target language (Fish Speech auto-detects from text)"
                            )
                            
                            optimize_for_memory = gr.Checkbox(
                                label="Optimize for Memory",
                                value=False,
                                info="Reduce VRAM usage (slower)"
                            )
                            
                            force_cpu = gr.Checkbox(
                                label="⚠️ Force CPU Mode (Experimental)",
                                value=False,
                                info="WARNING: May cause errors. For reliable CPU mode, set DEVICE=cpu in .env and restart backend."
                            )
                        
                        synthesize_btn = gr.Button("🎵 Generate Speech", variant="primary", size="lg")
                    
                    with gr.Column(scale=1):
                        status_output = gr.Markdown("Ready to synthesize")
                        audio_output = gr.Audio(
                            label="Generated Audio",
                            type="filepath"
                        )
                        metrics_output = gr.Markdown("")
                
                # Wire up synthesis
                synthesize_btn.click(
                    fn=synthesize_speech,
                    inputs=[
                        text_input,
                        speaker_file,
                        prompt_text,
                        language,
                        temperature,
                        top_p,
                        speed,
                        seed,
                        optimize_for_memory,
                        force_cpu
                    ],
                    outputs=[audio_output, status_output, metrics_output]
                )
            
            # Tips Tab
            with gr.Tab("💡 Tips"):
                gr.Markdown(
                    """
                    ### 💡 Tips for Best Results:
                    
                    1. **Reference Audio**: Use 10-30 seconds of clear speech for voice cloning
                    2. **Reference Transcript**: Providing the transcript improves cloning quality
                    3. **Text Length**: Keep text under 200 characters for 4GB GPUs, 600 for 6GB+
                    4. **Language Support**: Auto-detects from text (English, Chinese, Japanese, etc.)
                    5. **Temperature**: Lower (0.5-0.7) for consistent speech, higher (0.8-1.2) for variety
                    6. **Device Selection**: Set DEVICE in .env (auto/cpu/cuda) for best performance
                    """
                )
            
            # System Info Tab
            with gr.Tab("ℹ️ System Info"):
                system_info = gr.Markdown()
                refresh_btn = gr.Button("🔄 Refresh Info")
                
                def get_system_info():
                    health = get_health()
                    if not health:
                        return "❌ Unable to connect to API"
                    
                    device = health['device']
                    sys_info = health['system_info']
                    
                    info = f"""
                    ## 🖥️ Hardware Information
                    
                    **Status**: {health['status']}
                    
                    **Active Device**: `{device.upper()}`
                    """
                    
                    # GPU Information
                    if device == 'cuda':
                        info += f"""
                    
                    ### 🎮 NVIDIA GPU
                    - **Model**: {sys_info.get('gpu_name', 'N/A')}
                    - **Total VRAM**: {sys_info.get('gpu_memory_gb', 0):.1f} GB
                    - **Compute Capability**: {sys_info.get('compute_capability', 'N/A')}
                    - **Currently Allocated**: {health.get('gpu_memory_allocated_mb', 0):.1f} MB
                    - **Reserved Memory**: {health.get('gpu_memory_reserved_mb', 0):.1f} MB
                    - **CUDA Available**: ✅ Yes
                        """
                    elif device == 'mps':
                        info += f"""
                    
                    ### 🍎 Apple Silicon GPU
                    - **Model**: {sys_info.get('gpu_name', 'Apple M-series')}
                    - **Unified Memory**: Shared with system RAM
                    - **MPS Available**: ✅ Yes
                        """
                    else:
                        info += f"""
                    
                    ### 💻 CPU Mode
                    - **Processor**: {sys_info.get('cpu_model', 'Unknown')}
                    - **Note**: GPU acceleration not available or disabled
                        """
                    
                    # Configuration
                    info += f"""
                    
                    ### ⚙️ Configuration
                    - **Precision**: {sys_info.get('precision', 'N/A')}
                    - **Quantization**: {sys_info.get('quantization', 'none')}
                    - **Torch Compile**: {'✅ Enabled' if sys_info.get('compile_enabled') else '❌ Disabled'}
                    
                    ### 💾 Cache Statistics
                    - **VQ Cache Size**: {health['cache_stats'].get('vq_cache_size', 0)} items
                    - **Semantic Cache Size**: {health['cache_stats'].get('semantic_cache_size', 0)} items
                    
                    ### 💡 Tips
                    - Use "Force CPU Mode" in Advanced Settings to disable GPU
                    - Enable "Optimize for Memory" for 4GB GPUs
                    - Clear cache if experiencing memory issues
                    """
                    
                    return info
                
                refresh_btn.click(fn=get_system_info, outputs=system_info)
                demo.load(fn=get_system_info, outputs=system_info)
                
                clear_cache_btn = gr.Button("🗑️ Clear Cache")
                cache_status = gr.Markdown()
                clear_cache_btn.click(fn=clear_cache, outputs=cache_status)
        
        gr.Markdown(
            """
            ---
            **Powered by OpenAudio S1-Mini** | [GitHub](https://github.com/fishaudio/fish-speech) | [Documentation](https://speech.fish.audio)
            """
        )
    
    return demo


if __name__ == "__main__":
    # Check if API is available
    try:
        response = requests.get(f"{API_URL}/health", timeout=5)
        if response.status_code != 200:
            print(f"[WARNING] API returned status {response.status_code}")
    except:
        print(f"[ERROR] Cannot connect to API at {API_URL}")
        print("Please start the backend server first:")
        print("  python backend/app.py")
        sys.exit(1)
    
    # Launch Gradio
    demo = create_ui()
    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.getenv("GRADIO_PORT", "7860")),
        share=False,
        show_error=True
    )
