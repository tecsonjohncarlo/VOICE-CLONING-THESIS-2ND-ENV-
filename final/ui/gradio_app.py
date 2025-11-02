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
    optimize_for_memory
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
            'optimize_for_memory': optimize_for_memory
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


def create_ui():
    """Create Gradio interface"""
    
    # Check API health
    health = get_health()
    if health:
        device_info = f"🟢 Connected | Device: {health['device']} | {health['system_info'].get('gpu_name', 'CPU')}"
    else:
        device_info = "🔴 API not available. Please start the backend server."
    
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
                            placeholder="Enter text here... Use (emotion) markers for expressive speech!",
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
                                choices=["en", "zh", "ja", "ko", "fr", "de", "es", "ar"],
                                value="en",
                                label="Language",
                                info="Auto-detected, for reference only"
                            )
                            
                            optimize_for_memory = gr.Checkbox(
                                label="Optimize for Memory",
                                value=False,
                                info="Reduce VRAM usage (slower)"
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
                        optimize_for_memory
                    ],
                    outputs=[audio_output, status_output, metrics_output]
                )
            
            # Emotion Guide Tab
            with gr.Tab("🎭 Emotion Guide"):
                gr.Markdown(format_emotion_guide())
                
                gr.Markdown(
                    """
                    ### 💡 Tips for Best Results:
                    
                    1. **Reference Audio**: Use 10-30 seconds of clear speech
                    2. **Emotion Markers**: Place at the beginning or end of sentences
                    3. **Multiple Emotions**: You can use multiple markers in one text
                    4. **Language Support**: Emotions work best with English, Chinese, and Japanese
                    5. **Temperature**: Lower (0.5-0.7) for consistent speech, higher (0.8-1.2) for variety
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
                    
                    info = f"""
                    ## System Information
                    
                    **Status**: {health['status']}
                    
                    **Device**: {health['device']}
                    
                    **System Info**:
                    - Precision: {health['system_info'].get('precision', 'N/A')}
                    - Quantization: {health['system_info'].get('quantization', 'N/A')}
                    - Torch Compile: {health['system_info'].get('compile_enabled', 'N/A')}
                    """
                    
                    if health['device'] == 'cuda':
                        info += f"""
                    - GPU: {health['system_info'].get('gpu_name', 'N/A')}
                    - GPU Memory: {health['system_info'].get('gpu_memory_gb', 0):.1f} GB
                    - Compute Capability: {health['system_info'].get('compute_capability', 'N/A')}
                    - Allocated Memory: {health.get('gpu_memory_allocated_mb', 0):.1f} MB
                    - Reserved Memory: {health.get('gpu_memory_reserved_mb', 0):.1f} MB
                        """
                    
                    info += f"""
                    
                    **Cache Stats**:
                    - VQ Cache Size: {health['cache_stats'].get('vq_cache_size', 0)}
                    - Semantic Cache Size: {health['cache_stats'].get('semantic_cache_size', 0)}
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
