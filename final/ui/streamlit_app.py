"""
Streamlit Web UI for Optimized Fish Speech TTS
Alternative UI with similar functionality to Gradio
"""

import os
import sys
import requests
import streamlit as st
from pathlib import Path
import tempfile
import time

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


def synthesize_speech(text, speaker_file, prompt_text, language, 
                     temperature, top_p, speed, seed, optimize_for_memory):
    """Call TTS API and return audio with metrics"""
    
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
        files['speaker_file'] = speaker_file
    
    # Make request
    response = requests.post(
        f"{API_URL}/tts",
        data=data,
        files=files,
        timeout=300
    )
    
    if response.status_code != 200:
        error_detail = response.json().get('detail', 'Unknown error')
        raise Exception(error_detail)
    
    # Get metrics from headers
    metrics = {
        'latency_ms': response.headers.get('X-Latency-Ms', 'N/A'),
        'peak_vram_mb': response.headers.get('X-Peak-VRAM-Mb', 'N/A'),
        'gpu_util_pct': response.headers.get('X-GPU-Util-Pct', 'N/A'),
        'audio_duration_s': response.headers.get('X-Audio-Duration-S', 'N/A'),
        'rtf': response.headers.get('X-RTF', 'N/A')
    }
    
    return response.content, metrics


def main():
    """Main Streamlit app"""
    
    # Page config
    st.set_page_config(
        page_title="Fish Speech TTS",
        page_icon="🐟",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS
    st.markdown("""
        <style>
        .main-header {
            font-size: 2.5rem;
            font-weight: bold;
            color: #1f77b4;
            margin-bottom: 0.5rem;
        }
        .sub-header {
            font-size: 1.2rem;
            color: #666;
            margin-bottom: 2rem;
        }
        .metric-card {
            background: #f0f2f6;
            padding: 1rem;
            border-radius: 0.5rem;
            margin: 0.5rem 0;
        }
        </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.markdown('<div class="main-header">🐟 Optimized Fish Speech TTS</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Zero-Shot Voice Cloning with OpenAudio S1-Mini</div>', unsafe_allow_html=True)
    
    # Check API health
    health = get_health()
    if health:
        device_info = f"🟢 Connected | Device: {health['device']}"
        if health['device'] == 'cuda':
            device_info += f" | {health['system_info'].get('gpu_name', 'GPU')}"
        st.success(device_info)
    else:
        st.error("🔴 API not available. Please start the backend server.")
        st.stop()
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
        
        # Advanced settings
        with st.expander("🎛️ Advanced Parameters", expanded=False):
            temperature = st.slider(
                "Temperature",
                min_value=0.1,
                max_value=2.0,
                value=0.7,
                step=0.1,
                help="Higher = more random"
            )
            
            top_p = st.slider(
                "Top P",
                min_value=0.1,
                max_value=1.0,
                value=0.7,
                step=0.05,
                help="Nucleus sampling"
            )
            
            speed = st.slider(
                "Speed",
                min_value=0.5,
                max_value=2.0,
                value=1.0,
                step=0.1,
                help="Not implemented yet"
            )
            
            seed = st.number_input(
                "Seed (Optional)",
                min_value=0,
                value=0,
                help="For reproducibility"
            )
            
            language = st.selectbox(
                "Language",
                options=[
                    "en",  # English
                    "zh",  # Chinese
                    "ja",  # Japanese
                    "ko",  # Korean
                    "fr",  # French
                    "de",  # German
                    "es",  # Spanish
                    "pl",  # Polish
                    "ru",  # Russian
                    "it",  # Italian
                    "pt",  # Portuguese
                    "ar",  # Arabic
                    "hi",  # Hindi
                    "tr"   # Turkish
                ],
                format_func=lambda x: {
                    "en": "English",
                    "zh": "中文 (Chinese)",
                    "ja": "日本語 (Japanese)",
                    "ko": "한국어 (Korean)",
                    "fr": "Français (French)",
                    "de": "Deutsch (German)",
                    "es": "Español (Spanish)",
                    "pl": "Polski (Polish)",
                    "ru": "Русский (Russian)",
                    "it": "Italiano (Italian)",
                    "pt": "Português (Portuguese)",
                    "ar": "العربية (Arabic)",
                    "hi": "हिन्दी (Hindi)",
                    "tr": "Türkçe (Turkish)"
                }.get(x, x),
                index=0,
                help="Auto-detected, for reference only"
            )
            
            optimize_for_memory = st.checkbox(
                "Optimize for Memory",
                value=False,
                help="Reduce VRAM usage (slower)"
            )
        
        # System info
        with st.expander("ℹ️ System Info", expanded=False):
            if st.button("🔄 Refresh"):
                st.rerun()
            
            st.write("**Device:**", health['device'])
            st.write("**Precision:**", health['system_info'].get('precision', 'N/A'))
            st.write("**Quantization:**", health['system_info'].get('quantization', 'N/A'))
            
            if health['device'] == 'cuda':
                st.write("**GPU:**", health['system_info'].get('gpu_name', 'N/A'))
                st.write("**GPU Memory:**", f"{health['system_info'].get('gpu_memory_gb', 0):.1f} GB")
            
            st.write("**VQ Cache:**", health['cache_stats'].get('vq_cache_size', 0))
            st.write("**Semantic Cache:**", health['cache_stats'].get('semantic_cache_size', 0))
            
            if st.button("🗑️ Clear Cache"):
                try:
                    response = requests.post(f"{API_URL}/cache/clear", timeout=5)
                    if response.status_code == 200:
                        st.success("Cache cleared!")
                    else:
                        st.error("Failed to clear cache")
                except Exception as e:
                    st.error(f"Error: {e}")
    
    # Main content tabs
    tab1, tab2 = st.tabs(["🎙️ Synthesize", "💡 Tips"])
    
    with tab1:
        # Text input
        text_input = st.text_area(
            "Text to Synthesize",
            placeholder="Enter text here...",
            height=150,
            help="Enter the text you want to synthesize"
        )
        
        # Voice cloning section
        with st.expander("📁 Voice Cloning (Optional)", expanded=False):
            speaker_file = st.file_uploader(
                "Reference Audio (10-30 seconds recommended)",
                type=["wav", "mp3", "ogg", "flac"],
                help="Upload a reference audio file for voice cloning"
            )
            
            prompt_text = st.text_area(
                "Reference Transcript (Optional)",
                placeholder="Transcript of the reference audio...",
                height=80,
                help="Providing transcript improves quality"
            )
        
        # Generate button
        if st.button("🎵 Generate Speech", type="primary", use_container_width=True):
            if not text_input or not text_input.strip():
                st.error("❌ Text cannot be empty!")
            else:
                with st.spinner("Generating speech... This may take a moment."):
                    try:
                        # Call API
                        audio_bytes, metrics = synthesize_speech(
                            text=text_input,
                            speaker_file=speaker_file,
                            prompt_text=prompt_text,
                            language=language,
                            temperature=temperature,
                            top_p=top_p,
                            speed=speed,
                            seed=seed if seed > 0 else None,
                            optimize_for_memory=optimize_for_memory
                        )
                        
                        # Success message
                        st.success("✅ Synthesis completed successfully!")
                        
                        # Display audio
                        st.audio(audio_bytes, format="audio/wav")
                        
                        # Display metrics
                        st.subheader("📊 Performance Metrics")
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Latency", f"{metrics['latency_ms']} ms")
                            st.metric("Audio Duration", f"{metrics['audio_duration_s']}s")
                        with col2:
                            st.metric("Peak VRAM", f"{metrics['peak_vram_mb']} MB")
                            st.metric("GPU Utilization", f"{metrics['gpu_util_pct']}%")
                        with col3:
                            st.metric("Real-Time Factor", f"{metrics['rtf']}x")
                        
                        # Download button
                        st.download_button(
                            label="💾 Download Audio",
                            data=audio_bytes,
                            file_name="output.wav",
                            mime="audio/wav"
                        )
                        
                    except requests.exceptions.Timeout:
                        st.error("❌ Request timed out. Try shorter text or reference audio.")
                    except Exception as e:
                        st.error(f"❌ Error: {str(e)}")
    
    with tab2:
        st.header("💡 Tips for Best Results")
        
        st.markdown("""
        ### Voice Cloning Tips
        
        1. **Reference Audio**: Use 10-30 seconds of clear speech for best voice cloning
        2. **Reference Transcript**: Providing the transcript significantly improves cloning quality
        3. **Audio Quality**: Use clean audio without background noise
        4. **Text Length**: Keep text under 200 characters for 4GB GPUs, 600 for 6GB+ GPUs
        
        ### Performance Tips
        
        5. **Device Selection**: Set `DEVICE` in `.env` file:
           - `DEVICE=auto` - Smart auto-selection (recommended)
           - `DEVICE=cuda` - Force GPU mode
           - `DEVICE=cpu` - Force CPU mode
        
        6. **Temperature**: 
           - Lower (0.5-0.7) for consistent, predictable speech
           - Higher (0.8-1.2) for more variety and expressiveness
        
        7. **Language Support**: Auto-detects from text
           - Supported: English, Chinese, Japanese, Korean, French, German, Spanish, etc.
        
        ### Hardware Recommendations
        
        - **4GB GPU (RTX 3050)**: System will auto-use CPU mode (faster than overloaded GPU)
        - **6GB+ GPU (RTX 3060+)**: GPU mode works great
        - **CPU-only (i5+)**: ONNX optimization provides good performance (RTF 6-8x)
        """)
    
    # Footer
    st.markdown("---")
    st.markdown(
        "**Powered by OpenAudio S1-Mini** | "
        "[GitHub](https://github.com/fishaudio/fish-speech) | "
        "[Documentation](https://speech.fish.audio)"
    )


if __name__ == "__main__":
    # Check if API is available
    try:
        response = requests.get(f"{API_URL}/health", timeout=5)
        if response.status_code != 200:
            st.error(f"⚠️  Warning: API returned status {response.status_code}")
    except:
        st.error(f"❌ Error: Cannot connect to API at {API_URL}")
        st.info("Please start the backend server first: `python backend/app.py`")
        st.stop()
    
    main()
