"""
Example: Integrating Fish Speech into existing VoiceCloner
Shows how to add Fish Speech support to your current system
"""

import sys
from pathlib import Path

# Import your existing VoiceCloner
# from fewshot_voice import VoiceCloner

# Import Fish Speech wrapper
from fish_speech_wrapper import FishSpeechTTS


class EnhancedVoiceCloner:
    """
    Extended VoiceCloner with Fish Speech support
    Maintains backward compatibility with XTTS and RVC
    """
    
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_type = None  # 'xtts', 'rvc', or 'fish_speech'
        
        # Existing models
        self.xtts_model = None
        self.rvc_model = None
        self.speaker_encoder = None
        
        # NEW: Fish Speech
        self.fish_speech_model = None
        
        print(f"Enhanced Voice Cloner initialized (Device: {self.device})")
    
    def load_fish_speech(self, model_path="checkpoints/openaudio-s1-mini"):
        """
        Load Fish Speech model
        
        Args:
            model_path: Path to Fish Speech model directory
            
        Returns:
            bool: True if successful
        """
        try:
            print("\nLoading Fish Speech model...")
            self.fish_speech_model = FishSpeechTTS(
                model_path=model_path,
                device=self.device
            )
            self.model_type = 'fish_speech'
            print("✓ Fish Speech loaded successfully!")
            return True
            
        except FileNotFoundError as e:
            print(f"Error: {e}")
            print("\nTo download Fish Speech model:")
            print("  pip install huggingface_hub[cli]")
            print("  hf download fishaudio/openaudio-s1-mini --local-dir checkpoints/openaudio-s1-mini")
            return False
        except Exception as e:
            print(f"Error loading Fish Speech: {e}")
            return False
    
    def load_xtts(self):
        """Load XTTS model (existing implementation)"""
        # Your existing XTTS loading code
        pass
    
    def load_rvc(self, model_path):
        """Load RVC model (existing implementation)"""
        # Your existing RVC loading code
        pass
    
    def synthesize(self, text, output_path,
                   reference_audio_path=None,
                   speaker_embedding=None,
                   language="en",
                   output_format="mp3",
                   clean_reference=True,
                   enable_preview=True,
                   prompt_text=None,
                   compile_mode=True):
        """
        Enhanced synthesize method with Fish Speech support
        
        Args:
            text: Text to synthesize
            output_path: Output file path
            reference_audio_path: Reference audio for voice cloning
            speaker_embedding: Pre-computed speaker embedding (XTTS only)
            language: Language code
            output_format: Output format (wav, mp3)
            clean_reference: Clean reference audio (XTTS only)
            enable_preview: Enable audio preview
            prompt_text: Transcript of reference audio (Fish Speech only)
            compile_mode: Use compilation for speed (Fish Speech only)
            
        Returns:
            str: Path to generated audio file
        """
        
        if self.model_type == 'fish_speech':
            return self._synthesize_fish_speech(
                text=text,
                output_path=output_path,
                reference_audio_path=reference_audio_path,
                prompt_text=prompt_text,
                compile_mode=compile_mode,
                output_format=output_format,
                enable_preview=enable_preview
            )
        
        elif self.model_type == 'xtts':
            return self._synthesize_xtts(
                text=text,
                output_path=output_path,
                reference_audio_path=reference_audio_path,
                speaker_embedding=speaker_embedding,
                language=language,
                output_format=output_format,
                clean_reference=clean_reference,
                enable_preview=enable_preview
            )
        
        elif self.model_type == 'rvc':
            return self._synthesize_rvc(
                text=text,
                output_path=output_path,
                language=language,
                output_format=output_format,
                enable_preview=enable_preview
            )
        
        else:
            print("Error: No model loaded!")
            return None
    
    def _synthesize_fish_speech(self, text, output_path, reference_audio_path,
                               prompt_text, compile_mode, output_format,
                               enable_preview):
        """Synthesize using Fish Speech"""
        
        if not self.fish_speech_model:
            print("Error: Fish Speech model not loaded")
            return None
        
        try:
            print(f"\nSynthesizing with Fish Speech...")
            print(f"  Text: '{text[:50]}{'...' if len(text) > 50 else ''}'")
            
            # Generate audio
            temp_output = Path(output_path).with_suffix('.wav')
            
            audio = self.fish_speech_model.tts(
                text=text,
                speaker_wav=reference_audio_path,
                output_path=str(temp_output),
                prompt_text=prompt_text,
                compile_mode=compile_mode
            )
            
            # Convert format if needed
            if output_format.lower() == 'mp3' and FFMPEG_AVAILABLE:
                mp3_output = Path(output_path).with_suffix('.mp3')
                audio_segment = AudioSegment.from_wav(str(temp_output))
                audio_segment.export(str(mp3_output), format="mp3", bitrate="192k")
                temp_output.unlink()
                final_output = mp3_output
            else:
                final_output = temp_output
            
            # Preview if enabled
            if enable_preview and PLAYBACK_AVAILABLE:
                self._preview_audio(final_output)
            
            print(f"\n✓ Synthesis complete: {final_output}")
            return str(final_output)
            
        except Exception as e:
            print(f"\nFish Speech synthesis error: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _synthesize_xtts(self, text, output_path, reference_audio_path,
                        speaker_embedding, language, output_format,
                        clean_reference, enable_preview):
        """Synthesize using XTTS (existing implementation)"""
        # Your existing XTTS synthesis code
        pass
    
    def _synthesize_rvc(self, text, output_path, language, output_format,
                       enable_preview):
        """Synthesize using RVC (existing implementation)"""
        # Your existing RVC synthesis code
        pass
    
    def _preview_audio(self, audio_path):
        """Preview audio file"""
        if not PLAYBACK_AVAILABLE:
            print("Audio preview not available")
            return
        
        try:
            import simpleaudio
            audio_data, sample_rate = sf.read(audio_path)
            audio_data = (audio_data * 32767).astype(np.int16)
            
            if audio_data.ndim == 1:
                audio_data = np.column_stack((audio_data, audio_data))
            
            play_obj = simpleaudio.play_buffer(
                audio_data.tobytes(),
                num_channels=2,
                bytes_per_sample=2,
                sample_rate=sample_rate
            )
            
            print("\nPlaying preview... (Press Ctrl+C to stop)")
            play_obj.wait_done()
            
        except Exception as e:
            print(f"Preview error: {e}")
    
    def compare_models(self, text, reference_audio, output_dir="comparison"):
        """
        Compare output quality across different models
        
        Args:
            text: Text to synthesize
            reference_audio: Reference audio for voice cloning
            output_dir: Directory to save comparison outputs
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        results = {}
        
        # Test XTTS
        if self.xtts_model:
            print("\n" + "="*70)
            print("Testing XTTS...")
            print("="*70)
            self.model_type = 'xtts'
            xtts_output = self.synthesize(
                text=text,
                reference_audio_path=reference_audio,
                output_path=output_dir / "comparison_xtts.wav",
                enable_preview=False
            )
            results['xtts'] = xtts_output
        
        # Test Fish Speech
        if self.fish_speech_model:
            print("\n" + "="*70)
            print("Testing Fish Speech...")
            print("="*70)
            self.model_type = 'fish_speech'
            fish_output = self.synthesize(
                text=text,
                reference_audio_path=reference_audio,
                output_path=output_dir / "comparison_fish_speech.wav",
                enable_preview=False
            )
            results['fish_speech'] = fish_output
        
        print("\n" + "="*70)
        print("COMPARISON COMPLETE")
        print("="*70)
        print(f"\nOutputs saved in: {output_dir}")
        for model, path in results.items():
            print(f"  {model}: {path}")
        
        print("\nListen to both outputs and compare:")
        print("  1. Naturalness")
        print("  2. Voice similarity")
        print("  3. Pronunciation accuracy")
        print("  4. Audio quality")
        
        return results
    
    def get_model_info(self):
        """Get information about loaded models"""
        info = {
            'current_model': self.model_type,
            'available_models': []
        }
        
        if self.xtts_model:
            info['available_models'].append('xtts')
        if self.rvc_model:
            info['available_models'].append('rvc')
        if self.fish_speech_model:
            info['available_models'].append('fish_speech')
        
        return info
    
    def cleanup(self):
        """Cleanup resources"""
        if self.fish_speech_model:
            self.fish_speech_model.cleanup()


def main():
    """Interactive demo"""
    print("="*70)
    print("ENHANCED VOICE CLONER - FISH SPEECH INTEGRATION")
    print("="*70)
    
    cloner = EnhancedVoiceCloner()
    
    while True:
        print("\n" + "="*70)
        print("MAIN MENU")
        print("="*70)
        print("1. Load Fish Speech")
        print("2. Load XTTS")
        print("3. Synthesize speech")
        print("4. Compare models")
        print("5. View emotion guide (Fish Speech)")
        print("6. Model info")
        print("7. Exit")
        
        choice = input("\nChoice (1-7): ").strip()
        
        if choice == "1":
            model_path = input("\nModel path (default: checkpoints/openaudio-s1-mini): ").strip()
            model_path = model_path if model_path else "checkpoints/openaudio-s1-mini"
            cloner.load_fish_speech(model_path)
        
        elif choice == "2":
            cloner.load_xtts()
        
        elif choice == "3":
            if not cloner.model_type:
                print("\nError: Load a model first!")
                continue
            
            text = input("\nEnter text: ").strip()
            if not text:
                print("Text cannot be empty!")
                continue
            
            ref_audio = input("Reference audio path: ").strip()
            if not Path(ref_audio).exists():
                print(f"Error: File not found: {ref_audio}")
                continue
            
            output = input("Output path (default: output.wav): ").strip()
            output = output if output else "output.wav"
            
            # Fish Speech specific options
            if cloner.model_type == 'fish_speech':
                prompt_text = input("Prompt text (transcript of reference, optional): ").strip()
                prompt_text = prompt_text if prompt_text else None
            else:
                prompt_text = None
            
            cloner.synthesize(
                text=text,
                reference_audio_path=ref_audio,
                output_path=output,
                prompt_text=prompt_text
            )
        
        elif choice == "4":
            text = input("\nEnter text for comparison: ").strip()
            if not text:
                print("Text cannot be empty!")
                continue
            
            ref_audio = input("Reference audio path: ").strip()
            if not Path(ref_audio).exists():
                print(f"Error: File not found: {ref_audio}")
                continue
            
            cloner.compare_models(text, ref_audio)
        
        elif choice == "5":
            if cloner.fish_speech_model:
                cloner.fish_speech_model.print_emotion_guide()
            else:
                print("\nError: Fish Speech not loaded!")
        
        elif choice == "6":
            info = cloner.get_model_info()
            print(f"\nCurrent model: {info['current_model']}")
            print(f"Available models: {', '.join(info['available_models'])}")
        
        elif choice == "7":
            print("\nCleaning up...")
            cloner.cleanup()
            print("Goodbye!")
            break
        
        else:
            print("Invalid choice!")


if __name__ == "__main__":
    # Add necessary imports
    import torch
    import numpy as np
    
    main()
