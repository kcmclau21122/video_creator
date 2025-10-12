# ============================================================================
# AI Video Creator - Text-to-Speech Generator
# Converts narration text to natural-sounding speech
# ============================================================================

"""
TTS Generator
Generates high-quality speech from narration text
Supports multiple voices and emotions
"""

from pathlib import Path
from typing import List, Optional, Dict
import torch
import numpy as np
import scipy
from dataclasses import dataclass

# Bark - Best for natural, emotional speech
from transformers import AutoProcessor, BarkModel

# Alternative: Coqui TTS (if available)
try:
    from TTS.api import TTS as CoquiTTS
    COQUI_AVAILABLE = True
except ImportError:
    COQUI_AVAILABLE = False

from narrative_generator import NarrativeSegment
from config import Config


class TTSGenerator:
    """Generate speech from text using various TTS models"""
    
    # Available TTS models
    MODELS = {
        'bark': {
            'name': 'suno/bark',
            'size': '~8 GB',
            'vram': '6-8 GB',
            'quality': 'Excellent - Natural with emotion',
            'multilingual': True,
            'voices': ['male', 'female', 'neutral']
        },
        'bark-small': {
            'name': 'suno/bark-small',
            'size': '~2 GB',
            'vram': '4-6 GB',
            'quality': 'Good - Faster generation',
            'multilingual': True,
            'voices': ['male', 'female', 'neutral']
        }
    }
    
    # Voice presets for Bark
    BARK_VOICES = {
        'narrator_male': 'v2/en_speaker_6',  # Clear, authoritative
        'narrator_female': 'v2/en_speaker_9',  # Warm, engaging
        'storyteller': 'v2/en_speaker_3',  # Expressive
        'documentary': 'v2/en_speaker_5',  # Professional
        'casual_male': 'v2/en_speaker_0',  # Friendly male
        'casual_female': 'v2/en_speaker_1'  # Friendly female
    }
    
    def __init__(
        self,
        model_key: str = 'bark-small',
        voice: str = 'narrator_male',
        sample_rate: int = 24000
    ):
        """
        Initialize TTS generator
        
        Args:
            model_key: TTS model to use
            voice: Voice preset
            sample_rate: Audio sample rate
        """
        self.model_key = model_key
        self.voice = voice
        self.sample_rate = sample_rate
        self.model = None
        self.processor = None
        self.device = Config.DEVICE
        
    def load_model(self):
        """Load TTS model"""
        if self.model is not None:
            return
        
        model_info = self.MODELS.get(self.model_key)
        if not model_info:
            raise ValueError(f"Unknown model: {self.model_key}")
        
        print(f"\n{'='*70}")
        print(f"LOADING TTS MODEL: {self.model_key}")
        print(f"{'='*70}")
        print(f"Model: {model_info['name']}")
        print(f"Voice: {self.voice}")
        print("This may take a minute...\n")
        
        try:
            if 'bark' in self.model_key:
                self._load_bark()
            elif COQUI_AVAILABLE:
                self._load_coqui()
            else:
                raise ValueError("No TTS model available")
            
            print(f"✅ TTS model loaded\n")
            
        except Exception as e:
            print(f"✗ Error loading TTS model: {e}")
            raise
    
    def _load_bark(self):
        """Load Bark TTS model"""
        print("Loading Bark TTS...")
        
        model_name = self.MODELS[self.model_key]['name']
        
        # Load processor
        self.processor = AutoProcessor.from_pretrained(model_name)
        
        # Load model
        self.model = BarkModel.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if Config.USE_FP16 else torch.float32
        ).to(self.device)
        
        # Set voice preset
        self.voice_preset = self.BARK_VOICES.get(
            self.voice,
            self.BARK_VOICES['narrator_male']
        )
    
    def _load_coqui(self):
        """Load Coqui TTS model (alternative)"""
        if not COQUI_AVAILABLE:
            raise ImportError("Coqui TTS not installed")
        
        print("Loading Coqui TTS...")
        self.model = CoquiTTS(model_name="tts_models/en/ljspeech/tacotron2-DDC")
    
    def unload_model(self):
        """Unload TTS model"""
        if self.model:
            del self.model
            self.model = None
        if self.processor:
            del self.processor
            self.processor = None
        torch.cuda.empty_cache()
        print("✅ TTS model unloaded")
    
    def generate_speech_for_segments(
        self,
        narrative_segments: List[NarrativeSegment],
        output_dir: Path
    ) -> List[NarrativeSegment]:
        """
        Generate speech for all narrative segments
        
        Args:
            narrative_segments: List of narrative segments
            output_dir: Directory to save audio files
            
        Returns:
            Updated segments with audio_file paths
        """
        print(f"\n{'='*70}")
        print("GENERATING SPEECH FOR NARRATION")
        print(f"{'='*70}")
        print(f"Segments: {len(narrative_segments)}")
        print(f"Voice: {self.voice}")
        
        # Ensure model is loaded
        if not self.model:
            self.load_model()
        
        # Create output directory
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate speech for each segment
        for i, segment in enumerate(narrative_segments):
            print(f"\nGenerating {i+1}/{len(narrative_segments)}: {segment.narration_text[:50]}...")
            
            # Generate audio
            audio_array = self._generate_audio(
                segment.narration_text,
                emotion=segment.emotion
            )
            
            # Save to file
            audio_filename = f"narration_{i:03d}_{segment.sequence_index}.wav"
            audio_path = output_dir / audio_filename
            
            self._save_audio(audio_array, audio_path)
            
            # Update segment
            segment.audio_file = audio_path
            
            print(f"  ✅ Saved: {audio_filename}")
        
        print(f"\n✅ Generated speech for {len(narrative_segments)} segments")
        
        return narrative_segments
    
    def _generate_audio(self, text: str, emotion: str = "neutral") -> np.ndarray:
        """
        Generate audio from text
        
        Args:
            text: Text to convert to speech
            emotion: Emotion/tone (for compatible models)
            
        Returns:
            Audio array
        """
        if 'bark' in self.model_key:
            return self._generate_bark_audio(text, emotion)
        elif COQUI_AVAILABLE:
            return self._generate_coqui_audio(text)
        else:
            raise ValueError("No TTS model loaded")
    
    def _generate_bark_audio(self, text: str, emotion: str) -> np.ndarray:
        """Generate audio using Bark"""
        
        # Add emotion cues to text for Bark
        emotion_prefixes = {
            'cheerful': '♪ ',  # Musical note for happy tone
            'excited': '! ',  # Exclamation for excitement
            'nostalgic': '... ',  # Pause for reflective tone
            'calm': '~ ',  # Tilde for calm tone
            'dramatic': '!! '  # Double exclamation for drama
        }
        
        prefix = emotion_prefixes.get(emotion.lower(), '')
        text_with_emotion = prefix + text
        
        # Process text
        inputs = self.processor(
            text_with_emotion,
            voice_preset=self.voice_preset,
            return_tensors="pt"
        ).to(self.device)
        
        # Generate speech
        with torch.no_grad():
            speech_output = self.model.generate(
                **inputs,
                do_sample=True,
                fine_temperature=0.5,
                coarse_temperature=0.8
            )
        
        # Convert to numpy
        audio_array = speech_output.cpu().numpy().squeeze()
        
        return audio_array
    
    def _generate_coqui_audio(self, text: str) -> np.ndarray:
        """Generate audio using Coqui TTS"""
        
        # Generate speech
        audio_array = self.model.tts(text)
        
        return np.array(audio_array)
    
    def _save_audio(self, audio_array: np.ndarray, output_path: Path):
        """Save audio array to WAV file"""
        
        # Normalize audio
        audio_array = audio_array / np.max(np.abs(audio_array))
        
        # Convert to 16-bit PCM
        audio_int16 = (audio_array * 32767).astype(np.int16)
        
        # Save as WAV
        scipy.io.wavfile.write(
            str(output_path),
            self.sample_rate,
            audio_int16
        )
    
    def create_combined_narration(
        self,
        narrative_segments: List[NarrativeSegment],
        video_duration: float,
        output_path: Path
    ) -> Path:
        """
        Combine all narration segments into single audio track
        with silence padding to match video timing
        
        Args:
            narrative_segments: Segments with audio files
            video_duration: Total video duration
            output_path: Where to save combined audio
            
        Returns:
            Path to combined audio file
        """
        print(f"\n{'='*70}")
        print("COMBINING NARRATION AUDIO")
        print(f"{'='*70}")
        
        from pydub import AudioSegment
        # REMOVED: from pydub.silence import generate_silence
        
        # Create empty audio track
        combined_audio = AudioSegment.empty()
        current_time = 0.0
        
        for segment in narrative_segments:
            # Add silence until segment start time
            silence_duration = (segment.start_time - current_time) * 1000  # ms
            
            if silence_duration > 0:
                # CORRECTED: Use AudioSegment.silent() instead of generate_silence()
                silence = AudioSegment.silent(duration=int(silence_duration))
                combined_audio += silence
            
            # Load and add narration audio
            if segment.audio_file and segment.audio_file.exists():
                narration = AudioSegment.from_wav(str(segment.audio_file))
                combined_audio += narration
                current_time = segment.start_time + (len(narration) / 1000.0)
            else:
                print(f"  ⚠️  Missing audio for segment {segment.sequence_index}")
        
        # Add final silence to match video duration
        final_silence_duration = (video_duration - current_time) * 1000
        if final_silence_duration > 0:
            # CORRECTED: Use AudioSegment.silent() here as well
            final_silence = AudioSegment.silent(duration=int(final_silence_duration))
            combined_audio += final_silence
        
        # Export
        combined_audio.export(
            str(output_path),
            format="wav",
            parameters=["-ar", str(self.sample_rate)]
        )
        
        print(f"\n✅ Combined narration saved to: {output_path}")
        print(f"   Duration: {len(combined_audio)/1000:.2f}s")
        
        return output_path

if __name__ == "__main__":
    # Test TTS generator
    print("TTS Generator Module")
    print("\nAvailable models:")
    for key, info in TTSGenerator.MODELS.items():
        print(f"  • {key}: {info['quality']}")
    
    print("\nAvailable voices:")
    for voice_name in TTSGenerator.BARK_VOICES.keys():
        print(f"  • {voice_name}")