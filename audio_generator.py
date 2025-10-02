# ============================================================================
# AI Video Creator - Step 5: Enhanced Audio Generator (FIXED)
# ============================================================================

"""
Enhanced Audio Generator
Supports using existing music files or generating music with AI
FIXED: Forward reference type hints for TYPE_CHECKING imports
"""

from pathlib import Path
from typing import List, Dict, Optional, TYPE_CHECKING
import json
import torch
import scipy
import numpy as np
from transformers import AutoProcessor, MusicgenForConditionalGeneration
from datetime import datetime
from pydub import AudioSegment
from pydub.effects import normalize

from config import Config

# Use TYPE_CHECKING to avoid circular imports
if TYPE_CHECKING:
    from sequencing_engine import SequenceItem
    from scene_analyzer import SceneAnalysis


class AudioGenerator:
    """Generate or use existing audio for videos"""
    
    # Audio file extensions
    AUDIO_FORMATS = {'.mp3', '.wav', '.m4a', '.aac', '.ogg', '.flac', '.wma'}
    
    # Available AI models
    MODELS = {
        'musicgen-small': {
            'name': 'facebook/musicgen-small',
            'size': '~300M parameters',
            'vram': '4-6 GB',
            'quality': 'Good',
            'speed': 'Fast'
        },
        'musicgen-medium': {
            'name': 'facebook/musicgen-medium',
            'size': '~1.5B parameters',
            'vram': '8-10 GB',
            'quality': 'Better',
            'speed': 'Medium'
        },
        'musicgen-large': {
            'name': 'facebook/musicgen-large',
            'size': '~3.3B parameters',
            'vram': '14-16 GB',
            'quality': 'Best',
            'speed': 'Slow'
        }
    }
    
    def __init__(self, model_key: str = 'musicgen-medium', fade_duration: int = 0):
        """
        Initialize audio generator
        
        Args:
            model_key: Model to use for generation
            fade_duration: Seconds to fade at end (0 = no fade)
        """
        self.model_key = model_key
        self.model_info = self.MODELS.get(model_key)
        if not self.model_info:
            raise ValueError(f"Unknown model: {model_key}")
        
        self.model = None
        self.processor = None
        self.device = Config.DEVICE
        self.sample_rate = Config.AUDIO_SAMPLE_RATE
        self.fade_duration = fade_duration
        
    def find_music_files(self, input_folder: Path) -> List[Path]:
        """
        Find all music files in input folder
        
        Args:
            input_folder: Path to input folder
            
        Returns:
            List of audio file paths, sorted by filename
        """
        music_files = []
        
        for file_path in input_folder.rglob('*'):
            if file_path.suffix.lower() in self.AUDIO_FORMATS:
                music_files.append(file_path)
        
        # Sort by filename for consistent ordering
        music_files.sort(key=lambda x: x.name.lower())
        
        return music_files
    
    def prepare_audio_from_files(
        self,
        music_files: List[Path],
        target_duration: float,
        output_path: Path
    ) -> Path:
        """
        Prepare audio track from existing music files
        
        Args:
            music_files: List of music file paths
            target_duration: Target duration in seconds
            output_path: Where to save final audio
            
        Returns:
            Path to prepared audio file
        """
        print(f"\n{'='*70}")
        print("PREPARING AUDIO FROM EXISTING FILES")
        print(f"{'='*70}")
        print(f"Music files found: {len(music_files)}")
        print(f"Target duration: {target_duration:.2f} seconds ({target_duration/60:.2f} minutes)")
        print(f"Fade duration: {self.fade_duration} seconds")
        
        if not music_files:
            raise ValueError("No music files provided")
        
        # Load all audio files
        audio_segments = []
        for i, music_file in enumerate(music_files, 1):
            print(f"\nLoading {i}/{len(music_files)}: {music_file.name}")
            try:
                audio = AudioSegment.from_file(str(music_file))
                duration = len(audio) / 1000.0  # Convert to seconds
                print(f"  Duration: {duration:.2f}s, Format: {music_file.suffix}")
                audio_segments.append(audio)
            except Exception as e:
                print(f"  ⚠️  Could not load: {e}")
        
        if not audio_segments:
            raise ValueError("Could not load any music files")
        
        # Concatenate and loop audio to match target duration
        print("\nProcessing audio...")
        combined_audio = self._loop_audio(audio_segments, target_duration)
        
        # Apply fade out if specified
        if self.fade_duration > 0:
            print(f"Applying {self.fade_duration}s fade out...")
            combined_audio = self._apply_fade_out(combined_audio, self.fade_duration)
        
        # Normalize audio levels
        print("Normalizing audio levels...")
        combined_audio = normalize(combined_audio)
        
        # Export to WAV
        print(f"\nExporting to: {output_path}")
        combined_audio.export(
            str(output_path),
            format="wav",
            parameters=["-ar", str(self.sample_rate)]
        )
        
        actual_duration = len(combined_audio) / 1000.0
        print(f"\n✓ Audio prepared successfully!")
        print(f"  Final duration: {actual_duration:.2f}s")
        print(f"  Sample rate: {self.sample_rate} Hz")
        
        return output_path
    
    def _loop_audio(self, audio_segments: List[AudioSegment], target_duration: float) -> AudioSegment:
        """
        Loop audio segments to reach target duration
        
        Args:
            audio_segments: List of audio segments
            target_duration: Target duration in seconds
            
        Returns:
            Combined audio segment
        """
        target_ms = int(target_duration * 1000)
        
        # Start with empty audio
        combined = AudioSegment.empty()
        
        # Keep adding segments until we reach target duration
        while len(combined) < target_ms:
            for segment in audio_segments:
                combined += segment
                if len(combined) >= target_ms:
                    break
        
        # Trim to exact duration
        combined = combined[:target_ms]
        
        return combined
    
    def _apply_fade_out(self, audio: AudioSegment, fade_duration: int) -> AudioSegment:
        """
        Apply fade out to audio
        
        Args:
            audio: Audio segment
            fade_duration: Fade duration in seconds
            
        Returns:
            Audio with fade out applied
        """
        fade_ms = fade_duration * 1000
        
        # Don't fade if audio is shorter than fade duration
        if len(audio) < fade_ms:
            return audio
        
        return audio.fade_out(fade_ms)
    
    def generate_or_use_audio(
        self,
        input_folder: Path,
        sequence: Optional[List['SequenceItem']],  # FIXED: Quoted forward reference
        duration: float,
        output_path: Path,
        force_generate: bool = False
    ) -> Path:
        """
        Use existing music files if available, otherwise generate music
        
        Args:
            input_folder: Folder to search for music files
            sequence: Video sequence (for AI generation)
            duration: Target duration in seconds
            output_path: Where to save audio
            force_generate: Force AI generation even if files exist
            
        Returns:
            Path to audio file
        """
        # Check for existing music files
        music_files = self.find_music_files(input_folder) if not force_generate else []
        
        if music_files:
            print(f"\n{'='*70}")
            print(f"FOUND {len(music_files)} MUSIC FILE(S) IN INPUT FOLDER")
            print(f"{'='*70}")
            for i, f in enumerate(music_files, 1):
                print(f"{i}. {f.name}")
            print(f"\nUsing existing music instead of AI generation")
            
            return self.prepare_audio_from_files(music_files, duration, output_path)
        else:
            print(f"\n{'='*70}")
            print("NO MUSIC FILES FOUND - GENERATING WITH AI")
            print(f"{'='*70}")
            print("To use your own music, place MP3/WAV files in the input folder")
            
            return self.generate_soundtrack(sequence, duration, output_path)
    
    def load_model(self):
        """Load music generation model"""
        if self.model is not None:
            return
        
        print(f"\n{'='*70}")
        print(f"LOADING AUDIO MODEL: {self.model_key}")
        print(f"{'='*70}")
        print(f"Model: {self.model_info['name']}")
        print(f"Size: {self.model_info['size']}")
        print(f"This may take a minute...\n")
        
        try:
            # Load processor
            print("Loading processor...")
            self.processor = AutoProcessor.from_pretrained(self.model_info['name'])
            
            # Load model
            print("Loading model...")
            self.model = MusicgenForConditionalGeneration.from_pretrained(
                self.model_info['name'],
                dtype=torch.float16 if Config.USE_FP16 else torch.float32
            ).to(self.device)
            
            print(f"✓ Model loaded on {self.device}\n")
            
        except Exception as e:
            print(f"✗ Error loading model: {e}")
            raise
    
    def unload_model(self):
        """Unload model from memory"""
        if self.model:
            del self.model
            self.model = None
        if self.processor:
            del self.processor
            self.processor = None
        torch.cuda.empty_cache()
        print("✓ Model unloaded")
    
    def generate_soundtrack(
        self,
        sequence: Optional[List['SequenceItem']],  # FIXED: Quoted forward reference
        duration: float,
        output_path: Path
    ) -> Path:
        """
        Generate background music for entire video using AI
        
        Args:
            sequence: Video sequence
            duration: Total duration in seconds
            output_path: Where to save audio file
            
        Returns:
            Path to generated audio file
        """
        print(f"\n{'='*70}")
        print("GENERATING SOUNDTRACK WITH AI")
        print(f"{'='*70}")
        print(f"Duration: {duration:.2f} seconds ({duration/60:.2f} minutes)")
        print(f"Fade duration: {self.fade_duration} seconds")
        
        # Ensure model is loaded
        if not self.model:
            self.load_model()
        
        # Analyze overall mood/theme
        music_prompt = self._create_music_prompt(sequence) if sequence else "upbeat happy family video background music"
        print(f"\nMusic prompt: {music_prompt}")
        
        # Generate music
        print("\nGenerating music (this may take several minutes)...")
        audio = self._generate_music(music_prompt, duration)
        
        # Apply fade if specified
        if self.fade_duration > 0:
            print(f"\nApplying {self.fade_duration}s fade out...")
            audio = self._apply_fade_to_numpy(audio, duration, self.fade_duration)
        
        # Save to file
        self._save_audio(audio, output_path)
        
        print(f"\n✓ Soundtrack saved to: {output_path}")
        return output_path
    
    def _apply_fade_to_numpy(self, audio: np.ndarray, duration: float, fade_duration: int) -> np.ndarray:
        """
        Apply fade out to numpy audio array
        
        Args:
            audio: Audio array
            duration: Total duration in seconds
            fade_duration: Fade duration in seconds
            
        Returns:
            Audio with fade applied
        """
        if fade_duration <= 0 or fade_duration >= duration:
            return audio
        
        # Calculate fade start sample
        fade_start_sample = int((duration - fade_duration) * self.sample_rate)
        total_samples = len(audio)
        
        # Create fade curve
        fade_samples = total_samples - fade_start_sample
        fade_curve = np.linspace(1.0, 0.0, fade_samples)
        
        # Apply fade
        audio[fade_start_sample:] *= fade_curve
        
        return audio
    
    def _create_music_prompt(self, items: List['SequenceItem']) -> str:  # FIXED: Quoted forward reference
        """
        Create music generation prompt from scene analysis
        
        Args:
            items: Sequence items to analyze
            
        Returns:
            Music prompt string
        """
        if not items:
            return "upbeat happy family video background music"
        
        # Collect moods and themes
        moods = []
        scene_types = []
        
        for item in items:
            if hasattr(item, 'scene_analysis'):
                if hasattr(item.scene_analysis, 'mood') and item.scene_analysis.mood:
                    moods.append(item.scene_analysis.mood.lower())
                if hasattr(item.scene_analysis, 'scene_type') and item.scene_analysis.scene_type:
                    scene_types.append(item.scene_analysis.scene_type.lower())
        
        # Default to upbeat happy for family videos
        dominant_mood = "happy"
        
        # Try to extract meaningful mood
        if moods:
            mood_text = ' '.join(moods)
            if 'happy' in mood_text or 'joy' in mood_text or 'cheerful' in mood_text:
                dominant_mood = "happy"
            elif 'peaceful' in mood_text or 'calm' in mood_text:
                dominant_mood = "peaceful"
            elif 'energetic' in mood_text or 'exciting' in mood_text:
                dominant_mood = "energetic"
        
        # Determine scene setting
        outdoor_count = sum(1 for st in scene_types if 'outdoor' in st)
        
        if scene_types and outdoor_count > len(scene_types) / 2:
            setting = "bright outdoor"
        else:
            setting = "warm"
        
        # Create upbeat prompt
        prompt = f"upbeat {dominant_mood} {setting} family video background music, cheerful acoustic guitar"
        
        return prompt
    
    def _generate_music(self, prompt: str, duration: float) -> np.ndarray:
        """
        Generate music from text prompt
        
        Args:
            prompt: Text description of desired music
            duration: Duration in seconds
            
        Returns:
            Audio array
        """
        # Process prompt
        inputs = self.processor(
            text=[prompt],
            padding=True,
            return_tensors="pt"
        ).to(self.device)
        
        # Calculate number of tokens needed for duration
        tokens_per_second = 50 * (self.sample_rate / 32000)
        max_new_tokens = int(duration * tokens_per_second)
        
        # Clamp to reasonable limits
        max_new_tokens = max(256, min(max_new_tokens, 1500))
        
        # Generate
        with torch.no_grad():
            audio_values = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                guidance_scale=3.0
            )
        
        # Convert to numpy
        audio = audio_values[0, 0].cpu().numpy()
        
        # Adjust length to match desired duration
        target_samples = int(duration * self.sample_rate)
        current_samples = len(audio)
        
        if current_samples < target_samples:
            # Loop if too short
            repeats = (target_samples // current_samples) + 1
            audio = np.tile(audio, repeats)[:target_samples]
        elif current_samples > target_samples:
            # Trim if too long
            audio = audio[:target_samples]
        
        # Normalize
        audio = audio / np.max(np.abs(audio))
        
        return audio
    
    def _save_audio(self, audio: np.ndarray, output_path: Path):
        """
        Save audio to WAV file
        
        Args:
            audio: Audio array
            output_path: Output file path
        """
        # Ensure output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save as WAV
        scipy.io.wavfile.write(
            str(output_path),
            self.sample_rate,
            audio.astype(np.float32)
        )
    
    def export_metadata(self, metadata: dict, output_path: Path):
        """
        Export audio generation metadata
        
        Args:
            metadata: Metadata dictionary
            output_path: Output file path
        """
        with open(output_path, 'w') as f:
            json.dump(metadata, f, indent=2)


if __name__ == "__main__":
    # Test audio generator
    from config import Config
    
    print("Testing Enhanced Audio Generator...")
    
    # Check for music files
    generator = AudioGenerator(fade_duration=3)
    music_files = generator.find_music_files(Config.INPUT_DIR)
    
    if music_files:
        print(f"\nFound {len(music_files)} music file(s):")
        for f in music_files:
            print(f"  • {f.name}")
    else:
        print("\nNo music files found in input folder")
        print("Place MP3/WAV files in input folder to use them")