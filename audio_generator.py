# ============================================================================
# AI Video Creator - Step 5: Audio Generator
# ============================================================================

"""
Audio Generator
Generates background music and ambient sounds using AI models
"""

from pathlib import Path
from typing import List, Dict, Optional
import json
import torch
import scipy
import numpy as np
from transformers import AutoProcessor, MusicgenForConditionalGeneration
from datetime import datetime

from sequencing_engine import SequenceItem
from scene_analyzer import SceneAnalysis
from config import Config


class AudioGenerator:
    """Generate audio using AI models"""
    
    # Available models
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
    
    def __init__(self, model_key: str = 'musicgen-medium'):
        """
        Initialize audio generator
        
        Args:
            model_key: Model to use for generation
        """
        self.model_key = model_key
        self.model_info = self.MODELS.get(model_key)
        if not self.model_info:
            raise ValueError(f"Unknown model: {model_key}")
        
        self.model = None
        self.processor = None
        self.device = Config.DEVICE
        self.sample_rate = Config.AUDIO_SAMPLE_RATE
        
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
        sequence: List[SequenceItem],
        duration: float,
        output_path: Path
    ) -> Path:
        """
        Generate background music for entire video
        
        Args:
            sequence: Video sequence
            duration: Total duration in seconds
            output_path: Where to save audio file
            
        Returns:
            Path to generated audio file
        """
        print(f"\n{'='*70}")
        print("GENERATING SOUNDTRACK")
        print(f"{'='*70}")
        print(f"Duration: {duration:.2f} seconds ({duration/60:.2f} minutes)")
        
        # Ensure model is loaded
        if not self.model:
            self.load_model()
        
        # Analyze overall mood/theme
        music_prompt = self._create_music_prompt(sequence)
        print(f"\nMusic prompt: {music_prompt}")
        
        # Generate music
        print("\nGenerating music (this may take several minutes)...")
        audio = self._generate_music(music_prompt, duration)
        
        # Save to file
        self._save_audio(audio, output_path)
        
        print(f"\n✓ Soundtrack saved to: {output_path}")
        return output_path
    
    def generate_segmented_soundtrack(
        self,
        sequence: List[SequenceItem],
        output_dir: Path
    ) -> List[Path]:
        """
        Generate separate music for each scene group
        
        Args:
            sequence: Video sequence
            output_dir: Directory to save audio segments
            
        Returns:
            List of paths to generated audio files
        """
        print(f"\n{'='*70}")
        print("GENERATING SEGMENTED SOUNDTRACK")
        print(f"{'='*70}")
        
        # Ensure model is loaded
        if not self.model:
            self.load_model()
        
        # Group by scene group_id
        groups = {}
        for item in sequence:
            group_id = item.group_id
            if group_id not in groups:
                groups[group_id] = []
            groups[group_id].append(item)
        
        print(f"Found {len(groups)} scene groups\n")
        
        # Generate music for each group
        output_dir.mkdir(parents=True, exist_ok=True)
        audio_files = []
        
        for group_id, items in groups.items():
            # Calculate group duration
            start_time = items[0].start_time
            end_time = items[-1].start_time + items[-1].duration
            duration = end_time - start_time
            
            print(f"\nGroup {group_id}: {duration:.2f}s ({len(items)} items)")
            
            # Create prompt for this group
            prompt = self._create_music_prompt(items)
            print(f"Prompt: {prompt}")
            
            # Generate
            audio = self._generate_music(prompt, duration)
            
            # Save
            output_path = output_dir / f"segment_{group_id:03d}.wav"
            self._save_audio(audio, output_path)
            audio_files.append(output_path)
            
            print(f"✓ Saved: {output_path.name}")
        
        print(f"\n✓ Generated {len(audio_files)} audio segments")
        return audio_files
    
    def _create_music_prompt(self, items: List[SequenceItem]) -> str:
        """
        Create music generation prompt from scene analysis
        
        Args:
            items: Sequence items to analyze
            
        Returns:
            Music prompt string
        """
        # Collect moods and themes
        moods = []
        scene_types = []
        
        for item in items:
            if item.scene_analysis.mood:
                moods.append(item.scene_analysis.mood.lower())
            if item.scene_analysis.scene_type:
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
        
        if outdoor_count > len(scene_types) / 2:
            setting = "bright outdoor"
        else:
            setting = "warm"
        
        # Create upbeat prompt - ALWAYS upbeat for family videos
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
        # MusicGen generates at ~50 tokens per second at 32kHz
        # We use 44.1kHz, so adjust accordingly
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
    import json
    from sequencing_engine import SequenceItem
    from scene_analyzer import SceneAnalysis
    from media_ingester import MediaItem
    
    print("Testing Audio Generator...")
    
    # Load sequence
    sequence_file = Config.OUTPUT_DIR / "video_sequence.json"
    if not sequence_file.exists():
        print("✗ Sequence file not found")
        print("Run: python test_sequencing.py first")
        exit(1)
    
    with open(sequence_file, 'r') as f:
        sequence_data = json.load(f)
    
    print(f"Loaded sequence with {sequence_data['total_items']} items")
    print(f"Duration: {sequence_data['total_duration']:.2f} seconds")
    
    # For testing, generate short sample
    test_duration = min(10.0, sequence_data['total_duration'])
    
    # Create generator
    generator = AudioGenerator(model_key='musicgen-small')
    
    # Create simple test prompt
    print("\nGenerating test audio...")
    
    generator.load_model()
    
    # Generate
    inputs = generator.processor(
        text=["upbeat happy background music"],
        padding=True,
        return_tensors="pt"
    ).to(generator.device)
    
    with torch.no_grad():
        audio_values = generator.model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=True,
            guidance_scale=3.0
        )
    
    audio = audio_values[0, 0].cpu().numpy()
    
    # Save
    output_path = Config.OUTPUT_DIR / "test_audio.wav"
    generator._save_audio(audio, output_path)
    
    print(f"\n✓ Test audio saved to: {output_path}")
    print(f"Duration: {len(audio) / generator.sample_rate:.2f} seconds")
    
    generator.unload_model()