# ============================================================================
# AI Video Creator - Audio Mixer Module
# Mixes narration, background music, and sound effects
# ============================================================================

"""
Audio Mixer
Professional audio mixing for movie-style videos
Combines narration, music, and effects with ducking and normalization
"""

from pathlib import Path
from typing import List, Optional
import numpy as np
from pydub import AudioSegment
from pydub.effects import normalize, compress_dynamic_range
from pydub.silence import detect_nonsilent

from config import Config


class AudioMixer:
    """Mix multiple audio tracks for professional sound"""
    
    def __init__(
        self,
        narration_volume: float = 1.0,  # 100% volume
        music_volume: float = 0.3,      # 30% volume (background)
        ducking_amount: float = 0.5,    # Reduce music to 50% during narration
        fade_duration: int = 0
    ):
        """
        Initialize audio mixer
        
        Args:
            narration_volume: Narration volume multiplier (0.0-1.0)
            music_volume: Background music volume (0.0-1.0)
            ducking_amount: How much to reduce music during speech (0.0-1.0)
            fade_duration: Fade out duration in seconds
        """
        self.narration_volume = narration_volume
        self.music_volume = music_volume
        self.ducking_amount = ducking_amount
        self.fade_duration = fade_duration
        self.sample_rate = Config.AUDIO_SAMPLE_RATE
    
    def mix_audio_tracks(
        self,
        narration_path: Optional[Path],
        music_paths: List[Path],
        video_duration: float,
        output_path: Path
    ) -> Path:
        """
        Mix narration and music tracks
        
        Args:
            narration_path: Path to narration audio
            music_paths: List of background music files
            video_duration: Video duration in seconds
            output_path: Where to save mixed audio
            
        Returns:
            Path to mixed audio file
        """
        print(f"\n{'='*70}")
        print("MIXING AUDIO TRACKS")
        print(f"{'='*70}")
        print(f"Video duration: {video_duration:.2f}s")
        print(f"Narration: {narration_path.name if narration_path else 'None'}")
        print(f"Music tracks: {len(music_paths)}")
        
        # Load music track
        print("\n[1/4] Loading background music...")
        background_music = self._prepare_background_music(music_paths, video_duration)
        
        # Apply music volume
        background_music = background_music + (self.music_volume * 100 - 100)  # Convert to dB
        print(f"  ✅ Music volume: {self.music_volume*100:.0f}%")
        
        # Load narration if available
        if narration_path and narration_path.exists():
            print("\n[2/4] Loading narration...")
            narration = AudioSegment.from_wav(str(narration_path))
            
            # Apply narration volume
            narration = narration + (self.narration_volume * 100 - 100)
            print(f"  ✅ Narration volume: {self.narration_volume*100:.0f}%")
            
            # Apply ducking (lower music during speech)
            print("\n[3/4] Applying audio ducking...")
            final_audio = self._apply_ducking(background_music, narration)
        else:
            print("\n[2/4] No narration - using music only")
            final_audio = background_music
        
        # Final processing
        print("\n[4/4] Final audio processing...")
        final_audio = self._finalize_audio(final_audio, video_duration)
        
        # Export
        print(f"\nExporting mixed audio to: {output_path}")
        final_audio.export(
            str(output_path),
            format="wav",
            parameters=["-ar", str(self.sample_rate)]
        )
        
        actual_duration = len(final_audio) / 1000.0
        print(f"\n✅ Audio mixing complete!")
        print(f"   Final duration: {actual_duration:.2f}s")
        
        return output_path
    
    def _prepare_background_music(
        self,
        music_paths: List[Path],
        target_duration: float
    ) -> AudioSegment:
        """
        Load and loop background music to match video duration
        
        Args:
            music_paths: List of music file paths
            target_duration: Target duration in seconds
            
        Returns:
            Background music AudioSegment
        """
        if not music_paths:
            # Create silence if no music
            target_ms = int(target_duration * 1000)
            return AudioSegment.silent(duration=target_ms)
        
        # Load all music files
        music_segments = []
        for music_path in music_paths:
            print(f"  Loading: {music_path.name}")
            try:
                music = AudioSegment.from_file(str(music_path))
                music_segments.append(music)
            except Exception as e:
                print(f"  ⚠️  Could not load {music_path.name}: {e}")
        
        if not music_segments:
            # Fallback to silence
            target_ms = int(target_duration * 1000)
            return AudioSegment.silent(duration=target_ms)
        
        # Combine and loop
        combined_music = AudioSegment.empty()
        target_ms = int(target_duration * 1000)
        
        while len(combined_music) < target_ms:
            for segment in music_segments:
                combined_music += segment
                if len(combined_music) >= target_ms:
                    break
        
        # Trim to exact duration
        combined_music = combined_music[:target_ms]
        
        print(f"  ✅ Music prepared: {len(combined_music)/1000:.2f}s")
        
        return combined_music
    
    def _apply_ducking(
        self,
        background_music: AudioSegment,
        narration: AudioSegment
    ) -> AudioSegment:
        """
        Apply audio ducking - lower music volume during speech
        
        Args:
            background_music: Background music track
            narration: Narration/speech track
            
        Returns:
            Mixed audio with ducking applied
        """
        # Detect where speech is present
        speech_segments = detect_nonsilent(
            narration,
            min_silence_len=500,  # 500ms of silence to detect gaps
            silence_thresh=-50    # dB threshold for silence
        )
        
        print(f"  Detected {len(speech_segments)} speech segments")
        
        # Create a copy of background music
        ducked_music = background_music
        
        # Lower volume during speech segments
        for start_ms, end_ms in speech_segments:
            # Add fade margins (200ms before and after)
            fade_margin = 200
            duck_start = max(0, start_ms - fade_margin)
            duck_end = min(len(ducked_music), end_ms + fade_margin)
            
            # Calculate ducking level (dB reduction)
            duck_db = -10 * (1 - self.ducking_amount)  # Reduce by up to 10dB
            
            # Extract segment
            segment = ducked_music[duck_start:duck_end]
            
            # Apply ducking
            ducked_segment = segment + duck_db
            
            # Apply fade in/out to ducked segment for smooth transition
            ducked_segment = ducked_segment.fade_in(fade_margin).fade_out(fade_margin)
            
            # Replace in track
            ducked_music = (
                ducked_music[:duck_start] +
                ducked_segment +
                ducked_music[duck_end:]
            )
        
        # Overlay narration on ducked music
        # Ensure both tracks are same length
        max_len = max(len(ducked_music), len(narration))
        
        if len(ducked_music) < max_len:
            ducked_music += AudioSegment.silent(duration=max_len - len(ducked_music))
        if len(narration) < max_len:
            narration += AudioSegment.silent(duration=max_len - len(narration))
        
        # Mix tracks
        mixed_audio = ducked_music.overlay(narration)
        
        print(f"  ✅ Ducking applied")
        
        return mixed_audio
    
    def _finalize_audio(
        self,
        audio: AudioSegment,
        target_duration: float
    ) -> AudioSegment:
        """
        Final audio processing - compression, normalization, fade
        
        Args:
            audio: Audio to process
            target_duration: Target duration in seconds
            
        Returns:
            Processed audio
        """
        # Ensure correct duration
        target_ms = int(target_duration * 1000)
        if len(audio) > target_ms:
            audio = audio[:target_ms]
        elif len(audio) < target_ms:
            audio += AudioSegment.silent(duration=target_ms - len(audio))
        
        # Apply dynamic range compression for consistent volume
        print("  Applying compression...")
        audio = compress_dynamic_range(audio)
        
        # Normalize to prevent clipping
        print("  Normalizing audio...")
        audio = normalize(audio)
        
        # Apply fade out if specified
        if self.fade_duration > 0:
            fade_ms = self.fade_duration * 1000
            print(f"  Applying {self.fade_duration}s fade out...")
            audio = audio.fade_out(fade_ms)
        
        print("  ✅ Final processing complete")
        
        return audio
    
    def create_ambient_soundscape(
        self,
        scene_types: List[str],
        duration: float,
        output_path: Path
    ) -> Path:
        """
        Generate ambient sounds based on scene types
        (Feature for future enhancement - placeholder)
        
        Args:
            scene_types: Types of scenes (outdoor, indoor, etc.)
            duration: Duration in seconds
            output_path: Where to save ambient audio
            
        Returns:
            Path to ambient audio
        """
        # This would use sound effect libraries or generation
        # For now, return silence
        
        print("\n📝 Ambient soundscape generation not yet implemented")
        print("   Future feature: Add nature sounds, city ambiance, etc.")
        
        # Create silence for now
        target_ms = int(duration * 1000)
        silence = AudioSegment.silent(duration=target_ms)
        
        silence.export(
            str(output_path),
            format="wav",
            parameters=["-ar", str(self.sample_rate)]
        )
        
        return output_path


if __name__ == "__main__":
    # Test audio mixer
    from config import Config
    
    print("Audio Mixer Module")
    print("\nThis module mixes:")
    print("  • Narration (voice-over)")
    print("  • Background music")
    print("  • Sound effects (future)")
    print("\nFeatures:")
    print("  • Audio ducking (music lowers during speech)")
    print("  • Dynamic compression")
    print("  • Normalization")
    print("  • Fade effects")