# ============================================================================
# AI Video Creator - Step 6: Video Composer (FIXED - file_path handling)
# ============================================================================

"""
Video Composer
Combines media items, transitions, and audio into final video
FIXED: Handle both Path and string file_path types
"""

from pathlib import Path
from typing import List, Optional, Tuple
import json
from datetime import datetime
import numpy as np

from PIL import Image, ImageOps
from image_orientation import ImageOrientationFixer

from moviepy.editor import (
    VideoFileClip,
    ImageClip,
    AudioFileClip,
    ColorClip,
    CompositeVideoClip,
    concatenate_videoclips,
    concatenate_audioclips
)

from moviepy.video.fx.all import fadein, fadeout
from PIL import Image

from sequencing_engine import SequenceItem
from config import Config


class VideoComposer:
    """Compose final video from sequence and audio"""
    
    def __init__(self):
        """Initialize video composer"""
        self.output_resolution = Config.OUTPUT_RESOLUTION
        self.output_fps = Config.OUTPUT_FPS
        self.orientation_fixer = ImageOrientationFixer()
        
    def compose_video(
        self,
        sequence: List[SequenceItem],
        audio_path: Optional[Path] = None,
        output_path: Path = None
    ) -> Path:
        """
        Compose final video from sequence
        
        Args:
            sequence: List of sequence items
            audio_path: Path to audio file
            output_path: Where to save final video
            
        Returns:
            Path to output video
        """
        print(f"\n{'='*70}")
        print("COMPOSING VIDEO")
        print(f"{'='*70}")
        print(f"Items: {len(sequence)}")
        print(f"Resolution: {self.output_resolution[0]}x{self.output_resolution[1]}")
        print(f"FPS: {self.output_fps}")
        print(f"{'='*70}")
        
        # Create clips for each item
        clips = []
        
        for i, item in enumerate(sequence):
            # FIXED: Handle both Path and string types for file_path
            file_path = item.media_item.file_path
            if isinstance(file_path, str):
                file_path = Path(file_path)
            
            file_name = file_path.name
            file_type = item.media_item.file_type
            
            print(f"Processing {i+1}/{len(sequence)}: {file_name} ({file_type})")
            
            try:
                if file_type == 'image':
                    clip = self._create_image_clip(item)
                elif file_type == 'video':
                    clip = self._create_video_clip(item)
                else:
                    print(f"  ⚠  Unknown file type: {file_type}, skipping")
                    continue
                
                # Apply transitions
                if item.transition_in != 'cut' and item.transition_duration > 0:
                    clip = self._apply_transition(clip, item.transition_in, item.transition_duration)
                
                clips.append(clip)
                
            except Exception as e:
                print(f"  ⚠  Error processing {file_name}: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        if not clips:
            raise ValueError("No clips were successfully created")
        
        print(f"\n✓ Created {len(clips)} clips")
        
        # Concatenate all clips
        print("\nConcatenating clips...")
        final_video = concatenate_videoclips(clips, method="compose")
        
        # Add audio if provided
        if audio_path and audio_path.exists():
            print(f"\nAdding audio: {audio_path.name}")
            audio = AudioFileClip(str(audio_path))
            
            # Adjust audio duration to match video
            if audio.duration < final_video.duration:
                # Loop audio if too short
                num_loops = int(final_video.duration / audio.duration) + 1
                audio_list = [audio] * num_loops
                audio = concatenate_audioclips(audio_list)
            
            # Trim to exact duration
            audio = audio.subclip(0, final_video.duration)
            final_video = final_video.set_audio(audio)
        
        # Set output path
        if output_path is None:
            output_path = Config.OUTPUT_DIR / f"final_video_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
        
        # Render
        print(f"\n{'='*70}")
        print("RENDERING VIDEO")
        print(f"{'='*70}")
        print(f"Output: {output_path}")
        print(f"Duration: {final_video.duration:.2f}s ({final_video.duration/60:.2f}min)")
        
        # Write video file
        final_video.write_videofile(
            str(output_path),
            fps=self.output_fps,
            codec='libx264',
            audio_codec='aac',
            temp_audiofile='temp-audio.m4a',
            remove_temp=True,
            preset='medium',
            threads=4
        )
        
        # Clean up
        final_video.close()
        
        print(f"\n✓ Video saved: {output_path}")
        print(f"File size: {output_path.stat().st_size / (1024*1024):.2f} MB")
        
        return output_path
    
    def _create_image_clip(self, item: SequenceItem) -> ImageClip:
        """Create video clip from image with Ken Burns effect"""
        # FIXED: Ensure file_path is Path object
        file_path = item.media_item.file_path
        if isinstance(file_path, str):
            file_path = Path(file_path)
        
        # Load and fix orientation
        img = self.orientation_fixer.fix_orientation(
            file_path, 
            item.media_item.orientation
        )
        
        # Resize maintaining aspect ratio
        img = self._resize_image(img, self.output_resolution)
        
        # Convert to numpy array
        img_array = np.array(img)
        
        # Create clip
        clip = ImageClip(img_array)
        clip = clip.set_duration(item.duration)
        
        # Apply Ken Burns effect (zoom and pan)
        clip = self._apply_ken_burns(clip, item.duration)
        
        return clip
    
    def _create_video_clip(self, item: SequenceItem) -> VideoFileClip:
        """Create clip from video file"""
        # FIXED: Ensure file_path is Path object
        file_path = item.media_item.file_path
        if isinstance(file_path, str):
            file_path = Path(file_path)
        
        # Load video
        clip = VideoFileClip(str(file_path))
        
        # Trim to desired duration
        if clip.duration > item.duration:
            clip = clip.subclip(0, item.duration)
        elif clip.duration < item.duration:
            # Loop if needed
            num_loops = int(item.duration / clip.duration) + 1
            clips = [clip] * num_loops
            clip = concatenate_videoclips(clips)
            clip = clip.subclip(0, item.duration)
        
        # Resize to output resolution
        clip = clip.resize(self.output_resolution)
        
        return clip
    
    def _resize_image(self, img: Image.Image, target_size: Tuple[int, int]) -> Image.Image:
        """Resize image to fit target size while maintaining aspect ratio"""
        target_width, target_height = target_size
        img_width, img_height = img.size
        
        # Calculate scaling factor
        scale_w = target_width / img_width
        scale_h = target_height / img_height
        scale = max(scale_w, scale_h)  # Scale to fill
        
        # Resize
        new_width = int(img_width * scale)
        new_height = int(img_height * scale)
        img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        # Center crop to exact size
        left = (new_width - target_width) // 2
        top = (new_height - target_height) // 2
        right = left + target_width
        bottom = top + target_height
        
        img = img.crop((left, top, right, bottom))
        
        return img
    
    def _apply_ken_burns(self, clip: ImageClip, duration: float) -> ImageClip:
        """Apply Ken Burns effect (zoom and pan)"""
        import random
        
        # Random zoom parameters
        start_zoom = 1.0
        end_zoom = random.uniform(1.1, 1.3)
        
        # Random pan direction
        pan_x = random.uniform(-0.05, 0.05)
        pan_y = random.uniform(-0.05, 0.05)
        
        def effect(get_frame, t):
            """Apply zoom effect over time"""
            progress = t / duration
            zoom = start_zoom + (end_zoom - start_zoom) * progress
            
            frame = get_frame(t)
            h, w = frame.shape[:2]
            
            # Calculate new dimensions
            new_h, new_w = int(h * zoom), int(w * zoom)
            
            # Resize frame
            from scipy.ndimage import zoom as scipy_zoom
            frame_zoomed = scipy_zoom(frame, [zoom, zoom, 1], order=1)
            
            # Calculate crop position with pan
            center_y = int(new_h / 2 + pan_y * new_h * progress)
            center_x = int(new_w / 2 + pan_x * new_w * progress)
            
            start_y = max(0, center_y - h // 2)
            start_x = max(0, center_x - w // 2)
            
            # Ensure we don't go out of bounds
            start_y = min(start_y, new_h - h)
            start_x = min(start_x, new_w - w)
            
            # Crop to original size
            frame_cropped = frame_zoomed[start_y:start_y+h, start_x:start_x+w]
            
            return frame_cropped
        
        return clip.fl(effect)
    
    def _apply_transition(self, clip, transition_type: str, duration: float):
        """Apply transition effect to clip"""
        if transition_type == 'fade':
            clip = fadein(clip, duration)
        elif transition_type == 'dissolve':
            clip = fadein(clip, duration)
        # Add more transition types as needed
        
        return clip
    
    def export_metadata(self, metadata: dict, output_path: Path):
        """
        Export video composition metadata
        
        Args:
            metadata: Metadata dictionary
            output_path: Output file path
        """
        with open(output_path, 'w') as f:
            json.dump(metadata, f, indent=2)

if __name__ == "__main__":
    # Test video composer
    from config import Config
    import json
    
    print("Testing Video Composer...")
    
    # Load sequence
    sequence_file = Config.OUTPUT_DIR / "video_sequence.json"
    if not sequence_file.exists():
        print("Sequence file not found")
        print("Run: python test_sequencing.py first")
        exit(1)
    
    with open(sequence_file, 'r') as f:
        sequence_data = json.load(f)
    
    print(f"Loaded sequence with {sequence_data['total_items']} items")
    
    # For testing, just show what would happen
    print("\nVideo composer ready")
    print("Run: python test_video_composition.py to create video")