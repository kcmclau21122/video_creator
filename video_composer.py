# ============================================================================
# AI Video Creator - Step 6: Video Composer (CORRECTED)
# ============================================================================

"""
Video Composer
Combines media items, transitions, and audio into final video
"""

from pathlib import Path
from typing import List, Optional, Tuple
import json
from datetime import datetime
import numpy as np

from PIL import Image, ImageOps  # Add ImageOps if not already imported
from image_orientation import ImageOrientationFixer

from moviepy.editor import (
    VideoFileClip,
    ImageClip,
    AudioFileClip,
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
        print()
        
        # Create clips for each item
        clips = []
        
        for i, item in enumerate(sequence):
            print(f"Processing {i+1}/{len(sequence)}: {item.media_item.file_path.name}")
            
            try:
                if item.media_item.file_type == 'image':
                    clip = self._create_image_clip(item)
                else:
                    clip = self._create_video_clip(item)
                
                # Apply transitions
                if item.transition_in != 'cut' and item.transition_duration > 0:
                    clip = self._apply_transition(clip, item.transition_in, item.transition_duration)
                
                clips.append(clip)
                
            except Exception as e:
                print(f"  ⚠ Error processing {item.media_item.file_path.name}: {e}")
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
        print(f"Duration: {final_video.duration:.2f}s ({final_video.duration/60:.2f} min)")
        print("\nThis may take several minutes...")
        print()
        
        final_video.write_videofile(
            str(output_path),
            fps=self.output_fps,
            codec='libx264',
            audio_codec='aac',
            bitrate=Config.OUTPUT_BITRATE,
            preset='medium',
            threads=Config.NUM_CORES
        )
        
        # Clean up
        final_video.close()
        for clip in clips:
            clip.close()
        
        print(f"\n{'='*70}")
        print("VIDEO COMPOSITION COMPLETE")
        print(f"{'='*70}")
        
        return output_path
    
    def _create_image_clip(self, item: SequenceItem) -> ImageClip:
        """
        Create video clip from image with Ken Burns effect
        Now includes automatic orientation correction
        
        Args:
            item: Sequence item
            
        Returns:
            ImageClip
        """
        # Load image
        img = Image.open(item.media_item.file_path)
        
        # FIX ORIENTATION AUTOMATICALLY (NEW CODE)
        # This uses PIL's built-in method to fix orientation based on EXIF data
        img = ImageOps.exif_transpose(img)
        
        # Convert to RGB if needed
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Resize to fit output resolution while maintaining aspect ratio
        img = self._resize_image(img, self.output_resolution)
        
        # Create clip
        clip = ImageClip(np.array(img))
        clip = clip.set_duration(item.duration)
        clip = clip.set_fps(self.output_fps)
        
        # Apply Ken Burns effect (zoom/pan)
        clip = self._apply_ken_burns(clip, item.duration)
        
        return clip
    
    def _create_video_clip(self, item: SequenceItem) -> VideoFileClip:
        """
        Create clip from video file
        
        Args:
            item: Sequence item
            
        Returns:
            VideoFileClip
        """
        # Load video
        clip = VideoFileClip(str(item.media_item.file_path))
        
        # Resize to output resolution
        clip = clip.resize(self.output_resolution)
        
        # Trim to specified duration if needed
        if clip.duration > item.duration:
            clip = clip.subclip(0, item.duration)
        
        # Set FPS
        clip = clip.set_fps(self.output_fps)
        
        return clip
    
    def _resize_image(self, img: Image.Image, target_size: Tuple[int, int]) -> Image.Image:
        """
        Resize image to fit target size while maintaining aspect ratio
        Uses letterboxing/pillarboxing to avoid distortion
        
        Args:
            img: PIL Image
            target_size: (width, height)
            
        Returns:
            Resized PIL Image with black bars if needed
        """
        target_width, target_height = target_size
        img_width, img_height = img.size
        
        # Calculate aspect ratios
        target_ratio = target_width / target_height
        img_ratio = img_width / img_height
        
        # Determine scaling to fit within target (no cropping)
        if img_ratio > target_ratio:
            # Image is wider - fit to width
            new_width = target_width
            new_height = int(target_width / img_ratio)
        else:
            # Image is taller - fit to height
            new_height = target_height
            new_width = int(target_height * img_ratio)
        
        # Resize maintaining aspect ratio
        img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        # Create black background
        result = Image.new('RGB', target_size, (0, 0, 0))
        
        # Paste resized image centered
        x_offset = (target_width - new_width) // 2
        y_offset = (target_height - new_height) // 2
        result.paste(img, (x_offset, y_offset))
        
        return result

    
    def _apply_ken_burns(self, clip: ImageClip, duration: float) -> ImageClip:
        """
        Apply Ken Burns effect (slow zoom and pan)
        
        Args:
            clip: Image clip
            duration: Duration in seconds
            
        Returns:
            Clip with Ken Burns effect
        """
        # Define zoom range (1.0 to 1.15 = 15% zoom)
        zoom_start = 1.0
        zoom_end = 1.15
        
        def zoom_effect(get_frame, t):
            """Apply zoom at time t"""
            frame = get_frame(t)
            
            # Calculate zoom factor
            progress = min(t / duration, 1.0) if duration > 0 else 0
            zoom = zoom_start + (zoom_end - zoom_start) * progress
            
            # Get frame dimensions
            h, w = frame.shape[:2]
            
            # Calculate crop dimensions
            new_h = int(h / zoom)
            new_w = int(w / zoom)
            
            # Ensure dimensions are valid
            if new_h <= 0 or new_w <= 0:
                return frame
            
            # Center crop
            top = max(0, (h - new_h) // 2)
            left = max(0, (w - new_w) // 2)
            
            cropped = frame[top:top+new_h, left:left+new_w]
            
            # Resize back to original dimensions using PIL
            from PIL import Image as PILImage
            if len(cropped.shape) == 3:
                pil_img = PILImage.fromarray(cropped.astype('uint8'))
            else:
                pil_img = PILImage.fromarray(cropped.astype('uint8'), mode='L')
            
            pil_img = pil_img.resize((w, h), PILImage.Resampling.LANCZOS)
            
            return np.array(pil_img)
        
        # Apply effect
        return clip.fl(zoom_effect)
    
    def _apply_transition(self, clip, transition_type: str, duration: float):
        """
        Apply transition effect to clip
        
        Args:
            clip: Video clip
            transition_type: Type of transition
            duration: Transition duration
            
        Returns:
            Clip with transition
        """
        if transition_type == 'fade':
            clip = fadein(clip, duration)
        elif transition_type == 'dissolve':
            # Use fadein for dissolve effect
            clip = fadein(clip, duration)
        
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