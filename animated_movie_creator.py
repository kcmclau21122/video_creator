"""
Animated Movie Creator - Converts static images to fully animated videos
Uses Stable Video Diffusion and frame interpolation for smooth animations
"""

import os
import json
import torch
import numpy as np
from PIL import Image
from pathlib import Path
from typing import List, Dict, Tuple
import time

# Core dependencies
from diffusers import StableVideoDiffusionPipeline
from moviepy.editor import (
    ImageSequenceClip, 
    concatenate_videoclips, 
    AudioFileClip,
    CompositeVideoClip
)

# Frame interpolation for smooth transitions
try:
    from rife_ncnn_vulkan_python import Rife
    RIFE_AVAILABLE = True
except ImportError:
    RIFE_AVAILABLE = False
    print("⚠️  RIFE not available. Install with: pip install rife-ncnn-vulkan-python")


class AnimatedMovieCreator:
    """
    Creates fully animated videos from static images using AI models
    """
    
    def __init__(self, 
                 model_name="stabilityai/stable-video-diffusion-img2vid-xt",
                 device="cuda",
                 fps=25,
                 output_dir="output"):
        """
        Initialize the animation pipeline
        
        Args:
            model_name: HuggingFace model for image-to-video
            device: 'cuda' or 'cpu'
            fps: Frames per second for output video
            output_dir: Directory for output files
        """
        self.device = device
        self.fps = fps
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        print("\n" + "="*70)
        print("INITIALIZING ANIMATION ENGINE")
        print("="*70)
        
        # Check GPU availability
        if device == "cuda" and not torch.cuda.is_available():
            print("⚠️  CUDA not available, falling back to CPU")
            self.device = "cpu"
        
        # Load Stable Video Diffusion model
        print(f"\n📦 Loading animation model: {model_name}")
        print("   This may take a few minutes on first run...")
        
        try:
            self.svd_pipe = StableVideoDiffusionPipeline.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                variant="fp16" if self.device == "cuda" else None
            )
            
            if self.device == "cuda":
                self.svd_pipe.enable_model_cpu_offload()
            else:
                self.svd_pipe.to("cpu")
            
            print("✓ Animation model loaded successfully!")
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            raise
        
        # Initialize frame interpolator for smooth transitions
        self.interpolator = None
        if RIFE_AVAILABLE:
            try:
                self.interpolator = Rife()
                print("✓ Frame interpolator loaded (RIFE)")
            except:
                print("⚠️  Frame interpolator unavailable")
        
        print("="*70)
    
    def animate_single_image(self, 
                            image_path: str, 
                            num_frames: int = 25,
                            motion_intensity: int = 127,
                            target_size: Tuple[int, int] = (1024, 576)) -> List[np.ndarray]:
        """
        Animate a single static image using AI
        
        Args:
            image_path: Path to input image
            num_frames: Number of frames to generate (default 25 = 1 second at 25fps)
            motion_intensity: 0-255, controls amount of motion (127 = moderate)
            target_size: (width, height) for output frames
            
        Returns:
            List of frame arrays
        """
        print(f"\n🎬 Animating: {Path(image_path).name}")
        
        # Load and prepare image
        image = Image.open(image_path).convert("RGB")
        
        # Resize maintaining aspect ratio
        image.thumbnail(target_size, Image.Resampling.LANCZOS)
        
        # Create canvas and center image
        canvas = Image.new("RGB", target_size, (0, 0, 0))
        offset = ((target_size[0] - image.width) // 2,
                 (target_size[1] - image.height) // 2)
        canvas.paste(image, offset)
        
        print(f"   Generating {num_frames} animated frames...")
        print(f"   Motion intensity: {motion_intensity}/255")
        
        start_time = time.time()
        
        try:
            # Generate animated frames using SVD
            result = self.svd_pipe(
                canvas,
                num_frames=num_frames,
                motion_bucket_id=motion_intensity,
                decode_chunk_size=8,
                fps=self.fps,
                noise_aug_strength=0.02
            )
            
            frames = result.frames[0]
            
            # Convert PIL Images to numpy arrays
            frame_arrays = [np.array(frame) for frame in frames]
            
            elapsed = time.time() - start_time
            print(f"   ✓ Generated {len(frame_arrays)} frames in {elapsed:.1f}s")
            
            return frame_arrays
            
        except Exception as e:
            print(f"   ❌ Animation failed: {e}")
            # Fallback: return static frame repeated
            static_frame = np.array(canvas)
            return [static_frame] * num_frames
    
    def create_smooth_transition(self, 
                                frames_a: List[np.ndarray], 
                                frames_b: List[np.ndarray],
                                transition_frames: int = 10) -> List[np.ndarray]:
        """
        Create smooth transition between two animated clips using frame interpolation
        
        Args:
            frames_a: Last frames of clip A
            frames_b: First frames of clip B  
            transition_frames: Number of transition frames to generate
            
        Returns:
            List of transition frame arrays
        """
        if not self.interpolator:
            # Simple crossfade fallback
            last_frame = frames_a[-1]
            first_frame = frames_b[0]
            
            transition = []
            for i in range(transition_frames):
                alpha = i / transition_frames
                blended = (last_frame * (1 - alpha) + first_frame * alpha).astype(np.uint8)
                transition.append(blended)
            
            return transition
        
        # Use RIFE for smooth interpolation
        try:
            last_frame_pil = Image.fromarray(frames_a[-1])
            first_frame_pil = Image.fromarray(frames_b[0])
            
            # Generate intermediate frames
            transition = []
            for i in range(transition_frames):
                timestep = i / transition_frames
                interpolated = self.interpolator.process(
                    last_frame_pil, 
                    first_frame_pil, 
                    timestep=timestep
                )
                transition.append(np.array(interpolated))
            
            return transition
            
        except Exception as e:
            print(f"   ⚠️  Interpolation failed, using crossfade: {e}")
            # Fallback to crossfade
            transition = []
            for i in range(transition_frames):
                alpha = i / transition_frames
                blended = (frames_a[-1] * (1 - alpha) + frames_b[0] * alpha).astype(np.uint8)
                transition.append(blended)
            return transition
    
    def create_animated_movie(self,
                            sequence_file: str,
                            audio_file: str,
                            output_filename: str = "animated_movie.mp4",
                            frames_per_image: int = 50,
                            motion_intensity: int = 127,
                            transition_frames: int = 15) -> str:
        """
        Create complete animated movie from image sequence
        
        Args:
            sequence_file: JSON file with image sequence and metadata
            audio_file: Path to soundtrack
            output_filename: Output video filename
            frames_per_image: How many frames to generate per image (50 = 2 seconds at 25fps)
            motion_intensity: 0-255, amount of motion in animations
            transition_frames: Number of frames for transitions between scenes
            
        Returns:
            Path to output video file
        """
        print("\n" + "="*70)
        print("CREATING ANIMATED MOVIE")
        print("="*70)
        
        # Load sequence
        with open(sequence_file, 'r') as f:
            sequence_data = json.load(f)
        
        items = sequence_data.get('items', [])
        print(f"\n📸 Processing {len(items)} images")
        print(f"🎞️  {frames_per_image} frames per image ({frames_per_image/self.fps:.1f}s each)")
        print(f"🔄 {transition_frames} transition frames ({transition_frames/self.fps:.1f}s)")
        
        # Animate each image
        all_animated_clips = []

        for idx, item in enumerate(items, 1):
            print(f"\n[{idx}/{len(items)}]", end=" ")
            
            # Try different possible key names for the image path
            # CRITICAL: Include 'media_item_path' which is what the sequence JSON uses!
            image_path = (item.get('media_item_path') or      # <-- ADD THIS FIRST!
                        item.get('path') or 
                        item.get('file') or 
                        item.get('source') or 
                        item.get('image') or
                        item.get('input'))
            
            if image_path is None:
                print(f"⚠️  No path found in item: {item.keys()}")
                print(f"   Available keys: {list(item.keys())}")
                print(f"   Skipping this item")
                continue
                
            if not os.path.exists(image_path):
                print(f"⚠️  Image not found: {image_path}")
                continue
            
            # Generate animation for this image
            frames = self.animate_single_image(
                image_path,
                num_frames=frames_per_image,
                motion_intensity=motion_intensity
            )
            
            all_animated_clips.append(frames)
        
        if not all_animated_clips:
            raise ValueError("No images were successfully animated!")
        
        # Combine clips with smooth transitions
        print("\n" + "="*70)
        print("COMPOSING VIDEO WITH SMOOTH TRANSITIONS")
        print("="*70)
        
        final_frames = []
        
        for i, clip_frames in enumerate(all_animated_clips):
            print(f"\nAdding clip {i+1}/{len(all_animated_clips)}")
            
            # Add the animated clip
            final_frames.extend(clip_frames)
            
            # Add transition to next clip (except for last clip)
            if i < len(all_animated_clips) - 1:
                print(f"   Creating smooth transition to next scene...")
                transition = self.create_smooth_transition(
                    clip_frames,
                    all_animated_clips[i + 1],
                    transition_frames
                )
                final_frames.extend(transition)
        
        print(f"\n✓ Total frames: {len(final_frames)}")
        print(f"✓ Duration: {len(final_frames)/self.fps:.2f} seconds")
        
        # Create video clip from frames
        print("\n" + "="*70)
        print("RENDERING FINAL VIDEO")
        print("="*70)
        
        video_clip = ImageSequenceClip(final_frames, fps=self.fps)
        
        # Add audio
        if os.path.exists(audio_file):
            print(f"🎵 Adding soundtrack: {Path(audio_file).name}")
            audio = AudioFileClip(audio_file)
            
            # Match audio duration to video
            if audio.duration > video_clip.duration:
                audio = audio.subclip(0, video_clip.duration)
            
            video_clip = video_clip.set_audio(audio)
        else:
            print(f"⚠️  Audio file not found: {audio_file}")
        
        # Export final video
        output_path = self.output_dir / output_filename
        print(f"\n💾 Exporting to: {output_path}")
        print("   This may take several minutes...")
        
        video_clip.write_videofile(
            str(output_path),
            codec='libx264',
            audio_codec='aac',
            fps=self.fps,
            preset='medium',
            bitrate='8000k'
        )
        
        file_size = os.path.getsize(output_path) / (1024 * 1024)
        
        print("\n" + "="*70)
        print("🎬 ANIMATED MOVIE COMPLETE!")
        print("="*70)
        print(f"\n🎞️  Output: {output_path}")
        print(f"💾 Size: {file_size:.2f} MB")
        print(f"⏱️  Duration: {video_clip.duration:.2f} seconds")
        print(f"🎨 Animation: AI-Generated Motion")
        print(f"✨ Smooth Transitions: Enabled")
        print("\n🎥 Ready for YouTube!")
        
        return str(output_path)


def main():
    """
    Example usage
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='Create animated movies from static images')
    parser.add_argument('--sequence', required=True, help='Path to video_sequence.json')
    parser.add_argument('--audio', required=True, help='Path to soundtrack file')
    parser.add_argument('--output', default='animated_movie.mp4', help='Output filename')
    parser.add_argument('--fps', type=int, default=25, help='Frames per second')
    parser.add_argument('--frames-per-image', type=int, default=50, 
                       help='Frames to generate per image (50 = 2 seconds at 25fps)')
    parser.add_argument('--motion', type=int, default=127, 
                       help='Motion intensity 0-255 (127 = moderate)')
    parser.add_argument('--transition-frames', type=int, default=15,
                       help='Number of transition frames between scenes')
    parser.add_argument('--device', default='cuda', choices=['cuda', 'cpu'],
                       help='Device to run on')
    
    args = parser.parse_args()
    
    # Create animator
    animator = AnimatedMovieCreator(
        device=args.device,
        fps=args.fps
    )
    
    # Generate animated movie
    output_path = animator.create_animated_movie(
        sequence_file=args.sequence,
        audio_file=args.audio,
        output_filename=args.output,
        frames_per_image=args.frames_per_image,
        motion_intensity=args.motion,
        transition_frames=args.transition_frames
    )
    
    print(f"\n✓ Animation complete: {output_path}")


if __name__ == "__main__":
    main()