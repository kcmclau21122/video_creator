# video_creator_main.py
"""
Main application for creating YouTube videos from images and video clips
using multiple open-source LLMs and animation models
"""

import os
import json
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Tuple
import argparse

# Core imports
from PIL import Image
import torch
from moviepy.editor import ImageClip, VideoFileClip, concatenate_videoclips, AudioFileClip, CompositeAudioClip
import numpy as np

# Model imports
from transformers import AutoProcessor, AutoModelForCausalLM  # For image analysis
from diffusers import StableVideoDiffusionPipeline
import torchaudio


class MediaItem:
    """Represents a media item (image or video) with metadata"""
    def __init__(self, path: str):
        self.path = Path(path)
        self.timestamp = self._extract_timestamp()
        self.type = 'video' if self.path.suffix.lower() in ['.mp4', '.avi', '.mov'] else 'image'
        
    def _extract_timestamp(self) -> datetime:
        """Extract timestamp from EXIF data or file metadata"""
        try:
            if self.path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                from PIL import Image
                from PIL.ExifTags import TAGS
                
                img = Image.open(self.path)
                exif = img._getexif()
                
                if exif:
                    for tag_id, value in exif.items():
                        tag = TAGS.get(tag_id, tag_id)
                        if tag == 'DateTimeOriginal':
                            return datetime.strptime(value, '%Y:%m:%d %H:%M:%S')
            
            # Fallback to file creation time
            stat = self.path.stat()
            return datetime.fromtimestamp(stat.st_ctime)
            
        except Exception as e:
            print(f"Warning: Could not extract timestamp from {self.path}: {e}")
            return datetime.fromtimestamp(self.path.stat().st_ctime)


class SceneAnalyzer:
    """Analyzes images and videos to generate scene descriptions using Vision LLM"""
    
    def __init__(self, model_name: str = "microsoft/git-large-coco"):
        print(f"Loading scene analyzer: {model_name}")
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        
    def analyze(self, image_path: str) -> Dict[str, str]:
        """Analyze an image and return scene description"""
        image = Image.open(image_path).convert('RGB')
        
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        generated_ids = self.model.generate(pixel_values=inputs.pixel_values, max_length=50)
        caption = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
        return {
            'caption': caption,
            'path': image_path,
            'mood': self._infer_mood(caption)
        }
    
    def _infer_mood(self, caption: str) -> str:
        """Infer mood from caption for music generation"""
        mood_keywords = {
            'happy': ['smiling', 'celebration', 'joy', 'party', 'bright'],
            'calm': ['peaceful', 'quiet', 'serene', 'nature', 'sunset'],
            'energetic': ['action', 'sports', 'running', 'playing'],
            'dramatic': ['storm', 'dark', 'intense', 'conflict']
        }
        
        caption_lower = caption.lower()
        for mood, keywords in mood_keywords.items():
            if any(kw in caption_lower for kw in keywords):
                return mood
        
        return 'neutral'


class SimpleImageAnimator:
    """Animates still images using Stable Video Diffusion"""
    
    def __init__(self, animation_style='realistic', use_interpolation=True, enable_depth=False):
        self.animation_style = animation_style
        self.pipe = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
    def load_models(self):
        """Load Stable Video Diffusion pipeline with memory optimizations"""
        print("Loading Stable Video Diffusion model...")
        
        self.pipe = StableVideoDiffusionPipeline.from_pretrained(
            "stabilityai/stable-video-diffusion-img2vid-xt",
            torch_dtype=torch.float16,
            variant="fp16"
        )
        
        # Memory optimizations for SVD
        self.pipe.enable_model_cpu_offload()  # Moves models to CPU when not in use
        
        print("✓ Animation model loaded successfully")
        
    def animate_image(self, image_path: str, output_path: str, 
                     num_frames: int = 25, fps: int = 7) -> str:
        """Convert static image to short video clip"""
        if self.pipe is None:
            self.load_models()
            
        # Load and prepare image
        image = Image.open(image_path).convert('RGB')
        image = image.resize((1024, 576))  # SVD optimal resolution
        
        # Generate video frames
        print(f"Animating {image_path}...")
        frames = self.pipe(
            image, 
            num_frames=num_frames,
            decode_chunk_size=8,  # Process in chunks to save memory
            motion_bucket_id=127,  # Control motion amount (0-255)
            noise_aug_strength=0.02
        ).frames[0]
        
        # Save as video using moviepy
        from moviepy.editor import ImageSequenceClip
        clip = ImageSequenceClip([np.array(f) for f in frames], fps=fps)
        clip.write_videofile(output_path, codec='libx264', audio=False)
        
        return output_path


class MusicGenerator:
    """Generates background music based on scene analysis"""
    
    def __init__(self, model_name: str = "facebook/musicgen-small"):
        print(f"Loading music generator: {model_name}")
        from transformers import AutoProcessor, MusicgenForConditionalGeneration
        
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = MusicgenForConditionalGeneration.from_pretrained(model_name)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        
    def generate_soundtrack(self, scene_analyses: List[Dict], 
                          duration: float, output_path: str) -> str:
        """Generate music based on overall scene mood"""
        # Aggregate moods
        moods = [s['mood'] for s in scene_analyses]
        dominant_mood = max(set(moods), key=moods.count)
        
        # Create prompt
        mood_prompts = {
            'happy': "upbeat cheerful acoustic guitar melody",
            'calm': "peaceful ambient piano with soft strings",
            'energetic': "upbeat electronic music with drums",
            'dramatic': "cinematic orchestral music",
            'neutral': "light background instrumental music"
        }
        
        prompt = mood_prompts.get(dominant_mood, mood_prompts['neutral'])
        print(f"Generating {dominant_mood} soundtrack: '{prompt}'")
        
        inputs = self.processor(
            text=[prompt],
            padding=True,
            return_tensors="pt",
        ).to(self.device)
        
        # Generate audio
        max_tokens = int(duration * 50)  # 50 tokens per second
        audio_values = self.model.generate(**inputs, max_new_tokens=max_tokens)
        
        # Save audio
        sampling_rate = self.model.config.audio_encoder.sampling_rate
        torchaudio.save(output_path, audio_values[0].cpu(), sampling_rate)
        
        return output_path


class VideoCreator:
    """Main orchestrator for video creation"""
    
    def __init__(self, args):
        self.args = args
        self.scene_analyzer = SceneAnalyzer()
        self.animator = ImageAnimator() if args.animate_images else None
        self.music_gen = MusicGenerator() if args.generate_music else None
        
    def process_media(self, input_dir: str) -> Tuple[List[MediaItem], List[Dict]]:
        """Process all media files and sort by timestamp"""
        print(f"Processing media from {input_dir}...")
        
        # Collect all media files
        media_items = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.mp4', '*.avi', '*.mov']:
            media_items.extend([MediaItem(str(p)) for p in Path(input_dir).rglob(ext)])
        
        # Sort by timestamp
        media_items.sort(key=lambda x: x.timestamp)
        print(f"Found {len(media_items)} media files")
        
        # Analyze scenes
        scene_analyses = []
        for item in media_items:
            if item.type == 'image':
                analysis = self.scene_analyzer.analyze(str(item.path))
                scene_analyses.append(analysis)
                print(f"  {item.path.name}: {analysis['caption']} (mood: {analysis['mood']})")
        
        return media_items, scene_analyses
    
    def create_video(self, media_items: List[MediaItem], 
                    scene_analyses: List[Dict], output_path: str):
        """Assemble final video with transitions and audio"""
        print("\nCreating video...")
        
        clips = []
        temp_dir = Path("temp_clips")
        temp_dir.mkdir(exist_ok=True)
        
        for idx, item in enumerate(media_items):
            if item.type == 'image':
                # Option 1: Animate images
                if self.args.animate_images:
                    temp_video = temp_dir / f"animated_{idx}.mp4"
                    self.animator.animate_image(str(item.path), str(temp_video))
                    clip = VideoFileClip(str(temp_video))
                else:
                    # Option 2: Static image with duration
                    clip = ImageClip(str(item.path)).set_duration(self.args.image_duration)
            else:
                clip = VideoFileClip(str(item.path))
            
            # Resize to consistent dimensions
            clip = clip.resize(height=1080)
            clips.append(clip)
        
        # Concatenate all clips
        final_video = concatenate_videoclips(clips, method="compose")
        
        # Add music if requested
        if self.args.generate_music:
            music_path = temp_dir / "soundtrack.wav"
            self.music_gen.generate_soundtrack(
                scene_analyses, 
                final_video.duration, 
                str(music_path)
            )
            audio = AudioFileClip(str(music_path))
            final_video = final_video.set_audio(audio)
        
        # Export final video
        print(f"Exporting to {output_path}...")
        final_video.write_videofile(
            output_path,
            codec='libx264',
            audio_codec='aac',
            fps=30,
            preset='medium',
            bitrate='8000k'
        )
        
        print(f"✓ Video created successfully: {output_path}")
        
        # Cleanup
        for clip in clips:
            clip.close()


def main():
    parser = argparse.ArgumentParser(description='Create YouTube videos from images and clips')
    parser.add_argument('input_dir', help='Directory containing images and videos')
    parser.add_argument('--output', '-o', default='output.mp4', help='Output video path')
    parser.add_argument('--animate-images', action='store_true', 
                       help='Animate still images using Stable Video Diffusion')
    parser.add_argument('--image-duration', type=float, default=3.0,
                       help='Duration for static images (seconds)')
    parser.add_argument('--generate-music', action='store_true',
                       help='Generate background music based on scenes')
    parser.add_argument('--no-cuda', action='store_true',
                       help='Disable CUDA even if available')
    
    args = parser.parse_args()
    
    if args.no_cuda:
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
    
    # Create video
    creator = VideoCreator(args)
    media_items, scene_analyses = creator.process_media(args.input_dir)
    creator.create_video(media_items, scene_analyses, args.output)


if __name__ == "__main__":
    main()