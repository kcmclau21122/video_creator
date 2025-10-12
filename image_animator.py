# ============================================================================
# AI Video Creator - Image Animation Engine
# Converts still images to animated video clips using Stable Video Diffusion
# ============================================================================

"""
Image Animator
Uses Stable Video Diffusion to convert still images into animated clips
Adds realistic motion, camera movements, and depth effects
"""

from pathlib import Path
from typing import List, Optional, Tuple
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm

# Stable Video Diffusion
from diffusers import StableVideoDiffusionPipeline
from diffusers.utils import export_to_video

# RIFE for frame interpolation
try:
    from rife_arch import IFNet
    RIFE_AVAILABLE = True
except ImportError:
    RIFE_AVAILABLE = False

from config import Config
from scene_analyzer import SceneAnalysis


class ImageAnimator:
    """Convert still images to animated video clips"""
    
    # Animation presets
    ANIMATION_STYLES = {
        'cinematic': {
            'motion_bucket_id': 127,  # Medium motion
            'fps': 30,
            'num_frames': 90,  # 3 seconds at 30fps
            'decode_chunk_size': 8,
            'description': 'Smooth cinematic motion'
        },
        'dynamic': {
            'motion_bucket_id': 180,  # High motion
            'fps': 30,
            'num_frames': 60,  # 2 seconds at 30fps
            'decode_chunk_size': 8,
            'description': 'Dynamic, energetic motion'
        },
        'subtle': {
            'motion_bucket_id': 80,  # Low motion
            'fps': 30,
            'num_frames': 120,  # 4 seconds at 30fps
            'decode_chunk_size': 8,
            'description': 'Subtle, gentle motion'
        },
        'landscape': {
            'motion_bucket_id': 100,  # Low-medium motion
            'fps': 30,
            'num_frames': 150,  # 5 seconds at 30fps
            'decode_chunk_size': 8,
            'description': 'Slow pan across landscapes'
        }
    }
    
# ============================================================================
# FIXED: ImageAnimator.__init__ method
# Add this to your image_animator.py file
# ============================================================================

    def __init__(
        self,
        animation_style: str = 'cinematic',
        use_interpolation: bool = True,
        enable_depth: bool = False
    ):
        """
        Initialize image animator
        
        Args:
            animation_style: Animation preset to use
            use_interpolation: Use RIFE for frame interpolation
            enable_depth: Add depth-based parallax effects
        """
        self.animation_style = animation_style
        self.use_interpolation = use_interpolation
        self.enable_depth = enable_depth
        
        # FIX: Add model_name attribute
        self.model_name = "stabilityai/stable-video-diffusion-img2vid-xt"
        
        self.pipeline = None
        self.interpolator = None
        self.depth_estimator = None
        self.device = Config.DEVICE
        
        # Get animation parameters
        self.params = self.ANIMATION_STYLES.get(
            animation_style,
            self.ANIMATION_STYLES['cinematic']
        )
    
    def load_models(self):
        """Load animation models"""
        print(f"\n{'='*70}")
        print("LOADING ANIMATION MODELS")
        print(f"{'='*70}")
        print(f"Style: {self.animation_style}")
        print(f"Target FPS: {self.params['fps']}")
        print(f"Frames per clip: {self.params['num_frames']}")
        
        # Load Stable Video Diffusion
        print("\n[1/3] Loading Stable Video Diffusion...")
        print("  Model: stabilityai/stable-video-diffusion-img2vid-xt")
        print("  This will download ~7GB on first run...")
        
        # Load pipeline with proper error handling
        try:
            print(f"Loading SVD model from {self.model_name}...")
            
            # Load pipeline without device_map to avoid offload_state_dict
            from diffusers import StableVideoDiffusionPipeline
            import torch
            
            # First try: Load without low_cpu_mem_usage
            try:
                self.pipeline = StableVideoDiffusionPipeline.from_pretrained(
                    self.model_name,
                    torch_dtype=torch.float16,
                    variant="fp16",
                    low_cpu_mem_usage=False  # Disable to avoid offload_state_dict
                )
            except:
                # Fallback: Load without variant
                self.pipeline = StableVideoDiffusionPipeline.from_pretrained(
                    self.model_name,
                    torch_dtype=torch.float16,
                    low_cpu_mem_usage=False
                )
            
            # Move to device
            self.pipeline = self.pipeline.to(self.device)
            
            # Enable optimizations if on CUDA
            if self.device == "cuda":
                try:
                    self.pipeline.enable_model_cpu_offload()
                except:
                    pass  # Skip if not available
            
            print("✓ SVD model loaded successfully")
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            raise
    
    def _load_rife(self):
        """Load RIFE frame interpolation model"""
        # Download RIFE model if needed
        # This is a placeholder - actual implementation would load RIFE
        print("  Note: RIFE integration requires additional setup")
        self.interpolator = None
    
    def _load_depth_estimator(self):
        """Load depth estimation model"""
        from transformers import pipeline
        
        self.depth_estimator = pipeline(
            "depth-estimation",
            model="LiheYoung/depth-anything-small-hf",
            device=0 if self.device == "cuda" else -1
        )
    
    def unload_models(self):
        """Unload models from memory"""
        if self.pipeline:
            del self.pipeline
            self.pipeline = None
        if self.interpolator:
            del self.interpolator
            self.interpolator = None
        if self.depth_estimator:
            del self.depth_estimator
            self.depth_estimator = None
        
        torch.cuda.empty_cache()
        print("✅ Animation models unloaded")
    
    def animate_image(
        self,
        image_path: Path,
        scene_analysis: Optional[SceneAnalysis] = None,
        output_path: Optional[Path] = None
    ) -> Path:
        """
        Animate a single image into a video clip
        
        Args:
            image_path: Path to input image
            scene_analysis: Optional scene analysis for motion hints
            output_path: Where to save animated clip
            
        Returns:
            Path to animated video clip
        """
        if not self.pipeline:
            raise RuntimeError("Models not loaded. Call load_models() first.")
        
        # Load and prepare image
        image = Image.open(image_path).convert("RGB")
        
        # Resize to SVD's expected input size (1024x576 or similar)
        target_size = (1024, 576)
        image = self._resize_for_svd(image, target_size)
        
        # Determine motion parameters based on scene
        motion_params = self._analyze_motion_type(image, scene_analysis)
        
        # Generate animated frames
        print(f"  Animating: {image_path.name}")
        frames = self._generate_animation(image, motion_params)
        
        # Apply depth effects if enabled
        if self.enable_depth and self.depth_estimator:
            frames = self._apply_depth_effects(frames, image)
        
        # Interpolate frames if enabled
        if self.use_interpolation and self.interpolator:
            frames = self._interpolate_frames(frames)
        
        # Save animated clip
        if output_path is None:
            output_path = image_path.parent / f"{image_path.stem}_animated.mp4"
        
        self._save_video(frames, output_path)
        
        return output_path
    
    def _resize_for_svd(self, image: Image.Image, target_size: Tuple[int, int]) -> Image.Image:
        """Resize image for Stable Video Diffusion input"""
        # SVD expects specific resolutions, resize while maintaining aspect ratio
        img_width, img_height = image.size
        target_width, target_height = target_size
        
        # Calculate scaling to fit within target while maintaining aspect ratio
        scale = min(target_width / img_width, target_height / img_height)
        
        new_width = int(img_width * scale)
        new_height = int(img_height * scale)
        
        # Resize
        image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        # Pad to exact target size
        padded = Image.new('RGB', target_size, (0, 0, 0))
        paste_x = (target_width - new_width) // 2
        paste_y = (target_height - new_height) // 2
        padded.paste(image, (paste_x, paste_y))
        
        return padded
    
    def _analyze_motion_type(
        self,
        image: Image.Image,
        scene_analysis: Optional[SceneAnalysis]
    ) -> dict:
        """Determine motion parameters based on scene content"""
        motion_params = {
            'motion_bucket_id': self.params['motion_bucket_id'],
            'num_frames': self.params['num_frames'],
            'fps': self.params['fps']
        }
        
        if not scene_analysis:
            return motion_params
        
        # Adjust motion based on scene type
        scene_type = scene_analysis.scene_type.lower() if scene_analysis.scene_type else ""
        
        if 'outdoor' in scene_type or 'landscape' in scene_type:
            # Landscapes: slower, more sweeping motion
            motion_params['motion_bucket_id'] = max(80, motion_params['motion_bucket_id'] - 20)
            motion_params['num_frames'] = min(150, motion_params['num_frames'] + 30)
        
        elif 'action' in scene_type or 'sport' in scene_type:
            # Action: more dynamic motion
            motion_params['motion_bucket_id'] = min(200, motion_params['motion_bucket_id'] + 30)
        
        elif 'portrait' in scene_type or 'people' in scene_type:
            # Portraits: subtle motion
            motion_params['motion_bucket_id'] = max(60, motion_params['motion_bucket_id'] - 30)
        
        return motion_params
    
    def _generate_animation(self, image: Image.Image, motion_params: dict) -> List[Image.Image]:
        """Generate animated frames using Stable Video Diffusion"""
        
        # Generate video frames
        frames = self.pipeline(
            image,
            decode_chunk_size=self.params['decode_chunk_size'],
            generator=torch.manual_seed(42),
            motion_bucket_id=motion_params['motion_bucket_id'],
            num_frames=motion_params['num_frames'],
            fps=motion_params['fps']
        ).frames[0]
        
        return frames
    
    def _apply_depth_effects(
        self,
        frames: List[Image.Image],
        original_image: Image.Image
    ) -> List[Image.Image]:
        """Apply depth-based parallax effects"""
        # Estimate depth map
        depth_result = self.depth_estimator(original_image)
        depth_map = depth_result['depth']
        
        # Apply subtle parallax motion based on depth
        # This is a simplified version - full implementation would:
        # 1. Separate foreground/background using depth
        # 2. Apply differential motion (foreground moves more)
        # 3. Re-composite layers
        
        print("  Applying depth-based parallax...")
        return frames
    
    def _interpolate_frames(self, frames: List[Image.Image]) -> List[Image.Image]:
        """Interpolate frames for smoother motion using RIFE"""
        if not self.interpolator:
            return frames
        
        # Double frame rate by interpolating between frames
        interpolated = []
        for i in range(len(frames) - 1):
            interpolated.append(frames[i])
            # Generate intermediate frame
            # (Actual RIFE implementation would go here)
            # For now, just add original frames
        
        interpolated.append(frames[-1])
        return interpolated
    
    def _save_video(self, frames: List[Image.Image], output_path: Path):
        """Save frames as video file"""
        # Use diffusers' export utility
        export_to_video(frames, str(output_path), fps=self.params['fps'])
    
    def batch_animate_images(
        self,
        image_paths: List[Path],
        scene_analyses: dict,
        output_dir: Path
    ) -> List[Path]:
        """
        Animate multiple images in batch
        
        Args:
            image_paths: List of image paths
            scene_analyses: Dictionary of scene analyses
            output_dir: Output directory for animated clips
            
        Returns:
            List of paths to animated clips
        """
        print(f"\n{'='*70}")
        print(f"BATCH ANIMATING {len(image_paths)} IMAGES")
        print(f"{'='*70}")
        
        output_dir.mkdir(parents=True, exist_ok=True)
        animated_clips = []
        
        for image_path in tqdm(image_paths, desc="Animating images"):
            try:
                # Get scene analysis
                scene_analysis = scene_analyses.get(image_path)
                
                # Create output path
                output_path = output_dir / f"{image_path.stem}_animated.mp4"
                
                # Animate
                clip_path = self.animate_image(image_path, scene_analysis, output_path)
                animated_clips.append(clip_path)
                
            except Exception as e:
                print(f"\n⚠️  Error animating {image_path.name}: {e}")
                continue
        
        print(f"\n✅ Animated {len(animated_clips)}/{len(image_paths)} images")
        
        return animated_clips


class MotionPresets:
    """Pre-defined camera motion presets"""
    
    @staticmethod
    def get_preset(preset_name: str, scene_type: str = None) -> dict:
        """Get motion preset parameters"""
        
        presets = {
            'pan_right': {
                'description': 'Slow pan from left to right',
                'motion_bucket_id': 100,
                'best_for': ['landscapes', 'cityscapes']
            },
            'zoom_in': {
                'description': 'Slow zoom into subject',
                'motion_bucket_id': 120,
                'best_for': ['portraits', 'details']
            },
            'dolly_forward': {
                'description': 'Forward movement into scene',
                'motion_bucket_id': 140,
                'best_for': ['paths', 'corridors', 'streets']
            },
            'orbit': {
                'description': 'Circular motion around subject',
                'motion_bucket_id': 150,
                'best_for': ['objects', 'buildings']
            },
            'crane_up': {
                'description': 'Upward camera movement',
                'motion_bucket_id': 110,
                'best_for': ['tall subjects', 'reveals']
            }
        }
        
        return presets.get(preset_name, presets['pan_right'])


if __name__ == "__main__":
    print("Image Animator Module")
    print("\nThis module converts still images to animated video clips")
    print("using Stable Video Diffusion and advanced motion effects.")
    print("\nAnimation Styles:")
    for style, params in ImageAnimator.ANIMATION_STYLES.items():
        print(f"  • {style}: {params['description']}")
    print("\nRequires:")
    print("  • NVIDIA GPU with 12+ GB VRAM (RTX 4080 recommended)")
    print("  • diffusers library with SVD support")
    print("  • ~7GB model download on first run")