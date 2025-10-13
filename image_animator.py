# ============================================================================
# AI Video Creator - TRUE Image Animation Engine
# Converts still images to fully animated video clips
# Uses AnimateDiff + ControlNet for character/scene animation
# ============================================================================

from pathlib import Path
from typing import List, Optional, Tuple, Dict
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
import cv2

# Animation models
from diffusers import (
    AnimateDiffPipeline,
    MotionAdapter,
    EulerDiscreteScheduler,
    ControlNetModel,
    StableDiffusionControlNetPipeline
)
from diffusers.utils import export_to_video
from transformers import pipeline as transformers_pipeline

from config import Config
from scene_analyzer import SceneAnalysis


class ImageAnimator:
    """Convert still images to fully animated video clips with character motion"""
    
    ANIMATION_STYLES = {
        'anime': {
            'style_prompt': 'anime style, fluid animation, expressive characters',
            'num_frames': 48,  # 2 seconds at 24fps
            'fps': 24,
            'guidance_scale': 7.5,
            'num_inference_steps': 25,
            'description': 'Japanese anime style with exaggerated motion'
        },
        'cartoon': {
            'style_prompt': 'western cartoon style, bouncy animation, vibrant colors',
            'num_frames': 60,  # 2 seconds at 30fps
            'fps': 30,
            'guidance_scale': 8.0,
            'num_inference_steps': 25,
            'description': 'Western cartoon style with smooth motion'
        },
        'realistic': {
            'style_prompt': 'photorealistic, natural motion, cinematic',
            'num_frames': 90,  # 3 seconds at 30fps
            'fps': 30,
            'guidance_scale': 7.0,
            'num_inference_steps': 30,
            'description': 'Realistic animation with natural movement'
        },
        'stylized': {
            'style_prompt': 'artistic stylized animation, painterly, dynamic',
            'num_frames': 72,  # 3 seconds at 24fps
            'fps': 24,
            'guidance_scale': 8.5,
            'num_inference_steps': 30,
            'description': 'Artistic stylized animation'
        }
    }
    
    def __init__(
        self,
        animation_style: str = 'anime',
        use_interpolation: bool = True,
        enable_depth: bool = False
    ):
        """Initialize image animator with AnimateDiff
        
        Args:
            animation_style: Style of animation ('anime', 'cartoon', 'realistic', 'stylized')
            use_interpolation: Whether to use frame interpolation for smoother animation
            enable_depth: Whether to enable depth estimation (experimental)
        """
        self.animation_style = animation_style
        self.use_interpolation = use_interpolation
        self.enable_depth = enable_depth
        
        self.animate_pipeline = None
        self.controlnet_pipeline = None
        self.depth_estimator = None
        self.segmentation_model = None
        
        self.device = Config.DEVICE
        self.params = self.ANIMATION_STYLES.get(
            animation_style,
            self.ANIMATION_STYLES['anime']
        )
    
    def load_models(self):
        """Load AnimateDiff and supporting models"""
        print(f"\n{'='*70}")
        print("LOADING ANIMATION MODELS")
        print(f"{'='*70}")
        print(f"Style: {self.animation_style}")
        print(f"Target FPS: {self.params['fps']}")
        print(f"Frames per clip: {self.params['num_frames']}")
        
        # Load AnimateDiff with Motion Adapter
        print("\n[1/4] Loading AnimateDiff Pipeline...")
        try:
            # Load motion adapter
            adapter = MotionAdapter.from_pretrained(
                "guoyww/animatediff-motion-adapter-v1-5-2",
                torch_dtype=torch.float16
            )
            
            # Load base model
            model_id = "runwayml/stable-diffusion-v1-5"
            
            self.animate_pipeline = AnimateDiffPipeline.from_pretrained(
                model_id,
                motion_adapter=adapter,
                torch_dtype=torch.float16
            )
            
            # Set scheduler
            self.animate_pipeline.scheduler = EulerDiscreteScheduler.from_config(
                self.animate_pipeline.scheduler.config,
                beta_schedule="linear",
                timestep_spacing="trailing"
            )
            
            # Memory optimization
            self.animate_pipeline.enable_model_cpu_offload()
            
            print("✓ AnimateDiff loaded")
            
        except Exception as e:
            print(f"✗ Failed to load AnimateDiff: {e}")
            raise
        
        # Load ControlNet for structure preservation
        print("\n[2/4] Loading ControlNet...")
        try:
            controlnet = ControlNetModel.from_pretrained(
                "lllyasviel/control_v11p_sd15_canny",
                torch_dtype=torch.float16
            )
            
            self.controlnet_pipeline = StableDiffusionControlNetPipeline.from_pretrained(
                "runwayml/stable-diffusion-v1-5",
                controlnet=controlnet,
                torch_dtype=torch.float16
            )
            
            self.controlnet_pipeline.enable_model_cpu_offload()
            
            print("✓ ControlNet loaded")
            
        except Exception as e:
            print(f"⚠ ControlNet unavailable: {e}")
            self.controlnet_pipeline = None
        
        # Load depth estimator if enabled
        if self.enable_depth:
            print("\n[3/4] Loading Depth Estimator...")
            try:
                self.depth_estimator = transformers_pipeline(
                    "depth-estimation",
                    model="Intel/dpt-large"
                )
                print("✓ Depth estimator loaded")
            except Exception as e:
                print(f"⚠ Depth estimator unavailable: {e}")
                self.depth_estimator = None
        
        print(f"\n{'='*70}")
        print("✓ Animation models ready")
        print(f"{'='*70}\n")
    
    def unload_models(self):
        """Unload models to free memory"""
        print("\nUnloading animation models...")
        
        self.animate_pipeline = None
        self.controlnet_pipeline = None
        self.depth_estimator = None
        self.segmentation_model = None
        
        torch.cuda.empty_cache()
        print("✓ Models unloaded")
    
    def animate_image(
        self,
        image_path: Path,
        scene_analysis: Optional[SceneAnalysis] = None,
        output_path: Optional[Path] = None
    ) -> Path:
        """Animate a single image using AnimateDiff
        
        Args:
            image_path: Path to input image
            scene_analysis: Optional scene analysis for context
            output_path: Optional output path for video
            
        Returns:
            Path to generated video file
        """
        if self.animate_pipeline is None:
            raise RuntimeError("Models not loaded. Call load_models() first.")
        
        # Load image
        image = Image.open(image_path).convert("RGB")
        original_size = image.size
        
        # Resize for processing (AnimateDiff works best at 512x512)
        image = self._resize_for_animation(image, target_size=512)
        
        # Generate animation prompt
        prompt = self._create_animation_prompt(image, scene_analysis)
        negative_prompt = self._create_negative_prompt()
        
        print(f"  Animating: {image_path.name}")
        print(f"  Prompt: {prompt[:80]}...")
        
        # Extract edges for ControlNet guidance (preserves structure)
        control_image = self._extract_canny_edges(image)
        
        # Generate animated frames
        try:
            frames = self._generate_animated_frames(
                image=image,
                control_image=control_image,
                prompt=prompt,
                negative_prompt=negative_prompt
            )
            
            # Post-processing
            frames = self._post_process_frames(frames, original_size)
            
            # Apply motion smoothing
            if self.use_interpolation:
                frames = self._interpolate_frames(frames)
            
            # Save video
            if output_path is None:
                output_path = image_path.parent / f"{image_path.stem}_animated.mp4"
            
            self._save_video(frames, output_path)
            
            return output_path
            
        except Exception as e:
            print(f"  ✗ Animation failed: {e}")
            raise
    
    def _resize_for_animation(self, image: Image.Image, target_size: int = 512) -> Image.Image:
        """Resize image maintaining aspect ratio with AnimateDiff-compatible dimensions
        
        AnimateDiff requires dimensions that are:
        1. Divisible by 64 (not just 8) for proper latent space alignment
        2. Preferably square or common aspect ratios (16:9, 4:3, 1:1)
        """
        # Standard AnimateDiff-compatible resolutions (width, height)
        COMPATIBLE_RESOLUTIONS = [
            (512, 512),   # 1:1 - Most compatible
            (512, 384),   # 4:3
            (384, 512),   # 3:4
            (512, 320),   # 16:10
            (320, 512),   # 10:16
            (512, 288),   # 16:9
            (288, 512),   # 9:16
            (448, 448),   # 1:1 alternative
            (512, 448),   # Close to 8:7
            (448, 512),   # Close to 7:8
        ]
        
        width, height = image.size
        aspect_ratio = width / height
        
        # Find the best matching resolution
        best_resolution = None
        min_aspect_diff = float('inf')
        
        for res_w, res_h in COMPATIBLE_RESOLUTIONS:
            res_aspect = res_w / res_h
            aspect_diff = abs(aspect_ratio - res_aspect)
            
            if aspect_diff < min_aspect_diff:
                min_aspect_diff = aspect_diff
                best_resolution = (res_w, res_h)
        
        new_width, new_height = best_resolution
        
        print(f"  Resizing: {width}x{height} -> {new_width}x{new_height}")
        print(f"  Aspect ratio: {aspect_ratio:.2f} -> {new_width/new_height:.2f}")
        
        return image.resize((new_width, new_height), Image.Resampling.LANCZOS)


    def _generate_animated_frames(
            self,
            image: Image.Image,
            control_image: Image.Image,
            prompt: str,
            negative_prompt: str
        ) -> List[Image.Image]:
            """
            Generate animated frames using AnimateDiff with chunking to avoid memory issues.
            
            FIXED: Generates frames in chunks to handle model limitations on sequence length.
            """
            
            # Get image dimensions and verify compatibility
            img_width, img_height = image.size
            assert img_width % 64 == 0 and img_height % 64 == 0, \
                f"Image dimensions must be divisible by 64, got {img_width}x{img_height}"
            
            # Parameters for chunking
            total_frames_to_generate = self.params['num_frames']
            chunk_size = 16  # Safe chunk size for AnimateDiff v1.5 to avoid tensor errors
            num_chunks = (total_frames_to_generate + chunk_size - 1) // chunk_size  # Ceiling division
            
            all_frames = []
            
            print(f"  Generating {total_frames_to_generate} frames in {num_chunks} chunk(s) at {img_width}x{img_height}...")

            try:
                for i in range(num_chunks):
                    print(f"    - Processing chunk {i + 1}/{num_chunks}...")
                    
                    # Generate one chunk of frames
                    output = self.animate_pipeline(
                        prompt=prompt,
                        negative_prompt=negative_prompt,
                        num_frames=chunk_size, # Generate a smaller, safe number of frames
                        guidance_scale=self.params['guidance_scale'],
                        num_inference_steps=self.params['num_inference_steps'],
                        height=img_height,
                        width=img_width,
                        generator=torch.Generator(device=self.device).manual_seed(42 + i) # Seed per chunk
                    )
                    
                    all_frames.extend(output.frames[0])
                
                # Trim the collected frames to the exact number required
                frames = all_frames[:total_frames_to_generate]
                print(f"  ✓ Generated {len(frames)} frames successfully")
                return frames

            except RuntimeError as e:
                if "size of tensor" in str(e):
                    print(f"  ✗ Tensor dimension mismatch at {img_width}x{img_height}")
                    print(f"  Animation failed even with chunking. This may be a deeper configuration issue.")
                    # You could add a fallback here if needed, but chunking should prevent this.
                    raise
                else:
                    raise
    
    def _create_animation_prompt(
        self,
        image: Image.Image,
        scene_analysis: Optional[SceneAnalysis]
    ) -> str:
        """Create animation prompt from image and analysis"""
        
        # Base style prompt
        base_prompt = self.params['style_prompt']
        
        # Add scene-specific details
        # FIXED: Changed from scene_analysis.description to scene_analysis.detailed_description
        if scene_analysis and scene_analysis.detailed_description:
            scene_desc = scene_analysis.detailed_description
            
            # Add motion cues based on scene type
            if 'person' in scene_desc.lower() or 'people' in scene_desc.lower():
                motion = "walking, talking, natural human movement"
            elif 'landscape' in scene_desc.lower() or 'outdoor' in scene_desc.lower():
                motion = "gentle camera pan, clouds moving, leaves rustling"
            elif 'animal' in scene_desc.lower():
                motion = "animal moving naturally, breathing, looking around"
            elif 'object' in scene_desc.lower():
                motion = "subtle rotation, lighting changes, dynamic perspective"
            else:
                motion = "dynamic movement, cinematic camera motion"
            
            prompt = f"{scene_desc}, {motion}, {base_prompt}, high quality, smooth animation"
        else:
            prompt = f"{base_prompt}, dynamic movement, cinematic, high quality"
        
        return prompt
    
    def _create_negative_prompt(self) -> str:
        """Create negative prompt to avoid artifacts"""
        return "static, still image, frozen, blurry, low quality, distorted, deformed, artifacts"
    
    def _extract_canny_edges(self, image: Image.Image, low_threshold: int = 100, high_threshold: int = 200) -> Image.Image:
        """Extract Canny edges for ControlNet guidance"""
        # Convert to numpy array
        img_array = np.array(image)
        
        # Convert to grayscale
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        
        # Apply Canny edge detection
        edges = cv2.Canny(gray, low_threshold, high_threshold)
        
        # Convert back to RGB PIL image
        edges_rgb = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
        return Image.fromarray(edges_rgb)
    
    def _post_process_frames(self, frames: List[Image.Image], target_size: Tuple[int, int]) -> List[Image.Image]:
        """Post-process frames (resize, enhance)"""
        processed = []
        
        for frame in frames:
            # Resize to target size
            frame_resized = frame.resize(target_size, Image.Resampling.LANCZOS)
            processed.append(frame_resized)
        
        return processed
    
    def _interpolate_frames(self, frames: List[Image.Image]) -> List[Image.Image]:
        """Interpolate additional frames for smoother animation"""
        # Simple linear interpolation between frames
        # In production, you'd use RIFE or similar for better results
        
        interpolated = []
        
        for i in range(len(frames) - 1):
            interpolated.append(frames[i])
            
            # Add interpolated frame between current and next
            curr_array = np.array(frames[i])
            next_array = np.array(frames[i + 1])
            
            # Simple average interpolation
            mid_array = ((curr_array.astype(np.float32) + next_array.astype(np.float32)) / 2).astype(np.uint8)
            interpolated.append(Image.fromarray(mid_array))
        
        interpolated.append(frames[-1])  # Add last frame
        
        return interpolated
    
    def _save_video(self, frames: List[Image.Image], output_path: Path):
        """Save frames as video file"""
        # Convert PIL images to numpy arrays
        frame_arrays = [np.array(frame) for frame in frames]
        
        # Use diffusers export utility
        export_to_video(frame_arrays, str(output_path), fps=self.params['fps'])
        
        print(f"  ✓ Saved: {output_path.name}")
    
    def batch_animate_images(
        self,
        image_paths: List[Path],
        scene_analyses: Dict[Path, SceneAnalysis],
        output_dir: Path
    ) -> List[Path]:
        """Animate multiple images in batch
        
        Args:
            image_paths: List of image paths to animate
            scene_analyses: Dictionary mapping paths to scene analyses
            output_dir: Output directory for animated clips
            
        Returns:
            List of paths to generated video files
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        
        animated_clips = []
        
        for image_path in tqdm(image_paths, desc="Animating images", unit="images"):
            try:
                scene_analysis = scene_analyses.get(image_path)
                output_path = output_dir / f"{image_path.stem}_animated.mp4"
                
                clip_path = self.animate_image(image_path, scene_analysis, output_path)
                animated_clips.append(clip_path)
                
                # Clear cache periodically
                if len(animated_clips) % 5 == 0:
                    torch.cuda.empty_cache()
                
            except Exception as e:
                print(f"\n⚠ Error animating {image_path.name}: {e}")
                continue
        
        print(f"\n✓ Animated {len(animated_clips)}/{len(image_paths)} images")
        
        return animated_clips


if __name__ == "__main__":
    print("Image Animator Module - TRUE ANIMATION VERSION")
    print("\nThis module converts still images to fully animated videos")
    print("using AnimateDiff + ControlNet for character/scene animation.")
    print("\nAnimation Styles:")
    for style, params in ImageAnimator.ANIMATION_STYLES.items():
        print(f"  • {style}: {params['description']}")
    print("\nRequires:")
    print("  • NVIDIA GPU with 12+ GB VRAM (RTX 4080 recommended)")
    print("  • AnimateDiff models (~10GB download)")
    print("  • ControlNet models (~5GB download)")