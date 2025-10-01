# ============================================================================
# AI Video Creator - Configuration Loader
# ============================================================================

"""
Configuration Loader
Reads and parses Config.ini file, providing validated settings to the application
"""

import configparser
from pathlib import Path
from typing import List, Tuple, Optional
import torch


class Config:
    """Configuration management for AI Video Creator"""
    
    def __init__(self, config_file: str = "Config.ini"):
        """
        Initialize configuration from INI file
        
        Args:
            config_file: Path to configuration file
        """
        self.config_file = Path(config_file)
        
        if not self.config_file.exists():
            raise FileNotFoundError(
                f"Configuration file not found: {config_file}\n"
                f"Please create Config.ini from the template."
            )
        
        # Parse configuration file
        self.parser = configparser.ConfigParser()
        self.parser.read(self.config_file)
        
        # Load and validate all sections
        self._load_paths()
        self._load_hardware()
        self._load_media_ingestion()
        self._load_scene_analysis()
        self._load_sequencing()
        self._load_audio_generation()
        self._load_video_output()
        self._load_advanced()
        self._load_youtube()
        
        # Apply YouTube optimizations if enabled
        if self.youtube_optimized:
            self._apply_youtube_optimizations()
    
    def _get_bool(self, section: str, option: str, fallback: bool = False) -> bool:
        """Get boolean value from config"""
        return self.parser.getboolean(section, option, fallback=fallback)
    
    def _get_int(self, section: str, option: str, fallback: int = 0) -> int:
        """Get integer value from config"""
        return self.parser.getint(section, option, fallback=fallback)
    
    def _get_float(self, section: str, option: str, fallback: float = 0.0) -> float:
        """Get float value from config"""
        return self.parser.getfloat(section, option, fallback=fallback)
    
    def _get_str(self, section: str, option: str, fallback: str = "") -> str:
        """Get string value from config"""
        return self.parser.get(section, option, fallback=fallback)
    
    def _get_list(self, section: str, option: str, fallback: List[str] = None) -> List[str]:
        """Get comma-separated list from config"""
        value = self._get_str(section, option)
        if not value and fallback:
            return fallback
        return [item.strip() for item in value.split(',') if item.strip()]
    
    # ========================================================================
    # SECTION 1: PATHS
    # ========================================================================
    
    def _load_paths(self):
        """Load path configurations"""
        self.INPUT_DIR = Path(self._get_str('Paths', 'input_directory', 'input'))
        self.OUTPUT_DIR = Path(self._get_str('Paths', 'output_directory', 'output'))
        self.CACHE_DIR = Path(self._get_str('Paths', 'cache_directory', 'cache'))
        self.MODELS_DIR = Path(self._get_str('Paths', 'models_directory', 'models'))
        
        # Create directories if they don't exist
        self.INPUT_DIR.mkdir(parents=True, exist_ok=True)
        self.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        
        # Only create cache and models if not disabled
        if str(self.CACHE_DIR):
            self.CACHE_DIR.mkdir(parents=True, exist_ok=True)
        if str(self.MODELS_DIR):
            self.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    
    # ========================================================================
    # SECTION 2: HARDWARE
    # ========================================================================
    
    def _load_hardware(self):
        """Load hardware configurations"""
        device_str = self._get_str('Hardware', 'device', 'cuda').lower()
        
        # Validate and set device
        if device_str == 'cuda' and torch.cuda.is_available():
            self.DEVICE = 'cuda'
        elif device_str == 'mps' and torch.backends.mps.is_available():
            self.DEVICE = 'mps'
        else:
            self.DEVICE = 'cpu'
            if device_str != 'cpu':
                print(f"⚠️  Requested device '{device_str}' not available, using CPU")
        
        self.VRAM_GB = self._get_int('Hardware', 'vram_gb', 16)
        self.USE_FP16 = self._get_bool('Hardware', 'use_fp16', True)
        
        num_cores = self._get_int('Hardware', 'num_cores', 4)
        if num_cores == -1:
            import multiprocessing
            self.NUM_CORES = multiprocessing.cpu_count()
        else:
            self.NUM_CORES = num_cores
    
    # ========================================================================
    # SECTION 3: MEDIA INGESTION
    # ========================================================================
    
    def _load_media_ingestion(self):
        """Load media ingestion configurations"""
        self.ENABLE_METADATA_CACHE = self._get_bool('MediaIngestion', 'enable_metadata_cache', True)
        
        # Parse supported formats
        image_formats = self._get_list(
            'MediaIngestion', 
            'supported_image_formats',
            ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.heic', '.heif']
        )
        self.SUPPORTED_IMAGE_FORMATS = set(fmt.lower() for fmt in image_formats)
        
        video_formats = self._get_list(
            'MediaIngestion',
            'supported_video_formats',
            ['.mp4', '.mov', '.avi', '.mkv', '.wmv', '.flv', '.webm', '.m4v']
        )
        self.SUPPORTED_VIDEO_FORMATS = set(fmt.lower() for fmt in video_formats)
        
        # Image dimensions
        width = self._get_int('MediaIngestion', 'max_image_width', 1920)
        height = self._get_int('MediaIngestion', 'max_image_height', 1080)
        self.MAX_IMAGE_SIZE = (width, height)
    
    # ========================================================================
    # SECTION 4: SCENE ANALYSIS
    # ========================================================================
    
    def _load_scene_analysis(self):
        """Load scene analysis configurations"""
        self.VISION_MODEL = self._get_str('SceneAnalysis', 'vision_model', 'blip2-flan-t5-xl')
        self.ENABLE_SCENE_CACHE = self._get_bool('SceneAnalysis', 'enable_scene_cache', True)
        self.VIDEO_CLIP_SAMPLE_FPS = self._get_float('SceneAnalysis', 'video_sample_fps', 1.0)
        self.MAX_VIDEO_KEYFRAMES = self._get_int('SceneAnalysis', 'max_video_keyframes', 20)
        
        # Validate model selection
        valid_models = ['blip2-opt-2.7b', 'blip2-opt-6.7b', 'blip2-flan-t5-xl']
        if self.VISION_MODEL not in valid_models:
            print(f"⚠️  Invalid vision model '{self.VISION_MODEL}', using 'blip2-flan-t5-xl'")
            self.VISION_MODEL = 'blip2-flan-t5-xl'
    
    # ========================================================================
    # SECTION 5: SEQUENCING
    # ========================================================================
    
    def _load_sequencing(self):
        """Load sequencing configurations"""
        self.SORT_CHRONOLOGICALLY = self._get_bool('Sequencing', 'sort_chronologically', True)
        self.DEFAULT_IMAGE_DURATION = self._get_float('Sequencing', 'default_image_duration', 4.0)
        self.MIN_IMAGE_DURATION = self._get_float('Sequencing', 'min_image_duration', 2.0)
        self.MAX_IMAGE_DURATION = self._get_float('Sequencing', 'max_image_duration', 8.0)
        self.DEFAULT_TRANSITION_DURATION = self._get_float('Sequencing', 'default_transition_duration', 1.0)
        self.SCENE_GROUPING_THRESHOLD = self._get_float('Sequencing', 'scene_grouping_threshold', 0.3)
        self.ENABLE_KEN_BURNS = self._get_bool('Sequencing', 'enable_ken_burns', True)
        self.KEN_BURNS_ZOOM_FACTOR = self._get_float('Sequencing', 'ken_burns_zoom_factor', 1.15)
        
        # Validate ranges
        self.SCENE_GROUPING_THRESHOLD = max(0.0, min(1.0, self.SCENE_GROUPING_THRESHOLD))
        self.KEN_BURNS_ZOOM_FACTOR = max(1.0, min(2.0, self.KEN_BURNS_ZOOM_FACTOR))
    
    # ========================================================================
    # SECTION 6: AUDIO GENERATION
    # ========================================================================
    
    def _load_audio_generation(self):
        """Load audio generation configurations"""
        self.MUSIC_MODEL = self._get_str('AudioGeneration', 'music_model', 'musicgen-medium')
        self.ENABLE_AUDIO = self._get_bool('AudioGeneration', 'enable_audio', True)
        self.AUDIO_SAMPLE_RATE = self._get_int('AudioGeneration', 'audio_sample_rate', 44100)
        self.AUDIO_CHANNELS = self._get_int('AudioGeneration', 'audio_channels', 2)
        self.AUDIO_BITRATE = self._get_str('AudioGeneration', 'audio_bitrate', '192k')
        self.MUSIC_GUIDANCE_SCALE = self._get_float('AudioGeneration', 'music_guidance_scale', 3.0)
        self.DEFAULT_MUSIC_MOOD = self._get_str('AudioGeneration', 'default_music_mood', 'happy')
        self.MUSIC_STYLE = self._get_str('AudioGeneration', 'music_style', 'acoustic')
        
        # Validate model selection
        valid_music_models = ['musicgen-small', 'musicgen-medium', 'musicgen-large']
        if self.MUSIC_MODEL not in valid_music_models:
            print(f"⚠️  Invalid music model '{self.MUSIC_MODEL}', using 'musicgen-medium'")
            self.MUSIC_MODEL = 'musicgen-medium'
        
        # Validate audio channels
        if self.AUDIO_CHANNELS not in [1, 2]:
            print(f"⚠️  Invalid audio channels '{self.AUDIO_CHANNELS}', using 2 (stereo)")
            self.AUDIO_CHANNELS = 2
    
    # ========================================================================
    # SECTION 7: VIDEO OUTPUT
    # ========================================================================
    
    def _load_video_output(self):
        """Load video output configurations"""
        width = self._get_int('VideoOutput', 'output_width', 1920)
        height = self._get_int('VideoOutput', 'output_height', 1080)
        self.OUTPUT_RESOLUTION = (width, height)
        
        self.OUTPUT_FPS = self._get_int('VideoOutput', 'output_fps', 30)
        self.OUTPUT_BITRATE = self._get_str('VideoOutput', 'output_bitrate', '8000k')
        self.VIDEO_CODEC = self._get_str('VideoOutput', 'video_codec', 'libx264')
        self.ENCODING_PRESET = self._get_str('VideoOutput', 'encoding_preset', 'medium')
        self.MAINTAIN_ASPECT_RATIO = self._get_bool('VideoOutput', 'maintain_aspect_ratio', True)
        self.APPLY_VIDEO_FADES = self._get_bool('VideoOutput', 'apply_video_fades', True)
        self.FADE_DURATION = self._get_float('VideoOutput', 'fade_duration', 1.0)
        
        # Validate codec
        if self.VIDEO_CODEC not in ['libx264', 'libx265']:
            print(f"⚠️  Invalid codec '{self.VIDEO_CODEC}', using 'libx264'")
            self.VIDEO_CODEC = 'libx264'
        
        # Validate preset
        valid_presets = ['ultrafast', 'superfast', 'veryfast', 'faster', 'fast', 
                        'medium', 'slow', 'slower', 'veryslow']
        if self.ENCODING_PRESET not in valid_presets:
            print(f"⚠️  Invalid preset '{self.ENCODING_PRESET}', using 'medium'")
            self.ENCODING_PRESET = 'medium'
    
    # ========================================================================
    # SECTION 8: ADVANCED
    # ========================================================================
    
    def _load_advanced(self):
        """Load advanced configurations"""
        self.VERBOSE_LOGGING = self._get_bool('Advanced', 'verbose_logging', False)
        self.MAX_ITEMS_TO_PROCESS = self._get_int('Advanced', 'max_items_to_process', 0)
        self.EXPORT_INTERMEDIATE_FILES = self._get_bool('Advanced', 'export_intermediate_files', True)
        self.CLEANUP_TEMP_FILES = self._get_bool('Advanced', 'cleanup_temp_files', False)
        self.PARALLEL_ANALYSIS = self._get_bool('Advanced', 'parallel_analysis', False)
        self.ANALYSIS_BATCH_SIZE = self._get_int('Advanced', 'analysis_batch_size', 4)
    
    # ========================================================================
    # SECTION 9: YOUTUBE
    # ========================================================================
    
    def _load_youtube(self):
        """Load YouTube configurations"""
        self.youtube_optimized = self._get_bool('YouTube', 'youtube_optimized', True)
        self.VIDEO_CATEGORY = self._get_str('YouTube', 'video_category', 'family')
        self.TARGET_VIDEO_LENGTH = self._get_int('YouTube', 'target_video_length', 0)
        self.AUDIO_NORMALIZATION = self._get_bool('YouTube', 'audio_normalization', True)
        self.EMBED_METADATA = self._get_bool('YouTube', 'embed_metadata', True)
    
    def _apply_youtube_optimizations(self):
        """Apply YouTube-recommended settings"""
        # Force 1080p for YouTube
        if self.OUTPUT_RESOLUTION not in [(1920, 1080), (2560, 1440), (3840, 2160)]:
            print("ℹ️  YouTube optimization: Setting resolution to 1920x1080")
            self.OUTPUT_RESOLUTION = (1920, 1080)
        
        # Force 30 FPS for standard content
        if self.OUTPUT_FPS not in [24, 30, 60]:
            print("ℹ️  YouTube optimization: Setting FPS to 30")
            self.OUTPUT_FPS = 30
        
        # Ensure good bitrate for 1080p
        if self.OUTPUT_RESOLUTION == (1920, 1080):
            bitrate_value = int(self.OUTPUT_BITRATE.replace('k', ''))
            if bitrate_value < 8000:
                print("ℹ️  YouTube optimization: Setting bitrate to 8000k for 1080p")
                self.OUTPUT_BITRATE = '8000k'
        
        # Use H.264 for maximum compatibility
        if self.VIDEO_CODEC != 'libx264':
            print("ℹ️  YouTube optimization: Using H.264 codec")
            self.VIDEO_CODEC = 'libx264'
    
    # ========================================================================
    # UTILITY METHODS
    # ========================================================================
    
    def print_config(self):
        """Print current configuration"""
        print("\n" + "="*70)
        print("CURRENT CONFIGURATION")
        print("="*70)
        
        print("\n[Paths]")
        print(f"  Input: {self.INPUT_DIR}")
        print(f"  Output: {self.OUTPUT_DIR}")
        print(f"  Cache: {self.CACHE_DIR}")
        print(f"  Models: {self.MODELS_DIR}")
        
        print("\n[Hardware]")
        print(f"  Device: {self.DEVICE}")
        print(f"  VRAM: {self.VRAM_GB} GB")
        print(f"  FP16: {self.USE_FP16}")
        print(f"  CPU Cores: {self.NUM_CORES}")
        
        print("\n[Scene Analysis]")
        print(f"  Vision Model: {self.VISION_MODEL}")
        print(f"  Cache Enabled: {self.ENABLE_SCENE_CACHE}")
        
        print("\n[Sequencing]")
        print(f"  Chronological Sort: {self.SORT_CHRONOLOGICALLY}")
        print(f"  Image Duration: {self.DEFAULT_IMAGE_DURATION}s")
        print(f"  Ken Burns: {self.ENABLE_KEN_BURNS}")
        
        print("\n[Audio]")
        print(f"  Enabled: {self.ENABLE_AUDIO}")
        print(f"  Music Model: {self.MUSIC_MODEL}")
        print(f"  Sample Rate: {self.AUDIO_SAMPLE_RATE} Hz")
        print(f"  Channels: {self.AUDIO_CHANNELS}")
        
        print("\n[Video Output]")
        print(f"  Resolution: {self.OUTPUT_RESOLUTION[0]}x{self.OUTPUT_RESOLUTION[1]}")
        print(f"  FPS: {self.OUTPUT_FPS}")
        print(f"  Bitrate: {self.OUTPUT_BITRATE}")
        print(f"  Codec: {self.VIDEO_CODEC}")
        print(f"  Preset: {self.ENCODING_PRESET}")
        
        print("\n[YouTube]")
        print(f"  Optimized: {self.youtube_optimized}")
        print(f"  Category: {self.VIDEO_CATEGORY}")
        
        print("="*70 + "\n")
    
    def save_template(self, output_file: str = "Config.template.ini"):
        """Save a template configuration file with all options"""
        # This would create a template file
        # Implementation omitted for brevity
        pass
    
    def validate(self) -> bool:
        """
        Validate configuration settings
        
        Returns:
            True if valid, False otherwise
        """
        errors = []
        
        # Check paths exist
        if not self.INPUT_DIR.exists():
            errors.append(f"Input directory does not exist: {self.INPUT_DIR}")
        
        # Check image duration ranges
        if self.MIN_IMAGE_DURATION > self.MAX_IMAGE_DURATION:
            errors.append(f"MIN_IMAGE_DURATION ({self.MIN_IMAGE_DURATION}) > MAX_IMAGE_DURATION ({self.MAX_IMAGE_DURATION})")
        
        if self.DEFAULT_IMAGE_DURATION < self.MIN_IMAGE_DURATION or self.DEFAULT_IMAGE_DURATION > self.MAX_IMAGE_DURATION:
            errors.append(f"DEFAULT_IMAGE_DURATION ({self.DEFAULT_IMAGE_DURATION}) outside valid range")
        
        # Check VRAM requirements
        if self.DEVICE == 'cuda':
            if self.VISION_MODEL == 'blip2-flan-t5-xl' and self.VRAM_GB < 10:
                errors.append(f"Vision model requires ~10GB VRAM, but only {self.VRAM_GB}GB configured")
            elif self.VISION_MODEL == 'blip2-opt-6.7b' and self.VRAM_GB < 12:
                errors.append(f"Vision model requires ~12GB VRAM, but only {self.VRAM_GB}GB configured")
        
        # Print errors
        if errors:
            print("\n⚠️  CONFIGURATION ERRORS:")
            for error in errors:
                print(f"  • {error}")
            print()
            return False
        
        return True


# Singleton instance
_config_instance = None

def get_config(config_file: str = "Config.ini") -> Config:
    """
    Get global configuration instance
    
    Args:
        config_file: Path to configuration file
        
    Returns:
        Config instance
    """
    global _config_instance
    
    if _config_instance is None:
        _config_instance = Config(config_file)
    
    return _config_instance


if __name__ == "__main__":
    # Test configuration loading
    try:
        config = Config("Config.ini")
        config.print_config()
        
        if config.validate():
            print("✅ Configuration is valid")
        else:
            print("❌ Configuration has errors")
            
    except FileNotFoundError as e:
        print(f"❌ {e}")
        print("\nPlease create Config.ini from the provided template.")
    except Exception as e:
        print(f"❌ Error loading configuration: {e}")
        import traceback
        traceback.print_exc()