# ============================================================================
# AI Video Creator - Configuration
# ============================================================================

import torch
from pathlib import Path

class Config:
    """Application configuration"""
    
    # System Configuration
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_CORES = 20  # Intel i7-14700
    RAM_GB = 128
    VRAM_GB = 16  # RTX 4080
    
    # Model Configuration (optimized for your hardware)
    USE_FP16 = True  # Half precision for faster inference
    MAX_BATCH_SIZE = 8  # Adjust based on model and VRAM usage
    
    # Video Output Settings
    OUTPUT_RESOLUTION = (1920, 1080)  # 1080p
    OUTPUT_FPS = 30
    OUTPUT_CODEC = "libx264"
    OUTPUT_AUDIO_CODEC = "aac"
    OUTPUT_BITRATE = "10M"
    
    # Processing Settings
    MAX_IMAGE_SIZE = (1920, 1080)  # Resize large images
    VIDEO_CLIP_SAMPLE_FPS = 1  # Sample 1 frame per second for analysis
    
    # Cache Settings
    CACHE_DIR = Path("cache")
    ENABLE_CACHE = True
    
    # Paths
    INPUT_DIR = Path("input")
    OUTPUT_DIR = Path("output")
    MODELS_DIR = Path("models")
    TEMP_DIR = Path("temp")
    
    # Audio Settings
    AUDIO_SAMPLE_RATE = 44100
    AUDIO_CHANNELS = 2
    
    # Model Preferences (for Step 3+)
    VISION_MODEL = "Salesforce/blip2-opt-2.7b"  # Start with smaller model
    MUSIC_MODEL = "facebook/musicgen-small"     # 300M parameters
    
    @classmethod
    def create_directories(cls):
        """Create necessary directories"""
        for dir_path in [cls.CACHE_DIR, cls.INPUT_DIR, cls.OUTPUT_DIR, 
                        cls.MODELS_DIR, cls.TEMP_DIR]:
            dir_path.mkdir(exist_ok=True)
    
    @classmethod
    def print_info(cls):
        """Print configuration info"""
        print(f"Device: {cls.DEVICE}")
        if cls.DEVICE == "cuda":
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"CUDA Version: {torch.version.cuda}")
            print(f"VRAM Available: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        print(f"CPU Cores: {cls.NUM_CORES}")
        print(f"RAM: {cls.RAM_GB} GB")