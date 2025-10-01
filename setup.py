# ============================================================================
# AI Video Creator - Step 1: Environment Setup
# Windows 11 | Python 3.13 | NVIDIA RTX 4080
# ============================================================================

"""
SETUP INSTRUCTIONS:

1. Open PowerShell as Administrator and run:
   Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

2. Create project directory:
   mkdir C:\AIVideoCreator
   cd C:\AIVideoCreator

3. Create virtual environment:
   python -m venv venv

4. Activate virtual environment:
   .\venv\Scripts\Activate.ps1

5. Save this file as setup.py and run:
   python setup.py

6. After setup, run verification:
   python verify_setup.py
"""

import subprocess
import sys
import platform
import os
from pathlib import Path

class EnvironmentSetup:
    """Complete environment setup for AI Video Creator"""
    
    def __init__(self):
        self.project_root = Path.cwd()
        self.venv_path = self.project_root / "venv"
        self.requirements_file = self.project_root / "requirements.txt"
        self.config_file = self.project_root / "config.py"
        
    def verify_system(self):
        """Verify system requirements"""
        print("=" * 70)
        print("VERIFYING SYSTEM REQUIREMENTS")
        print("=" * 70)
        
        # Check Python version
        py_version = sys.version_info
        print(f"✓ Python Version: {py_version.major}.{py_version.minor}.{py_version.micro}")
        if py_version.major != 3 or py_version.minor != 13:
            print("⚠ Warning: Optimized for Python 3.13")
        
        # Check OS
        print(f"✓ Operating System: {platform.system()} {platform.release()}")
        if platform.system() != "Windows":
            print("⚠ Warning: Scripts optimized for Windows 11")
        
        # Check if in virtual environment
        in_venv = sys.prefix != sys.base_prefix
        print(f"✓ Virtual Environment: {'Active' if in_venv else 'Not Active'}")
        if not in_venv:
            print("⚠ Warning: Please activate virtual environment first")
            
        print()
        
    def create_requirements_txt(self):
        """Generate requirements.txt with all dependencies"""
        print("Creating requirements.txt...")
        
        requirements = """# ============================================================================
# AI Video Creator - Core Dependencies
# Optimized for Windows 11 | Python 3.13 | NVIDIA RTX 4080 (16GB VRAM)
# ============================================================================

# Deep Learning & AI Frameworks
# PyTorch with CUDA 12.1 support for RTX 4080
--index-url https://download.pytorch.org/whl/cu121
torch==2.1.2+cu121
torchvision==0.16.2+cu121
torchaudio==2.1.2+cu121

# Transformers & AI Models
transformers==4.36.2
accelerate==0.25.0
sentencepiece==0.1.99
protobuf==4.25.1
safetensors==0.4.1

# Vision & Image Processing
pillow==10.2.0
opencv-python==4.9.0.80
opencv-contrib-python==4.9.0.80
imageio==2.33.1
imageio-ffmpeg==0.4.9
scikit-image==0.22.0

# Video Processing & Editing
moviepy==1.0.3
ffmpeg-python==0.2.0

# Audio Processing & Generation
librosa==0.10.1
soundfile==0.12.1
audioread==3.0.1
pydub==0.25.1

# Metadata Extraction
piexif==1.1.3
ExifRead==3.0.0

# Data Processing & Utilities
numpy==1.26.3
pandas==2.1.4
scipy==1.11.4

# Progress Bars & UI
tqdm==4.66.1
rich==13.7.0

# Configuration & Settings
python-dotenv==1.0.0
pyyaml==6.0.1

# File Handling
pathlib2==2.3.7.post1

# HTTP & API (for future YouTube upload)
requests==2.31.0
google-auth==2.26.2
google-auth-oauthlib==1.2.0
google-auth-httplib2==0.2.0
google-api-python-client==2.111.0

# Testing & Development
pytest==7.4.3
pytest-cov==4.1.0

# Optional: Advanced features
einops==0.7.0
timm==0.9.12
"""
        
        self.requirements_file.write_text(requirements)
        print(f"✓ Created {self.requirements_file}")
        print()
        
    def create_config(self):
        """Create configuration file"""
        print("Creating config.py...")
        
        config = """# ============================================================================
# AI Video Creator - Configuration
# ============================================================================

import torch
from pathlib import Path

class Config:
    \"\"\"Application configuration\"\"\"
    
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
        \"\"\"Create necessary directories\"\"\"
        for dir_path in [cls.CACHE_DIR, cls.INPUT_DIR, cls.OUTPUT_DIR, 
                        cls.MODELS_DIR, cls.TEMP_DIR]:
            dir_path.mkdir(exist_ok=True)
    
    @classmethod
    def print_info(cls):
        \"\"\"Print configuration info\"\"\"
        print(f"Device: {cls.DEVICE}")
        if cls.DEVICE == "cuda":
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"CUDA Version: {torch.version.cuda}")
            print(f"VRAM Available: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        print(f"CPU Cores: {cls.NUM_CORES}")
        print(f"RAM: {cls.RAM_GB} GB")
"""
        
        self.config_file.write_text(config)
        print(f"✓ Created {self.config_file}")
        print()
        
    def create_verification_script(self):
        """Create verification script"""
        print("Creating verify_setup.py...")
        
        verify_script = """# ============================================================================
# Environment Verification Script
# ============================================================================

import sys
import subprocess
from pathlib import Path

def check_package(package_name, import_name=None):
    \"\"\"Check if package is installed and importable\"\"\"
    if import_name is None:
        import_name = package_name
    
    try:
        __import__(import_name)
        print(f"✓ {package_name}")
        return True
    except ImportError:
        print(f"✗ {package_name} - NOT INSTALLED")
        return False

def verify_pytorch_cuda():
    \"\"\"Verify PyTorch CUDA setup\"\"\"
    try:
        import torch
        print(f"\\n{'='*70}")
        print("PYTORCH & CUDA VERIFICATION")
        print('='*70)
        print(f"PyTorch Version: {torch.__version__}")
        print(f"CUDA Available: {torch.cuda.is_available()}")
        
        if torch.cuda.is_available():
            print(f"CUDA Version: {torch.version.cuda}")
            print(f"cuDNN Version: {torch.backends.cudnn.version()}")
            print(f"GPU Device: {torch.cuda.get_device_name(0)}")
            print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
            
            # Test tensor operation on GPU
            x = torch.rand(5, 3).cuda()
            print(f"✓ GPU tensor operation successful")
            return True
        else:
            print("⚠ CUDA not available - will use CPU (slower)")
            return False
            
    except Exception as e:
        print(f"✗ PyTorch verification failed: {e}")
        return False

def verify_ffmpeg():
    \"\"\"Verify FFmpeg installation\"\"\"
    print(f"\\n{'='*70}")
    print("FFMPEG VERIFICATION")
    print('='*70)
    
    try:
        result = subprocess.run(['ffmpeg', '-version'], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            version_line = result.stdout.split('\\n')[0]
            print(f"✓ {version_line}")
            return True
        else:
            print("✗ FFmpeg not found")
            return False
    except FileNotFoundError:
        print("✗ FFmpeg not found in PATH")
        print("  Install from: https://github.com/BtbN/FFmpeg-Builds/releases")
        return False

def main():
    print('='*70)
    print("AI VIDEO CREATOR - SETUP VERIFICATION")
    print('='*70)
    
    print(f"\\nPython Version: {sys.version}")
    print(f"Python Executable: {sys.executable}")
    
    print(f"\\n{'='*70}")
    print("CHECKING CORE PACKAGES")
    print('='*70)
    
    packages = [
        ("torch", "torch"),
        ("torchvision", "torchvision"),
        ("transformers", "transformers"),
        ("pillow", "PIL"),
        ("opencv-python", "cv2"),
        ("moviepy", "moviepy.editor"),
        ("numpy", "numpy"),
        ("pandas", "pandas"),
        ("librosa", "librosa"),
        ("soundfile", "soundfile"),
        ("piexif", "piexif"),
        ("tqdm", "tqdm"),
        ("requests", "requests"),
    ]
    
    results = []
    for pkg_name, import_name in packages:
        results.append(check_package(pkg_name, import_name))
    
    # Verify PyTorch CUDA
    cuda_ok = verify_pytorch_cuda()
    
    # Verify FFmpeg
    ffmpeg_ok = verify_ffmpeg()
    
    # Load config
    print(f"\\n{'='*70}")
    print("CONFIGURATION")
    print('='*70)
    
    try:
        from config import Config
        Config.print_info()
        Config.create_directories()
        print("\\n✓ Directories created")
    except Exception as e:
        print(f"✗ Config error: {e}")
    
    # Summary
    print(f"\\n{'='*70}")
    print("SUMMARY")
    print('='*70)
    
    total = len(results)
    passed = sum(results)
    
    print(f"Packages: {passed}/{total} installed")
    print(f"CUDA: {'✓' if cuda_ok else '✗'}")
    print(f"FFmpeg: {'✓' if ffmpeg_ok else '✗'}")
    
    if passed == total and cuda_ok and ffmpeg_ok:
        print("\\n🎉 SETUP COMPLETE! Ready to build AI Video Creator.")
    else:
        print("\\n⚠ Some components missing. Please check errors above.")
        
if __name__ == "__main__":
    main()
"""
        
        verify_file = self.project_root / "verify_setup.py"
        verify_file.write_text(verify_script)
        print(f"✓ Created {verify_file}")
        print()
        
    def create_install_script(self):
        """Create Windows batch file for easy installation"""
        print("Creating install.bat...")
        
        batch_script = """@echo off
REM ============================================================================
REM AI Video Creator - Automated Installation Script
REM Windows 11 | Python 3.13 | NVIDIA RTX 4080
REM ============================================================================

echo ====================================================================
echo AI VIDEO CREATOR - INSTALLATION
echo ====================================================================
echo.

REM Check if virtual environment exists
if not exist "venv\\Scripts\\activate.bat" (
    echo Creating virtual environment...
    python -m venv venv
    echo.
)

REM Activate virtual environment
echo Activating virtual environment...
call venv\\Scripts\\activate.bat
echo.

REM Upgrade pip
echo Upgrading pip...
python -m pip install --upgrade pip
echo.

REM Install requirements
echo Installing packages (this may take 10-15 minutes)...
echo.
pip install -r requirements.txt
echo.

REM Create directories
echo Creating project directories...
python -c "from config import Config; Config.create_directories()"
echo.

REM Verify installation
echo.
echo ====================================================================
echo VERIFYING INSTALLATION
echo ====================================================================
echo.
python verify_setup.py

echo.
echo ====================================================================
echo INSTALLATION COMPLETE
echo ====================================================================
echo.
echo To activate environment in future: venv\\Scripts\\activate
echo.
pause
"""
        
        batch_file = self.project_root / "install.bat"
        batch_file.write_text(batch_script)
        print(f"✓ Created {batch_file}")
        print()
        
    def create_readme(self):
        """Create README with instructions"""
        print("Creating README.md...")
        
        readme = """# AI Video Creator - Environment Setup

## System Specifications
- **OS**: Windows 11 Home Edition
- **RAM**: 128 GB
- **CPU**: Intel i7-14700, 3400 MHz, 20 Cores
- **GPU**: NVIDIA GeForce RTX 4080 (16GB VRAM)
- **Storage**: SSD
- **Python**: 3.13

## Quick Start

### 1. Install Prerequisites

#### Python 3.13
Download and install from: https://www.python.org/downloads/

#### FFmpeg
1. Download from: https://github.com/BtbN/FFmpeg-Builds/releases
2. Extract to `C:\\ffmpeg`
3. Add `C:\\ffmpeg\\bin` to System PATH

#### CUDA Toolkit (for GPU acceleration)
Download CUDA 12.1 from: https://developer.nvidia.com/cuda-downloads

### 2. Setup Project
```powershell
# Create project directory
mkdir C:\\AIVideoCreator
cd C:\\AIVideoCreator

# Create virtual environment
python -m venv venv

# Activate virtual environment
.\\venv\\Scripts\\Activate.ps1

# Run setup
python setup.py

# Run automated installation
install.bat