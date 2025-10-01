# ============================================================================
# Environment Verification Script
# ============================================================================

import sys
import subprocess
from pathlib import Path

def check_package(package_name, import_name=None):
    """Check if package is installed and importable"""
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
    """Verify PyTorch CUDA setup"""
    try:
        import torch
        print(f"\n{'='*70}")
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
    """Verify FFmpeg installation"""
    print(f"\n{'='*70}")
    print("FFMPEG VERIFICATION")
    print('='*70)
    
    try:
        result = subprocess.run(['ffmpeg', '-version'], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            version_line = result.stdout.split('\n')[0]
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
    
    print(f"\nPython Version: {sys.version}")
    print(f"Python Executable: {sys.executable}")
    
    print(f"\n{'='*70}")
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
    print(f"\n{'='*70}")
    print("CONFIGURATION")
    print('='*70)
    
    try:
        from config import Config
        Config.print_info()
        Config.create_directories()
        print("\n✓ Directories created")
    except Exception as e:
        print(f"✗ Config error: {e}")
    
    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print('='*70)
    
    total = len(results)
    passed = sum(results)
    
    print(f"Packages: {passed}/{total} installed")
    print(f"CUDA: {'✓' if cuda_ok else '✗'}")
    print(f"FFmpeg: {'✓' if ffmpeg_ok else '✗'}")
    
    if passed == total and cuda_ok and ffmpeg_ok:
        print("\n🎉 SETUP COMPLETE! Ready to build AI Video Creator.")
    else:
        print("\n⚠ Some components missing. Please check errors above.")
        
if __name__ == "__main__":
    main()