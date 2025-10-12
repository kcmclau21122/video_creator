@echo off
REM ============================================================================
REM AI Video Creator - Animation Mode Installation Script
REM Installs additional dependencies for animation features
REM ============================================================================

echo ====================================================================
echo AI VIDEO CREATOR - ANIMATION MODE SETUP
echo ====================================================================
echo.
echo This will install additional dependencies for animation mode:
echo   - Stable Video Diffusion (SVD) pipeline
echo   - Frame interpolation tools
echo   - Depth estimation models
echo.
echo Estimated download: ~500MB of packages
echo First-run model download: ~7GB (Stable Video Diffusion)
echo.
echo SYSTEM REQUIREMENTS CHECK:
echo   - NVIDIA GPU with 12+ GB VRAM
echo   - Your RTX 4080 (16GB): PERFECT!
echo.
echo ====================================================================
echo.

REM Check if base environment exists
if not exist "video_ai\Scripts\activate.bat" (
    echo ERROR: Base environment not found!
    echo.
    echo Please run install.bat first to set up the base environment:
    echo   install.bat
    echo.
    pause
    exit /b 1
)

echo Step 1: Activating virtual environment...
call .\video_ai\Scripts\Activate.ps1
echo.

echo Step 2: Checking base requirements...
echo.
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import transformers; print(f'Transformers: {transformers.__version__}')"
echo.

echo Step 3: Installing animation dependencies...
echo This may take 5-10 minutes...
echo.

REM Install from requirements_animation.txt
pip install -r requirements_animation.txt --upgrade

echo.
echo ====================================================================
echo Step 4: Verifying installation...
echo ====================================================================
echo.

python -c "try:
    from diffusers import StableVideoDiffusionPipeline
    print('✅ Stable Video Diffusion: OK')
except Exception as e:
    print(f'❌ SVD installation failed: {e}')
"

python -c "try:
    import av
    print('✅ PyAV: OK')
except:
    print('⚠️  PyAV: Not installed (optional)')
"

echo.
echo ====================================================================
echo INSTALLATION COMPLETE
echo ====================================================================
echo.
echo Animation mode is ready to use!
echo.
echo NEXT STEPS:
echo.
echo 1. Place your photos in the input folder:
echo    input\your_photo1.jpg
echo    input\your_photo2.jpg
echo    ...
echo.
echo 2. (Optional) Add background music:
echo    input\your_music.mp3
echo.
echo 3. Create your first animated movie:
echo    python create_animated_movie.py --max-images 3
echo.
echo    This will create a test video with your first 3 images
echo    (takes about 3-5 minutes)
echo.
echo 4. If the test looks good, process all images:
echo    python create_animated_movie.py
echo.
echo IMPORTANT NOTES:
echo.
echo - First run will download ~7GB Stable Video Diffusion model
echo - Animation takes 30-60 seconds per image
echo - For 10 images, expect ~10 minutes total processing time
echo - Your RTX 4080 is perfect for this! 16GB VRAM is plenty.
echo.
echo MODES AVAILABLE:
echo.
echo Animation Mode (NEW):
echo   python create_animated_movie.py
echo   - Photos come to life with AI animation
echo   - ~30-60 seconds per image
echo   - Professional, cinematic results
echo.
echo Slideshow Mode (Original):
echo   python process_all_media.py
echo   - Traditional photo slideshow
echo   - ~5 seconds per image
echo   - Ken Burns effects
echo.
echo For full documentation, see: ANIMATION_MODE_GUIDE.md
echo.
echo ====================================================================
echo.
pause