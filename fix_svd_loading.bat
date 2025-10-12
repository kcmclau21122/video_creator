@echo off
REM ============================================================================
REM Fix Stable Video Diffusion Loading Error
REM Installs compatible versions of diffusers and transformers
REM ============================================================================

echo ====================================================================
echo FIXING STABLE VIDEO DIFFUSION LOADING ERROR
echo ====================================================================
echo.
echo The error is caused by version mismatch between diffusers and transformers
echo.
echo This will:
echo   1. Uninstall development version of diffusers
echo   2. Install stable release versions
echo   3. Use compatible transformers version
echo.
echo ====================================================================
echo.

pause

echo Step 1: Uninstalling problematic versions...
pip uninstall diffusers -y
pip uninstall transformers -y

echo.
echo Step 2: Installing compatible stable versions...
echo.

REM Install specific compatible versions
pip install diffusers==0.27.2
pip install transformers==4.40.0
pip install accelerate==0.29.3

echo.
echo Step 3: Verifying installation...
echo.

python -c "import diffusers; print(f'Diffusers: {diffusers.__version__}')"
python -c "import transformers; print(f'Transformers: {transformers.__version__}')"
python -c "try:
    from diffusers import StableVideoDiffusionPipeline
    print('✅ StableVideoDiffusionPipeline: Available')
except Exception as e:
    print(f'❌ Error: {e}')
"

echo.
echo ====================================================================
echo FIX COMPLETE
echo ====================================================================
echo.
echo Now try again: python create_animated_movie.py --max-images 3
echo.
pause