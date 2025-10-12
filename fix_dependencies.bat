@echo off
REM ============================================================================
REM Fix NumPy Dependency Conflicts
REM Downgrades NumPy to compatible version for all packages
REM ============================================================================

echo ====================================================================
echo FIXING NUMPY DEPENDENCY CONFLICTS
echo ====================================================================
echo.
echo The animation installation upgraded NumPy to 2.2.6, but some of your
echo existing packages require NumPy 1.x
echo.
echo This script will:
echo   1. Downgrade NumPy to 1.26.4 (compatible with all packages)
echo   2. Verify all dependencies are satisfied
echo.
echo ====================================================================
echo.

pause

echo Downgrading NumPy to compatible version...
pip install "numpy>=1.24.0,<2.0.0"

echo.
echo ====================================================================
echo Verifying installation...
echo ====================================================================
echo.

python -c "import numpy; print(f'NumPy version: {numpy.__version__}')"
python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
python -c "import transformers; print(f'Transformers version: {transformers.__version__}')"
python -c "try:
    from diffusers import StableVideoDiffusionPipeline
    print('✅ Stable Video Diffusion: OK')
except Exception as e:
    print(f'❌ SVD Error: {e}')
"

echo.
echo ====================================================================
echo DEPENDENCY CHECK COMPLETE
echo ====================================================================
echo.
echo If all checks passed, you're ready to use animation mode!
echo.
echo Test with: python create_animated_movie.py --max-images 3
echo.
pause