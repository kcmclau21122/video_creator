@echo off
REM ============================================================================
REM AI Video Creator - Automated Installation Script
REM Windows 11 | Python 3.13 | NVIDIA RTX 4080
REM ============================================================================

echo ====================================================================
echo AI VIDEO CREATOR - INSTALLATION
echo ====================================================================
echo.

REM Check if virtual environment exists
if not exist "video_ai\Scripts\activate.bat" (
    echo Creating virtual environment...
    python -m venv video_ai
    echo.
)

REM Activate virtual environment
echo Activating virtual environment...
call .\video_ai\Scripts\Activate.ps1
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
echo To activate environment in future: venv\Scripts\activate
echo.
pause