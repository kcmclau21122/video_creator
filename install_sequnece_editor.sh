#!/bin/bash
# Installation script for Video Sequence Editor

echo "=================================================="
echo "  AI Video Creator - Sequence Editor Setup"
echo "=================================================="
echo ""

# Check Python version
echo "[1/4] Checking Python version..."
python_version=$(python3 --version 2>&1 | grep -oP '\d+\.\d+')
required_version="3.8"

if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" != "$required_version" ]; then
    echo "❌ Error: Python 3.8 or higher required (found: $python_version)"
    exit 1
fi
echo "✓ Python $python_version detected"
echo ""

# Check if virtual environment exists
echo "[2/4] Checking virtual environment..."
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi
echo "✓ Virtual environment ready"
echo ""

# Activate virtual environment
echo "[3/4] Activating virtual environment..."
source venv/bin/activate || . venv/Scripts/activate

# Install dependencies
echo "[4/4] Installing dependencies..."
echo ""

# Check which version to install
echo "Which version would you like to install?"
echo "1) Streamlit (Web-based, recommended)"
echo "2) Tkinter (Desktop native)"
echo "3) Both"
read -p "Enter choice [1-3]: " choice

case $choice in
    1)
        echo "Installing Streamlit version..."
        pip install streamlit pillow
        INSTALLED="streamlit"
        ;;
    2)
        echo "Installing Tkinter version..."
        pip install pillow
        INSTALLED="tkinter"
        ;;
    3)
        echo "Installing both versions..."
        pip install streamlit pillow
        INSTALLED="both"
        ;;
    *)
        echo "Invalid choice. Exiting."
        exit 1
        ;;
esac

echo ""
echo "=================================================="
echo "  Installation Complete! ✓"
echo "=================================================="
echo ""

# Show next steps based on installation
if [ "$INSTALLED" = "streamlit" ] || [ "$INSTALLED" = "both" ]; then
    echo "To run Streamlit version:"
    echo "  $ source venv/bin/activate"
    echo "  $ streamlit run sequence_editor_gui.py"
    echo ""
fi

if [ "$INSTALLED" = "tkinter" ] || [ "$INSTALLED" = "both" ]; then
    echo "To run Tkinter version:"
    echo "  $ source venv/bin/activate"
    echo "  $ python sequence_editor_tkinter.py"
    echo ""
fi

echo "First time setup:"
echo "  1. Add your media files to input/ folder"
echo "  2. Run: python process_all_media.py"
echo "  3. Then use the sequence editor to modify"
echo ""
echo "For more information, see SEQUENCE_EDITOR_README.md"
echo ""