# AI Video Creator

An intelligent video creation pipeline that automatically generates YouTube-ready videos from your photos and video clips using AI. The system analyzes scenes, sequences content chronologically, generates background music, and composes everything into a polished final video.

## 🎬 Features

- **Automatic Scene Analysis**: Uses BLIP-2 vision AI to understand image content, mood, and themes
- **Intelligent Sequencing**: Chronologically orders media based on EXIF metadata with smart scene grouping
- **AI-Generated Soundtrack**: Creates custom background music using Meta's MusicGen model
- **Professional Transitions**: Applies fade, dissolve, and Ken Burns effects
- **Automatic Orientation Fixing**: Corrects rotated/upside-down images based on EXIF data
- **HEIC Support**: Automatically converts Apple HEIC images to JPEG
- **Caching System**: Speeds up re-processing by caching analysis results

## 📋 Requirements

### System Requirements
- **GPU**: NVIDIA GPU with 8GB+ VRAM recommended (RTX 4080 or better)
- **RAM**: 16GB+ system RAM
- **Storage**: ~20GB for AI models + space for media and output

### Python Version
- Python 3.8 or higher

## 🚀 Installation

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/ai-video-creator.git
cd ai-video-creator
```

### 2. Install Dependencies
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install transformers accelerate
pip install pillow pillow-heif
pip install moviepy
pip install opencv-python
pip install scipy numpy
pip install piexif exifread
pip install tqdm
```

### 3. Create Required Directories
```bash
mkdir -p input output cache models
```

### 4. Download AI Models (First Time Only)
```bash
python model_downloader.py
# Follow the interactive prompts to download the vision model
# Recommended: blip2-flan-t5-xl (~10GB)
```

## 📁 Project Structure

```
ai-video-creator/
├── input/              # Place your photos and videos here
├── output/             # Generated videos and intermediate files
├── cache/              # Cached analysis results
├── models/             # Downloaded AI models
├── config.py           # Configuration settings
├── process_all_media.py    # Main pipeline script
└── [other Python modules]
```

## 🎯 Quick Start

### Basic Usage (Full Pipeline)
```bash
# 1. Add your photos/videos to the input/ folder
# 2. Run the complete pipeline:
python process_all_media.py
```

This will:
1. Ingest all media files and extract metadata
2. Analyze scenes with AI (~3-5 seconds per image)
3. Create an intelligent sequence
4. Generate background music
5. Compose the final video

### Output
- **Final Video**: `output/final_video.mp4`
- **Intermediate Files**: 
  - `output/media_manifest.json` - Media inventory
  - `output/scene_analysis.json` - AI scene analysis
  - `output/video_sequence.json` - Sequencing data
  - `output/soundtrack.wav` - Generated music

## 📝 Command-Line Scripts

### Main Pipeline Script

#### `process_all_media.py`
Main script for creating videos with multiple options.

**Flags:**
- (none) - Full pipeline with audio
- `--no-audio` - Skip audio generation (faster, silent video)
- `--skip-analysis` - Use existing analysis files, only compose video

**Examples:**
```bash
# Full pipeline with audio
python process_all_media.py

# Fast mode without audio generation
python process_all_media.py --no-audio

# Re-compose video with different settings (if you've already run analysis)
python process_all_media.py --skip-analysis

# Fast re-compose without audio
python process_all_media.py --skip-analysis --no-audio
```

**Estimated Times:**
- Scene Analysis: ~3-5 seconds per image
- Audio Generation: ~2-5 minutes for a 3-minute video
- Video Composition: ~1 second per 10 seconds of output

---

### Image Orientation Tools

#### `check_and_fix_images.py`
Standalone tool to detect and fix orientation issues in images before processing.

**Flags:**
- (none) - Check only (no modifications)
- `--fix` or `-f` - Fix images (overwrites files)
- `--help` or `-h` - Show help message
- `[folder]` - Specify folder (default: "input")

**Examples:**
```bash
# Check images for orientation issues
python check_and_fix_images.py

# Check specific folder
python check_and_fix_images.py path/to/folder

# Fix images in-place
python check_and_fix_images.py --fix

# Fix images in specific folder
python check_and_fix_images.py path/to/folder --fix
```

#### `image_orientation.py`
Advanced image orientation tool with batch processing capabilities.

**Commands:**
- `check <folder>` - Check all images in folder
- `fix <folder>` - Fix images (creates corrected copies)
- `fix <folder> --in-place` - Fix images (overwrites originals)
- `fix-image <image_file>` - Fix single image
- `fix-image <image_file> --in-place` - Fix single image in-place

**Examples:**
```bash
# Check all images in input folder
python image_orientation.py check input

# Fix all images (creates corrected/ subfolder)
python image_orientation.py fix input

# Fix all images in-place (overwrites originals)
python image_orientation.py fix input --in-place

# Fix single image
python image_orientation.py fix-image input/photo.jpg

# Fix single image in-place
python image_orientation.py fix-image input/photo.jpg --in-place
```

---

### Diagnostic Tools

#### `diagnose_issue.py`
Diagnostic script to troubleshoot processing issues.

**Flags:** None (runs automatically)

**Usage:**
```bash
python diagnose_issue.py
```

**Output:**
- Counts files in input folder
- Checks manifest status
- Verifies scene analysis
- Identifies missing or incomplete processing

#### `check_problem_images.py`
Checks specific problematic images that failed to process.

**Flags:** None

**Usage:**
```bash
python check_problem_images.py
```

---

### Model Management

#### `model_downloader.py`
Interactive tool for downloading and managing AI models.

**Flags:** None (interactive menu)

**Usage:**
```bash
python model_downloader.py
```

**Menu Options:**
1. List available models
2. Download recommended model (blip2-flan-t5-xl)
3. Download specific model
4. Check disk usage
5. Delete model
6. Exit

**Available Models:**
- `blip2-opt-2.7b` - Fast, 6-8GB VRAM (~5.4GB download)
- `blip2-opt-6.7b` - Better quality, 12-14GB VRAM (~13GB download)
- `blip2-flan-t5-xl` - Best quality, 10-12GB VRAM (~10GB download) ⭐ Recommended

---

### Legacy/Testing Scripts

#### `run_all_steps.py`
Legacy master script that runs all pipeline steps sequentially.

**Flags:** None

**Usage:**
```bash
python run_all_steps.py
```

**Note:** Use `process_all_media.py` instead for better control and options.

---

## ⚙️ Configuration

Edit `config.py` to customize:

### Paths
```python
INPUT_DIR = Path("input")           # Source media folder
OUTPUT_DIR = Path("output")         # Output folder
CACHE_DIR = Path("cache")           # Cache folder
MODELS_DIR = Path("models")         # AI models folder
```

### Video Output Settings
```python
OUTPUT_RESOLUTION = (1920, 1080)    # 1080p, 720p: (1280, 720)
OUTPUT_FPS = 30                     # Frames per second
OUTPUT_BITRATE = "8000k"            # Video quality
```

### Audio Settings
```python
AUDIO_SAMPLE_RATE = 44100           # Audio quality (Hz)
```

### Image Processing
```python
MAX_IMAGE_SIZE = (1920, 1080)       # Max dimensions before resize
```

### Performance
```python
DEVICE = "cuda"                     # "cuda" for GPU, "cpu" for CPU only
USE_FP16 = True                     # Half-precision (faster, less VRAM)
NUM_CORES = 8                       # CPU cores for rendering
```

## 🎨 Supported File Formats

### Images
- JPEG/JPG
- PNG
- BMP
- TIFF/TIF
- HEIC/HEIF (automatically converted)

### Videos
- MP4
- MOV
- AVI
- MKV
- WMV
- FLV
- WEBM
- M4V

## 📊 Processing Pipeline

```
1. MEDIA INGESTION
   ├─ Scan input folder for media files
   ├─ Convert HEIC images to JPEG
   ├─ Extract EXIF metadata (dates, GPS, camera info)
   └─ Sort chronologically

2. SCENE ANALYSIS (AI)
   ├─ Load BLIP-2 vision model
   ├─ Generate captions for each image
   ├─ Analyze: mood, colors, scene type, objects, lighting
   └─ Cache results for future runs

3. CONTENT SEQUENCING
   ├─ Group similar scenes together
   ├─ Calculate optimal display duration
   ├─ Select appropriate transitions
   └─ Create timeline

4. AUDIO GENERATION (AI)
   ├─ Load MusicGen model
   ├─ Analyze overall mood from scenes
   ├─ Generate custom soundtrack
   └─ Match duration to video

5. VIDEO COMPOSITION
   ├─ Load and resize images/videos
   ├─ Apply Ken Burns effect to images
   ├─ Add transitions (fade, dissolve)
   ├─ Sync audio track
   └─ Render final MP4
```

## 🔧 Troubleshooting

### "No media files found"
- Ensure files are in the `input/` folder
- Check file extensions are supported
- Verify files aren't corrupted

### "Out of memory" / CUDA errors
- Reduce `OUTPUT_RESOLUTION` in config.py
- Set `USE_FP16 = True` in config.py
- Use smaller model (blip2-opt-2.7b)
- Close other GPU-intensive applications

### Images appear rotated/upside-down
```bash
# Run orientation fixer before processing
python check_and_fix_images.py --fix
```

### Only some images processed
```bash
# Run diagnostic to identify issues
python diagnose_issue.py

# Then re-run full pipeline
python process_all_media.py
```

### Audio generation takes too long
```bash
# Skip audio generation
python process_all_media.py --no-audio

# Or use faster model in audio_generator.py:
# Change: AudioGenerator(model_key='musicgen-small')
```

### Processing seems stuck
- Press Ctrl+C to cancel
- Delete cache files: `rm -rf cache/*`
- Re-run with fresh start

## 📈 Performance Tips

### For Faster Processing
1. **Use GPU**: Ensure CUDA is properly installed
2. **Enable FP16**: Set `USE_FP16 = True` in config.py
3. **Skip Audio**: Use `--no-audio` flag during testing
4. **Cache Results**: Don't delete cache folder between runs
5. **Resize Large Images**: Set lower `MAX_IMAGE_SIZE` in config.py

### For Better Quality
1. **Use Larger Model**: blip2-flan-t5-xl for scene analysis
2. **Higher Bitrate**: Increase `OUTPUT_BITRATE` in config.py
3. **Higher FPS**: Set `OUTPUT_FPS = 60` for smoother motion
4. **Longer Durations**: Adjust durations in sequencing_engine.py

## 🎓 Advanced Usage

### Custom Scene Duration
Edit `sequencing_engine.py`:
```python
DEFAULT_IMAGE_DURATION = 4.0  # seconds per image
MIN_IMAGE_DURATION = 2.0
MAX_IMAGE_DURATION = 8.0
```

### Custom Music Prompts
Edit `audio_generator.py` `_create_music_prompt()` method to change music style.

### Custom Transitions
Edit `sequencing_engine.py` `_select_transition()` to adjust transition logic.

### Batch Processing Multiple Folders
```bash
for folder in folder1 folder2 folder3; do
    cp -r $folder/* input/
    python process_all_media.py
    mv output/final_video.mp4 output/${folder}_video.mp4
    rm input/*
done
```

## 🐛 Known Issues

1. **HEIC Images**: Some HEIC conversions may lose quality. Convert to JPEG beforehand if possible.
2. **Very Long Videos**: Videos >20 minutes may take significant time to render.
3. **Mixed Orientations**: Portrait and landscape images are letterboxed/pillarboxed (black bars).

## 📄 License

This project uses open-source models:
- BLIP-2: Salesforce (BSD-3-Clause License)
- MusicGen: Meta AI (CC-BY-NC 4.0)
- MoviePy: MIT License

## 🤝 Contributing

Contributions welcome! Areas for improvement:
- Additional transition effects
- More music style options
- Video stabilization
- Face detection for better cropping
- Cloud processing support

## 📧 Support

For issues and questions:
1. Check this README
2. Run diagnostic tools
3. Check existing GitHub issues
4. Create new issue with error logs

## 🎉 Example Workflows

### Quick Test (5 images)
```bash
# Place 5 images in input/
python process_all_media.py --no-audio  # ~2-3 minutes
```

### Full Family Video (100 images)
```bash
# Place all images in input/
python process_all_media.py  # ~20-30 minutes
# Output: 6-8 minute video with music
```

### Re-render with Different Music
```bash
# First run completed, want different audio
python process_all_media.py --skip-analysis
# Only regenerates audio and re-composes video (~5-10 minutes)
```

### Professional Quality (Wedding, etc.)
```bash
# Edit config.py:
# OUTPUT_RESOLUTION = (3840, 2160)  # 4K
# OUTPUT_FPS = 60
# OUTPUT_BITRATE = "20000k"

python process_all_media.py
# Warning: Much longer processing time
```

---

**Created with ❤️ using AI** | Ready for YouTube! 🎬