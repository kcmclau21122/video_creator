# AI Video Creator

An intelligent video creation pipeline that automatically generates YouTube-ready videos from your photos and video clips using AI. The system analyzes scenes, sequences content chronologically, generates or uses background music, and composes everything into a polished final video.

## 🎬 Features

- **Automatic Scene Analysis**: Uses BLIP-2 vision AI to understand image content, mood, and themes
- **Intelligent Sequencing**: Chronologically orders media based on EXIF metadata with smart scene grouping
- **Flexible Audio Options**: 
  - **Use Your Own Music**: Automatically uses MP3/WAV files from input folder
  - **AI-Generated Soundtrack**: Creates custom background music using Meta's MusicGen model
  - **Music Looping**: Automatically loops music to match video length
  - **Fade Control**: Optional fade-out effect at end of video
- **Professional Transitions**: Applies fade, dissolve, and Ken Burns effects
- **Optimized Image Duration**: 6-10 seconds per image (adjustable based on content)
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

### FFmpeg (Required for Audio Processing)
- **Windows**: Download from https://ffmpeg.org/download.html and add to PATH
- **Linux**: `sudo apt-get install ffmpeg`
- **Mac**: `brew install ffmpeg`

## 🚀 Installation

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/ai-video-creator.git
cd ai-video-creator
```

### 2. Install Dependencies
```bash
# Install PyTorch with CUDA support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install other dependencies
pip install transformers accelerate
pip install pillow pillow-heif
pip install moviepy
pip install opencv-python
pip install scipy numpy
pip install piexif exifread
pip install tqdm
pip install pydub
```

### 3. Install FFmpeg (for Audio)
See requirements above for your operating system.

### 4. Create Required Directories
```bash
mkdir -p input output cache models
```

### 5. Download AI Models (First Time Only)
```bash
python model_downloader.py
# Follow the interactive prompts to download the vision model
# Recommended: blip2-flan-t5-xl (~10GB)
```

## 🎯 Quick Start

### Using Your Own Music (Recommended)
```bash
# 1. Add your photos/videos to the input/ folder
# 2. Add your music files (MP3, WAV, etc.) to the input/ folder
# 3. Run the pipeline:
python process_all_media.py

# With fade-out effect:
python process_all_media.py --fade 5
```

### Using AI-Generated Music
```bash
# 1. Add your photos/videos to the input/ folder (no music files)
# 2. Run the pipeline:
python process_all_media.py

# Or force AI generation even if music files exist:
python process_all_media.py --force-generate
```

## 🎵 Audio System

### Automatic Audio Selection
The system automatically detects music files in your input folder:

1. **Music files found**: Uses your music (loops if needed to match video length)
2. **No music files**: Generates AI music based on scene analysis
3. **Force generation**: Use `--force-generate` flag to ignore music files

### Supported Audio Formats
- MP3
- WAV
- M4A
- AAC
- OGG
- FLAC
- WMA

### Music Looping
If your music is shorter than the video, it will automatically loop:
- Files are played in alphabetical order by filename
- Seamless looping between tracks
- Example: `01_intro.mp3`, `02_main.mp3`, `03_outro.mp3` will loop in order

### Fade Control
Control the fade-out effect at the end of your video:
```bash
# No fade (default)
python process_all_media.py

# 3 second fade
python process_all_media.py --fade 3

# 5 second fade
python process_all_media.py --fade 5
```

## 📝 Command Line Options

### Full Command Reference
```bash
python process_all_media.py [OPTIONS]

Options:
  --no-audio           Skip audio entirely (silent video)
  --skip-analysis      Use existing analysis, only compose video
  --fade N             Add N-second fade at end (0 = no fade)
  --force-generate     Generate AI music even if music files exist

Examples:
  python process_all_media.py                    # Auto: use files or generate
  python process_all_media.py --fade 5           # With 5-second fade
  python process_all_media.py --force-generate   # Always generate music
  python process_all_media.py --no-audio         # Silent video
  python process_all_media.py --skip-analysis    # Re-compose only
```

## ⚙️ Configuration

### Image Duration Settings
Edit `sequencing_engine.py`:
```python
DEFAULT_IMAGE_DURATION = 7.0  # Default seconds per image
MIN_IMAGE_DURATION = 6.0      # Minimum duration
MAX_IMAGE_DURATION = 10.0     # Maximum duration
```

### Audio Settings
Edit `config.py`:
```python
AUDIO_SAMPLE_RATE = 44100     # Audio quality (Hz)
```

### Video Output Settings
Edit `config.py`:
```python
OUTPUT_RESOLUTION = (1920, 1080)    # 1080p
OUTPUT_FPS = 30                      # Frames per second
OUTPUT_BITRATE = "8000k"             # Video quality
```

## 📊 Processing Pipeline

```
1. MEDIA INGESTION
   ├─ Scan input folder for media files
   ├─ Detect music files (MP3, WAV, etc.)
   ├─ Convert HEIC images to JPEG
   ├─ Extract EXIF metadata
   └─ Sort chronologically

2. SCENE ANALYSIS (AI)
   ├─ Load BLIP-2 vision model
   ├─ Generate captions for each image
   ├─ Analyze: mood, colors, scene type
   └─ Cache results

3. CONTENT SEQUENCING
   ├─ Group similar scenes together
   ├─ Calculate duration (6-10 seconds per image)
   ├─ Select appropriate transitions
   └─ Create timeline

4. AUDIO PREPARATION
   ├─ Check for music files in input folder
   ├─ IF music files found:
   │  ├─ Load and concatenate files
   │  ├─ Loop to match video duration
   │  └─ Apply fade if specified
   └─ ELSE:
      ├─ Load MusicGen AI model
      ├─ Analyze overall mood from scenes
      └─ Generate custom soundtrack

5. VIDEO COMPOSITION
   ├─ Load and resize images/videos
   ├─ Apply Ken Burns effect to images
   ├─ Add transitions (fade, dissolve)
   ├─ Sync audio track
   └─ Render final MP4
```

## 🎨 Example Workflows

### Family Video with Your Music
```bash
# Setup
input/
  ├─ family_001.jpg
  ├─ family_002.jpg
  ├─ ...
  └─ my_favorite_song.mp3

# Run with fade
python process_all_media.py --fade 5

# Output: 
# - Video with your music
# - 5-second fade at end
# - 6-10 seconds per image
```

### Wedding Video with Multiple Songs
```bash
# Setup (songs play in order, loop if needed)
input/
  ├─ wedding_photos/
  │   ├─ ceremony_01.jpg
  │   ├─ ceremony_02.jpg
  │   └─ ...
  ├─ 01_prelude.mp3
  ├─ 02_ceremony.mp3
  └─ 03_reception.mp3

# Run
python process_all_media.py --fade 10

# Output:
# - Songs play in order: 01, 02, 03, then loop
# - 10-second fade at end
```

### AI-Generated Music
```bash
# No music files in input folder
python process_all_media.py

# Or force AI generation
python process_all_media.py --force-generate --fade 3
```

### Quick Test (No Audio)
```bash
# Fast processing for testing
python process_all_media.py --no-audio
```

## 🔧 Troubleshooting

### "pydub.exceptions.CouldntDecodeError"
**Problem**: FFmpeg not found or not in PATH

**Solution**:
```bash
# Windows: Add ffmpeg.exe to PATH or project folder
# Linux: sudo apt-get install ffmpeg
# Mac: brew install ffmpeg
```

### Music Not Playing
**Problem**: Music files not detected

**Solution**:
- Ensure files are in `input/` folder (not subdirectory)
- Check file extensions: `.mp3`, `.wav`, `.m4a`, etc.
- Verify files aren't corrupted

### Music Too Short
**Problem**: Music shorter than video

**Solution**: This is automatic! The system will:
- Loop your music files seamlessly
- Play files in alphabetical order
- Match exact video duration

### No Fade Effect
**Problem**: Fade not working

**Solution**:
- Use `--fade N` flag where N > 0
- Example: `python process_all_media.py --fade 5`

## 📈 Performance Tips

### For Faster Processing
1. Use existing music files (faster than AI generation)
2. Skip audio during testing: `--no-audio`
3. Enable caching (don't delete cache folder)
4. Use GPU acceleration

### For Better Audio Quality
1. Use high-quality source music (320kbps MP3 or WAV)
2. Provide music longer than video (avoid excessive looping)
3. Use fade-out for professional finish: `--fade 5`

## 🎓 Advanced Usage

### Custom Fade Duration per Project
```bash
# Short fade (3s) for upbeat videos
python process_all_media.py --fade 3

# Long fade (10s) for emotional videos
python process_all_media.py --fade 10

# No fade for loop videos
python process_all_media.py --fade 0
```

### Mix AI Music with Custom Intro/Outro
```bash
# Place only intro/outro music in input
# Let AI fill the middle by using --force-generate
# (Not yet implemented - feature request)
```

### Batch Processing
```bash
# Process multiple folders with same music
for folder in vacation_2023 vacation_2024; do
    cp music/*.mp3 $folder/
    cp -r $folder/* input/
    python process_all_media.py --fade 5
    mv output/final_video.mp4 output/${folder}_video.mp4
    rm input/*
done
```

## 🆕 What's New

### Version 2.0 Updates
- ✅ **Music File Support**: Use your own MP3/WAV files
- ✅ **Automatic Music Looping**: Seamlessly extends short music
- ✅ **Fade Control**: Configurable fade-out duration
- ✅ **Longer Image Duration**: 6-10 seconds (was 2-8 seconds)
- ✅ **Better Audio Quality**: Using pydub for audio processing

## 🛠 Known Issues

1. **Music Files in Subdirectories**: Currently only detects music in root input folder
2. **Very Long Music**: If music is much longer than video, excess is trimmed
3. **Fade on Short Videos**: Fade duration shouldn't exceed video duration

## 📄 License

This project uses open-source models:
- BLIP-2: Salesforce (BSD-3-Clause License)
- MusicGen: Meta AI (CC-BY-NC 4.0)
- MoviePy: MIT License
- Pydub: MIT License

## 🤝 Contributing

Contributions welcome! Priority areas:
- Music crossfade between loops
- Volume normalization across tracks
- Audio filters (EQ, compression)
- Multiple audio track support
- Custom music mapping to scenes

## 📧 Support

For issues and questions:
1. Check this README
2. Run diagnostic tools
3. Check existing GitHub issues
4. Create new issue with error logs

---

**Created with ❤️ using AI** | Ready for YouTube! 🎬