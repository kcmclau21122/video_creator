#!/usr/bin/env python3
# ============================================================================
# Check Problem Images
# ============================================================================

"""
Verify the problematic images and see if they can be loaded
"""

from pathlib import Path
from PIL import Image
from config import Config
import json


def check_images():
    """Check problematic images"""
    
    problem_files = [
        "IMG_1414.heic.jpeg",
        "IMG_1370.heic.jpeg", 
        "IMG_1417.heic.jpeg",
        "IMG_1368.heic.jpeg"
    ]
    
    print("="*70)
    print("CHECKING PROBLEM IMAGES")
    print("="*70)
    
    # Load manifest to get file info
    manifest_path = Config.OUTPUT_DIR / "media_manifest.json"
    if manifest_path.exists():
        with open(manifest_path, 'r') as f:
            manifest = json.load(f)
        
        manifest_dict = {Path(item['file_path']).name: item for item in manifest['items']}
    else:
        manifest_dict = {}
    
    for filename in problem_files:
        print(f"\n[{filename}]")
        print("-"*70)
        
        # Find in input folder
        file_path = None
        for path in Config.INPUT_DIR.rglob(filename):
            file_path = path
            break
        
        if not file_path:
            print("❌ File not found in input folder")
            continue
        
        # Check file exists and size
        if file_path.exists():
            size_mb = file_path.stat().st_size / (1024 * 1024)
            print(f"✓ File exists: {size_mb:.2f} MB")
        else:
            print("❌ File doesn't exist")
            continue
        
        # Check manifest data
        if filename in manifest_dict:
            item = manifest_dict[filename]
            print(f"  Width: {item['width']}")
            print(f"  Height: {item['height']}")
            print(f"  Datetime: {item['datetime_original']}")
        
        # Try to load with PIL
        try:
            img = Image.open(file_path)
            print(f"✓ PIL can load: {img.size}, {img.mode}")
            
            # Try converting to RGB
            if img.mode != 'RGB':
                rgb_img = img.convert('RGB')
                print(f"✓ Converted to RGB: {rgb_img.size}")
            else:
                print(f"✓ Already RGB")
                
        except Exception as e:
            print(f"❌ PIL error: {e}")
            
        # Check if it's really a JPEG
        try:
            with open(file_path, 'rb') as f:
                header = f.read(10)
                if header[:2] == b'\xff\xd8':
                    print(f"✓ Valid JPEG header")
                else:
                    print(f"❌ Invalid JPEG header: {header[:10]}")
        except Exception as e:
            print(f"❌ Error reading file: {e}")
    
    print("\n" + "="*70)
    print("RECOMMENDATION")
    print("="*70)
    print("\nOption 1: Include with defaults (current solution)")
    print("  - Run: python process_all_media.py")
    print("  - Will use default analysis for these 4 files")
    print("  - They'll be in chronological order with 4-second duration")
    
    print("\nOption 2: Try to fix the files")
    print("  - These appear to be HEIC-to-JPEG conversions")
    print("  - Original HEIC files might be better quality")
    print("  - Consider re-converting with better tool")
    
    print("\nOption 3: Manually re-save")
    print("  - Open each file in an image editor")
    print("  - Save as new JPEG")
    print("  - Then re-run analysis")


if __name__ == "__main__":
    check_images()