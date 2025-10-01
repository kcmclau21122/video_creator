#!/usr/bin/env python3
# ============================================================================
# Check and Fix Image Orientations - Standalone Tool
# ============================================================================

"""
Simple tool to check and fix image orientations in your input folder
Run this BEFORE processing your video to catch orientation issues early
"""

from pathlib import Path
from PIL import Image, ImageOps


def check_and_fix_images(input_folder: str = "input", fix: bool = False):
    """
    Check (and optionally fix) all images in the input folder
    
    Args:
        input_folder: Path to folder with images
        fix: If True, fixes images in-place
    """
    folder = Path(input_folder)
    
    if not folder.exists():
        print(f"❌ Folder not found: {folder}")
        return
    
    print("\n" + "="*70)
    print("IMAGE ORIENTATION CHECKER")
    print("="*70)
    print(f"Folder: {folder}")
    print(f"Mode: {'FIX (overwrites files)' if fix else 'CHECK ONLY'}")
    print("="*70 + "\n")
    
    # Find all images
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.heic', '.heif'}
    images = [f for f in folder.rglob('*') if f.suffix.lower() in image_extensions]
    
    if not images:
        print("❌ No images found!")
        return
    
    print(f"Found {len(images)} images\n")
    
    # Check each image
    needs_fixing = []
    
    for img_path in images:
        try:
            img = Image.open(img_path)
            exif = img._getexif()
            
            if exif and 274 in exif:  # 274 is orientation tag
                orientation = exif[274]
                
                if orientation != 1:  # 1 = normal orientation
                    orientation_desc = {
                        2: "Mirrored horizontal",
                        3: "Upside down (180°)",
                        4: "Mirrored vertical",
                        5: "Mirrored + 90° CCW",
                        6: "Rotated 90° CW",
                        7: "Mirrored + 90° CW",
                        8: "Rotated 90° CCW"
                    }
                    
                    desc = orientation_desc.get(orientation, f"Unknown ({orientation})")
                    
                    print(f"🔄 {img_path.name}")
                    print(f"   Issue: {desc}")
                    
                    needs_fixing.append((img_path, orientation, desc))
                    
                    if fix:
                        # Fix the orientation
                        img_fixed = ImageOps.exif_transpose(img)
                        img_fixed.save(img_path, quality=95)
                        print(f"   ✅ FIXED!")
                    
                    print()
                    
        except Exception as e:
            print(f"⚠️  Could not check {img_path.name}: {e}\n")
    
    # Summary
    print("="*70)
    print("SUMMARY")
    print("="*70)
    print(f"Total images: {len(images)}")
    print(f"Need fixing: {len(needs_fixing)}")
    print(f"Correct orientation: {len(images) - len(needs_fixing)}")
    
    if needs_fixing and not fix:
        print("\n" + "="*70)
        print("TO FIX THESE ISSUES:")
        print("="*70)
        print("Run this script again with --fix:")
        print(f"  python check_and_fix_images.py --fix")
        print("\nOr add this line to video_composer.py:")
        print("  img = ImageOps.exif_transpose(img)")
    elif needs_fixing and fix:
        print(f"\n✅ Fixed {len(needs_fixing)} images!")
    else:
        print("\n✅ All images have correct orientation!")


if __name__ == "__main__":
    import sys
    
    fix_mode = "--fix" in sys.argv or "-f" in sys.argv
    
    # Get folder from arguments or use default
    folder = "input"
    for arg in sys.argv[1:]:
        if not arg.startswith("-"):
            folder = arg
            break
    
    if "--help" in sys.argv or "-h" in sys.argv:
        print("Usage:")
        print("  python check_and_fix_images.py              # Check only")
        print("  python check_and_fix_images.py --fix        # Fix images")
        print("  python check_and_fix_images.py input --fix  # Fix specific folder")
        print()
        print("This will check all images in the folder for orientation issues")
        print("and optionally fix them in-place.")
    else:
        check_and_fix_images(folder, fix_mode)