# ============================================================================
# AI Video Creator - Image Orientation Fixer
# ============================================================================

"""
Image Orientation Fixer
Detects and corrects image orientation based on EXIF data
Fixes upside-down and rotated images automatically
"""

from pathlib import Path
from PIL import Image, ImageOps
import piexif
from typing import Optional, Tuple


class ImageOrientationFixer:
    """Fix image orientation issues"""
    
    # EXIF Orientation tag values
    ORIENTATION_MAP = {
        1: 0,      # Normal
        2: 0,      # Mirrored horizontal
        3: 180,    # Rotated 180
        4: 0,      # Mirrored vertical
        5: 0,      # Mirrored horizontal then rotated 90 CCW
        6: 270,    # Rotated 90 CW
        7: 0,      # Mirrored horizontal then rotated 90 CW
        8: 90      # Rotated 90 CCW
    }
    
    def __init__(self):
        """Initialize orientation fixer"""
        pass
    
    def get_orientation(self, image_path: Path) -> Optional[int]:
        """
        Get EXIF orientation value from image
        
        Args:
            image_path: Path to image file
            
        Returns:
            Orientation value (1-8) or None if not found
        """
        try:
            img = Image.open(image_path)
            
            # Try to get EXIF data
            exif_data = img._getexif()
            
            if exif_data:
                # Orientation tag is 274
                orientation = exif_data.get(274)
                return orientation
            
            return None
            
        except Exception as e:
            print(f"⚠️  Could not read orientation from {image_path.name}: {e}")
            return None
    
    def fix_orientation(self, image_path: Path, output_path: Path = None, in_place: bool = False) -> Path:
        """
        Fix image orientation based on EXIF data
        
        Args:
            image_path: Path to image file
            output_path: Where to save corrected image (None = auto-generate)
            in_place: If True, overwrites original file
            
        Returns:
            Path to corrected image
        """
        try:
            # Open image
            img = Image.open(image_path)
            
            # Get orientation
            orientation = self.get_orientation(image_path)
            
            if orientation is None or orientation == 1:
                # No orientation data or already correct
                print(f"✓ {image_path.name}: Orientation is correct")
                return image_path
            
            print(f"🔄 {image_path.name}: Fixing orientation (EXIF value: {orientation})")
            
            # Apply orientation fix using PIL's built-in method
            # This is the recommended way - it handles all orientation cases
            img_corrected = ImageOps.exif_transpose(img)
            
            # Determine output path
            if in_place:
                save_path = image_path
            elif output_path:
                save_path = output_path
            else:
                # Create new filename
                stem = image_path.stem
                suffix = image_path.suffix
                save_path = image_path.parent / f"{stem}_corrected{suffix}"
            
            # Save corrected image
            # Remove EXIF orientation tag since we've applied it
            img_corrected.save(save_path, quality=95, optimize=True)
            
            print(f"✅ {image_path.name}: Corrected and saved to {save_path.name}")
            
            return save_path
            
        except Exception as e:
            print(f"✗ Error fixing orientation for {image_path.name}: {e}")
            return image_path
    
    def fix_orientation_detailed(self, image_path: Path, output_path: Path = None) -> Tuple[Path, dict]:
        """
        Fix orientation with detailed information about what was done
        
        Args:
            image_path: Path to image file
            output_path: Where to save corrected image
            
        Returns:
            Tuple of (output_path, info_dict)
        """
        info = {
            'original_path': str(image_path),
            'orientation_value': None,
            'rotation_applied': 0,
            'was_corrected': False,
            'output_path': str(image_path)
        }
        
        try:
            img = Image.open(image_path)
            orientation = self.get_orientation(image_path)
            
            info['orientation_value'] = orientation
            
            if orientation and orientation != 1:
                # Apply correction
                img_corrected = ImageOps.exif_transpose(img)
                
                # Determine rotation for info
                info['rotation_applied'] = self.ORIENTATION_MAP.get(orientation, 0)
                info['was_corrected'] = True
                
                # Save
                if output_path:
                    save_path = output_path
                else:
                    stem = image_path.stem
                    suffix = image_path.suffix
                    save_path = image_path.parent / f"{stem}_corrected{suffix}"
                
                img_corrected.save(save_path, quality=95, optimize=True)
                info['output_path'] = str(save_path)
                
                return save_path, info
            else:
                return image_path, info
                
        except Exception as e:
            info['error'] = str(e)
            return image_path, info
    
    def check_orientation(self, image_path: Path) -> dict:
        """
        Check orientation without fixing
        
        Args:
            image_path: Path to image file
            
        Returns:
            Dictionary with orientation information
        """
        result = {
            'path': str(image_path),
            'orientation': None,
            'needs_correction': False,
            'description': 'Normal'
        }
        
        orientation = self.get_orientation(image_path)
        result['orientation'] = orientation
        
        if orientation:
            descriptions = {
                1: 'Normal (Correct)',
                2: 'Mirrored horizontal',
                3: 'Rotated 180° (Upside down)',
                4: 'Mirrored vertical',
                5: 'Mirrored horizontal + rotated 90° CCW',
                6: 'Rotated 90° clockwise',
                7: 'Mirrored horizontal + rotated 90° CW',
                8: 'Rotated 90° counter-clockwise'
            }
            
            result['description'] = descriptions.get(orientation, 'Unknown')
            result['needs_correction'] = orientation != 1
        
        return result
    
    def batch_check(self, folder: Path) -> list:
        """
        Check orientation of all images in a folder
        
        Args:
            folder: Path to folder containing images
            
        Returns:
            List of dictionaries with orientation info
        """
        results = []
        
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
        
        for file_path in folder.rglob('*'):
            if file_path.suffix.lower() in image_extensions:
                result = self.check_orientation(file_path)
                results.append(result)
        
        return results
    
    def batch_fix(self, folder: Path, in_place: bool = False, create_subfolder: bool = True) -> dict:
        """
        Fix orientation of all images in a folder
        
        Args:
            folder: Path to folder containing images
            in_place: If True, overwrites original files
            create_subfolder: If True, saves corrected images to 'corrected' subfolder
            
        Returns:
            Dictionary with summary of fixes
        """
        summary = {
            'total_files': 0,
            'corrected': 0,
            'already_correct': 0,
            'errors': 0,
            'corrected_files': []
        }
        
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
        
        # Create output folder if needed
        if create_subfolder and not in_place:
            output_folder = folder / 'corrected'
            output_folder.mkdir(exist_ok=True)
        else:
            output_folder = folder
        
        for file_path in folder.rglob('*'):
            if file_path.suffix.lower() in image_extensions:
                summary['total_files'] += 1
                
                try:
                    orientation = self.get_orientation(file_path)
                    
                    if orientation and orientation != 1:
                        # Needs correction
                        if in_place:
                            output_path = None
                        else:
                            output_path = output_folder / file_path.name
                        
                        corrected_path = self.fix_orientation(file_path, output_path, in_place)
                        summary['corrected'] += 1
                        summary['corrected_files'].append(str(corrected_path))
                    else:
                        summary['already_correct'] += 1
                        
                except Exception as e:
                    summary['errors'] += 1
                    print(f"✗ Error processing {file_path.name}: {e}")
        
        return summary


def fix_single_image(image_path: str, in_place: bool = False):
    """
    Convenience function to fix a single image
    
    Args:
        image_path: Path to image file
        in_place: If True, overwrites original file
    """
    fixer = ImageOrientationFixer()
    fixer.fix_orientation(Path(image_path), in_place=in_place)


def check_folder_orientations(folder_path: str):
    """
    Convenience function to check all images in a folder
    
    Args:
        folder_path: Path to folder
    """
    fixer = ImageOrientationFixer()
    results = fixer.batch_check(Path(folder_path))
    
    print(f"\n{'='*70}")
    print("ORIENTATION CHECK RESULTS")
    print(f"{'='*70}\n")
    
    needs_fixing = [r for r in results if r['needs_correction']]
    
    print(f"Total images: {len(results)}")
    print(f"Need correction: {len(needs_fixing)}")
    print(f"Already correct: {len(results) - len(needs_fixing)}")
    
    if needs_fixing:
        print(f"\n{'='*70}")
        print("IMAGES NEEDING CORRECTION:")
        print(f"{'='*70}\n")
        
        for result in needs_fixing:
            print(f"📷 {Path(result['path']).name}")
            print(f"   {result['description']}")
            print()


def fix_folder(folder_path: str, in_place: bool = False):
    """
    Convenience function to fix all images in a folder
    
    Args:
        folder_path: Path to folder
        in_place: If True, overwrites original files
    """
    fixer = ImageOrientationFixer()
    
    print(f"\n{'='*70}")
    print("BATCH ORIENTATION FIX")
    print(f"{'='*70}")
    print(f"Folder: {folder_path}")
    print(f"Mode: {'In-place (overwrites originals)' if in_place else 'Create corrected copies'}")
    print()
    
    if in_place:
        confirm = input("⚠️  This will modify original files. Continue? (y/n): ")
        if confirm.lower() != 'y':
            print("Cancelled")
            return
    
    summary = fixer.batch_fix(Path(folder_path), in_place=in_place)
    
    print(f"\n{'='*70}")
    print("BATCH FIX COMPLETE")
    print(f"{'='*70}")
    print(f"Total files: {summary['total_files']}")
    print(f"Corrected: {summary['corrected']}")
    print(f"Already correct: {summary['already_correct']}")
    print(f"Errors: {summary['errors']}")
    
    if summary['corrected'] > 0:
        print(f"\n✅ {summary['corrected']} images corrected!")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage:")
        print("  Check folder:     python image_orientation.py check <folder>")
        print("  Fix folder:       python image_orientation.py fix <folder>")
        print("  Fix in-place:     python image_orientation.py fix <folder> --in-place")
        print("  Fix single image: python image_orientation.py fix-image <image_file>")
        sys.exit(1)
    
    command = sys.argv[1]
    
    if command == 'check' and len(sys.argv) >= 3:
        check_folder_orientations(sys.argv[2])
    
    elif command == 'fix' and len(sys.argv) >= 3:
        in_place = '--in-place' in sys.argv
        fix_folder(sys.argv[2], in_place)
    
    elif command == 'fix-image' and len(sys.argv) >= 3:
        in_place = '--in-place' in sys.argv
        fix_single_image(sys.argv[2], in_place)
    
    else:
        print("Invalid command or arguments")
        print("Use: python image_orientation.py check <folder>")
        print("Or:  python image_orientation.py fix <folder>")