# ============================================================================
# AI Video Creator - HEIC to JPEG Converter (Maximum Quality + Size Control)
# ============================================================================

from pathlib import Path
from typing import Optional, List
from PIL import Image
import pillow_heif
import piexif
import sys


class HEICConverter:
    """Convert HEIC/HEIF images to JPEG with maximum quality and size control"""
    
    def __init__(self, output_quality: int = 100, optimize: bool = True):
        self.output_quality = output_quality
        self.optimize = optimize
        pillow_heif.register_heif_opener()
    
    def _extract_exif_fields(self, exif_bytes):
        """Helper to extract key EXIF fields from EXIF bytes."""
        if not exif_bytes:
            return {}
        try:
            exif_dict = piexif.load(exif_bytes)
            fields = {
                "DateTimeOriginal": exif_dict["Exif"].get(piexif.ExifIFD.DateTimeOriginal),
                "CameraModel": exif_dict["0th"].get(piexif.ImageIFD.Model),
                "GPSPresent": bool(exif_dict.get("GPS")),
            }
            return fields
        except Exception:
            return {}
    
    def convert_heic(self, heic_path: Path, output_path: Path = None) -> Optional[Path]:
        try:
            if output_path is None:
                output_path = heic_path.with_suffix('.jpeg')
            
            if output_path.exists():
                print(f"Already converted: {heic_path.name}")
                return output_path
            
            # Get original file size for comparison
            original_size_mb = heic_path.stat().st_size / (1024 * 1024)
            print(f"Converting: {heic_path.name} ({original_size_mb:.2f} MB)")
            
            # Open HEIC
            img = Image.open(heic_path)
            exif_bytes = img.info.get("exif")
            original_exif = self._extract_exif_fields(exif_bytes)
            
            # Store original dimensions
            original_width, original_height = img.size
            
            # Convert to RGB if required
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            # Save with EXIF preserved if possible
            # Use quality 100 and optimize to balance quality and size
            save_kwargs = {
                'format': 'JPEG',
                'quality': self.output_quality,
                'optimize': self.optimize,
                'subsampling': 0,  # No chroma subsampling for maximum quality
                'progressive': True  # Progressive JPEG for better compression
            }
            
            if exif_bytes:
                try:
                    exif_dict = piexif.load(exif_bytes)
                    exif_bytes = piexif.dump(exif_dict)
                    save_kwargs['exif'] = exif_bytes
                except Exception as e:
                    print(f"  Warning: EXIF copy failed ({e}), saving without EXIF.")
            
            img.save(output_path, **save_kwargs)
            
            # Verify result
            new_img = Image.open(output_path)
            new_exif_bytes = new_img.info.get("exif")
            new_exif = self._extract_exif_fields(new_exif_bytes)
            new_width, new_height = new_img.size
            new_size_mb = output_path.stat().st_size / (1024 * 1024)
            
            print(f"  Output: {output_path.name} ({new_size_mb:.2f} MB)")
            print(f"  Resolution: {original_width}x{original_height} -> {new_width}x{new_height} (preserved: {original_width==new_width and original_height==new_height})")
            print(f"  Size change: {((new_size_mb/original_size_mb - 1) * 100):+.1f}%")
            print(f"  EXIF preserved: {original_exif == new_exif}")
            
            return output_path
        
        except Exception as e:
            print(f"Error converting {heic_path.name}: {e}")
            return None
    
    def convert_folder(self, folder: Path) -> List[Path]:
        heic_extensions = {'.heic', '.heif', '.HEIC', '.HEIF'}
        heic_files = [f for f in folder.rglob('*') if f.suffix in heic_extensions]
        
        if not heic_files:
            print("No HEIC files found.")
            return []
        
        print(f"\nConverting {len(heic_files)} HEIC files with quality {self.output_quality}...")
        print("="*70)
        converted_files = []
        
        total_original_size = 0
        total_new_size = 0
        
        for heic_path in heic_files:
            total_original_size += heic_path.stat().st_size
            jpeg_path = self.convert_heic(heic_path)
            if jpeg_path:
                converted_files.append(jpeg_path)
                total_new_size += jpeg_path.stat().st_size
            print()  # Blank line between files
        
        print("="*70)
        print(f"Converted {len(converted_files)} files successfully")
        if converted_files:
            original_mb = total_original_size / (1024 * 1024)
            new_mb = total_new_size / (1024 * 1024)
            print(f"Total original size: {original_mb:.2f} MB")
            print(f"Total new size: {new_mb:.2f} MB")
            print(f"Overall size change: {((new_mb/original_mb - 1) * 100):+.1f}%")
        print()
        return converted_files


def select_folder():
    """
    Allow user to select folder via GUI or manual input
    """
    try:
        # Try to use tkinter for GUI folder selection
        import tkinter as tk
        from tkinter import filedialog
        
        root = tk.Tk()
        root.withdraw()  # Hide the main window
        root.attributes('-topmost', True)  # Bring dialog to front
        
        folder_path = filedialog.askdirectory(
            title="Select folder containing HEIC images"
        )
        
        root.destroy()
        
        if folder_path:
            return Path(folder_path)
        else:
            print("No folder selected.")
            return None
            
    except ImportError:
        # Fallback to manual input if tkinter not available
        print("GUI not available. Please enter folder path manually.")
        return manual_folder_input()


def manual_folder_input():
    """
    Manual folder path input
    """
    print("\n" + "="*70)
    print("HEIC TO JPEG CONVERTER")
    print("="*70)
    print("\nEnter the full path to the folder containing HEIC images:")
    print("Example: C:\\Users\\YourName\\Pictures")
    print("Or type 'quit' to exit\n")
    
    while True:
        folder_input = input("Folder path: ").strip()
        
        if folder_input.lower() == 'quit':
            return None
        
        folder_path = Path(folder_input)
        
        if folder_path.exists() and folder_path.is_dir():
            return folder_path
        else:
            print(f"Error: '{folder_input}' is not a valid folder path.")
            print("Please try again or type 'quit' to exit.\n")


def main():
    """
    Main function for standalone execution
    """
    print("\n" + "="*70)
    print("HEIC TO JPEG CONVERTER - Maximum Quality Mode")
    print("="*70)
    print("\nThis tool converts HEIC/HEIF images to JPEG format")
    print("with maximum quality (100) and optimized file size.\n")
    
    # Check if folder was provided as command line argument
    if len(sys.argv) > 1:
        folder_path = Path(sys.argv[1])
        if not folder_path.exists() or not folder_path.is_dir():
            print(f"Error: '{sys.argv[1]}' is not a valid folder path.")
            folder_path = select_folder()
    else:
        # Try GUI folder selection
        folder_path = select_folder()
    
    if not folder_path:
        print("Exiting...")
        return
    
    print(f"\nSelected folder: {folder_path}")
    print("-"*70)
    
    # Settings
    print("\nConversion Settings:")
    print("  Quality: 100 (Maximum)")
    print("  Subsampling: None (Maximum quality)")
    print("  Optimization: Enabled (Better compression)")
    print("  Resolution: Preserved (Same as original)")
    print("  EXIF Data: Preserved (Dates, camera, GPS)")
    
    confirm = input("\nProceed with conversion? (y/n): ").strip().lower()
    if confirm != 'y':
        print("Cancelled.")
        return
    
    # Create converter and process files
    converter = HEICConverter(output_quality=100, optimize=True)
    converted_files = converter.convert_folder(folder_path)
    
    # Summary
    print("\n" + "="*70)
    print("CONVERSION COMPLETE")
    print("="*70)
    print(f"Total files converted: {len(converted_files)}")
    print(f"Location: {folder_path}")
    print("\nConverted JPEG files are saved in the same folder as originals.")
    print("Original HEIC files are preserved (not deleted).")
    print("\nNote: JPEG files may be larger than HEIC due to format differences,")
    print("but image quality and resolution are maximally preserved.")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()