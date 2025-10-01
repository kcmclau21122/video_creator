# ============================================================================
# AI Video Creator - HEIC to JPEG Converter
# ============================================================================

"""
HEIC to JPEG Converter
Automatically converts HEIC/HEIF images to JPEG format
"""

from pathlib import Path
from typing import Optional, List
from PIL import Image
import pillow_heif


class HEICConverter:
    """Convert HEIC/HEIF images to JPEG"""
    
    def __init__(self, output_quality: int = 95):
        """
        Initialize HEIC converter
        
        Args:
            output_quality: JPEG quality (1-100, 95 recommended)
        """
        self.output_quality = output_quality
        
        # Register HEIF opener with PIL
        pillow_heif.register_heif_opener()
    
    def convert_heic(self, heic_path: Path, output_path: Path = None) -> Optional[Path]:
        """
        Convert single HEIC file to JPEG
        
        Args:
            heic_path: Path to HEIC file
            output_path: Where to save JPEG (None = auto-generate)
            
        Returns:
            Path to converted JPEG file, or None if failed
        """
        try:
            # Generate output path if not provided
            if output_path is None:
                output_path = heic_path.with_suffix('.jpeg')
            
            # Check if already converted
            if output_path.exists():
                print(f"✓ Already converted: {heic_path.name}")
                return output_path
            
            print(f"🔄 Converting: {heic_path.name}")
            
            # Open HEIC file
            img = Image.open(heic_path)
            
            # Convert to RGB if needed
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            # Save as JPEG
            img.save(output_path, 'JPEG', quality=self.output_quality, optimize=True)
            
            print(f"✅ Converted: {heic_path.name} → {output_path.name}")
            
            return output_path
            
        except Exception as e:
            print(f"✗ Error converting {heic_path.name}: {e}")
            return None
    
    def convert_folder(self, folder: Path) -> List[Path]:
        """
        Convert all HEIC files in a folder
        
        Args:
            folder: Path to folder
            
        Returns:
            List of converted JPEG paths
        """
        # Find all HEIC files
        heic_extensions = {'.heic', '.heif', '.HEIC', '.HEIF'}
        heic_files = [f for f in folder.rglob('*') if f.suffix in heic_extensions]
        
        if not heic_files:
            return []
        
        print(f"Converting {len(heic_files)} HEIC files...")
        
        # Convert each file
        converted_files = []
        for heic_path in heic_files:
            jpeg_path = self.convert_heic(heic_path)
            if jpeg_path:
                converted_files.append(jpeg_path)
        
        print(f"✅ Converted {len(converted_files)} files\n")
        
        return converted_files