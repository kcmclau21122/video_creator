# ============================================================================
# AI Video Creator - Step 2: Metadata Extractor
# ============================================================================

"""
Metadata Extraction Module
Extracts EXIF and metadata from images and videos
"""

from pathlib import Path
from typing import Dict, Optional
from datetime import datetime
import piexif
from PIL import Image
import cv2
import exifread


class MetadataExtractor:
    """Extract metadata from images and videos"""
    
    def extract_video_metadata(self, video_path: Path) -> Dict:
        """
        Extract metadata from video file
        
        Args:
            video_path: Path to video file
            
        Returns:
            Dictionary with metadata
        """
        metadata = {
            'datetime_original': None,
            'width': None,
            'height': None,
            'duration': None,
            'gps_latitude': None,
            'gps_longitude': None,
            'camera_make': None,
            'camera_model': None,
            'orientation': None
        }
        
        try:
            # Open video with OpenCV
            cap = cv2.VideoCapture(str(video_path))
            
            if cap.isOpened():
                # Get dimensions
                metadata['width'] = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                metadata['height'] = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                
                # Get duration
                fps = cap.get(cv2.CAP_PROP_FPS)
                frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
                if fps > 0:
                    metadata['duration'] = frame_count / fps
                
                cap.release()
            
            # Try to get datetime from file modification time
            stat = video_path.stat()
            metadata['datetime_original'] = datetime.fromtimestamp(stat.st_mtime)
            
            # If no datetime, try filename parsing
            if not metadata['datetime_original']:
                metadata['datetime_original'] = self._parse_datetime_from_filename(video_path)
            
        except Exception as e:
            print(f"⚠  Could not extract video metadata from {video_path.name}: {e}")
        
        return metadata

    def extract_image_metadata(self, image_path: Path) -> Dict:
        """
        Extract metadata from image file
        
        Args:
            image_path: Path to image file
            
        Returns:
            Dictionary with metadata
        """
        metadata = {
            'datetime_original': None,
            'width': None,
            'height': None,
            'duration': None,
            'gps_latitude': None,
            'gps_longitude': None,
            'camera_make': None,
            'camera_model': None,
            'orientation': None
        }
        
        try:
            # Try PIL first
            with Image.open(image_path) as img:
                metadata['width'], metadata['height'] = img.size
                
                # Get EXIF data
                exif_data = img._getexif()
                
                if exif_data:
                    # Extract datetime
                    datetime_str = exif_data.get(36867) or exif_data.get(36868) or exif_data.get(306)
                    if datetime_str:
                        metadata['datetime_original'] = self._parse_datetime(datetime_str)
                    
                    # Camera info
                    metadata['camera_make'] = exif_data.get(271)  # Make
                    metadata['camera_model'] = exif_data.get(272)  # Model
                    metadata['orientation'] = exif_data.get(274)  # Orientation
                    
                    # GPS data
                    gps_info = exif_data.get(34853)
                    if gps_info:
                        lat, lon = self._extract_gps(gps_info)
                        metadata['gps_latitude'] = lat
                        metadata['gps_longitude'] = lon
            
        except Exception as e:
            # Fallback to exifread
            try:
                metadata.update(self._extract_with_exifread(image_path))
            except Exception as e2:
                print(f"⚠ Could not extract metadata from {image_path.name}: {e2}")
        
        return metadata
    
    def extract_image_metadata(self, image_path: Path) -> Dict:
        """
        Extract metadata from image file
        Prefers JPEG over HEIC if both exist
        Falls back to filename date if EXIF missing
        
        Args:
            image_path: Path to image file
            
        Returns:
            Dictionary with metadata
        """
        metadata = {
            'datetime_original': None,
            'width': None,
            'height': None,
            'duration': None,
            'gps_latitude': None,
            'gps_longitude': None,
            'camera_make': None,
            'camera_model': None,
            'orientation': None
        }
        
        # If this is a HEIC file, check if JPEG version exists
        actual_path = image_path
        if image_path.suffix.lower() in ['.heic', '.heif']:
            jpeg_path = image_path.with_suffix('.jpeg')
            if jpeg_path.exists():
                print(f"  Using JPEG version: {jpeg_path.name}")
                actual_path = jpeg_path
        
        try:
            # Try PIL first
            with Image.open(actual_path) as img:
                metadata['width'], metadata['height'] = img.size
                
                # Get EXIF data
                exif_data = img._getexif()
                
                if exif_data:
                    # Extract datetime
                    datetime_str = exif_data.get(36867) or exif_data.get(36868) or exif_data.get(306)
                    if datetime_str:
                        metadata['datetime_original'] = self._parse_datetime(datetime_str)
                    
                    # Camera info
                    metadata['camera_make'] = exif_data.get(271)  # Make
                    metadata['camera_model'] = exif_data.get(272)  # Model
                    metadata['orientation'] = exif_data.get(274)  # Orientation
                    
                    # GPS data
                    gps_info = exif_data.get(34853)
                    if gps_info:
                        lat, lon = self._extract_gps(gps_info)
                        metadata['gps_latitude'] = lat
                        metadata['gps_longitude'] = lon
            
        except Exception as e:
            # Fallback to exifread
            try:
                metadata.update(self._extract_with_exifread(actual_path))
            except Exception as e2:
                print(f"⚠  Could not extract metadata from {actual_path.name}: {e2}")
        
        # If no datetime found in EXIF, try to parse from filename
        if not metadata['datetime_original']:
            metadata['datetime_original'] = self._parse_datetime_from_filename(actual_path)
        
        return metadata

    def _parse_datetime_from_filename(self, file_path: Path) -> Optional[datetime]:
        """
        Parse datetime from filename if it contains a date pattern
        Supports formats like: 20231002_193645375_iOS.jpeg, IMG_20231002_193645.jpg
        
        Args:
            file_path: Path to file
            
        Returns:
            datetime object or None
        """
        import re
        
        filename = file_path.stem  # Get filename without extension
        
        # Pattern 1: YYYYMMDD_HHMMSS (with optional milliseconds)
        # Examples: 20231002_193645, 20231002_193645375
        pattern1 = r'(\d{4})(\d{2})(\d{2})[-_](\d{2})(\d{2})(\d{2})'
        match = re.search(pattern1, filename)
        
        if match:
            try:
                year, month, day, hour, minute, second = match.groups()
                dt = datetime(
                    int(year), int(month), int(day),
                    int(hour), int(minute), int(second)
                )
                print(f"  Extracted date from filename: {dt.strftime('%Y-%m-%d %H:%M:%S')}")
                return dt
            except ValueError:
                pass
        
        # Pattern 2: YYYYMMDD only (no time)
        # Examples: 20231002, IMG_20231002
        pattern2 = r'(\d{4})(\d{2})(\d{2})'
        match = re.search(pattern2, filename)
        
        if match:
            try:
                year, month, day = match.groups()
                dt = datetime(int(year), int(month), int(day))
                print(f"  Extracted date from filename: {dt.strftime('%Y-%m-%d')}")
                return dt
            except ValueError:
                pass
        
        # Pattern 3: YYYY-MM-DD format
        # Examples: 2023-10-02
        pattern3 = r'(\d{4})-(\d{2})-(\d{2})'
        match = re.search(pattern3, filename)
        
        if match:
            try:
                year, month, day = match.groups()
                dt = datetime(int(year), int(month), int(day))
                print(f"  Extracted date from filename: {dt.strftime('%Y-%m-%d')}")
                return dt
            except ValueError:
                pass
        
        return None
    
    def _extract_with_exifread(self, image_path: Path) -> Dict:
        """Extract metadata using exifread library"""
        metadata = {}
        
        with open(image_path, 'rb') as f:
            tags = exifread.process_file(f, details=False)
            
            # DateTime
            datetime_tag = tags.get('EXIF DateTimeOriginal') or tags.get('Image DateTime')
            if datetime_tag:
                metadata['datetime_original'] = self._parse_datetime(str(datetime_tag))
            
            # Camera
            if 'Image Make' in tags:
                metadata['camera_make'] = str(tags['Image Make'])
            if 'Image Model' in tags:
                metadata['camera_model'] = str(tags['Image Model'])
            
            # GPS
            if 'GPS GPSLatitude' in tags and 'GPS GPSLongitude' in tags:
                lat = self._convert_gps_to_decimal(
                    tags['GPS GPSLatitude'].values,
                    str(tags.get('GPS GPSLatitudeRef', 'N'))
                )
                lon = self._convert_gps_to_decimal(
                    tags['GPS GPSLongitude'].values,
                    str(tags.get('GPS GPSLongitudeRef', 'E'))
                )
                metadata['gps_latitude'] = lat
                metadata['gps_longitude'] = lon
        
        return metadata
    
    def _parse_datetime(self, datetime_str: str) -> Optional[datetime]:
        """Parse datetime string from EXIF"""
        if not datetime_str:
            return None
        
        # Common EXIF datetime format: "YYYY:MM:DD HH:MM:SS"
        formats = [
            "%Y:%m:%d %H:%M:%S",
            "%Y-%m-%d %H:%M:%S",
            "%Y/%m/%d %H:%M:%S",
            "%Y:%m:%d",
            "%Y-%m-%d",
        ]
        
        for fmt in formats:
            try:
                return datetime.strptime(datetime_str, fmt)
            except ValueError:
                continue
        
        return None
    
    def _extract_gps(self, gps_info: dict) -> tuple:
        """Extract GPS coordinates from EXIF GPS info"""
        try:
            lat = gps_info.get(2)  # Latitude
            lat_ref = gps_info.get(1, b'N').decode() if isinstance(gps_info.get(1), bytes) else gps_info.get(1, 'N')
            lon = gps_info.get(4)  # Longitude
            lon_ref = gps_info.get(3, b'E').decode() if isinstance(gps_info.get(3), bytes) else gps_info.get(3, 'E')
            
            if lat and lon:
                # Convert to decimal degrees
                lat_decimal = self._dms_to_decimal(lat, lat_ref)
                lon_decimal = self._dms_to_decimal(lon, lon_ref)
                return lat_decimal, lon_decimal
        except Exception:
            pass
        
        return None, None
    
    def _dms_to_decimal(self, dms: tuple, ref: str) -> float:
        """Convert degrees, minutes, seconds to decimal degrees"""
        degrees = float(dms[0])
        minutes = float(dms[1])
        seconds = float(dms[2])
        
        decimal = degrees + (minutes / 60.0) + (seconds / 3600.0)
        
        if ref in ['S', 'W']:
            decimal = -decimal
        
        return decimal
    
    def _convert_gps_to_decimal(self, values, ref: str) -> float:
        """Convert GPS values to decimal (for exifread)"""
        degrees = float(values[0].num) / float(values[0].den)
        minutes = float(values[1].num) / float(values[1].den)
        seconds = float(values[2].num) / float(values[2].den)
        
        decimal = degrees + (minutes / 60.0) + (seconds / 3600.0)
        
        if ref in ['S', 'W']:
            decimal = -decimal
        
        return decimal


if __name__ == "__main__":
    # Test metadata extraction
    extractor = MetadataExtractor()
    
    from config import Config
    
    # Test on first image in input folder
    input_folder = Config.INPUT_DIR
    
    image_files = list(input_folder.glob("*.jpg")) + list(input_folder.glob("*.jpeg"))
    video_files = list(input_folder.glob("*.mp4"))
    
    if image_files:
        print(f"\nTesting image metadata extraction on: {image_files[0].name}")
        metadata = extractor.extract_image_metadata(image_files[0])
        print("Metadata:", metadata)
    else:
        print("No image files found in input folder")
    
    if video_files:
        print(f"\nTesting video metadata extraction on: {video_files[0].name}")
        metadata = extractor.extract_video_metadata(video_files[0])
        print("Metadata:", metadata)
    else:
        print("No video files found in input folder")