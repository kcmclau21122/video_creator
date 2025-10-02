# ============================================================================
# AI Video Creator - Step 2: Media Ingestion Module
# ============================================================================

"""
Media Ingestion Module
Handles loading, validating, and organizing media files (images and videos)
"""

import os
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime
import json
from dataclasses import dataclass, asdict
from tqdm import tqdm

from metadata_extractor import MetadataExtractor
from config import Config
from heic_converter import HEICConverter


@dataclass
class MediaItem:
    """Represents a single media item with metadata"""
    file_path: Path
    file_type: str  # 'image' or 'video'
    file_format: str  # 'jpg', 'mp4', etc.
    file_size: int  # bytes
    datetime_original: Optional[datetime]
    width: Optional[int]
    height: Optional[int]
    duration: Optional[float]  # seconds, for videos only
    gps_latitude: Optional[float]
    gps_longitude: Optional[float]
    camera_make: Optional[str]
    camera_model: Optional[str]
    orientation: Optional[int]
    
    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization"""
        result = asdict(self)
        # Convert Path to string
        result['file_path'] = str(self.file_path)
        # Convert datetime to ISO format
        if self.datetime_original:
            result['datetime_original'] = self.datetime_original.isoformat()
        return result
    
    @classmethod
    def from_dict(cls, data: dict) -> 'MediaItem':
        """Create MediaItem from dictionary"""
        if 'file_path' in data:
            data['file_path'] = Path(data['file_path'])
        if 'datetime_original' in data and data['datetime_original']:
            data['datetime_original'] = datetime.fromisoformat(data['datetime_original'])
        return cls(**data)


class MediaIngester:
    """Handles media file ingestion and organization"""
    
    # Supported formats
    IMAGE_FORMATS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.heic', '.heif'}
    VIDEO_FORMATS = {'.mp4', '.mov', '.avi', '.mkv', '.wmv', '.flv', '.webm', '.m4v'}
    
    def __init__(self, cache_enabled: bool = True):
        """
        Initialize MediaIngester
        
        Args:
            cache_enabled: Whether to use cached metadata
        """
        self.metadata_extractor = MetadataExtractor()
        self.cache_enabled = cache_enabled
        self.cache_dir = Config.CACHE_DIR / "metadata"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.heic_converter = HEICConverter()
        
    def process_folder(self, input_folder: Path, use_cache: bool = True) -> List[MediaItem]:
        """
        Process all media files in a folder
        
        Args:
            input_folder: Path to folder containing media files
            use_cache: Whether to use cached metadata
            
        Returns:
            List of MediaItem objects
        """
        print(f"\n{'='*70}")
        print(f"PROCESSING MEDIA FOLDER: {input_folder}")
        print(f"{'='*70}\n")
        
        # Convert HEIC files first
        heic_files = list(input_folder.glob("*.heic")) + list(input_folder.glob("*.HEIC"))
        if heic_files:
            print(f"Found {len(heic_files)} HEIC files - converting to JPEG...\n")
            self.heic_converter.convert_folder(input_folder)

        # Find all media files
        media_files = self._find_media_files(input_folder)
        
        if not media_files:
            print("⚠  No media files found!")
            return []
        
        print(f"Found {len(media_files)} media files")
        
        # Load cache if enabled
        cache_data = {}
        if use_cache and self.cache_enabled:
            cache_data = self._load_cache()
        
        # Process each file
        media_items = []
        for file_path in tqdm(media_files, desc="Processing files"):
            # Check cache first
            cache_key = self._get_cache_key(file_path)
            
            if cache_key in cache_data:
                media_item = MediaItem.from_dict(cache_data[cache_key])
                # Update file path in case it moved
                media_item.file_path = file_path
            else:
                # Extract metadata
                media_item = self._process_file(file_path)
                
                # Only cache if processing was successful
                if media_item and self.cache_enabled:
                    cache_data[cache_key] = media_item.to_dict()
            
            if media_item:
                media_items.append(media_item)
        
        # Save cache
        if self.cache_enabled:
            self._save_cache(cache_data)
        
        print(f"\n✓ Successfully processed {len(media_items)} files")
        
        return media_items
    
    def sort_by_datetime(self, media_items: List[MediaItem]) -> List[MediaItem]:
        """
        Sort media items chronologically
        
        Args:
            media_items: List of MediaItem objects
            
        Returns:
            Sorted list of MediaItem objects
        """
        # Separate items with and without datetime
        with_datetime = [item for item in media_items if item.datetime_original]
        without_datetime = [item for item in media_items if not item.datetime_original]
        
        # Sort items with datetime
        with_datetime.sort(key=lambda x: x.datetime_original)
        
        # For items without datetime, use file modification time
        for item in without_datetime:
            item.datetime_original = datetime.fromtimestamp(
                item.file_path.stat().st_mtime
            )
        
        without_datetime.sort(key=lambda x: x.datetime_original)
        
        if without_datetime:
            print(f"\n⚠ {len(without_datetime)} files missing EXIF datetime, using file modification time")
        
        # Combine lists
        return with_datetime + without_datetime
    
    def validate_media(self, file_path: Path) -> bool:
        """
        Validate if file is a supported media format
        
        Args:
            file_path: Path to file
            
        Returns:
            True if valid, False otherwise
        """
        if not file_path.exists():
            return False
        
        if not file_path.is_file():
            return False
        
        suffix = file_path.suffix.lower()
        return suffix in self.IMAGE_FORMATS or suffix in self.VIDEO_FORMATS
    
    def get_statistics(self, media_items: List[MediaItem]) -> Dict:
        """
        Get statistics about media collection
        
        Args:
            media_items: List of MediaItem objects
            
        Returns:
            Dictionary with statistics
        """
        stats = {
            'total_files': len(media_items),
            'images': sum(1 for item in media_items if item.file_type == 'image'),
            'videos': sum(1 for item in media_items if item.file_type == 'video'),
            'total_size_mb': sum(item.file_size for item in media_items) / (1024 * 1024),
            'with_gps': sum(1 for item in media_items if item.gps_latitude and item.gps_longitude),
            'date_range': None,
            'total_video_duration': 0
        }
        
        # Calculate date range
        dates = [item.datetime_original for item in media_items if item.datetime_original]
        if dates:
            stats['date_range'] = {
                'earliest': min(dates).isoformat(),
                'latest': max(dates).isoformat()
            }
        
        # Calculate total video duration
        video_durations = [item.duration for item in media_items 
                          if item.file_type == 'video' and item.duration]
        if video_durations:
            stats['total_video_duration'] = sum(video_durations)
        
        return stats
    
    def _find_media_files(self, folder: Path) -> List[Path]:
        """Find all media files in folder recursively"""
        media_files = []
        
        for root, dirs, files in os.walk(folder):
            for file in files:
                file_path = Path(root) / file
                if self.validate_media(file_path):
                    media_files.append(file_path)
        
        return media_files
    
    def _process_file(self, file_path: Path) -> Optional[MediaItem]:
        """Process a single media file"""
        try:
            suffix = file_path.suffix.lower()
            
            # Determine file type
            if suffix in self.IMAGE_FORMATS:
                file_type = 'image'
                metadata = self.metadata_extractor.extract_image_metadata(file_path)
            else:
                file_type = 'video'
                metadata = self.metadata_extractor.extract_video_metadata(file_path)
            
            # Create MediaItem
            media_item = MediaItem(
                file_path=file_path,
                file_type=file_type,
                file_format=suffix[1:],  # Remove the dot
                file_size=file_path.stat().st_size,
                datetime_original=metadata.get('datetime_original'),
                width=metadata.get('width'),
                height=metadata.get('height'),
                duration=metadata.get('duration'),
                gps_latitude=metadata.get('gps_latitude'),
                gps_longitude=metadata.get('gps_longitude'),
                camera_make=metadata.get('camera_make'),
                camera_model=metadata.get('camera_model'),
                orientation=metadata.get('orientation')
            )
            
            return media_item
            
        except Exception as e:
            print(f"✗ Error processing {file_path.name}: {e}")
            return None
    
    def _get_cache_key(self, file_path: Path) -> str:
        """Generate cache key for file"""
        stat = file_path.stat()
        return f"{file_path.name}_{stat.st_size}_{stat.st_mtime}"
    
    def _load_cache(self) -> dict:
        """Load metadata cache"""
        cache_file = self.cache_dir / "media_cache.json"
        if cache_file.exists():
            try:
                with open(cache_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                print(f"⚠ Could not load cache: {e}")
        return {}
    
    def _save_cache(self, cache_data: dict):
        """Save metadata cache"""
        cache_file = self.cache_dir / "media_cache.json"
        try:
            # Convert all Path objects to strings before saving
            serializable_cache = {}
            for key, value in cache_data.items():
                if isinstance(value, dict):
                    # Deep copy and convert Path to string
                    item_copy = value.copy()
                    if 'file_path' in item_copy and isinstance(item_copy['file_path'], Path):
                        item_copy['file_path'] = str(item_copy['file_path'])
                    serializable_cache[key] = item_copy
                else:
                    serializable_cache[key] = value
            
            with open(cache_file, 'w') as f:
                json.dump(serializable_cache, f, indent=2)
        except Exception as e:
            print(f"⚠ Could not save cache: {e}")

    
    def export_manifest(self, media_items: List[MediaItem], output_path: Path):
        """
        Export media manifest to JSON file
        
        Args:
            media_items: List of MediaItem objects
            output_path: Path to output JSON file
        """
        manifest = {
            'created_at': datetime.now().isoformat(),
            'total_items': len(media_items),
            'statistics': self.get_statistics(media_items),
            'items': [item.to_dict() for item in media_items]
        }
        
        with open(output_path, 'w') as f:
            json.dump(manifest, f, indent=2)
        
        print(f"\n✓ Manifest exported to: {output_path}")


if __name__ == "__main__":
    # Test the ingester
    ingester = MediaIngester()
    
    # Process input folder
    media_items = ingester.process_folder(Config.INPUT_DIR)
    
    if media_items:
        # Sort by datetime
        sorted_items = ingester.sort_by_datetime(media_items)
        
        # Print statistics
        stats = ingester.get_statistics(sorted_items)
        print(f"\n{'='*70}")
        print("STATISTICS")
        print(f"{'='*70}")
        print(f"Total Files: {stats['total_files']}")
        print(f"Images: {stats['images']}")
        print(f"Videos: {stats['videos']}")
        print(f"Total Size: {stats['total_size_mb']:.2f} MB")
        print(f"Files with GPS: {stats['with_gps']}")
        if stats['date_range']:
            print(f"Date Range: {stats['date_range']['earliest']} to {stats['date_range']['latest']}")
        if stats['total_video_duration'] > 0:
            print(f"Total Video Duration: {stats['total_video_duration']:.2f} seconds")
        
        # Export manifest
        manifest_path = Config.OUTPUT_DIR / "media_manifest.json"
        ingester.export_manifest(sorted_items, manifest_path)