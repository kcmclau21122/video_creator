#!/usr/bin/env python3
# ============================================================================
# AI Video Creator - Duplicate Detection Module
# ============================================================================

"""
Duplicate Detection Module
Detects and removes duplicate media files based on:
1. File hash (exact duplicates)
2. Image perceptual hash (similar images)
3. HEIC/JPEG pairs (converted duplicates)
"""

from pathlib import Path
from typing import List, Dict, Set, Tuple, Optional
import hashlib
from PIL import Image
import imagehash
from dataclasses import dataclass
from media_ingester import MediaItem


@dataclass
class DuplicateGroup:
    """Group of duplicate files"""
    primary_file: Path  # The file to keep
    duplicates: List[Path]  # Files to remove
    reason: str  # Why they're duplicates
    similarity: float  # How similar (0.0-1.0)


class DuplicateDetector:
    """Detect duplicate media files"""
    
    def __init__(self, 
                 perceptual_threshold: int = 5,  # Lower = more strict
                 prefer_jpeg_over_heic: bool = True):
        """
        Initialize duplicate detector
        
        Args:
            perceptual_threshold: Maximum hamming distance for perceptual hash
            prefer_jpeg_over_heic: Prefer JPEG over HEIC when both exist
        """
        self.perceptual_threshold = perceptual_threshold
        self.prefer_jpeg_over_heic = prefer_jpeg_over_heic
        
    def find_duplicates(self, media_items: List[MediaItem]) -> List[DuplicateGroup]:
        """
        Find all duplicates in media collection
        
        Args:
            media_items: List of MediaItem objects
            
        Returns:
            List of DuplicateGroup objects
        """
        print(f"\n{'='*70}")
        print("DETECTING DUPLICATES")
        print(f"{'='*70}")
        print(f"Checking {len(media_items)} files...")
        
        duplicate_groups = []
        processed_files = set()
        
        # Step 1: Find exact duplicates (by file hash)
        print("\n[1/3] Checking for exact duplicates...")
        exact_dupes = self._find_exact_duplicates(media_items)
        duplicate_groups.extend(exact_dupes)
        for group in exact_dupes:
            processed_files.add(group.primary_file)
            processed_files.update(group.duplicates)
        print(f"  Found {len(exact_dupes)} exact duplicate groups")
        
        # Step 2: Find HEIC/JPEG pairs
        print("\n[2/3] Checking for HEIC/JPEG pairs...")
        heic_jpeg_pairs = self._find_heic_jpeg_pairs(media_items)
        duplicate_groups.extend(heic_jpeg_pairs)
        for group in heic_jpeg_pairs:
            processed_files.add(group.primary_file)
            processed_files.update(group.duplicates)
        print(f"  Found {len(heic_jpeg_pairs)} HEIC/JPEG pairs")
        
        # Step 3: Find perceptually similar images
        print("\n[3/3] Checking for similar images...")
        # Only check images not already marked as duplicates
        remaining_images = [
            item for item in media_items 
            if item.file_type == 'image' 
            and item.file_path not in processed_files
        ]
        similar_images = self._find_similar_images(remaining_images)
        duplicate_groups.extend(similar_images)
        print(f"  Found {len(similar_images)} similar image groups")
        
        # Summary
        total_duplicates = sum(len(g.duplicates) for g in duplicate_groups)
        print(f"\n{'='*70}")
        print(f"SUMMARY: Found {total_duplicates} duplicate files in {len(duplicate_groups)} groups")
        print(f"{'='*70}")
        
        return duplicate_groups
    
    def remove_duplicates(self, 
                         media_items: List[MediaItem],
                         duplicate_groups: List[DuplicateGroup]) -> List[MediaItem]:
        """
        Remove duplicates from media items list
        
        Args:
            media_items: Original list of media items
            duplicate_groups: Groups of duplicates found
            
        Returns:
            Filtered list with duplicates removed
        """
        # Create set of all duplicate files to remove
        duplicates_to_remove = set()
        for group in duplicate_groups:
            duplicates_to_remove.update(group.duplicates)
        
        # Filter out duplicates
        filtered_items = [
            item for item in media_items 
            if item.file_path not in duplicates_to_remove
        ]
        
        removed_count = len(media_items) - len(filtered_items)
        print(f"\n✓ Removed {removed_count} duplicate files")
        print(f"✓ Keeping {len(filtered_items)} unique files")
        
        return filtered_items
    
    def _find_exact_duplicates(self, media_items: List[MediaItem]) -> List[DuplicateGroup]:
        """Find files with identical content (by MD5 hash)"""
        hash_to_files: Dict[str, List[MediaItem]] = {}
        
        # Group files by hash
        for item in media_items:
            file_hash = self._compute_file_hash(item.file_path)
            if file_hash:
                if file_hash not in hash_to_files:
                    hash_to_files[file_hash] = []
                hash_to_files[file_hash].append(item)
        
        # Create duplicate groups for files with same hash
        duplicate_groups = []
        for file_hash, items in hash_to_files.items():
            if len(items) > 1:
                # Keep the one with the earliest date, or first alphabetically
                primary = self._select_primary_file(items)
                duplicates = [item.file_path for item in items if item != primary]
                
                group = DuplicateGroup(
                    primary_file=primary.file_path,
                    duplicates=duplicates,
                    reason="Exact duplicate (identical file content)",
                    similarity=1.0
                )
                duplicate_groups.append(group)
        
        return duplicate_groups
    
    def _find_heic_jpeg_pairs(self, media_items: List[MediaItem]) -> List[DuplicateGroup]:
        """Find HEIC files that have been converted to JPEG"""
        duplicate_groups = []
        
        # Group files by stem (filename without extension)
        stem_to_files: Dict[str, List[MediaItem]] = {}
        for item in media_items:
            if item.file_type == 'image':
                stem = item.file_path.stem
                # Handle .heic.jpeg pattern
                if stem.endswith('.heic'):
                    stem = stem[:-5]  # Remove .heic from stem
                
                if stem not in stem_to_files:
                    stem_to_files[stem] = []
                stem_to_files[stem].append(item)
        
        # Find HEIC/JPEG pairs
        for stem, items in stem_to_files.items():
            if len(items) < 2:
                continue
            
            # Look for HEIC and JPEG/JPG versions
            heic_files = [
                item for item in items 
                if item.file_path.suffix.lower() in ['.heic', '.heif']
            ]
            jpeg_files = [
                item for item in items 
                if item.file_path.suffix.lower() in ['.jpg', '.jpeg']
            ]
            
            if heic_files and jpeg_files:
                # We have both HEIC and JPEG versions
                if self.prefer_jpeg_over_heic:
                    # Keep JPEG, remove HEIC
                    primary = jpeg_files[0]
                    duplicates = [item.file_path for item in heic_files]
                    duplicates.extend([item.file_path for item in jpeg_files[1:]])
                else:
                    # Keep HEIC, remove JPEG
                    primary = heic_files[0]
                    duplicates = [item.file_path for item in jpeg_files]
                    duplicates.extend([item.file_path for item in heic_files[1:]])
                
                group = DuplicateGroup(
                    primary_file=primary.file_path,
                    duplicates=duplicates,
                    reason="HEIC/JPEG pair (same image, different format)",
                    similarity=1.0
                )
                duplicate_groups.append(group)
        
        return duplicate_groups
    
    def _find_similar_images(self, media_items: List[MediaItem]) -> List[DuplicateGroup]:
        """Find perceptually similar images using perceptual hashing"""
        if not media_items:
            return []
        
        # Compute perceptual hashes - FIX: Use Path as key instead of MediaItem
        image_hashes: Dict[Path, imagehash.ImageHash] = {}
        path_to_item: Dict[Path, MediaItem] = {}
        
        for item in media_items:
            try:
                img = Image.open(item.file_path)
                # Use difference hash (dHash) - good for finding similar images
                phash = imagehash.dhash(img)
                image_hashes[item.file_path] = phash
                path_to_item[item.file_path] = item
            except Exception as e:
                print(f"  ⚠️  Could not hash {item.file_path.name}: {e}")
                continue
        
        # Find similar pairs
        duplicate_groups = []
        processed = set()
        
        paths_list = list(image_hashes.keys())
        for i, path1 in enumerate(paths_list):
            if path1 in processed:
                continue
            
            hash1 = image_hashes[path1]
            similar_paths = []
            
            # Compare with remaining items
            for path2 in paths_list[i+1:]:
                if path2 in processed:
                    continue
                
                hash2 = image_hashes[path2]
                distance = hash1 - hash2  # Hamming distance
                
                if distance <= self.perceptual_threshold:
                    similar_paths.append(path2)
                    processed.add(path2)
            
            if similar_paths:
                # Keep the highest quality/earliest date
                all_paths = [path1] + similar_paths
                all_items = [path_to_item[p] for p in all_paths]
                primary = self._select_primary_file(all_items)
                duplicates = [p for p in all_paths if p != primary.file_path]
                
                # Calculate similarity
                avg_distance = sum(
                    image_hashes[primary.file_path] - image_hashes[path] 
                    for path in similar_paths
                ) / len(similar_paths)
                similarity = 1.0 - (avg_distance / 64.0)  # Normalize to 0-1
                
                group = DuplicateGroup(
                    primary_file=primary.file_path,
                    duplicates=duplicates,
                    reason=f"Perceptually similar (distance: {avg_distance:.1f})",
                    similarity=similarity
                )
                duplicate_groups.append(group)
                processed.add(path1)
        
        return duplicate_groups
    
    def _compute_file_hash(self, file_path: Path, algorithm: str = 'md5') -> Optional[str]:
        """Compute hash of file content"""
        try:
            hash_obj = hashlib.new(algorithm)
            with open(file_path, 'rb') as f:
                # Read in chunks for large files
                while chunk := f.read(8192):
                    hash_obj.update(chunk)
            return hash_obj.hexdigest()
        except Exception as e:
            print(f"  ⚠️  Could not hash {file_path.name}: {e}")
            return None
    
    def _select_primary_file(self, items: List[MediaItem]) -> MediaItem:
        """
        Select which file to keep from a group of duplicates
        Priority:
        1. Largest file size (usually highest quality)
        2. Earliest datetime
        3. First alphabetically
        """
        if not items:
            raise ValueError("Empty items list")
        
        if len(items) == 1:
            return items[0]
        
        # Sort by: size (desc), datetime (asc), filename (asc)
        def sort_key(item):
            return (
                -item.file_size,  # Negative for descending (largest first)
                item.datetime_original or datetime.max,  # Earliest date
                item.file_path.name.lower()  # Alphabetically
            )
        
        from datetime import datetime
        sorted_items = sorted(items, key=sort_key)
        return sorted_items[0]
    
    def export_report(self, duplicate_groups: List[DuplicateGroup], output_path: Path):
        """Export duplicate detection report"""
        import json
        
        report = {
            'total_groups': len(duplicate_groups),
            'total_duplicates': sum(len(g.duplicates) for g in duplicate_groups),
            'groups': []
        }
        
        for group in duplicate_groups:
            report['groups'].append({
                'primary_file': str(group.primary_file),
                'duplicates': [str(d) for d in group.duplicates],
                'reason': group.reason,
                'similarity': group.similarity
            })
        
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\n✓ Duplicate report saved to: {output_path}")


def print_duplicate_report(duplicate_groups: List[DuplicateGroup]):
    """Print detailed report of duplicates found"""
    if not duplicate_groups:
        print("\n✓ No duplicates found!")
        return
    
    print(f"\n{'='*70}")
    print("DUPLICATE DETECTION REPORT")
    print(f"{'='*70}\n")
    
    for i, group in enumerate(duplicate_groups, 1):
        print(f"Group {i}: {group.reason}")
        print(f"  Similarity: {group.similarity*100:.1f}%")
        print(f"  → KEEP: {group.primary_file.name}")
        print(f"  → REMOVE ({len(group.duplicates)}):")
        for dup in group.duplicates:
            print(f"     - {dup.name}")
        print()


def remove_duplicates_interactive(media_items: List[MediaItem]) -> List[MediaItem]:
    """
    Interactive duplicate removal
    
    Args:
        media_items: List of media items
        
    Returns:
        Filtered list with duplicates removed
    """
    detector = DuplicateDetector(perceptual_threshold=5, prefer_jpeg_over_heic=True)
    duplicate_groups = detector.find_duplicates(media_items)
    
    if not duplicate_groups:
        return media_items
    
    # Show report
    print_duplicate_report(duplicate_groups)
    
    # Ask for confirmation
    print(f"{'='*70}")
    total_to_remove = sum(len(g.duplicates) for g in duplicate_groups)
    print(f"Found {total_to_remove} duplicates to remove")
    print(f"This will keep {len(media_items) - total_to_remove} unique files")
    
    response = input("\nRemove duplicates? (y/n): ").lower().strip()
    
    if response == 'y':
        filtered_items = detector.remove_duplicates(media_items, duplicate_groups)
        
        # Export report
        report_path = Config.OUTPUT_DIR / "duplicate_report.json"
        detector.export_report(duplicate_groups, report_path)
        
        return filtered_items
    else:
        print("Keeping all files (including duplicates)")
        return media_items


if __name__ == "__main__":
    # Test duplicate detection
    from config import Config
    from media_ingester import MediaIngester
    
    print("Testing Duplicate Detection...")
    
    # Ingest media
    ingester = MediaIngester()
    media_items = ingester.process_folder(Config.INPUT_DIR)
    
    if not media_items:
        print("No media files found")
        exit(1)
    
    print(f"\nOriginal count: {len(media_items)} files")
    
    # Detect and remove duplicates
    filtered_items = remove_duplicates_interactive(media_items)
    
    print(f"\nFinal count: {len(filtered_items)} unique files")