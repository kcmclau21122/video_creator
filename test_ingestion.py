# ============================================================================
# AI Video Creator - Step 2: Test Ingestion
# ============================================================================

"""
Test script for media ingestion module
"""

from pathlib import Path
from media_ingester import MediaIngester
from config import Config


def test_ingestion():
    """Test media ingestion functionality"""
    
    print("="*70)
    print("TESTING MEDIA INGESTION MODULE")
    print("="*70)
    
    # Initialize ingester
    ingester = MediaIngester()
    
    # Check if input folder has files
    input_folder = Config.INPUT_DIR
    
    if not input_folder.exists():
        print(f"\n✗ Input folder does not exist: {input_folder}")
        print("Please create it and add some image/video files")
        return
    
    # Count files
    all_files = list(input_folder.iterdir())
    if not all_files:
        print(f"\n✗ Input folder is empty: {input_folder}")
        print("Please add some image/video files")
        return
    
    print(f"\n✓ Input folder: {input_folder}")
    print(f"✓ Found {len(all_files)} files")
    
    # Process media
    print("\nProcessing media files...")
    media_items = ingester.process_folder(input_folder)
    
    if not media_items:
        print("\n✗ No valid media files found")
        return
    
    # Sort by datetime
    print("\nSorting by datetime...")
    sorted_items = ingester.sort_by_datetime(media_items)
    
    # Display results
    print(f"\n{'='*70}")
    print("PROCESSED MEDIA FILES")
    print(f"{'='*70}\n")
    
    for i, item in enumerate(sorted_items[:10], 1):  # Show first 10
        print(f"{i}. {item.file_path.name}")
        print(f"   Type: {item.file_type} ({item.file_format})")
        print(f"   Size: {item.file_size / 1024:.2f} KB")
        print(f"   Dimensions: {item.width}x{item.height}")
        if item.datetime_original:
            print(f"   Date: {item.datetime_original.strftime('%Y-%m-%d %H:%M:%S')}")
        if item.duration:
            print(f"   Duration: {item.duration:.2f} seconds")
        if item.camera_make or item.camera_model:
            print(f"   Camera: {item.camera_make} {item.camera_model}")
        if item.gps_latitude and item.gps_longitude:
            print(f"   GPS: {item.gps_latitude:.6f}, {item.gps_longitude:.6f}")
        print()
    
    if len(sorted_items) > 10:
        print(f"... and {len(sorted_items) - 10} more files\n")
    
    # Statistics
    stats = ingester.get_statistics(sorted_items)
    print(f"{'='*70}")
    print("STATISTICS")
    print(f"{'='*70}")
    print(f"Total Files: {stats['total_files']}")
    print(f"Images: {stats['images']}")
    print(f"Videos: {stats['videos']}")
    print(f"Total Size: {stats['total_size_mb']:.2f} MB")
    print(f"Files with GPS: {stats['with_gps']}")
    
    if stats['date_range']:
        print(f"Earliest: {stats['date_range']['earliest']}")
        print(f"Latest: {stats['date_range']['latest']}")
    
    if stats['total_video_duration'] > 0:
        print(f"Total Video Duration: {stats['total_video_duration']:.2f} seconds")
    
    # Export manifest
    print(f"\n{'='*70}")
    print("EXPORTING MANIFEST")
    print(f"{'='*70}")
    
    manifest_path = Config.OUTPUT_DIR / "media_manifest.json"
    ingester.export_manifest(sorted_items, manifest_path)
    
    print(f"\n{'='*70}")
    print("✓ TEST COMPLETE")
    print(f"{'='*70}")
    print("\nNext steps:")
    print("1. Check the manifest file:", manifest_path)
    print("2. Verify media items are sorted correctly")
    print("3. Ready for Step 3: Scene Analysis")


if __name__ == "__main__":
    test_ingestion()