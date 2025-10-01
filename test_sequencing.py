# ============================================================================
# AI Video Creator - Step 4: Test Sequencing
# ============================================================================

"""
Test script for sequencing engine
"""

from pathlib import Path
import json

from media_ingester import MediaIngester
from scene_analyzer import SceneAnalysis
from sequencing_engine import SequencingEngine
from config import Config


def test_sequencing():
    """Test sequencing functionality"""
    
    print("="*70)
    print("TESTING SEQUENCING ENGINE")
    print("="*70)
    
    # Step 1: Load media
    print("\n[Step 1] Loading media files...")
    ingester = MediaIngester()
    media_items = ingester.process_folder(Config.INPUT_DIR)
    
    if not media_items:
        print("\n✗ No media files found")
        return
    
    # Sort by datetime
    sorted_items = ingester.sort_by_datetime(media_items)
    print(f"✓ Loaded {len(sorted_items)} media files")
    
    # Step 2: Load scene analyses
    print("\n[Step 2] Loading scene analyses...")
    analyses_file = Config.OUTPUT_DIR / "scene_analysis.json"
    
    if not analyses_file.exists():
        print("✗ Scene analysis file not found")
        print("Please run: python test_scene_analysis.py first")
        return
    
    with open(analyses_file, 'r') as f:
        analyses_data = json.load(f)
    
    # Convert to dictionary of SceneAnalysis objects
    scene_analyses = {}
    for path_str, analysis_dict in analyses_data['analyses'].items():
        path = Path(path_str)
        scene_analyses[path] = SceneAnalysis.from_dict(analysis_dict)
    
    print(f"✓ Loaded {len(scene_analyses)} scene analyses")
    
    # Step 3: Create sequence
    print("\n[Step 3] Creating optimized sequence...")
    engine = SequencingEngine()
    sequence = engine.create_sequence(sorted_items, scene_analyses)
    
    # Step 4: Display sequence
    print(f"\n{'='*70}")
    print("SEQUENCE PREVIEW")
    print(f"{'='*70}\n")
    
    for i, item in enumerate(sequence[:10], 1):  # Show first 10
        print(f"{i}. {item.media_item.file_path.name}")
        print(f"   Start: {item.start_time:.1f}s | Duration: {item.duration:.1f}s")
        print(f"   Transition: {item.transition_in} ({item.transition_duration:.1f}s)")
        print(f"   Group: {item.group_id}")
        print(f"   Caption: {item.scene_analysis.caption}")
        print()
    
    if len(sequence) > 10:
        print(f"... and {len(sequence) - 10} more items\n")
    
    # Step 5: Statistics
    stats = engine.get_sequence_statistics(sequence)
    print(f"{'='*70}")
    print("SEQUENCE STATISTICS")
    print(f"{'='*70}")
    print(f"Total Items: {stats['total_items']}")
    print(f"Total Duration: {stats['total_duration_minutes']:.2f} minutes")
    print(f"Scene Groups: {stats['num_groups']}")
    print(f"Average Duration: {stats['avg_item_duration']:.2f} seconds")
    print(f"Range: {stats['shortest_item']:.2f}s - {stats['longest_item']:.2f}s")
    
    print(f"\nTransition Types:")
    for trans, count in stats['transition_counts'].items():
        print(f"  {trans}: {count}")
    
    # Step 6: Export
    print(f"\n{'='*70}")
    print("EXPORTING SEQUENCE")
    print(f"{'='*70}")
    
    output_path = Config.OUTPUT_DIR / "video_sequence.json"
    engine.export_sequence(sequence, output_path)
    
    print(f"\n{'='*70}")
    print("✓ TEST COMPLETE")
    print(f"{'='*70}")
    print("\nSequence saved to:", output_path)
    print("\nNext steps:")
    print("1. Review the sequence")
    print("2. Adjust durations/transitions if needed")
    print("3. Ready for Step 5: Audio Generation")


if __name__ == "__main__":
    test_sequencing()