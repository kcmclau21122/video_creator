# ============================================================================
# AI Video Creator - Step 6: Test Video Composition
# ============================================================================

"""
Test script for video composition
"""

from pathlib import Path
import json

from video_composer import VideoComposer
from sequencing_engine import SequenceItem
from scene_analyzer import SceneAnalysis
from media_ingester import MediaItem, MediaIngester
from config import Config


def reconstruct_sequence(sequence_data: dict, media_items: list) -> list[SequenceItem]:
    """Reconstruct SequenceItem objects from JSON"""
    from typing import List
    
    # Create lookup by path
    media_lookup = {str(item.file_path): item for item in media_items}
    
    sequence = []
    for item_data in sequence_data['items']:
        path_str = item_data['media_item_path']
        
        if path_str not in media_lookup:
            print(f"Warning: {path_str} not found in media items")
            continue
        
        media_item = media_lookup[path_str]
        
        # Create simple scene analysis
        class SimpleAnalysis:
            def __init__(self, data):
                self.caption = data.get('caption', '')
                self.scene_type = data.get('scene_type', '')
                self.mood = data.get('mood', '')
        
        seq_item = SequenceItem(
            media_item=media_item,
            scene_analysis=SimpleAnalysis(item_data),
            sequence_index=item_data['sequence_index'],
            start_time=item_data['start_time'],
            duration=item_data['duration'],
            transition_in=item_data['transition_in'],
            transition_duration=item_data['transition_duration'],
            group_id=item_data['group_id']
        )
        
        sequence.append(seq_item)
    
    return sequence


def test_video_composition():
    """Test video composition functionality"""
    
    print("="*70)
    print("TESTING VIDEO COMPOSITION")
    print("="*70)
    
    # Step 1: Load media items
    print("\n[Step 1] Loading media files...")
    ingester = MediaIngester()
    media_items = ingester.process_folder(Config.INPUT_DIR)
    
    if not media_items:
        print("No media files found")
        return
    
    print(f"Loaded {len(media_items)} media files")
    
    # Step 2: Load sequence
    print("\n[Step 2] Loading video sequence...")
    sequence_file = Config.OUTPUT_DIR / "video_sequence.json"
    
    if not sequence_file.exists():
        print("Sequence file not found")
        print("Run: python test_sequencing.py first")
        return
    
    with open(sequence_file, 'r') as f:
        sequence_data = json.load(f)
    
    print(f"Loaded sequence: {sequence_data['total_items']} items")
    
    # Reconstruct sequence
    sequence = reconstruct_sequence(sequence_data, media_items)
    print(f"Reconstructed {len(sequence)} sequence items")
    
    # Step 3: Check for audio
    print("\n[Step 3] Checking for audio...")
    audio_path = Config.OUTPUT_DIR / "soundtrack.wav"
    
    if audio_path.exists():
        print(f"Found audio: {audio_path.name}")
        size_mb = audio_path.stat().st_size / (1024 * 1024)
        print(f"Size: {size_mb:.2f} MB")
    else:
        print("No audio found (will create video without sound)")
        print("Run: python test_audio_generation.py to generate audio first")
        audio_path = None
    
    # Step 4: Compose video
    print("\n[Step 4] Composing video...")
    print("\nWARNING: This will take several minutes!")
    print("Progress will be shown below...")
    
    confirm = input("\nProceed with video composition? (y/n): ").lower().strip()
    
    if confirm != 'y':
        print("\nCancelled")
        return
    
    composer = VideoComposer()
    
    output_path = Config.OUTPUT_DIR / "final_video.mp4"
    
    try:
        result_path = composer.compose_video(
            sequence=sequence,
            audio_path=audio_path,
            output_path=output_path
        )
        
        # Get file info
        file_size = result_path.stat().st_size / (1024 * 1024)
        
        print(f"\n{'='*70}")
        print("VIDEO COMPOSITION COMPLETE")
        print(f"{'='*70}")
        print(f"Output: {result_path}")
        print(f"Size: {file_size:.2f} MB")
        print(f"Resolution: {Config.OUTPUT_RESOLUTION[0]}x{Config.OUTPUT_RESOLUTION[1]}")
        print(f"FPS: {Config.OUTPUT_FPS}")
        
        print(f"\n{'='*70}")
        print("READY FOR YOUTUBE")
        print(f"{'='*70}")
        print("\nYour video is ready to upload!")
        print(f"\nTo play: {result_path}")
        
    except Exception as e:
        print(f"\nError composing video: {e}")
        import traceback
        traceback.print_exc()
    
    print(f"\n{'='*70}")
    print("TEST COMPLETE")
    print(f"{'='*70}")

def test_video_composition_auto():
    """Automatic video composition without prompts (for run_all_steps.py)"""
    
    print("="*70)
    print("TESTING VIDEO COMPOSITION")
    print("="*70)
    
    # Step 1: Load media items
    print("\n[Step 1] Loading media files...")
    ingester = MediaIngester()
    media_items = ingester.process_folder(Config.INPUT_DIR)
    
    if not media_items:
        print("No media files found")
        return
    
    print(f"Loaded {len(media_items)} media files")
    
    # Step 2: Load sequence
    print("\n[Step 2] Loading video sequence...")
    sequence_file = Config.OUTPUT_DIR / "video_sequence.json"
    
    if not sequence_file.exists():
        print("Sequence file not found")
        return
    
    with open(sequence_file, 'r') as f:
        sequence_data = json.load(f)
    
    print(f"Loaded sequence: {sequence_data['total_items']} items")
    
    # Reconstruct sequence
    sequence = reconstruct_sequence(sequence_data, media_items)
    print(f"Reconstructed {len(sequence)} sequence items")
    
    # Step 3: Check for audio
    print("\n[Step 3] Checking for audio...")
    audio_path = Config.OUTPUT_DIR / "soundtrack.wav"
    
    if audio_path.exists():
        print(f"Found audio: {audio_path.name}")
    else:
        print("No audio found (creating silent video)")
        audio_path = None
    
    # Step 4: Compose video
    print("\n[Step 4] Composing video...")
    
    composer = VideoComposer()
    output_path = Config.OUTPUT_DIR / "final_video.mp4"
    
    result_path = composer.compose_video(
        sequence=sequence,
        audio_path=audio_path,
        output_path=output_path
    )
    
    file_size = result_path.stat().st_size / (1024 * 1024)
    
    print(f"\n{'='*70}")
    print("VIDEO COMPOSITION COMPLETE")
    print(f"{'='*70}")
    print(f"Output: {result_path}")
    print(f"Size: {file_size:.2f} MB")

if __name__ == "__main__":
    test_video_composition()