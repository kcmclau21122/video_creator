# ============================================================================
# AI Video Creator - Step 5: Test Audio Generation (CORRECTED)
# ============================================================================

"""
Test script for audio generation - generates ONE soundtrack for entire video
"""

from pathlib import Path
import json

from audio_generator import AudioGenerator
from sequencing_engine import SequenceItem
from scene_analyzer import SceneAnalysis
from media_ingester import MediaItem
from config import Config


def reconstruct_sequence_items(sequence_data: dict) -> list:
    """Reconstruct SequenceItem objects from JSON data"""
    items = []
    
    for item_data in sequence_data['items']:
        # Create minimal objects for audio generation
        # We only need the scene analysis data
        class SimpleAnalysis:
            def __init__(self, data):
                self.mood = data.get('mood', '')
                self.scene_type = data.get('scene_type', '')
                self.caption = data.get('caption', '')
        
        class SimpleItem:
            def __init__(self, data):
                self.scene_analysis = SimpleAnalysis(data)
                self.start_time = data['start_time']
                self.duration = data['duration']
        
        items.append(SimpleItem(item_data))
    
    return items


def test_audio_generation():
    """Test audio generation functionality"""
    
    print("="*70)
    print("TESTING AUDIO GENERATION")
    print("="*70)
    
    # Load sequence
    print("\n[Step 1] Loading video sequence...")
    sequence_file = Config.OUTPUT_DIR / "video_sequence.json"
    
    if not sequence_file.exists():
        print("✗ Sequence file not found")
        print("Please run: python test_sequencing.py first")
        return
    
    with open(sequence_file, 'r') as f:
        sequence_data = json.load(f)
    
    total_duration = sequence_data['total_duration']
    num_items = sequence_data['total_items']
    
    print(f"✓ Loaded sequence")
    print(f"  Items: {num_items}")
    print(f"  Duration: {total_duration:.2f}s ({total_duration/60:.2f} min)")
    
    # Reconstruct sequence items
    print("\n[Step 2] Analyzing scenes for music generation...")
    sequence_items = reconstruct_sequence_items(sequence_data)
    print(f"✓ Prepared {len(sequence_items)} scene analyses")
    
    # For testing, limit to shorter duration
    test_duration = min(30.0, total_duration)
    print(f"\n[Step 3] Generating soundtrack...")
    print(f"  Full video duration: {total_duration:.2f}s")
    print(f"  Test duration: {test_duration:.2f}s (for faster testing)")
    
    # Choose model
    model_key = 'musicgen-small'  # Fast for testing
    print(f"  Using model: {model_key}")
    
    # Create generator
    generator = AudioGenerator(model_key=model_key)
    
    # Generate ONE soundtrack for entire sequence
    output_path = Config.OUTPUT_DIR / "soundtrack.wav"
    
    try:
        soundtrack_path = generator.generate_soundtrack(
            sequence=sequence_items,
            duration=test_duration,
            output_path=output_path
        )
        
        actual_duration = output_path.stat().st_size / (generator.sample_rate * 4)  # 32-bit float
        
        print(f"\n{'='*70}")
        print("AUDIO GENERATION COMPLETE")
        print(f"{'='*70}")
        print(f"Output: {soundtrack_path}")
        print(f"Duration: {actual_duration:.2f} seconds")
        print(f"Sample rate: {generator.sample_rate} Hz")
        print(f"File size: {output_path.stat().st_size / (1024*1024):.2f} MB")
        
    except Exception as e:
        print(f"\n✗ Error generating audio: {e}")
        import traceback
        traceback.print_exc()
    finally:
        generator.unload_model()
    
    print(f"\n{'='*70}")
    print("✓ TEST COMPLETE")
    print(f"{'='*70}")
    print("\nGenerated ONE continuous soundtrack for entire video")
    print("\nNext steps:")
    print("1. Listen to the generated audio")
    print("2. Generate full-length version if satisfied")
    print("3. Ready for Step 6: Video Composition")
    print("\nTo play the audio:")
    print(f"  Open: {output_path}")


if __name__ == "__main__":
    test_audio_generation()