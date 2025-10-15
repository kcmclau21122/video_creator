#!/usr/bin/env python3
# ============================================================================
# AI Video Creator - Main Entry Point with Animation Options (FIXED)
# Choose between slideshow or TRUE animation using AnimateDiff
# ============================================================================

"""
Main pipeline with animation mode selection
Usage:
    # Slideshow mode (fast, Ken Burns effects)
    python main_with_animation_option.py --input photos/ --mode slideshow
    
    # Animated mode (slow, TRUE animation with AnimateDiff)
    python main_with_animation_option.py --input photos/ --mode animated
    
    # Both modes (creates two separate videos)
    python main_with_animation_option.py --input photos/ --mode both
"""

import sys
import argparse
from pathlib import Path
from config import Config
from media_ingester import MediaIngester
from scene_analyzer import SceneAnalyzer
from sequencing_engine import SequencingEngine
# FIXED: Explicitly import from the correct module
from image_animator import ImageAnimator as AnimateDiffAnimator
from audio_generator import AudioGenerator
from video_composer import VideoComposer
from duplicate_detector import DuplicateDetector
import json
import time


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Create video from photos and clips with animation options',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Fast slideshow with effects
  python main_with_animation_option.py --input photos/ --mode slideshow
  
  # TRUE animation (takes longer but creates real movement)
  python main_with_animation_option.py --input photos/ --mode animated
  
  # Create both versions
  python main_with_animation_option.py --input photos/ --mode both
"""
    )
    
    # Input/output
    parser.add_argument(
        '--input',
        type=str,
        default='input',
        help='Input directory containing photos and videos'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default='output',
        help='Output directory for generated videos'
    )
    
    # Mode selection
    parser.add_argument(
        '--mode',
        choices=['slideshow', 'animated', 'both'],
        default='slideshow',
        help='Video creation mode (default: slideshow)'
    )
    
    # Animation options
    parser.add_argument(
        '--style',
        choices=['anime', 'cartoon', 'realistic', 'stylized'],
        default='realistic',
        help='Animation style (only for animated mode)'
    )
    
    # Audio options
    parser.add_argument(
        '--music-volume',
        type=float,
        default=0.5,
        help='Background music volume (0.0-1.0)'
    )
    
    parser.add_argument(
        '--fade',
        type=int,
        default=3,
        help='Audio fade out duration in seconds'
    )
    
    # Processing options
    parser.add_argument(
        '--no-duplicates',
        action='store_true',
        help='Skip duplicate detection'
    )
    
    parser.add_argument(
        '--motion',
        type=int,
        default=127,
        help='Motion intensity for animation (1-255, default: 127)'
    )
    
    return parser.parse_args()

def reconstruct_sequence(sequence_data: dict, sorted_items: list) -> list:
    """
    Reconstruct SequenceItem objects from JSON data
    
    NOTE: This function is useful if you need to load a sequence from a JSON file.
    However, if you already have the sequence as SequenceItem objects in memory,
    you can pass them directly to compose_video() without reconstruction.
    
    Args:
        sequence_data: Loaded JSON sequence data
        sorted_items: List of MediaItem objects
        
    Returns:
        List of SequenceItem objects
    """
    from sequencing_engine import SequenceItem
    
    # Create lookup by path (as string)
    media_lookup = {str(item.file_path): item for item in sorted_items}
    
    sequence = []
    for item_data in sequence_data['items']:
        path_str = item_data['media_item_path']
        
        if path_str not in media_lookup:
            print(f"Warning: {path_str} not found in media items")
            continue
        
        media_item = media_lookup[path_str]
        
        # Create simple scene analysis from JSON data
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

def create_slideshow(sorted_items, scene_analyses, args):
    """Create slideshow video with Ken Burns effects"""
    print("\n" + "="*70)
    print("CREATING SLIDESHOW VERSION")
    print("="*70)
    
    # Create sequence
    engine = SequencingEngine()
    sequence = engine.create_sequence(sorted_items, scene_analyses)
    
    sequence_path = Config.OUTPUT_DIR / "slideshow_sequence.json"
    engine.export_sequence(sequence, sequence_path)
    
    stats = engine.get_sequence_statistics(sequence)
    total_duration = stats['total_duration_seconds']
    
    print(f"✓ Sequence duration: {stats['total_duration_minutes']:.2f} minutes")
    
    # Prepare audio
    audio_gen = AudioGenerator(fade_duration=args.fade)
    music_files = audio_gen.find_music_files(Path(args.input))
    
    audio_path = Config.OUTPUT_DIR / "slideshow_audio.wav"
    
    if music_files:
        print(f"Using music: {music_files[0].name}")
        audio_path = audio_gen.generate_or_use_audio(
            input_folder=Config.INPUT_DIR,
            sequence=sorted_items,
            duration=total_duration,
            output_path=audio_path,
            force_generate=False
        )
    else:
        print("Generating AI music...")
        audio_gen.generate_background_music(
            duration_seconds=total_duration,
            output_path=audio_path
        )
    
    # Compose video
    composer = VideoComposer()
    output_path = Config.OUTPUT_DIR / "slideshow_movie.mp4"
    
    try:
        # FIXED: Pass the sequence list directly (it's already created above)
        final_video = composer.compose_video(
            sequence=sequence,  # Pass list, not path
            audio_path=audio_path,
            output_path=output_path
            # music_volume parameter removed - not supported
        )
        
        print(f"\n✓ Slideshow created: {final_video}")
        return final_video
        
    except Exception as e:
        print(f"\n❌ Slideshow composition failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def create_animated(sorted_items, scene_analyses, args):
    """
    Create animated video from images.
    FIXED: Properly loads and passes sequence objects to VideoComposer
    """
    # Separate images and videos for processing
    image_items = [item for item in sorted_items if item.file_type == 'image']
    video_items = [item for item in sorted_items if item.file_type == 'video']

    if not image_items:
        print("\n⚠️  No images found to animate. If you have videos, a movie will be created from them.")
        if not video_items:
            return None
    else:
        print(f"\nImages to animate: {len(image_items)}")
        print(f"Video clips to include: {len(video_items)}")
        print(f"⏱  Each image takes 60-120 seconds to animate.")
        print(f"⏱  Estimated animation time: ~{len(image_items) * 1.5:.1f} minutes")

    animated_clips = []
    if image_items:
        # Initialize and run the animator
        animator = AnimateDiffAnimator(
            animation_style=args.style,
            use_interpolation=True,
            enable_depth=False
        )
        animator.load_models()

        animated_dir = Config.OUTPUT_DIR / "animated_clips"
        animated_dir.mkdir(parents=True, exist_ok=True)

        image_paths = [Path(item.file_path) for item in image_items]
        
        # Create a dictionary for scene analyses with Path keys for the animator
        scene_dict_for_animator = {Path(p): scene_analyses.get(p) for p in [i.file_path for i in image_items]}

        print("\n🎬 Starting animation process...")
        start_time = time.time()
        animated_clips = animator.batch_animate_images(
            image_paths=image_paths,
            scene_analyses=scene_dict_for_animator,
            output_dir=animated_dir
        )
        elapsed_time = time.time() - start_time
        print(f"\n✓ Animation completed in {elapsed_time/60:.1f} minutes")
        animator.unload_models()

    # ====================================================================
    # ROBUST REMAPPING LOGIC
    # ====================================================================
    # 1. Create a map from original image paths to new animated clip paths
    path_map = {
        image_items[i].file_path: str(animated_clips[i])
        for i in range(len(animated_clips))
    }

    # 2. Create a new, updated scene analyses dictionary
    updated_scene_analyses = {}
    for original_path, analysis in scene_analyses.items():
        new_path_key = path_map.get(original_path, original_path)
        updated_scene_analyses[new_path_key] = analysis

    # 3. Modify the original `sorted_items` list in-place
    for item in sorted_items:
        if item.file_path in path_map:
            item.file_path = path_map[item.file_path]
            item.file_type = 'video'
            item.file_format = 'mp4'
    
    # Now `sorted_items` contains updated paths, and `updated_scene_analyses`
    # has keys that match those new paths.
    # ====================================================================

    print("\n" + "="*70)
    print("CREATING SEQUENCE")
    print("="*70)

    engine = SequencingEngine()
    sequence = engine.create_sequence(sorted_items, updated_scene_analyses)
    
    sequence_path = Config.OUTPUT_DIR / "animated_sequence.json"
    engine.export_sequence(sequence, sequence_path)
    
    stats = engine.get_sequence_statistics(sequence)

    if not sequence or stats.get('total_items', 0) == 0:
        print("\n❌ ERROR: Sequence is empty after processing.")
        print("⚠️  This can happen if analysis data is missing or corrupt.")
        return None

    total_duration = stats.get('total_duration_seconds', 0)
    print(f"✓ Sequence created: {stats['total_items']} clips")
    print(f"Duration: {stats['total_duration_minutes']:.2f} minutes")

    # Generate Audio
    print("\n" + "="*70)
    print("GENERATING AUDIO")
    print("="*70)
    
    audio_gen = AudioGenerator(fade_duration=args.fade)
    audio_path = Config.OUTPUT_DIR / "animated_audio.wav"
    
    audio_path = audio_gen.generate_or_use_audio(
        input_folder=Config.INPUT_DIR,
        sequence=sequence,
        duration=total_duration,
        output_path=audio_path,
        force_generate=False
    )
    
    print("\n" + "="*70)
    print("COMPOSING ANIMATED VIDEO")
    print("="*70)
    
    try:
        composer = VideoComposer()
        output_path = Config.OUTPUT_DIR / args.output if hasattr(args, 'output') and args.output else Config.OUTPUT_DIR / "animated_movie.mp4"
        
        # FIXED: Use the sequence list directly (already created above)
        # No need to reconstruct since 'sequence' is already a list of SequenceItem objects
        final_video = composer.compose_video(
            sequence=sequence,  # Pass the list directly
            audio_path=audio_path,
            output_path=output_path
        )
        
        print("\n" + "="*70)
        print("✅ ANIMATED VIDEO COMPLETE!")
        print("="*70)
        print(f"Output: {final_video}")
        print(f"Duration: {stats['total_duration_minutes']:.2f} minutes")
        
        return final_video
        
    except Exception as e:
        print("="*70)
        print(f"❌ Video composition failed: {e}")
        import traceback
        traceback.print_exc()
        return None
    
def main():
    """Main execution flow"""
    args = parse_arguments()
    
    # Update Config paths
    Config.INPUT_DIR = Path(args.input)
    Config.OUTPUT_DIR = Path(args.output)
    Config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*70)
    print("AI VIDEO CREATOR")
    print("="*70)
    print(f"Mode: {args.mode.upper()}")
    print(f"Input: {args.input}")
    print(f"Output: {args.output}")
    print(f"Style: {args.style}")
    print("="*70)
    
    # STEP 1: Ingest Media
    print("\n" + "="*70)
    print("[STEP 1/7] MEDIA INGESTION")
    print("="*70)
    
    ingester = MediaIngester()
    media_items = ingester.process_folder(Config.INPUT_DIR)
    
    if not media_items:
        print("✗ No media files found!")
        return
    
    # Remove duplicates
    if not args.no_duplicates:
        detector = DuplicateDetector()
        duplicate_groups = detector.find_duplicates(media_items)
        if duplicate_groups:
            media_items = detector.remove_duplicates(media_items, duplicate_groups)
    
    # Sort chronologically
    sorted_items = ingester.sort_by_datetime(media_items)
    
    image_count = sum(1 for item in sorted_items if item.file_type == 'image')
    video_count = sum(1 for item in sorted_items if item.file_type == 'video')
    
    print(f"✓ Found {image_count} images and {video_count} videos")
    
    # STEP 2: Scene Analysis
    print("\n" + "="*70)
    print("[STEP 2/7] SCENE ANALYSIS")
    print("="*70)
    
    analyzer = SceneAnalyzer(model_key='blip2-flan-t5-xl')
    scene_analyses = analyzer.analyze_media_collection(sorted_items)
    
    analysis_path = Config.OUTPUT_DIR / "scene_analysis.json"
    analyzer.export_analysis(scene_analyses, analysis_path)
    
    analyzer.unload_model()
    
    print(f"✓ Analyzed {len(scene_analyses)} scenes")
    
    # STEP 3+: Create video(s) based on mode
    if args.mode == 'slideshow':
        create_slideshow(sorted_items, scene_analyses, args)
    
    elif args.mode == 'animated':
        create_animated(sorted_items, scene_analyses, args)
    
    elif args.mode == 'both':
        print("\n🎞️ Creating both slideshow and animated versions...")
        # Create copies to avoid modifying the same list
        create_slideshow(sorted_items.copy(), scene_analyses, args)
        create_animated(sorted_items.copy(), scene_analyses, args)
    
    print("\n" + "="*70)
    print("🎉 ALL DONE!")
    print("="*70)
    print(f"📁 Output directory: {Config.OUTPUT_DIR}")
    print("="*70)

if __name__ == "__main__":
    main()