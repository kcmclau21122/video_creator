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
    
    final_video = composer.compose_video(
        sequence_path=sequence_path,
        audio_path=audio_path,
        output_path=output_path,
        music_volume=args.music_volume
    )
    
    print(f"\n✓ Slideshow created: {final_video}")
    return final_video


"""
Replace the create_animated function in main_with_animation_option.py with this fixed version
"""

def create_animated(sorted_items, scene_analyses, args):
    """Create animated video from images"""
    
    # Separate images and videos
    image_items = [item for item in sorted_items if item.file_type == 'image']
    video_items = [item for item in sorted_items if item.file_type == 'video']
    
    if not image_items:
        print("\n⚠️  No images found to animate")
        return None
    
    print(f"Images to animate: {len(image_items)}")
    print(f"Video clips: {len(video_items)}")
    print(f"⏱  Each image takes 60-120 seconds")
    print(f"⏱  Estimated time: ~{len(image_items) * 1.5:.1f} minutes")
    
    # Initialize animator
    animator = AnimateDiffAnimator(
        animation_style=args.style,
        use_interpolation=True,
        enable_depth=False
    )
    
    # Load AnimateDiff models
    animator.load_models()
    
    # Create output directory
    animated_dir = Config.OUTPUT_DIR / "animated_clips"
    animated_dir.mkdir(parents=True, exist_ok=True)
    
    # Convert paths to Path objects
    image_paths = [Path(item.file_path) for item in image_items]
    
    # Create scene analyses dict with Path keys
    scene_dict = {}
    for item in image_items:
        path_key = Path(item.file_path)
        scene_dict[path_key] = scene_analyses.get(item.file_path)
    
    # Animate all images
    print("\n🎬 Starting animation process...")
    start_time = time.time()
    
    animated_clips = animator.batch_animate_images(
        image_paths=image_paths,
        scene_analyses=scene_dict,
        output_dir=animated_dir
    )
    
    elapsed_time = time.time() - start_time
    print(f"\n✓ Animation completed in {elapsed_time/60:.1f} minutes")
    
    # Unload models to free memory
    animator.unload_models()
    
    # Update items to point to animated clips
    for i, item in enumerate(image_items):
        if i < len(animated_clips):
            item.file_path = str(animated_clips[i])
            item.file_type = 'video'
            item.file_format = 'mp4'
    
    # Recreate sorted_items with updated paths
    updated_items = []
    for item in sorted_items:
        if item.file_type == 'image' and item in image_items:
            idx = image_items.index(item)
            updated_items.append(image_items[idx])
        else:
            updated_items.append(item)
    
    # Create sequence with animated clips
    engine = SequencingEngine()
    sequence = engine.create_sequence(updated_items, scene_analyses)
    
    sequence_path = Config.OUTPUT_DIR / "animated_sequence.json"
    engine.export_sequence(sequence, sequence_path)
    
    stats = engine.get_sequence_statistics(sequence)
    total_duration = stats['total_duration_seconds']
    
    print(f"✓ Sequence exported to: {sequence_path}")
    print(f"✓ Sequence duration: {stats['total_duration_minutes']:.2f} minutes")
    
    # FIXED: Prepare audio with proper error handling
    audio_gen = AudioGenerator(fade_duration=args.fade)
    music_files = audio_gen.find_music_files(Path(args.input))
    
    audio_path = Config.OUTPUT_DIR / "animated_audio.wav"
    
    try:
        if music_files:
            print(f"Using music: {music_files[0].name}")
            # Use the correct method name
            audio_path = audio_gen.use_existing_music(
                music_files[0],
                total_duration,
                audio_path
            )
        else:
            print("No music files found - generating AI music...")
            # Use the generate method
            audio_path = audio_gen.generate_background_music(
                duration_seconds=total_duration,
                output_path=audio_path,
                scene_analyses=list(scene_analyses.values())
            )
    except Exception as e:
        print(f"⚠️  Audio preparation failed: {e}")
        print("Continuing without audio...")
        audio_path = None
    
    # Compose final video
    print("\n" + "="*70)
    print("COMPOSING FINAL VIDEO")
    print("="*70)

    composer = VideoComposer()

    output_path = Config.OUTPUT_DIR / "animated_video.mp4"

    try:
        # FIXED: Pass sequence (not sequence_path) and removed music_volume parameter
        final_video = composer.compose_video(
            sequence=sequence,  # Changed from sequence_path
            audio_path=audio_path,
            output_path=output_path
            # Removed music_volume parameter - not supported by compose_video()
        )
        
        print("\n" + "="*70)
        print("✓ ANIMATED VIDEO CREATED SUCCESSFULLY!")
        print("="*70)
        print(f"Output: {final_video}")
        print(f"Duration: {stats['total_duration_minutes']:.2f} minutes")
        
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