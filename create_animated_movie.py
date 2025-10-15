#!/usr/bin/env python3
# ============================================================================
# AI Video Creator - Animated Movie Creation Pipeline (FULLY FIXED)
# Creates fully animated movies from still photos
# ============================================================================

"""
Complete animated movie creation pipeline using AnimateDiff
Usage:
    python create_animated_movie.py                        # Default settings
    python create_animated_movie.py --style cartoon        # Cartoon style
    python create_animated_movie.py --no-duplicates        # Skip duplicate detection

FIXED ISSUES:
1. Removed unsupported motion_intensity parameter
2. Fixed scene_analyses path mapping after animation
3. Added defensive checks for empty sequences
4. Proper error handling throughout
"""

import sys
import argparse
from pathlib import Path
from config import Config
from media_ingester import MediaIngester
from scene_analyzer import SceneAnalyzer
from sequencing_engine import SequencingEngine
from image_animator import ImageAnimator
from audio_generator import AudioGenerator
from video_composer import VideoComposer
from duplicate_detector import DuplicateDetector
import json


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Create fully animated movie from photos'
    )
    
    # Animation style
    parser.add_argument(
        '--style',
        choices=['anime', 'cartoon', 'realistic', 'stylized'],
        default='realistic',
        help='Animation style (default: realistic)'
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
        '--no-interpolation',
        action='store_true',
        help='Disable frame interpolation (faster but less smooth)'
    )
    
    parser.add_argument(
        '--enable-depth',
        action='store_true',
        help='Enable depth estimation (experimental)'
    )
    
    return parser.parse_args()


def main():
    """Main execution flow with comprehensive error handling"""
    args = parse_arguments()
    
    print("\n" + "="*70)
    print("AI VIDEO CREATOR - ANIMATED MOVIE PIPELINE")
    print("Powered by AnimateDiff")
    print("="*70)
    print(f"Animation Style: {args.style}")
    print(f"Output Directory: {Config.OUTPUT_DIR}")
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
    
    # Remove duplicates if requested
    if not args.no_duplicates:
        detector = DuplicateDetector()
        duplicate_groups = detector.find_duplicates(media_items)
        if duplicate_groups:
            media_items = detector.remove_duplicates(media_items, duplicate_groups)
    
    # Sort chronologically
    sorted_items = ingester.sort_by_datetime(media_items)
    
    image_items = [item for item in sorted_items if item.file_type == 'image']
    video_items = [item for item in sorted_items if item.file_type == 'video']
    
    print(f"✓ Found {len(image_items)} images and {len(video_items)} videos")
    
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
    
    # STEP 3: Animate Images
    print("\n" + "="*70)
    print("[STEP 3/7] IMAGE ANIMATION WITH ANIMATEDIFF")
    print("="*70)
    
    animated_clips = []
    
    if image_items:
        print(f"Images to animate: {len(image_items)}")
        print(f"Animation style: {args.style}")
        print("⚠  This creates TRUE ANIMATION with movement and dynamics")
        print("⚠  Each image takes 60-120 seconds to animate")
        print(f"⚠  Total time estimate: ~{len(image_items) * 1.5:.1f} minutes")
        
        # Initialize animator with AnimateDiff
        animator = ImageAnimator(
            animation_style=args.style,
            use_interpolation=not args.no_interpolation,
            enable_depth=args.enable_depth
        )
        
        # Load AnimateDiff model
        animator.load_models()
        
        # Create output directory for animated clips
        animated_dir = Config.OUTPUT_DIR / "animated_clips"
        animated_dir.mkdir(parents=True, exist_ok=True)
        
        # Convert MediaItem paths to Path objects
        image_paths = [Path(item.file_path) for item in image_items]
        
        # Create scene analyses dict with Path keys for the animator
        scene_dict_for_animator = {
            Path(item.file_path): scene_analyses.get(item.file_path) 
            for item in image_items
        }
        
        # Animate each image - NO motion_intensity parameter
        animated_clips = animator.batch_animate_images(
            image_paths=image_paths,
            scene_analyses=scene_dict_for_animator,
            output_dir=animated_dir
        )
        
        # Unload animator models to free GPU memory
        animator.unload_models()
        
        print(f"\n✓ Animated {len(animated_clips)} images")
        
        # ================================================================
        # CRITICAL FIX: Update scene_analyses with new animated clip paths
        # ================================================================
        
        # 1. Create path mapping from original images to animated clips
        path_map = {
            image_items[i].file_path: str(animated_clips[i])
            for i in range(len(animated_clips))
        }
        
        # 2. Create updated scene_analyses with new paths as keys
        updated_scene_analyses = {}
        for original_path, analysis in scene_analyses.items():
            # If this path was animated, use the new animated clip path
            new_path = path_map.get(original_path, original_path)
            updated_scene_analyses[new_path] = analysis
        
        # Replace the old scene_analyses with the updated one
        scene_analyses = updated_scene_analyses
        
        # 3. Update sorted_items to point to animated clips
        for item in sorted_items:
            if item.file_path in path_map:
                item.file_path = path_map[item.file_path]
                item.file_type = 'video'
                item.file_format = 'mp4'
        
        print("✓ Scene analyses remapped to animated clip paths")
        
    else:
        print("No images to animate (video-only input)")
    
    # STEP 4: Create Sequence
    print("\n" + "="*70)
    print("[STEP 4/7] SEQUENCING ANIMATED CLIPS")
    print("="*70)
    
    engine = SequencingEngine()
    sequence = engine.create_sequence(sorted_items, scene_analyses)
    
    # DEFENSIVE: Check if sequence is empty
    if not sequence:
        print("\n❌ ERROR: Failed to create sequence!")
        print("⚠️  No items were added to the sequence.")
        print("Possible causes:")
        print("  - Scene analyses don't match media item paths")
        print("  - All items were filtered out")
        print("\nDebug info:")
        print(f"  Media items: {len(sorted_items)}")
        print(f"  Scene analyses: {len(scene_analyses)}")
        if sorted_items:
            print(f"  Sample item path: {sorted_items[0].file_path}")
        if scene_analyses:
            print(f"  Sample analysis key: {list(scene_analyses.keys())[0]}")
        return
    
    sequence_path = Config.OUTPUT_DIR / "video_sequence.json"
    engine.export_sequence(sequence, sequence_path)
    
    # Get statistics with defensive checks
    stats = engine.get_sequence_statistics(sequence)
    
    if not stats or 'total_duration_seconds' not in stats:
        print("\n❌ ERROR: Failed to get sequence statistics!")
        return
    
    total_duration = stats['total_duration_seconds']
    
    print(f"✓ Sequence created: {stats['total_items']} items")
    print(f"✓ Total duration: {stats['total_duration_minutes']:.2f} minutes")
    
    # STEP 5: Generate Audio
    print("\n" + "="*70)
    print("[STEP 5/7] AUDIO GENERATION")
    print("="*70)
    
    audio_gen = AudioGenerator(fade_duration=args.fade)
    
    # Look for music files in input directory
    music_files = audio_gen.find_music_files(Config.INPUT_DIR)
    
    audio_path = Config.OUTPUT_DIR / "soundtrack.wav"
    
    if music_files:
        print(f"Found {len(music_files)} music file(s)")
        audio_path = audio_gen.prepare_audio_from_files(
            music_files=music_files,
            target_duration=total_duration,
            output_path=audio_path
        )
    else:
        print("No music files found, generating AI soundtrack...")
        audio_path = audio_gen.generate_soundtrack(
            sequence=sequence,
            duration=total_duration,
            output_path=audio_path
        )
    
    # Apply volume adjustment
    if args.music_volume != 1.0:
        try:
            from pydub import AudioSegment
            audio = AudioSegment.from_wav(str(audio_path))
            # Convert 0-1 to dB adjustment
            db_adjustment = (args.music_volume - 1.0) * 20
            audio = audio + db_adjustment
            audio.export(str(audio_path), format="wav")
            print(f"✓ Audio volume adjusted to {args.music_volume}")
        except Exception as e:
            print(f"⚠️  Could not adjust volume: {e}")
    
    print(f"✓ Audio ready")
    
    # STEP 6: Compose Video
    print("\n" + "="*70)
    print("[STEP 6/7] VIDEO COMPOSITION")
    print("="*70)
    
    composer = VideoComposer()
    output_path = Config.OUTPUT_DIR / f"animated_movie_{args.style}.mp4"
    
    try:
        result_path = composer.compose_video(
            sequence=sequence,
            audio_path=audio_path,
            output_path=output_path
        )
        
        # STEP 7: Summary
        print("\n" + "="*70)
        print("🎬 ANIMATED MOVIE COMPLETE!")
        print("="*70)
        
        file_size = result_path.stat().st_size / (1024 * 1024)
        
        print(f"\n🎞️  Output: {result_path}")
        print(f"💾 Size: {file_size:.2f} MB")
        print(f"⏱️  Duration: {total_duration/60:.2f} minutes")
        print(f"🎨 Animation Style: {args.style}")
        print(f"✨ Fully Animated Clips: {len(animated_clips)}")
        if video_items:
            print(f"🎥 Video Clips: {len(video_items)}")
        print("\n🎥 Ready for YouTube!")
        
    except Exception as e:
        print(f"\n❌ ERROR during video composition: {e}")
        import traceback
        traceback.print_exc()
        return


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Process interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)