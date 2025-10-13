#!/usr/bin/env python3
# ============================================================================
# AI Video Creator - TRUE ANIMATED Movie Pipeline
# Creates fully animated movies from still photos
# ============================================================================

"""
True Animated Movie Creator
Converts still photos into fully animated clips with character movement
and scene dynamics, then creates a seamless animated movie

Usage:
    python create_animated_movie.py                    # Default anime style
    python create_animated_movie.py --style cartoon    # Western cartoon
    python create_animated_movie.py --style realistic  # Photorealistic
    python create_animated_movie.py --no-interpolation # Skip interpolation
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
from tqdm import tqdm
import time


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Create fully animated movie from still photos'
    )
    
    parser.add_argument(
        '--style',
        choices=['anime', 'cartoon', 'realistic', 'stylized'],
        default='anime',
        help='Animation style (default: anime)'
    )
    
    parser.add_argument(
        '--no-interpolation',
        action='store_true',
        help='Skip frame interpolation (faster but less smooth)'
    )
    
    parser.add_argument(
        '--enable-depth',
        action='store_true',
        help='Enable depth-based effects (experimental)'
    )
    
    parser.add_argument(
        '--music-volume',
        type=float,
        default=0.5,
        help='Background music volume (0.0-1.0, default: 0.5)'
    )
    
    parser.add_argument(
        '--fade',
        type=int,
        default=3,
        help='Fade out duration in seconds (default: 3)'
    )
    
    parser.add_argument(
        '--force-generate-music',
        action='store_true',
        help='Generate music even if MP3 files exist'
    )
    
    parser.add_argument(
        '--skip-duplicates',
        action='store_true',
        help='Skip duplicate detection'
    )
    
    parser.add_argument(
        '--max-images',
        type=int,
        default=0,
        help='Limit number of images (0 = no limit, for testing)'
    )
    
    parser.add_argument(
        '--skip-analysis',
        action='store_true',
        help='Skip scene analysis (use cached results)'
    )
    
    return parser.parse_args()


def create_animated_movie(args):
    """Main pipeline for creating animated movies"""
    
    start_time = time.time()
    
    print("\n" + "="*70)
    print("TRUE ANIMATED MOVIE CREATOR")
    print("="*70)
    print(f"Animation Style: {args.style}")
    print(f"Frame Interpolation: {'Disabled' if args.no_interpolation else 'Enabled'}")
    
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
    if not args.skip_duplicates:
        detector = DuplicateDetector()
        duplicate_groups = detector.find_duplicates(media_items)
        if duplicate_groups:
            media_items = detector.remove_duplicates(media_items, duplicate_groups)
    
    # Sort chronologically
    sorted_items = ingester.sort_by_datetime(media_items)
    
    # Limit for testing
    if args.max_images > 0:
        sorted_items = sorted_items[:args.max_images]
        print(f"⚠ Limited to {len(sorted_items)} items for testing")
    
    # Separate images from videos
    image_items = [item for item in sorted_items if item.file_type == 'image']
    video_items = [item for item in sorted_items if item.file_type == 'video']
    
    print(f"✓ Found {len(image_items)} images and {len(video_items)} videos")
    
    # Export manifest
    manifest_path = Config.OUTPUT_DIR / "media_manifest.json"
    ingester.export_manifest(sorted_items, manifest_path)
    
    # STEP 2: Scene Analysis
    if args.skip_analysis:
        print("\n" + "="*70)
        print("[STEP 2/7] SCENE ANALYSIS - SKIPPED")
        print("="*70)
        
        analysis_path = Config.OUTPUT_DIR / "scene_analysis.json"
        with open(analysis_path, 'r') as f:
            analysis_data = json.load(f)
        
        from scene_analyzer import SceneAnalysis
        scene_analyses = {
            Path(k): SceneAnalysis.from_dict(v) 
            for k, v in analysis_data['analyses'].items()
        }
    else:
        print("\n" + "="*70)
        print("[STEP 2/7] SCENE ANALYSIS")
        print("="*70)
        
        analyzer = SceneAnalyzer(model_key='blip2-flan-t5-xl')
        scene_analyses = analyzer.analyze_media_collection(sorted_items)
        
        analysis_path = Config.OUTPUT_DIR / "scene_analysis.json"
        analyzer.export_analysis(scene_analyses, analysis_path)
        
        analyzer.unload_model()
    
    print(f"✓ Analyzed {len(scene_analyses)} scenes")
    
    # STEP 3: TRUE IMAGE ANIMATION (NEW!)
    print("\n" + "="*70)
    print("[STEP 3/7] TRUE IMAGE ANIMATION WITH AI")
    print("="*70)
    print(f"Images to animate: {len(image_items)}")
    print(f"Animation style: {args.style}")
    print("⚠ This creates FULL ANIMATION with character/scene movement")
    print("⚠ Each image takes 60-120 seconds to animate")
    print("⚠ Total time estimate: ~{:.1f} minutes".format(len(image_items) * 1.5))
    
    if image_items:
        # Initialize animator
        animator = ImageAnimator(
            animation_style=args.style,
            use_interpolation=not args.no_interpolation,
            enable_depth=args.enable_depth
        )
        
        # Load models
        animator.load_models()
        
        # Create output directory for animated clips
        animated_dir = Config.OUTPUT_DIR / "animated_clips"
        animated_dir.mkdir(parents=True, exist_ok=True)
        
        # Animate each image
        animated_clips = animator.batch_animate_images(
            image_paths=[item.file_path for item in image_items],
            scene_analyses=scene_analyses,
            output_dir=animated_dir
        )
        
        # Unload animator models
        animator.unload_models()
        
        print(f"\n✓ Animated {len(animated_clips)} images")
        
        # Update media items to point to animated clips
        animated_items_map = {
            image_items[i].file_path: animated_clips[i]
            for i in range(len(animated_clips))
        }
        
        # Replace original image paths with animated clip paths
        for item in sorted_items:
            if item.file_path in animated_items_map:
                item.file_path = animated_items_map[item.file_path]
                item.file_type = 'video'
                item.file_format = 'mp4'
    else:
        print("No images to animate (video-only input)")
    
    # STEP 4: Create Sequence
    print("\n" + "="*70)
    print("[STEP 4/7] SEQUENCING ANIMATED CLIPS")
    print("="*70)
    
    engine = SequencingEngine()
    sequence = engine.create_sequence(sorted_items, scene_analyses)
    
    sequence_path = Config.OUTPUT_DIR / "video_sequence.json"
    engine.export_sequence(sequence, sequence_path)
    
    stats = engine.get_sequence_statistics(sequence)
    total_duration = stats['total_duration_seconds']
    
    print(f"✓ Created sequence: {stats['total_duration_minutes']:.2f} minutes")
    
    # STEP 5: Generate/Load Background Music
    print("\n" + "="*70)
    print("[STEP 5/7] AUDIO PREPARATION")
    print("="*70)
    
    audio_gen = AudioGenerator(fade_duration=args.fade)
    music_files = audio_gen.find_music_files(Config.INPUT_DIR)
    
    audio_path = Config.OUTPUT_DIR / "soundtrack.wav"
    
    if music_files and not args.force_generate_music:
        print(f"Found {len(music_files)} music file(s)")
        audio_path = audio_gen.prepare_audio_from_files(
            music_files=music_files,
            target_duration=total_duration,
            output_path=audio_path
        )
    else:
        print("Generating AI music...")
        audio_path = audio_gen.generate_soundtrack(
            sequence=sequence,
            duration=total_duration,
            output_path=audio_path
        )
    
    # Apply music volume
    from pydub import AudioSegment
    audio = AudioSegment.from_wav(str(audio_path))
    audio = audio + (args.music_volume * 100 - 100)
    audio.export(str(audio_path), format="wav")
    
    print(f"✓ Audio prepared: {total_duration/60:.2f} minutes")
    
    # STEP 6: Compose Final Video
    print("\n" + "="*70)
    print("[STEP 6/7] VIDEO COMPOSITION")
    print("="*70)
    print("Combining animated clips with audio...")
    
    composer = VideoComposer()
    output_path = Config.OUTPUT_DIR / f"animated_movie_{args.style}.mp4"
    
    result_path = composer.compose_video(
        sequence=sequence,
        audio_path=audio_path,
        output_path=output_path
    )
    
    # STEP 7: Final Summary
    elapsed_time = time.time() - start_time
    
    print("\n" + "="*70)
    print("🎬 ANIMATED MOVIE COMPLETE!")
    print("="*70)
    
    file_size = result_path.stat().st_size / (1024 * 1024)
    
    print(f"\n🎞️  Output: {result_path}")
    print(f"💾 Size: {file_size:.2f} MB")
    print(f"⏱️  Duration: {total_duration/60:.2f} minutes")
    print(f"⏰ Processing time: {elapsed_time/60:.1f} minutes")
    print(f"🎨 Animation: {args.style}")
    print(f"🎵 Music: {'Provided' if music_files and not args.force_generate_music else 'AI-Generated'}")
    
    if args.enable_depth:
        print(f"🌄 Depth Effects: Enabled")
    if not args.no_interpolation:
        print(f"✨ Frame Interpolation: Enabled")
    
    print(f"\n🎥 Ready for YouTube!")
    print(f"🎞️  Your photos are now FULLY ANIMATED!")
    print(f"✨ Characters move, scenes are dynamic, transitions are seamless!")
    
    return result_path


def main():
    """Main entry point"""
    args = parse_arguments()
    
    # Warn about requirements
    print("\n" + "="*70)
    print("⚠️  TRUE ANIMATED MOVIE REQUIREMENTS")
    print("="*70)
    print("This mode uses AnimateDiff which requires:")
    print("  • NVIDIA GPU with 12+ GB VRAM (you have RTX 4080 ✅)")
    print("  • ~15GB model download on first run")
    print("  • 60-120 seconds per image for full animation")
    print("  • Additional dependencies:")
    print("    - pip install git+https://github.com/huggingface/diffusers.git")
    print("    - pip install controlnet-aux xformers")
    print("="*70)
    
    confirm = input("\nContinue with animated movie creation? (y/n): ").lower().strip()
    if confirm != 'y':
        print("Cancelled")
        return
    
    try:
        create_animated_movie(args)
    except KeyboardInterrupt:
        print("\n\n⚠️ Process interrupted by user")
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()