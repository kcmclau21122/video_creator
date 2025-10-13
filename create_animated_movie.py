#!/usr/bin/env python3
# ============================================================================
# AI Video Creator - Animated Movie Creation Pipeline (SVD Version)
# Creates fully animated movies from still photos
# ============================================================================

"""
Complete animated movie creation pipeline using Stable Video Diffusion
Usage:
    python create_animated_movie.py                        # Default settings
    python create_animated_movie.py --style realistic     # Realistic animation
    python create_animated_movie.py --motion 180          # High motion
    python create_animated_movie.py --no-duplicates       # Skip duplicate detection
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
        choices=['anime', 'cartoon', 'realistic', 'stylized', 'cinematic'],
        default='realistic',
        help='Animation style (default: realistic)'
    )
    
    # Motion control
    parser.add_argument(
        '--motion',
        type=int,
        default=127,
        help='Motion intensity 1-255 (default: 127, moderate)'
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
    """Main execution flow"""
    args = parse_arguments()
    
    print("\n" + "="*70)
    print("AI VIDEO CREATOR - ANIMATED MOVIE PIPELINE")
    print("Powered by Stable Video Diffusion")
    print("="*70)
    print(f"Animation Style: {args.style}")
    print(f"Motion Intensity: {args.motion}/255")
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
    print("[STEP 3/7] IMAGE ANIMATION WITH STABLE VIDEO DIFFUSION")
    print("="*70)
    print(f"Images to animate: {len(image_items)}")
    print(f"Animation style: {args.style}")
    print("⚠  This creates TRUE ANIMATION with movement and dynamics")
    print("⚠  Each image takes 60-120 seconds to animate")
    print(f"⚠  Total time estimate: ~{len(image_items) * 1.5:.1f} minutes")
    
    if image_items:
        # Initialize animator with SVD
        animator = ImageAnimator(
            animation_style=args.style,
            use_interpolation=not args.no_interpolation,
            enable_depth=args.enable_depth
        )
        
        # Load Stable Video Diffusion model
        animator.load_models()
        
        # Create output directory for animated clips
        animated_dir = Config.OUTPUT_DIR / "animated_clips"
        animated_dir.mkdir(parents=True, exist_ok=True)
        
        # Convert MediaItem paths to Path objects
        image_paths = [Path(item.file_path) for item in image_items]
        
        # Match scene analyses to images
        scene_list = [scene_analyses.get(str(item.file_path)) for item in image_items]
        
        # Animate each image
        animated_clips = animator.batch_animate_images(
            image_paths=image_paths,
            scene_analyses=scene_list,
            output_dir=animated_dir,
            motion_intensity=args.motion
        )
        
        # Unload animator models to free GPU memory
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
    
    if music_files:
        print(f"Found {len(music_files)} music file(s) in input directory")
        audio_path = audio_gen.use_existing_music(
            music_files[0],
            total_duration,
            Config.OUTPUT_DIR / "soundtrack.wav"
        )
    else:
        print("No music files found - generating AI music...")
        audio_gen.generate_background_music(
            duration_seconds=total_duration,
            output_path=audio_path
        )
    
    print(f"✓ Audio ready: {audio_path}")
    
    # STEP 6: Compose Final Video
    print("\n" + "="*70)
    print("[STEP 6/7] VIDEO COMPOSITION")
    print("="*70)
    
    composer = VideoComposer()
    
    output_path = Config.OUTPUT_DIR / "animated_movie.mp4"
    
    final_video = composer.compose_video(
        sequence_path=sequence_path,
        audio_path=audio_path,
        output_path=output_path,
        music_volume=args.music_volume
    )
    
    # STEP 7: Final Report
    print("\n" + "="*70)
    print("[STEP 7/7] COMPLETED!")
    print("="*70)
    print(f"✓ Output: {final_video}")
    print(f"✓ Duration: {stats['total_duration_minutes']:.2f} minutes")
    print(f"✓ Animated scenes: {len(animated_clips)}")
    print(f"✓ Video clips: {len(video_items)}")
    print("="*70)
    
    print("\n🎉 Animated movie creation complete!")
    print(f"📁 All files saved to: {Config.OUTPUT_DIR}")
    

if __name__ == "__main__":
    main()