#!/usr/bin/env python3
# ============================================================================
# AI Video Creator - Unified Pipeline Script (WITH DUPLICATE DETECTION)
# ============================================================================

"""
Complete processing script - runs all steps to create final video
NOW INCLUDES: Automatic duplicate detection and removal

Usage:
    python process_all_media.py                      # Full pipeline with deduplication
    python process_all_media.py --no-audio           # Full pipeline without audio
    python process_all_media.py --skip-analysis      # Just compose (if already analyzed)
    python process_all_media.py --fade 5             # Add 5 second fade at end
    python process_all_media.py --force-generate     # Generate music even if files exist
    python process_all_media.py --skip-duplicates    # Skip duplicate detection
"""

import sys
from pathlib import Path
from config import Config
from media_ingester import MediaIngester
from scene_analyzer import SceneAnalyzer, SceneAnalysis
from sequencing_engine import SequencingEngine
from audio_generator import AudioGenerator
from video_composer import VideoComposer
from duplicate_detector import DuplicateDetector, print_duplicate_report
import json
from tqdm import tqdm


def parse_arguments():
    """Parse command line arguments"""
    args = {
        'skip_audio': '--no-audio' in sys.argv,
        'skip_analysis': '--skip-analysis' in sys.argv,
        'force_generate': '--force-generate' in sys.argv,
        'skip_duplicates': '--skip-duplicates' in sys.argv,
        'fade_duration': 0
    }
    
    # Parse fade duration
    if '--fade' in sys.argv:
        try:
            fade_idx = sys.argv.index('--fade')
            if fade_idx + 1 < len(sys.argv):
                args['fade_duration'] = int(sys.argv[fade_idx + 1])
        except (ValueError, IndexError):
            print("âš ï¸  Invalid --fade value, using 0 (no fade)")
    
    return args


def step_0_remove_duplicates(media_items, skip_duplicates=False):
    """Step 0: Detect and remove duplicates"""
    if skip_duplicates:
        print("\n" + "="*70)
        print("[STEP 0/6] DUPLICATE DETECTION - SKIPPED")
        print("="*70)
        return media_items
    
    print("\n" + "="*70)
    print(f"[STEP 0/6] DUPLICATE DETECTION")
    print("="*70)
    
    detector = DuplicateDetector(
        perceptual_threshold=5,  # Adjust for strictness
        prefer_jpeg_over_heic=True  # Prefer JPEG over HEIC
    )
    
    duplicate_groups = detector.find_duplicates(media_items)
    
    # If no duplicates found, return original list
    if not duplicate_groups:
        print("\n✓ No duplicates found")
        return media_items
    
    # Show report
    print_duplicate_report(duplicate_groups)
    
    # Calculate savings
    total_to_remove = sum(len(g.duplicates) for g in duplicate_groups)
    total_size_mb = sum(
        sum(next((item.file_size for item in media_items if item.file_path == dup), 0) 
            for dup in group.duplicates)
        for group in duplicate_groups
    ) / (1024 * 1024)
    
    print(f"{'='*70}")
    print(f"SUMMARY:")
    print(f"  • Duplicates found: {total_to_remove} files")
    print(f"  • Space savings: {total_size_mb:.2f} MB")
    print(f"  • Files to keep: {len(media_items) - total_to_remove}")
    print(f"{'='*70}")
    
    response = input("\nRemove duplicates? (y/n): ").lower().strip()
    
    if response == 'y':
        # Actually remove the duplicates
        filtered_items = detector.remove_duplicates(media_items, duplicate_groups)
        
        # Export report
        report_path = Config.OUTPUT_DIR / "duplicate_report.json"
        detector.export_report(duplicate_groups, report_path)
        
        print(f"\n✓ Duplicates removed successfully")
        return filtered_items
    else:
        print("\n⚠️ Keeping all files (including duplicates)")
        return media_items

def step_1_ingest_media():
    """Step 1: Ingest all media files"""
    print("\n" + "="*70)
    print("[STEP 1/6] MEDIA INGESTION")
    print("="*70)
    
    ingester = MediaIngester(cache_enabled=True)
    media_items = ingester.process_folder(Config.INPUT_DIR, use_cache=True)
    
    if not media_items:
        print("No media files found!")
        return None
    
    print(f"Found {len(media_items)} media files")
    
    # Sort chronologically
    sorted_items = ingester.sort_by_datetime(media_items)
    
    return sorted_items


def step_2_analyze_scenes(sorted_items):
    """Step 2: Analyze all scenes with AI"""
    print("\n" + "="*70)
    print(f"[STEP 2/6] SCENE ANALYSIS - {len(sorted_items)} files")
    print("="*70)
    print("âš ï¸  Estimated time: 3-5 seconds per image")
    print(f"âš ï¸  Total: ~{len(sorted_items) * 3 / 60:.1f} minutes\n")
    
    analyzer = SceneAnalyzer(model_key='blip2-flan-t5-xl', cache_enabled=True)
    analyzer.load_model()
    
    # Analyze ALL items
    scene_analyses = {}
    
    for item in tqdm(sorted_items, desc="Analyzing images"):
        try:
            if item.file_type == 'image':
                analysis = analyzer.analyze_image(item.file_path, use_cache=True)
                scene_analyses[item.file_path] = analysis
            elif item.file_type == 'video':
                analyses = analyzer.analyze_video(item.file_path, sample_fps=1.0, use_cache=True)
                if analyses:
                    scene_analyses[item.file_path] = analyses[0]
        except Exception as e:
            print(f"\nâš ï¸  Could not analyze {item.file_path.name}: {e}")
            continue
    
    analyzer.unload_model()
    
    print(f"Successfully analyzed {len(scene_analyses)}/{len(sorted_items)} items")
    
    # Create default analysis for failed items
    for item in sorted_items:
        if item.file_path not in scene_analyses:
            print(f"  â†' Using default analysis for: {item.file_path.name}")
            default_analysis = SceneAnalysis(
                media_item_path=item.file_path,
                caption="Photo",
                detailed_description="",
                mood="neutral",
                dominant_colors=["blue", "white"],
                scene_type="unknown",
                objects=[],
                lighting="natural",
                weather=None,
                time_of_day=None
            )
            scene_analyses[item.file_path] = default_analysis
    
    # Export scene analysis
    analysis_path = Config.OUTPUT_DIR / "scene_analysis.json"
    analyzer.export_analysis(scene_analyses, analysis_path)
    
    return scene_analyses


def step_3_create_sequence(sorted_items, scene_analyses):
    """Step 3: Create video sequence"""
    print("\n" + "="*70)
    print(f"[STEP 3/6] SEQUENCING - {len(sorted_items)} items")
    print("="*70)
    
    engine = SequencingEngine()
    sequence = engine.create_sequence(sorted_items, scene_analyses, sort_chronologically=True)
    
    # Export sequence
    sequence_path = Config.OUTPUT_DIR / "video_sequence.json"
    engine.export_sequence(sequence, sequence_path)
    
    # Print statistics
    stats = engine.get_sequence_statistics(sequence)
    print(f"Sequence created:")
    print(f"Total items: {stats['total_items']}")
    print(f"Duration: {stats['total_duration_minutes']:.2f} minutes")
    print(f"Scene groups: {stats['num_groups']}")
    print(f"Avg item duration: {stats['avg_item_duration']:.1f}s")
    
    return sequence, stats


def step_4_generate_audio(total_duration, skip_audio=False, fade_duration=0, force_generate=False):
    """Step 4: Generate or use existing background music"""
    if skip_audio:
        print("\n" + "="*70)
        print("[STEP 4/6] AUDIO GENERATION - SKIPPED")
        print("="*70)
        return None
    
    print("\n" + "="*70)
    print(f"[STEP 4/6] AUDIO PREPARATION - {total_duration/60:.1f} minutes")
    print("="*70)
    
    if fade_duration > 0:
        print(f"Fade out: {fade_duration} seconds before end")
    else:
        print("Fade out: Disabled")
    
    if force_generate:
        print("Mode: Force AI generation (ignoring music files)")
    else:
        print("Mode: Auto (use music files if available, otherwise generate)")
    
    print()
    
    generator = AudioGenerator(model_key='musicgen-small', fade_duration=fade_duration)
    audio_path = Config.OUTPUT_DIR / "soundtrack.wav"
    
    # This will automatically choose between existing files or generation
    result_path = generator.generate_or_use_audio(
        input_folder=Config.INPUT_DIR,
        sequence=None,  # Only needed if generating
        duration=total_duration,
        output_path=audio_path,
        force_generate=force_generate
    )
    
    return result_path


def step_5_compose_video(audio_path=None):
    """Step 5: Compose final video"""
    print("\n" + "="*70)
    print("[STEP 5/6] VIDEO COMPOSITION")
    print("="*70)
    
    # Load sequence
    sequence_path = Config.OUTPUT_DIR / "video_sequence.json"
    if not sequence_path.exists():
        print("âŒ Sequence file not found!")
        return None
    
    with open(sequence_path, 'r') as f:
        sequence_data = json.load(f)
    
    print(f"â€¢ Items: {sequence_data['total_items']}")
    print(f"â€¢ Duration: {sequence_data['total_duration']/60:.1f} minutes")
    print(f"â€¢ Audio: {'Yes' if audio_path else 'No'}")
    print("\nâš ï¸  This is the longest step - rendering all clips")
    print(f"âš ï¸  Estimated time: {sequence_data['total_duration'] / 10:.1f} minutes\n")
    
    # Load full data
    from media_ingester import MediaItem
    
    manifest_path = Config.OUTPUT_DIR / "media_manifest.json"
    with open(manifest_path, 'r') as f:
        manifest_data = json.load(f)
    
    media_items_dict = {}
    for item_dict in manifest_data['items']:
        media_item = MediaItem.from_dict(item_dict)
        media_items_dict[media_item.file_path] = media_item
    
    analysis_path = Config.OUTPUT_DIR / "scene_analysis.json"
    with open(analysis_path, 'r') as f:
        analysis_data = json.load(f)
    
    scene_analyses_dict = {}
    for path_str, analysis_dict in analysis_data['analyses'].items():
        scene_analyses_dict[Path(path_str)] = SceneAnalysis.from_dict(analysis_dict)
    
    # Reconstruct sequence items
    from sequencing_engine import SequenceItem
    
    sequence_items = []
    for item_dict in sequence_data['items']:
        media_path = Path(item_dict['media_item_path'])
        
        if media_path not in media_items_dict or media_path not in scene_analyses_dict:
            print(f"âš ï¸  Skipping {media_path.name} - missing data")
            continue
        
        seq_item = SequenceItem(
            media_item=media_items_dict[media_path],
            scene_analysis=scene_analyses_dict[media_path],
            sequence_index=item_dict['sequence_index'],
            start_time=item_dict['start_time'],
            duration=item_dict['duration'],
            transition_in=item_dict['transition_in'],
            transition_duration=item_dict['transition_duration'],
            group_id=item_dict['group_id']
        )
        sequence_items.append(seq_item)
    
    print(f"Loaded {len(sequence_items)} sequence items\n")
    
    # Compose video
    composer = VideoComposer()
    output_path = Config.OUTPUT_DIR / "final_video.mp4"
    
    composer.compose_video(
        sequence=sequence_items,
        audio_path=audio_path,
        output_path=output_path
    )
    
    return output_path


def main():
    """Main pipeline"""
    
    # Parse arguments
    args = parse_arguments()
    
    print("="*70)
    print("AI VIDEO CREATOR - UNIFIED PIPELINE")
    print("="*70)
    print(f"Input folder: {Config.INPUT_DIR}")
    print(f"Output folder: {Config.OUTPUT_DIR}")
    print(f"Duplicate detection: {'Disabled' if args['skip_duplicates'] else 'Enabled'}")
    print(f"Audio: {'Disabled' if args['skip_audio'] else 'Enabled'}")
    print(f"Fade duration: {args['fade_duration']} seconds")
    print(f"Force generation: {'Yes' if args['force_generate'] else 'No'}")
    print(f"Mode: {'Compose only' if args['skip_analysis'] else 'Full pipeline'}")
    
    if args['skip_analysis']:
        # Just do composition
        print("\nâš ï¸  Skipping analysis - using existing data files")
        
        # Check if files exist
        required_files = [
            Config.OUTPUT_DIR / "media_manifest.json",
            Config.OUTPUT_DIR / "scene_analysis.json",
            Config.OUTPUT_DIR / "video_sequence.json"
        ]
        
        missing = [f for f in required_files if not f.exists()]
        if missing:
            print(f"\nâŒ Missing required files:")
            for f in missing:
                print(f"  â€¢ {f.name}")
            print("\nRun without --skip-analysis first to generate these files.")
            return
        
        # Load duration from sequence
        with open(Config.OUTPUT_DIR / "video_sequence.json", 'r') as f:
            sequence_data = json.load(f)
        total_duration = sequence_data['total_duration']
        
        # Generate audio if needed
        audio_path = step_4_generate_audio(
            total_duration, 
            args['skip_audio'],
            args['fade_duration'],
            args['force_generate']
        )
        
        # Compose video
        output_path = step_5_compose_video(audio_path)
        
    else:
        # Full pipeline
        print("\nâš ï¸  This will run the complete pipeline:")
        print("  0. Detect and remove duplicates" + (" (SKIPPED)" if args['skip_duplicates'] else ""))
        print("  1. Ingest media files")
        print("  2. Analyze scenes with AI (takes longest)")
        print("  3. Create sequence (6-10s per image)")
        print("  4. Prepare audio" + (" (SKIPPED)" if args['skip_audio'] else " (use files or generate)"))
        if args['fade_duration'] > 0:
            print(f"     â†' Fade out: {args['fade_duration']}s")
        if args['force_generate']:
            print("     â†' Force AI generation")
        print("  5. Compose video")
        
        confirm = input("\nContinue? (y/n): ").lower().strip()
        if confirm != 'y':
            print("Cancelled")
            return
        
        # Run all steps
        media_items = step_1_ingest_media()
        if not media_items:
            return
        
        # NEW: Duplicate detection and removal
        media_items = step_0_remove_duplicates(media_items, args['skip_duplicates'])
        
        # Export manifest with deduplicated items
        ingester = MediaIngester()
        manifest_path = Config.OUTPUT_DIR / "media_manifest.json"
        ingester.export_manifest(media_items, manifest_path)
        
        scene_analyses = step_2_analyze_scenes(media_items)
        sequence, stats = step_3_create_sequence(media_items, scene_analyses)
        audio_path = step_4_generate_audio(
            stats['total_duration_seconds'],
            args['skip_audio'],
            args['fade_duration'],
            args['force_generate']
        )
        output_path = step_5_compose_video(audio_path)
    
    # Final summary
    if output_path and output_path.exists():
        file_size = output_path.stat().st_size / (1024 * 1024)
        
        print("\n" + "="*70)
        print("VIDEO CREATION COMPLETE!")
        print("="*70)
        print(f"\nOutput file: {output_path}")
        print(f"File size: {file_size:.2f} MB")
        
        # Check for music files used
        from audio_generator import AudioGenerator
        gen = AudioGenerator()
        music_files = gen.find_music_files(Config.INPUT_DIR)
        if music_files and not args['force_generate']:
            print(f"\nMusic: Used {len(music_files)} existing file(s)")
        elif not args['skip_audio']:
            print("\nMusic: AI-generated")
        
        if args['fade_duration'] > 0:
            print(f"Fade out: {args['fade_duration']}s")
        
        # Show duplicate info if applicable
        duplicate_report = Config.OUTPUT_DIR / "duplicate_report.json"
        if duplicate_report.exists():
            with open(duplicate_report, 'r') as f:
                report_data = json.load(f)
            print(f"\nDuplicates removed: {report_data['total_duplicates']} files")
        
        print(f"\nðŸŽ¬ Ready to upload to YouTube!")
    else:
        print("\nâŒ Video creation failed")


if __name__ == "__main__":
    main()