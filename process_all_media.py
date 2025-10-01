#!/usr/bin/env python3
# ============================================================================
# AI Video Creator - Unified Pipeline Script
# ============================================================================

"""
Complete processing script - runs all steps to create final video
Usage:
    python process_all_media.py              # Full pipeline with audio
    python process_all_media.py --no-audio   # Full pipeline without audio
    python process_all_media.py --skip-analysis  # Just compose (if already analyzed)
"""

import sys
from pathlib import Path
from config import Config
from media_ingester import MediaIngester
from scene_analyzer import SceneAnalyzer, SceneAnalysis
from sequencing_engine import SequencingEngine
from audio_generator import AudioGenerator
from video_composer import VideoComposer
import json
from tqdm import tqdm


def step_1_ingest_media():
    """Step 1: Ingest all media files"""
    print("\n" + "="*70)
    print("[STEP 1/5] MEDIA INGESTION")
    print("="*70)
    
    ingester = MediaIngester(cache_enabled=True)
    media_items = ingester.process_folder(Config.INPUT_DIR, use_cache=True)
    
    if not media_items:
        print("❌ No media files found!")
        return None
    
    print(f"✓ Found {len(media_items)} media files")
    
    # Sort chronologically
    sorted_items = ingester.sort_by_datetime(media_items)
    
    # Export manifest
    manifest_path = Config.OUTPUT_DIR / "media_manifest.json"
    ingester.export_manifest(sorted_items, manifest_path)
    
    return sorted_items


def step_2_analyze_scenes(sorted_items):
    """Step 2: Analyze all scenes with AI"""
    print("\n" + "="*70)
    print(f"[STEP 2/5] SCENE ANALYSIS - {len(sorted_items)} files")
    print("="*70)
    print("⚠️  Estimated time: 3-5 seconds per image")
    print(f"⚠️  Total: ~{len(sorted_items) * 3 / 60:.1f} minutes\n")
    
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
            print(f"\n⚠️  Could not analyze {item.file_path.name}: {e}")
            continue
    
    analyzer.unload_model()
    
    print(f"\n✓ Successfully analyzed {len(scene_analyses)}/{len(sorted_items)} items")
    
    # Create default analysis for failed items
    for item in sorted_items:
        if item.file_path not in scene_analyses:
            print(f"  → Using default analysis for: {item.file_path.name}")
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
    print(f"[STEP 3/5] SEQUENCING - {len(sorted_items)} items")
    print("="*70)
    
    engine = SequencingEngine()
    sequence = engine.create_sequence(sorted_items, scene_analyses, sort_chronologically=True)
    
    # Export sequence
    sequence_path = Config.OUTPUT_DIR / "video_sequence.json"
    engine.export_sequence(sequence, sequence_path)
    
    # Print statistics
    stats = engine.get_sequence_statistics(sequence)
    print(f"\n✓ Sequence created:")
    print(f"  • Total items: {stats['total_items']}")
    print(f"  • Duration: {stats['total_duration_minutes']:.2f} minutes")
    print(f"  • Scene groups: {stats['num_groups']}")
    
    return sequence, stats


def step_4_generate_audio(total_duration, skip_audio=False):
    """Step 4: Generate background music"""
    if skip_audio:
        print("\n" + "="*70)
        print("[STEP 4/5] AUDIO GENERATION - SKIPPED")
        print("="*70)
        return None
    
    print("\n" + "="*70)
    print(f"[STEP 4/5] AUDIO GENERATION - {total_duration/60:.1f} minutes")
    print("="*70)
    print("⚠️  This will take several minutes\n")
    
    generator = AudioGenerator(model_key='musicgen-small')
    audio_path = Config.OUTPUT_DIR / "soundtrack.wav"
    
    generator.load_model()
    
    prompt = "upbeat happy family video background music, cheerful acoustic"
    print(f"Generating music: '{prompt}'")
    
    audio = generator._generate_music(prompt, total_duration)
    generator._save_audio(audio, audio_path)
    generator.unload_model()
    
    print(f"\n✓ Audio saved: {audio_path.name}")
    
    return audio_path


def step_5_compose_video(audio_path=None):
    """Step 5: Compose final video"""
    print("\n" + "="*70)
    print("[STEP 5/5] VIDEO COMPOSITION")
    print("="*70)
    
    # Load sequence
    sequence_path = Config.OUTPUT_DIR / "video_sequence.json"
    if not sequence_path.exists():
        print("❌ Sequence file not found!")
        return None
    
    with open(sequence_path, 'r') as f:
        sequence_data = json.load(f)
    
    print(f"• Items: {sequence_data['total_items']}")
    print(f"• Duration: {sequence_data['total_duration']/60:.1f} minutes")
    print(f"• Audio: {'Yes' if audio_path else 'No'}")
    print("\n⚠️  This is the longest step - rendering all clips")
    print(f"⚠️  Estimated time: {sequence_data['total_duration'] / 10:.1f} minutes\n")
    
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
            print(f"⚠️  Skipping {media_path.name} - missing data")
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
    
    print(f"✓ Loaded {len(sequence_items)} sequence items\n")
    
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
    skip_audio = '--no-audio' in sys.argv
    skip_analysis = '--skip-analysis' in sys.argv
    
    print("="*70)
    print("AI VIDEO CREATOR - UNIFIED PIPELINE")
    print("="*70)
    print(f"Input folder: {Config.INPUT_DIR}")
    print(f"Output folder: {Config.OUTPUT_DIR}")
    print(f"Audio: {'Disabled' if skip_audio else 'Enabled'}")
    print(f"Mode: {'Compose only' if skip_analysis else 'Full pipeline'}")
    
    if skip_analysis:
        # Just do composition
        print("\n⚠️  Skipping analysis - using existing data files")
        
        # Check if files exist
        required_files = [
            Config.OUTPUT_DIR / "media_manifest.json",
            Config.OUTPUT_DIR / "scene_analysis.json",
            Config.OUTPUT_DIR / "video_sequence.json"
        ]
        
        missing = [f for f in required_files if not f.exists()]
        if missing:
            print(f"\n❌ Missing required files:")
            for f in missing:
                print(f"  • {f.name}")
            print("\nRun without --skip-analysis first to generate these files.")
            return
        
        # Load duration from sequence
        with open(Config.OUTPUT_DIR / "video_sequence.json", 'r') as f:
            sequence_data = json.load(f)
        total_duration = sequence_data['total_duration']
        
        # Generate audio if needed
        audio_path = step_4_generate_audio(total_duration, skip_audio)
        
        # Compose video
        output_path = step_5_compose_video(audio_path)
        
    else:
        # Full pipeline
        print("\n⚠️  This will run the complete pipeline:")
        print("  1. Ingest media files")
        print("  2. Analyze scenes with AI (takes longest)")
        print("  3. Create sequence")
        print("  4. Generate audio" + (" (SKIPPED)" if skip_audio else ""))
        print("  5. Compose video")
        
        confirm = input("\nContinue? (y/n): ").lower().strip()
        if confirm != 'y':
            print("Cancelled")
            return
        
        # Run all steps
        sorted_items = step_1_ingest_media()
        if not sorted_items:
            return
        
        scene_analyses = step_2_analyze_scenes(sorted_items)
        sequence, stats = step_3_create_sequence(sorted_items, scene_analyses)
        audio_path = step_4_generate_audio(stats['total_duration_seconds'], skip_audio)
        output_path = step_5_compose_video(audio_path)
    
    # Final summary
    if output_path and output_path.exists():
        file_size = output_path.stat().st_size / (1024 * 1024)
        
        print("\n" + "="*70)
        print("✓ VIDEO CREATION COMPLETE!")
        print("="*70)
        print(f"\nOutput file: {output_path}")
        print(f"File size: {file_size:.2f} MB")
        print(f"\n🎬 Ready to upload to YouTube!")
    else:
        print("\n❌ Video creation failed")


if __name__ == "__main__":
    main()