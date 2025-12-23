#!/usr/bin/env python3
# ============================================================================
# AI Video Creator - Movie Creation Pipeline
# Creates movies with narration, dialog, and background music
# ============================================================================

"""
Complete movie creation pipeline with narration
Usage:
    python create_movie.py                         # Default family style
    python create_movie.py --style documentary     # Documentary style
    python create_movie.py --voice narrator_female # Female narrator
    python create_movie.py --no-narration          # Music only
    python create_movie.py --title "Our Summer"    # Custom title
"""

import sys
import argparse
from pathlib import Path
from config import Config
from media_ingester import MediaIngester
from scene_analyzer import SceneAnalyzer
from sequencing_engine import SequencingEngine
from narrative_generator import NarrativeGenerator
from tts_generator import TTSGenerator
from audio_mixer import AudioMixer
from video_composer import VideoComposer
from duplicate_detector import DuplicateDetector
import json


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Create a movie with narration from photos and videos'
    )
    
    # Narrative options
    parser.add_argument(
        '--style',
        choices=['documentary', 'story', 'travel', 'family', 'adventure'],
        default='family',
        help='Narrative style (default: family)'
    )
    
    parser.add_argument(
        '--voice',
        choices=list(TTSGenerator.BARK_VOICES.keys()),
        default='narrator_male',
        help='Voice for narration (default: narrator_male)'
    )
    
    parser.add_argument(
        '--title',
        type=str,
        default='The McLaurin Story',
        help='Video title for narrative generation'
    )
    
    # Model selection
    parser.add_argument(
        '--narrative-model',
        choices=list(NarrativeGenerator.MODELS.keys()),
        default='phi-3-mini',
        help='LLM for narrative generation (default: phi-3-mini)'
    )
    
    parser.add_argument(
        '--tts-model',
        choices=list(TTSGenerator.MODELS.keys()),
        default='bark',
        help='TTS model (default: bark-small)'
    )
    
    # Audio options
    parser.add_argument(
        '--no-narration',
        action='store_true',
        help='Skip narration generation (music only)'
    )
    
    parser.add_argument(
        '--narration-volume',
        type=float,
        default=1.0,
        help='Narration volume (0.0-1.0, default: 1.0)'
    )
    
    parser.add_argument(
        '--music-volume',
        type=float,
        default=0.3,
        help='Background music volume (0.0-1.0, default: 0.3)'
    )

    parser.add_argument(
        '--picture-duration',
        type=float,
        default=7.0,
        help='Base duration for images in seconds (default: 7.0, adjusted ±1-2s based on complexity)'
    )

    parser.add_argument(
        '--fade',
        type=float,
        default=3.0,
        help='Fade transition duration in seconds (default: 3.0, applies to video fades and audio fade-out)'
    )
    
    # Processing options
    parser.add_argument(
        '--skip-duplicates',
        action='store_true',
        help='Skip duplicate detection'
    )
    
    parser.add_argument(
        '--skip-analysis',
        action='store_true',
        help='Use existing scene analysis'
    )

    args = parser.parse_args()

    # Validate parameters
    if args.picture_duration < 1.0 or args.picture_duration > 60.0:
        parser.error("--picture-duration must be between 1.0 and 60.0 seconds")
    if args.fade < 0.1 or args.fade > 10.0:
        parser.error("--fade must be between 0.1 and 10.0 seconds")

    return args


def create_movie_with_narration(args):
    """Main movie creation pipeline"""
    
    print("="*70)
    print("AI VIDEO CREATOR - MOVIE MODE")
    print("="*70)
    print(f"Style: {args.style}")
    print(f"Voice: {args.voice}")
    print(f"Title: {args.title}")
    print(f"Narration: {'Disabled' if args.no_narration else 'Enabled'}")
    print(f"Music volume: {args.music_volume*100:.0f}%")
    print("="*70)
    
    # Step 1: Ingest media
    print("\n" + "="*70)
    print("[STEP 1/7] MEDIA INGESTION")
    print("="*70)
    
    ingester = MediaIngester()
    media_items = ingester.process_folder(Config.INPUT_DIR)
    
    if not media_items:
        print("❌ No media files found!")
        return
    
    # Remove duplicates
    if not args.skip_duplicates:
        detector = DuplicateDetector()
        duplicate_groups = detector.find_duplicates(media_items)
        if duplicate_groups:
            media_items = detector.remove_duplicates(media_items, duplicate_groups)
    
    # Sort chronologically
    sorted_items = ingester.sort_by_datetime(media_items)
    
    # Export manifest
    manifest_path = Config.OUTPUT_DIR / "media_manifest.json"
    ingester.export_manifest(sorted_items, manifest_path)
    
    print(f"✅ Processed {len(sorted_items)} media files")
    
    # Step 2: Scene Analysis
    if args.skip_analysis:
        print("\n" + "="*70)
        print("[STEP 2/7] SCENE ANALYSIS - SKIPPED")
        print("="*70)
        
        # Load existing analysis
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
        
        # Export analysis
        analysis_path = Config.OUTPUT_DIR / "scene_analysis.json"
        analyzer.export_analysis(scene_analyses, analysis_path)
        
        analyzer.unload_model()
    
    print(f"✅ Analyzed {len(scene_analyses)} scenes")
    
    # Step 3: Create Sequence
    print("\n" + "="*70)
    print("[STEP 3/7] SEQUENCING")
    print("="*70)

    engine = SequencingEngine(
        base_image_duration=args.picture_duration,
        fade_duration=args.fade
    )
    sequence = engine.create_sequence(sorted_items, scene_analyses)
    
    # Export sequence
    sequence_path = Config.OUTPUT_DIR / "video_sequence.json"
    engine.export_sequence(sequence, sequence_path)
    
    stats = engine.get_sequence_statistics(sequence)
    total_duration = stats['total_duration_seconds']
    
    print(f"✅ Created sequence: {stats['total_duration_minutes']:.2f} minutes")
    
    # Step 4: Generate Narrative (NEW!)
    narration_path = None
    
    if not args.no_narration:
        print("\n" + "="*70)
        print("[STEP 4/7] NARRATIVE GENERATION")
        print("="*70)
        
        # Generate narrative script
        narrator = NarrativeGenerator(
            model_key=args.narrative_model,
            style=args.style
        )
        
        narrative_segments = narrator.generate_narrative(
            sequence=sequence,
            video_title=args.title
        )
        
        # Export script
        script_path = Config.OUTPUT_DIR / "narrative_script.json"
        narrator.export_script(narrative_segments, script_path)
        
        narrator.unload_model()
        
        print(f"✅ Generated {len(narrative_segments)} narrative segments")
        
        # Step 5: Generate Speech (NEW!)
        print("\n" + "="*70)
        print("[STEP 5/7] SPEECH GENERATION")
        print("="*70)
        
        tts = TTSGenerator(
            model_key=args.tts_model,
            voice=args.voice
        )
        
        # Generate speech for each segment
        narration_dir = Config.OUTPUT_DIR / "narration"
        narrative_segments = tts.generate_speech_for_segments(
            narrative_segments,
            narration_dir
        )
        
        # Combine into single track with timing
        narration_path = Config.OUTPUT_DIR / "narration_track.wav"
        tts.create_combined_narration(
            narrative_segments,
            total_duration,
            narration_path
        )
        
        tts.unload_model()
        
        print(f"✅ Generated speech narration")
    else:
        print("\n" + "="*70)
        print("[STEP 4/7] NARRATIVE GENERATION - SKIPPED")
        print("[STEP 5/7] SPEECH GENERATION - SKIPPED")
        print("="*70)
    
    # Step 6: Mix Audio (NEW!)
    print("\n" + "="*70)
    print("[STEP 6/7] AUDIO MIXING")
    print("="*70)
    
    # Find music files
    from audio_generator import AudioGenerator
    audio_gen = AudioGenerator()
    music_files = audio_gen.find_music_files(Config.INPUT_DIR)
    
    if not music_files:
        print("⚠️  No music files found - generating with AI...")
        # Generate background music
        music_path = Config.OUTPUT_DIR / "generated_music.wav"
        audio_gen.generate_soundtrack(
            sequence=sequence,
            duration=total_duration,
            output_path=music_path
        )
        music_files = [music_path]
    
    # Mix narration and music
    mixer = AudioMixer(
        narration_volume=args.narration_volume,
        music_volume=args.music_volume,
        ducking_amount=0.6,  # Reduce music by 60% during speech
        fade_duration=args.fade
    )
    
    final_audio_path = Config.OUTPUT_DIR / "final_audio.wav"
    mixer.mix_audio_tracks(
        narration_path=narration_path,
        music_paths=music_files,
        video_duration=total_duration,
        output_path=final_audio_path
    )
    
    print("✅ Audio mixing complete")
    
    # Step 7: Compose Video
    print("\n" + "="*70)
    print("[STEP 7/7] VIDEO COMPOSITION")
    print("="*70)

    composer = VideoComposer(fade_duration=args.fade)
    output_path = Config.OUTPUT_DIR / f"movie_{args.style}.mp4"
    
    result_path = composer.compose_video(
        sequence=sequence,
        audio_path=final_audio_path,
        output_path=output_path
    )
    
    # Final Summary
    file_size = result_path.stat().st_size / (1024 * 1024)
    
    print("\n" + "="*70)
    print("🎬 MOVIE CREATION COMPLETE!")
    print("="*70)
    print(f"\nOutput: {result_path}")
    print(f"Size: {file_size:.2f} MB")
    print(f"Duration: {total_duration/60:.2f} minutes")
    print(f"Style: {args.style}")
    
    if narration_path:
        print(f"Narration: {args.voice}")
        print(f"Script: {script_path}")
    
    print(f"\n🎥 Ready for YouTube!")
    print(f"💬 Your movie tells a story with narration and music!")
    
    return result_path


def main():
    """Main entry point"""
    args = parse_arguments()
    
    try:
        create_movie_with_narration(args)
    except KeyboardInterrupt:
        print("\n\n❌ Cancelled by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()