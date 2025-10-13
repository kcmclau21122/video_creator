#!/usr/bin/env python3
# ============================================================================
# AI Video Creator - HYBRID Movie Pipeline (UPDATED)
# Intelligently combines TRUE animation and slideshow
# ============================================================================

"""
Hybrid Movie Creator - TRUE ANIMATION VERSION
For large photo collections (60-200+ images):
- TRUE ANIMATION for KEY moments (full character/scene motion)
- Slideshow for other photos (portraits, similar scenes)
- Best balance: Quality animation + Processing speed

Usage:
    python create_hybrid_movie.py                     # Auto-select 30% for animation
    python create_hybrid_movie.py --animate-percent 50  # Animate 50% of images
    python create_hybrid_movie.py --animate-top 20    # Animate top 20 images only
    python create_hybrid_movie.py --style anime       # Anime-style animation
"""

import sys
import argparse
from pathlib import Path
from config import Config
from media_ingester import MediaIngester
from scene_analyzer import SceneAnalyzer, SceneAnalysis
from sequencing_engine import SequencingEngine
from image_animator import ImageAnimator
from audio_generator import AudioGenerator
from video_composer import VideoComposer
from duplicate_detector import DuplicateDetector
import json
from tqdm import tqdm
from typing import List, Set
import time


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Create hybrid animated/slideshow movie for large collections'
    )
    
    # Selection strategy
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        '--animate-percent',
        type=int,
        default=30,
        help='Percentage of images to animate (default: 30%%)'
    )
    group.add_argument(
        '--animate-top',
        type=int,
        help='Animate only top N images by quality score'
    )
    group.add_argument(
        '--animate-landscapes',
        action='store_true',
        help='Animate only landscape/outdoor scenes'
    )
    group.add_argument(
        '--animate-every',
        type=int,
        help='Animate every Nth image (e.g., every 5th image)'
    )
    
    # Animation options
    parser.add_argument(
        '--style',
        choices=['anime', 'cartoon', 'realistic', 'stylized'],
        default='anime',
        help='Animation style for animated clips'
    )
    
    parser.add_argument(
        '--slideshow-duration',
        type=float,
        default=4.0,
        help='Duration for slideshow images (seconds, default: 4.0)'
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
        help='Fade out duration in seconds'
    )
    
    # Processing options
    parser.add_argument(
        '--skip-duplicates',
        action='store_true',
        help='Skip duplicate detection'
    )
    
    parser.add_argument(
        '--batch-size',
        type=int,
        default=10,
        help='Process animation in batches of N images (default: 10)'
    )
    
    return parser.parse_args()


class HybridMovieCreator:
    """Creates hybrid animated/slideshow movies with TRUE animation"""
    
    def __init__(self, args):
        self.args = args
        self.animated_clips = {}
    
    def select_images_to_animate(
        self,
        media_items: List,
        scene_analyses: dict
    ) -> Set[Path]:
        """
        Intelligently select which images to animate
        
        Returns:
            Set of image paths to animate
        """
        print(f"\n{'='*70}")
        print("SELECTING IMAGES FOR TRUE ANIMATION")
        print(f"{'='*70}")
        
        # Filter to images only
        image_items = [item for item in media_items if item.file_type == 'image']
        total_images = len(image_items)
        
        if total_images == 0:
            return set()
        
        # Score each image
        scored_images = []
        for item in image_items:
            analysis = scene_analyses.get(item.file_path)
            score = self._score_image_for_animation(item, analysis)
            scored_images.append((item.file_path, score, analysis))
        
        # Sort by score
        scored_images.sort(key=lambda x: x[1], reverse=True)
        
        # Apply selection strategy
        selected = set()
        
        if self.args.animate_top:
            # Animate top N by score
            n = min(self.args.animate_top, total_images)
            selected = {img[0] for img in scored_images[:n]}
            print(f"Strategy: Animating top {n} images by quality score")
        
        elif self.args.animate_landscapes:
            # Animate only landscapes
            for path, score, analysis in scored_images:
                if analysis and analysis.scene_type:
                    scene_type = analysis.scene_type.lower()
                    if any(kw in scene_type for kw in ['outdoor', 'landscape', 'nature', 'scenic']):
                        selected.add(path)
            print(f"Strategy: Animating {len(selected)} landscape/outdoor scenes")
        
        elif self.args.animate_every:
            # Animate every Nth image
            for i in range(0, total_images, self.args.animate_every):
                selected.add(image_items[i].file_path)
            print(f"Strategy: Animating every {self.args.animate_every}th image")
        
        else:
            # Animate by percentage
            n = int(total_images * (self.args.animate_percent / 100))
            n = max(1, min(n, total_images))
            selected = {img[0] for img in scored_images[:n]}
            print(f"Strategy: Animating top {self.args.animate_percent}% ({n} images)")
        
        # Show selection summary
        print(f"\n📊 Selection Summary:")
        print(f"  • Total images: {total_images}")
        print(f"  • Selected for TRUE ANIMATION: {len(selected)}")
        print(f"  • Remaining for slideshow: {total_images - len(selected)}")
        
        return selected
    
    def _score_image_for_animation(self, item, analysis: SceneAnalysis) -> float:
        """Score image for animation priority (higher = more suitable)"""
        score = 50.0  # Base score
        
        if not analysis:
            return score
        
        # Scene type bonuses
        if analysis.scene_type:
            scene_type = analysis.scene_type.lower()
            
            # High priority for action/dynamic scenes
            if any(kw in scene_type for kw in ['action', 'sport', 'activity', 'motion']):
                score += 30
            
            # Good for landscapes
            if any(kw in scene_type for kw in ['outdoor', 'landscape', 'nature', 'scenic']):
                score += 20
            
            # Good for people/portraits
            if any(kw in scene_type for kw in ['person', 'people', 'portrait', 'group']):
                score += 15
            
            # Reduce for static scenes
            if any(kw in scene_type for kw in ['indoor', 'object', 'still']):
                score -= 10
        
        # Mood bonuses
        if analysis.mood and analysis.mood != 'neutral':
            score += 10
        
        return score
    
    def create_hybrid_movie(self):
        """Main pipeline for creating hybrid movies"""
        
        start_time = time.time()
        
        print("\n" + "="*70)
        print("HYBRID MOVIE CREATOR - TRUE ANIMATION VERSION")
        print("="*70)
        print(f"Animation Style: {self.args.style}")
        print(f"Slideshow Duration: {self.args.slideshow_duration}s")
        
        # STEP 1: Ingest Media
        print("\n" + "="*70)
        print("[STEP 1/8] MEDIA INGESTION")
        print("="*70)
        
        ingester = MediaIngester()
        media_items = ingester.ingest_directory(Config.INPUT_DIR)
        
        if not media_items:
            print("✗ No media files found!")
            return
        
        # Remove duplicates
        if not self.args.skip_duplicates:
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
        print("[STEP 2/8] SCENE ANALYSIS")
        print("="*70)
        
        analyzer = SceneAnalyzer(model_key='blip2-flan-t5-xl')
        scene_analyses = analyzer.analyze_media_collection(sorted_items)
        
        analysis_path = Config.OUTPUT_DIR / "scene_analysis.json"
        analyzer.export_analysis(scene_analyses, analysis_path)
        
        analyzer.unload_model()
        
        print(f"✓ Analyzed {len(scene_analyses)} scenes")
        
        # STEP 3: Select Images to Animate
        images_to_animate = self.select_images_to_animate(sorted_items, scene_analyses)
        images_for_slideshow = set([item.file_path for item in image_items]) - images_to_animate
        
        print(f"\n📊 PROCESSING BREAKDOWN:")
        print(f"  • Images for TRUE ANIMATION: {len(images_to_animate)}")
        print(f"  • Images for SLIDESHOW: {len(images_for_slideshow)}")
        print(f"  • Video clips: {len(video_items)}")
        print(f"  • Total items: {len(sorted_items)}")
        
        # Estimate time
        animation_time = len(images_to_animate) * 90  # ~90 sec per image with TRUE animation
        print(f"\n⏱️  Estimated animation time: {animation_time/60:.1f} minutes")
        
        # STEP 4: Animate Selected Images
        if images_to_animate:
            print("\n" + "="*70)
            print(f"[STEP 3/8] TRUE ANIMATION OF {len(images_to_animate)} IMAGES")
            print("="*70)
            print("⚠️  This creates FULL animation with character/scene movement")
            
            animator = ImageAnimator(
                animation_style=self.args.style,
                use_interpolation=True,
                enable_depth=False
            )
            
            animator.load_models()
            
            # Create output directory
            animated_dir = Config.OUTPUT_DIR / "animated_clips"
            animated_dir.mkdir(parents=True, exist_ok=True)
            
            # Animate in batches
            paths_to_animate = list(images_to_animate)
            batch_size = self.args.batch_size
            
            for i in range(0, len(paths_to_animate), batch_size):
                batch = paths_to_animate[i:i+batch_size]
                batch_num = i // batch_size + 1
                total_batches = (len(paths_to_animate) + batch_size - 1) // batch_size
                
                print(f"\n📦 Processing batch {batch_num}/{total_batches} ({len(batch)} images)")
                
                animated_clips = animator.batch_animate_images(
                    image_paths=batch,
                    scene_analyses=scene_analyses,
                    output_dir=animated_dir
                )
                
                # Map original paths to animated clips
                for j, clip_path in enumerate(animated_clips):
                    self.animated_clips[batch[j]] = clip_path
            
            animator.unload_models()
            
            print(f"\n✓ Animated {len(self.animated_clips)} images with TRUE animation")
        
        # STEP 5: Update Media Items
        print("\n" + "="*70)
        print("[STEP 4/8] PREPARING FINAL SEQUENCE")
        print("="*70)
        
        # Replace animated images with their clips
        for item in sorted_items:
            if item.file_path in self.animated_clips:
                item.file_path = self.animated_clips[item.file_path]
                item.file_type = 'video'
                item.file_format = 'mp4'
                print(f"  🎬 {item.file_path.name} → TRUE ANIMATION")
            elif item.file_path in images_for_slideshow:
                print(f"  📸 {item.file_path.name} → slideshow")
        
        # STEP 6: Create Sequence
        print("\n" + "="*70)
        print("[STEP 5/8] SEQUENCING")
        print("="*70)
        
        # Adjust durations
        engine = SequencingEngine()
        original_default = engine.DEFAULT_IMAGE_DURATION
        engine.DEFAULT_IMAGE_DURATION = self.args.slideshow_duration
        
        sequence = engine.create_sequence(sorted_items, scene_analyses)
        
        engine.DEFAULT_IMAGE_DURATION = original_default
        
        sequence_path = Config.OUTPUT_DIR / "video_sequence.json"
        engine.export_sequence(sequence, sequence_path)
        
        stats = engine.get_sequence_statistics(sequence)
        total_duration = stats['total_duration_seconds']
        
        print(f"✓ Final video duration: {stats['total_duration_minutes']:.2f} minutes")
        
        # STEP 7: Generate Audio
        print("\n" + "="*70)
        print("[STEP 6/8] AUDIO PREPARATION")
        print("="*70)
        
        audio_gen = AudioGenerator(fade_duration=self.args.fade)
        music_files = audio_gen.find_music_files(Config.INPUT_DIR)
        
        audio_path = Config.OUTPUT_DIR / "soundtrack.wav"
        
        if music_files:
            audio_path = audio_gen.prepare_audio_from_files(
                music_files=music_files,
                target_duration=total_duration,
                output_path=audio_path
            )
        else:
            audio_path = audio_gen.generate_soundtrack(
                sequence=sequence,
                duration=total_duration,
                output_path=audio_path
            )
        
        # Apply volume
        from pydub import AudioSegment
        audio = AudioSegment.from_wav(str(audio_path))
        audio = audio + (self.args.music_volume * 100 - 100)
        audio.export(str(audio_path), format="wav")
        
        print(f"✓ Audio ready")
        
        # STEP 8: Compose Final Video
        print("\n" + "="*70)
        print("[STEP 7/8] VIDEO COMPOSITION")
        print("="*70)
        
        composer = VideoComposer()
        output_path = Config.OUTPUT_DIR / f"hybrid_movie_{self.args.style}.mp4"
        
        result_path = composer.compose_video(
            sequence=sequence,
            audio_path=audio_path,
            output_path=output_path
        )
        
        # STEP 9: Summary
        elapsed_time = time.time() - start_time
        
        print("\n" + "="*70)
        print("🎬 HYBRID MOVIE COMPLETE!")
        print("="*70)
        
        file_size = result_path.stat().st_size / (1024 * 1024)
        
        print(f"\n🎞️  Output: {result_path}")
        print(f"💾 Size: {file_size:.2f} MB")
        print(f"⏱️  Duration: {total_duration/60:.2f} minutes")
        print(f"⏰ Processing time: {elapsed_time/60:.1f} minutes")
        print(f"\n📊 Composition:")
        print(f"  • TRUE ANIMATED clips: {len(self.animated_clips)}")
        print(f"  • Slideshow images: {len(images_for_slideshow)}")
        print(f"  • Video clips: {len(video_items)}")
        print(f"  • Total items: {len(sorted_items)}")
        print(f"\n🎨 Style: {self.args.style} animation + {self.args.slideshow_duration}s slideshow")
        
        print(f"\n🎥 Ready for YouTube!")
        print(f"✨ Perfect balance: Quality TRUE animation + Fast processing!")
        print(f"🎞️  Key moments are FULLY ANIMATED with character movement!")


def main():
    """Main entry point"""
    args = parse_arguments()
    
    print("\n" + "="*70)
    print("⚠️  HYBRID MOVIE REQUIREMENTS")
    print("="*70)
    print("This mode uses AnimateDiff for selected images:")
    print("  • NVIDIA GPU with 12+ GB VRAM (you have RTX 4080 ✅)")
    print("  • ~15GB model download on first run")
    print("  • 60-120 seconds per animated image")
    print("  • Slideshow images process instantly")
    print("="*70)
    
    confirm = input("\nContinue? (y/n): ").lower().strip()
    if confirm != 'y':
        print("Cancelled")
        return
    
    creator = HybridMovieCreator(args)
    
    try:
        creator.create_hybrid_movie()
    except KeyboardInterrupt:
        print("\n\n⚠️  Processing interrupted!")
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()