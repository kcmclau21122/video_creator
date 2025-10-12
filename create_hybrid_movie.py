#!/usr/bin/env python3
# ============================================================================
# AI Video Creator - HYBRID Movie Pipeline
# Intelligently combines animation and slideshow for large collections
# ============================================================================

"""
Hybrid Movie Creator
For large photo collections (60-200+ images):
- Animates KEY moments (landscapes, important scenes)
- Slideshow for other photos (portraits, similar scenes)
- Best of both worlds: Quality + Speed

Usage:
    python create_hybrid_movie.py                     # Auto-select 30% for animation
    python create_hybrid_movie.py --animate-percent 50  # Animate 50% of images
    python create_hybrid_movie.py --animate-top 20    # Animate top 20 images only
    python create_hybrid_movie.py --animate-landscapes # Animate only landscapes
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
        '--animation-style',
        choices=['cinematic', 'dynamic', 'subtle', 'landscape'],
        default='cinematic',
        help='Style for animated clips'
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
    """Creates hybrid animated/slideshow movies"""
    
    def __init__(self, args):
        self.args = args
        self.animated_clips = {}  # Map of image_path -> animated_clip_path
    
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
        print("SELECTING IMAGES FOR ANIMATION")
        print(f"{'='*70}")
        
        # Filter to images only
        image_items = [item for item in media_items if item.file_type == 'image']
        total_images = len(image_items)
        
        if self.args.animate_landscapes:
            return self._select_landscapes(image_items, scene_analyses)
        elif self.args.animate_top:
            return self._select_top_n(image_items, scene_analyses, self.args.animate_top)
        elif self.args.animate_every:
            return self._select_every_nth(image_items, self.args.animate_every)
        else:
            # Default: animate percent
            n_to_animate = int(total_images * self.args.animate_percent / 100)
            return self._select_top_n(image_items, scene_analyses, n_to_animate)
    
    def _select_landscapes(
        self,
        image_items: List,
        scene_analyses: dict
    ) -> Set[Path]:
        """Select landscape and outdoor scenes"""
        selected = set()
        
        for item in image_items:
            analysis = scene_analyses.get(item.file_path)
            if not analysis:
                continue
            
            scene_type = (analysis.scene_type or '').lower()
            if any(keyword in scene_type for keyword in ['outdoor', 'landscape', 'nature', 'scenic']):
                selected.add(item.file_path)
        
        print(f"Selected {len(selected)} landscape/outdoor images for animation")
        return selected
    
    def _select_top_n(
        self,
        image_items: List,
        scene_analyses: dict,
        n: int
    ) -> Set[Path]:
        """Select top N images by quality score"""
        # Score images based on:
        # - Complexity (more objects = higher score)
        # - Outdoor scenes (better for animation)
        # - Image quality (resolution)
        
        scored_items = []
        
        for item in image_items:
            analysis = scene_analyses.get(item.file_path)
            score = 0
            
            if analysis:
                # Object complexity
                score += len(analysis.objects or []) * 10
                
                # Scene type bonus
                scene_type = (analysis.scene_type or '').lower()
                if 'outdoor' in scene_type or 'landscape' in scene_type:
                    score += 30
                elif 'nature' in scene_type:
                    score += 25
                
                # Resolution bonus
                if item.width and item.height:
                    megapixels = (item.width * item.height) / 1_000_000
                    score += min(megapixels * 5, 20)  # Cap at 20 points
            
            scored_items.append((item.file_path, score))
        
        # Sort by score and take top N
        scored_items.sort(key=lambda x: x[1], reverse=True)
        selected = set([path for path, score in scored_items[:n]])
        
        print(f"Selected top {len(selected)} images for animation based on quality score")
        print(f"Score range: {scored_items[0][1]:.0f} (best) to {scored_items[min(n-1, len(scored_items)-1)][1]:.0f}")
        
        return selected
    
    def _select_every_nth(
        self,
        image_items: List,
        n: int
    ) -> Set[Path]:
        """Select every Nth image"""
        selected = set([item.file_path for i, item in enumerate(image_items) if i % n == 0])
        print(f"Selected every {n}th image = {len(selected)} images for animation")
        return selected
    
    def create_hybrid_movie(self):
        """Main hybrid movie creation pipeline"""
        
        print("="*70)
        print("🎬 HYBRID MOVIE CREATOR")
        print("="*70)
        print("Combines animation (slow, high quality) with slideshow (fast)")
        print("Perfect for large collections (60-200+ images)")
        print("="*70)
        
        # STEP 1: Ingest Media
        print("\n" + "="*70)
        print("[STEP 1/8] MEDIA INGESTION")
        print("="*70)
        
        ingester = MediaIngester()
        media_items = ingester.process_folder(Config.INPUT_DIR)
        
        if not media_items:
            print("❌ No media files found!")
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
        
        print(f"✅ Found {len(image_items)} images and {len(video_items)} videos")
        
        # STEP 2: Scene Analysis
        print("\n" + "="*70)
        print("[STEP 2/8] SCENE ANALYSIS")
        print("="*70)
        
        analyzer = SceneAnalyzer(model_key='blip2-flan-t5-xl')
        scene_analyses = analyzer.analyze_media_collection(sorted_items)
        
        analysis_path = Config.OUTPUT_DIR / "scene_analysis.json"
        analyzer.export_analysis(scene_analyses, analysis_path)
        
        analyzer.unload_model()
        
        print(f"✅ Analyzed {len(scene_analyses)} scenes")
        
        # STEP 3: Select Images to Animate
        images_to_animate = self.select_images_to_animate(sorted_items, scene_analyses)
        images_for_slideshow = set([item.file_path for item in image_items]) - images_to_animate
        
        print(f"\n📊 PROCESSING BREAKDOWN:")
        print(f"  • Images to ANIMATE: {len(images_to_animate)}")
        print(f"  • Images for SLIDESHOW: {len(images_for_slideshow)}")
        print(f"  • Video clips: {len(video_items)}")
        print(f"  • Total items: {len(sorted_items)}")
        
        # Estimate time
        animation_time = len(images_to_animate) * 45  # ~45 sec per image
        print(f"\n⏱️  Estimated animation time: {animation_time/60:.1f} minutes")
        
        # STEP 4: Animate Selected Images
        if images_to_animate:
            print("\n" + "="*70)
            print(f"[STEP 3/8] ANIMATING {len(images_to_animate)} SELECTED IMAGES")
            print("="*70)
            
            animator = ImageAnimator(
                animation_style=self.args.animation_style,
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
            
            print(f"\n✅ Animated {len(self.animated_clips)} images")
        
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
                print(f"  🎬 {item.file_path.name} → animated")
            elif item.file_path in images_for_slideshow:
                print(f"  📸 {item.file_path.name} → slideshow")
        
        # STEP 6: Create Sequence
        print("\n" + "="*70)
        print("[STEP 5/8] SEQUENCING")
        print("="*70)
        
        # Adjust durations
        engine = SequencingEngine()
        # Override default duration for slideshow images
        original_default = engine.DEFAULT_IMAGE_DURATION
        engine.DEFAULT_IMAGE_DURATION = self.args.slideshow_duration
        
        sequence = engine.create_sequence(sorted_items, scene_analyses)
        
        # Restore
        engine.DEFAULT_IMAGE_DURATION = original_default
        
        sequence_path = Config.OUTPUT_DIR / "video_sequence.json"
        engine.export_sequence(sequence, sequence_path)
        
        stats = engine.get_sequence_statistics(sequence)
        total_duration = stats['total_duration_seconds']
        
        print(f"✅ Final video duration: {stats['total_duration_minutes']:.2f} minutes")
        
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
        
        print(f"✅ Audio ready")
        
        # STEP 8: Compose Final Video
        print("\n" + "="*70)
        print("[STEP 7/8] VIDEO COMPOSITION")
        print("="*70)
        
        composer = VideoComposer()
        output_path = Config.OUTPUT_DIR / "hybrid_movie.mp4"
        
        result_path = composer.compose_video(
            sequence=sequence,
            audio_path=audio_path,
            output_path=output_path
        )
        
        # STEP 9: Summary
        print("\n" + "="*70)
        print("🎬 HYBRID MOVIE COMPLETE!")
        print("="*70)
        
        file_size = result_path.stat().st_size / (1024 * 1024)
        
        print(f"\n📽️  Output: {result_path}")
        print(f"💾 Size: {file_size:.2f} MB")
        print(f"⏱️  Duration: {total_duration/60:.2f} minutes")
        print(f"\n📊 Composition:")
        print(f"  • Animated clips: {len(self.animated_clips)}")
        print(f"  • Slideshow images: {len(images_for_slideshow)}")
        print(f"  • Video clips: {len(video_items)}")
        print(f"  • Total items: {len(sorted_items)}")
        print(f"\n🎨 Style: {self.args.animation_style} animation + {self.args.slideshow_duration}s slideshow")
        
        print(f"\n🎥 Ready for YouTube!")
        print(f"✨ Best of both worlds: Quality animation + Fast processing!")
        
        return result_path


def main():
    """Main entry point"""
    args = parse_arguments()
    
    creator = HybridMovieCreator(args)
    
    try:
        creator.create_hybrid_movie()
    except KeyboardInterrupt:
        print("\n\n❌ Cancelled by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()