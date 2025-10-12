#!/usr/bin/env python3
# ============================================================================
# AI Video Creator - Optimized Large Batch Processing
# Memory-efficient processing for 100+ images
# ============================================================================

"""
Optimized Large Batch Processor
For processing 100-200+ images efficiently:
- Processes in small batches to prevent memory issues
- Auto-cleanup between batches
- Resume capability if interrupted
- Progress tracking and time estimates

Usage:
    python optimize_large_batch.py                    # Auto batch size
    python optimize_large_batch.py --batch-size 5     # Custom batch size
    python optimize_large_batch.py --resume           # Resume interrupted job
"""

import argparse
import json
from pathlib import Path
from typing import List, Dict, Set
import time
import gc
import torch

from config import Config
from media_ingester import MediaIngester
from scene_analyzer import SceneAnalyzer
from image_animator import ImageAnimator
from sequencing_engine import SequencingEngine
from audio_generator import AudioGenerator
from video_composer import VideoComposer


class OptimizedBatchProcessor:
    """Memory-optimized batch processor for large collections"""
    
    def __init__(self, batch_size: int = 10, resume: bool = False):
        self.batch_size = batch_size
        self.resume = resume
        self.progress_file = Config.OUTPUT_DIR / "batch_progress.json"
        self.progress = self._load_progress() if resume else {}
    
    def _load_progress(self) -> Dict:
        """Load previous progress if resuming"""
        if self.progress_file.exists():
            with open(self.progress_file, 'r') as f:
                return json.load(f)
        return {}
    
    def _save_progress(self):
        """Save current progress"""
        with open(self.progress_file, 'w') as f:
            json.dump(self.progress, f, indent=2)
    
    def _cleanup_memory(self):
        """Aggressive memory cleanup"""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    
    def process_large_collection(
        self,
        animation_style: str = 'cinematic',
        music_volume: float = 0.5,
        fade_duration: int = 3
    ):
        """Process large collection in optimized batches"""
        
        start_time = time.time()
        
        print("="*70)
        print("🚀 OPTIMIZED LARGE BATCH PROCESSOR")
        print("="*70)
        print(f"Batch size: {self.batch_size} images")
        print(f"Resume mode: {'Enabled' if self.resume else 'Disabled'}")
        print("="*70)
        
        # STEP 1: Quick scan to count images
        print("\n[SCAN] Counting images...")
        
        ingester = MediaIngester()
        media_items = ingester.process_folder(Config.INPUT_DIR)
        sorted_items = ingester.sort_by_datetime(media_items)
        
        image_items = [item for item in sorted_items if item.file_type == 'image']
        video_items = [item for item in sorted_items if item.file_type == 'video']
        
        total_images = len(image_items)
        
        print(f"Found: {total_images} images, {len(video_items)} videos")
        
        if total_images == 0:
            print("❌ No images to process!")
            return
        
        # Estimate time
        seconds_per_image = 45
        total_seconds = total_images * seconds_per_image
        print(f"\n⏱️  Estimated time: {total_seconds/60:.0f} minutes ({total_seconds/3600:.1f} hours)")
        
        # STEP 2: Scene Analysis (if not done)
        if not self.progress.get('scene_analysis_done', False):
            print("\n" + "="*70)
            print("[STEP 1] SCENE ANALYSIS")
            print("="*70)
            
            analyzer = SceneAnalyzer(model_key='blip2-flan-t5-xl')
            scene_analyses = analyzer.analyze_media_collection(sorted_items)
            
            analysis_path = Config.OUTPUT_DIR / "scene_analysis.json"
            analyzer.export_analysis(scene_analyses, analysis_path)
            
            analyzer.unload_model()
            self._cleanup_memory()
            
            self.progress['scene_analysis_done'] = True
            self._save_progress()
            
            print(f"✅ Scene analysis complete")
        else:
            print("\n[STEP 1] Scene analysis already done (resuming)")
            analysis_path = Config.OUTPUT_DIR / "scene_analysis.json"
            with open(analysis_path, 'r') as f:
                analysis_data = json.load(f)
            
            from scene_analyzer import SceneAnalysis
            scene_analyses = {
                Path(k): SceneAnalysis.from_dict(v) 
                for k, v in analysis_data['analyses'].items()
            }
        
        # STEP 3: Batch Animation
        print("\n" + "="*70)
        print("[STEP 2] BATCH ANIMATION")
        print("="*70)
        
        animated_dir = Config.OUTPUT_DIR / "animated_clips"
        animated_dir.mkdir(parents=True, exist_ok=True)
        
        # Track completed batches
        completed_batches = set(self.progress.get('completed_batches', []))
        
        # Split into batches
        total_batches = (total_images + self.batch_size - 1) // self.batch_size
        
        animator = None
        batch_start_time = time.time()
        
        for batch_idx in range(total_batches):
            batch_num = batch_idx + 1
            
            # Skip if already processed
            if batch_idx in completed_batches:
                print(f"\n✓ Batch {batch_num}/{total_batches} already completed (skipping)")
                continue
            
            # Get batch items
            start_idx = batch_idx * self.batch_size
            end_idx = min(start_idx + self.batch_size, total_images)
            batch_items = image_items[start_idx:end_idx]
            
            print(f"\n{'='*70}")
            print(f"📦 BATCH {batch_num}/{total_batches}")
            print(f"{'='*70}")
            print(f"Images: {len(batch_items)} (#{start_idx+1} to #{end_idx})")
            
            # Estimate time for this batch
            if batch_idx > 0:
                elapsed = time.time() - batch_start_time
                avg_time_per_batch = elapsed / batch_idx
                remaining_batches = total_batches - batch_idx
                eta_seconds = avg_time_per_batch * remaining_batches
                print(f"⏱️  ETA: {eta_seconds/60:.0f} minutes ({eta_seconds/3600:.1f} hours)")
            
            # Load animator if not loaded
            if animator is None:
                animator = ImageAnimator(
                    animation_style=animation_style,
                    use_interpolation=True,
                    enable_depth=False
                )
                animator.load_models()
            
            # Animate batch
            try:
                batch_paths = [item.file_path for item in batch_items]
                animated_clips = animator.batch_animate_images(
                    image_paths=batch_paths,
                    scene_analyses=scene_analyses,
                    output_dir=animated_dir
                )
                
                # Mark batch as complete
                completed_batches.add(batch_idx)
                self.progress['completed_batches'] = list(completed_batches)
                self._save_progress()
                
                print(f"✅ Batch {batch_num} complete")
                
                # Memory cleanup every 3 batches
                if batch_num % 3 == 0:
                    print("🧹 Cleaning up memory...")
                    self._cleanup_memory()
                
            except Exception as e:
                print(f"\n❌ Error in batch {batch_num}: {e}")
                print("Progress saved. You can resume with --resume flag")
                raise
        
        # Unload animator
        if animator:
            animator.unload_models()
            self._cleanup_memory()
        
        print(f"\n✅ All {total_batches} batches animated!")
        
        # STEP 4: Create Final Sequence
        print("\n" + "="*70)
        print("[STEP 3] CREATING FINAL SEQUENCE")
        print("="*70)
        
        # Update media items to point to animated clips
        for item in sorted_items:
            if item.file_type == 'image':
                animated_path = animated_dir / f"{item.file_path.stem}_animated.mp4"
                if animated_path.exists():
                    item.file_path = animated_path
                    item.file_type = 'video'
                    item.file_format = 'mp4'
        
        # Create sequence
        engine = SequencingEngine()
        sequence = engine.create_sequence(sorted_items, scene_analyses)
        
        sequence_path = Config.OUTPUT_DIR / "video_sequence.json"
        engine.export_sequence(sequence, sequence_path)
        
        stats = engine.get_sequence_statistics(sequence)
        total_duration = stats['total_duration_seconds']
        
        print(f"✅ Sequence created: {stats['total_duration_minutes']:.2f} minutes")
        
        # STEP 5: Generate Audio
        print("\n" + "="*70)
        print("[STEP 4] AUDIO PREPARATION")
        print("="*70)
        
        audio_gen = AudioGenerator(fade_duration=fade_duration)
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
        audio = audio + (music_volume * 100 - 100)
        audio.export(str(audio_path), format="wav")
        
        print(f"✅ Audio ready")
        
        # STEP 6: Compose Final Video
        print("\n" + "="*70)
        print("[STEP 5] VIDEO COMPOSITION")
        print("="*70)
        
        composer = VideoComposer()
        output_path = Config.OUTPUT_DIR / "large_batch_movie.mp4"
        
        result_path = composer.compose_video(
            sequence=sequence,
            audio_path=audio_path,
            output_path=output_path
        )
        
        # Clean up progress file
        if self.progress_file.exists():
            self.progress_file.unlink()
        
        # Final summary
        total_time = time.time() - start_time
        
        print("\n" + "="*70)
        print("🎬 LARGE BATCH PROCESSING COMPLETE!")
        print("="*70)
        
        file_size = result_path.stat().st_size / (1024 * 1024)
        
        print(f"\n📽️  Output: {result_path}")
        print(f"💾 Size: {file_size:.2f} MB")
        print(f"⏱️  Duration: {total_duration/60:.2f} minutes")
        print(f"🕒 Processing time: {total_time/60:.0f} minutes ({total_time/3600:.1f} hours)")
        print(f"\n📊 Processed:")
        print(f"  • Images animated: {total_images}")
        print(f"  • Video clips: {len(video_items)}")
        print(f"  • Total items: {len(sorted_items)}")
        print(f"  • Average time per image: {total_time/total_images:.1f} seconds")
        
        print(f"\n🎥 Ready for YouTube!")
        print(f"✨ Successfully processed {total_images} images!")
        
        return result_path


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Optimized batch processing for large image collections'
    )
    
    parser.add_argument(
        '--batch-size',
        type=int,
        default=10,
        help='Images per batch (default: 10, lower = less memory)'
    )
    
    parser.add_argument(
        '--resume',
        action='store_true',
        help='Resume interrupted processing'
    )
    
    parser.add_argument(
        '--animation-style',
        choices=['cinematic', 'dynamic', 'subtle', 'landscape'],
        default='cinematic',
        help='Animation style'
    )
    
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
    
    return parser.parse_args()


def main():
    """Main entry point"""
    args = parse_arguments()
    
    processor = OptimizedBatchProcessor(
        batch_size=args.batch_size,
        resume=args.resume
    )
    
    try:
        processor.process_large_collection(
            animation_style=args.animation_style,
            music_volume=args.music_volume,
            fade_duration=args.fade
        )
    except KeyboardInterrupt:
        print("\n\n⚠️  Processing interrupted!")
        print("Progress has been saved.")
        print("Resume with: python optimize_large_batch.py --resume")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        print("\n💡 You can resume with: python optimize_large_batch.py --resume")


if __name__ == "__main__":
    main()