#!/usr/bin/env python3
# ============================================================================
# Diagnostic Script - Check Processing Status
# ============================================================================

"""
Diagnose why only 5 files were processed instead of 83
"""

from pathlib import Path
import json
from config import Config


def diagnose():
    """Run diagnostics"""
    
    print("="*70)
    print("DIAGNOSTIC REPORT")
    print("="*70)
    
    # Check input folder
    print(f"\n[1] INPUT FOLDER: {Config.INPUT_DIR}")
    print("-"*70)
    
    image_formats = {'.jpg', '.jpeg', '.png', '.heic'}
    video_formats = {'.mp4', '.mov', '.avi'}
    
    images = []
    videos = []
    
    for f in Config.INPUT_DIR.rglob("*"):
        if f.is_file():
            if f.suffix.lower() in image_formats:
                images.append(f)
            elif f.suffix.lower() in video_formats:
                videos.append(f)
    
    print(f"Images found: {len(images)}")
    print(f"Videos found: {len(videos)}")
    print(f"Total media: {len(images) + len(videos)}")
    
    # Check manifest
    print(f"\n[2] MEDIA MANIFEST")
    print("-"*70)
    
    manifest_path = Config.OUTPUT_DIR / "media_manifest.json"
    if manifest_path.exists():
        with open(manifest_path, 'r') as f:
            manifest = json.load(f)
        print(f"✓ Manifest exists")
        print(f"  Items in manifest: {manifest['total_items']}")
        print(f"  Statistics:")
        print(f"    Images: {manifest['statistics']['images']}")
        print(f"    Videos: {manifest['statistics']['videos']}")
    else:
        print(f"❌ Manifest not found")
    
    # Check scene analysis
    print(f"\n[3] SCENE ANALYSIS")
    print("-"*70)
    
    analysis_path = Config.OUTPUT_DIR / "scene_analysis.json"
    if analysis_path.exists():
        with open(analysis_path, 'r') as f:
            analysis = json.load(f)
        
        analyzed_count = len(analysis['analyses'])
        print(f"✓ Scene analysis exists")
        print(f"  Items analyzed: {analyzed_count}")
        print(f"  Model used: {analysis['model_used']}")
        
        if manifest_path.exists():
            manifest_count = manifest['total_items']
            missing = manifest_count - analyzed_count
            
            if missing > 0:
                print(f"\n⚠️  PROBLEM FOUND:")
                print(f"  {missing} files in manifest NOT analyzed!")
                print(f"  Only {analyzed_count}/{manifest_count} files were analyzed")
                
                # Show which files were analyzed
                analyzed_files = set(Path(p) for p in analysis['analyses'].keys())
                print(f"\n  Analyzed files:")
                for i, path in enumerate(list(analyzed_files)[:5], 1):
                    print(f"    {i}. {path.name}")
                if len(analyzed_files) > 5:
                    print(f"    ... and {len(analyzed_files)-5} more")
            else:
                print(f"✓ All files analyzed")
    else:
        print(f"❌ Scene analysis not found")
    
    # Check sequence
    print(f"\n[4] VIDEO SEQUENCE")
    print("-"*70)
    
    sequence_path = Config.OUTPUT_DIR / "video_sequence.json"
    if sequence_path.exists():
        with open(sequence_path, 'r') as f:
            sequence = json.load(f)
        
        print(f"✓ Sequence exists")
        print(f"  Items in sequence: {sequence['total_items']}")
        print(f"  Duration: {sequence['total_duration']:.1f}s ({sequence['total_duration']/60:.1f} min)")
        print(f"  Scene groups: {sequence['num_groups']}")
        
        if analysis_path.exists():
            if sequence['total_items'] != analyzed_count:
                print(f"\n⚠️  WARNING:")
                print(f"  Sequence has {sequence['total_items']} items")
                print(f"  But {analyzed_count} files were analyzed")
    else:
        print(f"❌ Sequence not found")
    
    # Check for cached analyses
    print(f"\n[5] CACHE STATUS")
    print("-"*70)
    
    cache_dir = Config.CACHE_DIR / "scene_analysis"
    if cache_dir.exists():
        cached_files = list(cache_dir.glob("*.json"))
        print(f"✓ Cache directory exists")
        print(f"  Cached analyses: {len(cached_files)}")
    else:
        print(f"❌ No cache directory")
    
    # Final diagnosis
    print(f"\n{'='*70}")
    print("DIAGNOSIS")
    print("="*70)
    
    if manifest_path.exists() and analysis_path.exists():
        manifest_count = manifest['total_items']
        analyzed_count = len(analysis['analyses'])
        
        if analyzed_count < manifest_count:
            print(f"\n❌ ROOT CAUSE IDENTIFIED:")
            print(f"\n  Only {analyzed_count} out of {manifest_count} files were analyzed.")
            print(f"  This is why only {analyzed_count} files appear in the video.")
            print(f"\n  SOLUTION:")
            print(f"  Run: python process_all_media.py")
            print(f"  This will analyze ALL {manifest_count} files.")
        else:
            print(f"\n✓ All files have been analyzed!")
            print(f"  You should have all {manifest_count} files in your video.")
    else:
        print(f"\n⚠️  Missing required files. Need to run full processing.")
        print(f"  Run: python process_all_media.py")
    
    print()


if __name__ == "__main__":
    diagnose()
    