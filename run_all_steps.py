# ============================================================================
# AI Video Creator - Master Script
# Run all steps sequentially
# ============================================================================

"""
Master script to run entire video creation pipeline
"""

import sys
from pathlib import Path
from config import Config


def run_step(step_name: str, script_name: str) -> bool:
    """
    Run a step script
    
    Args:
        step_name: Name of the step
        script_name: Python script to run
        
    Returns:
        True if successful, False otherwise
    """
    print(f"\n{'#'*70}")
    print(f"# {step_name}")
    print(f"{'#'*70}\n")
    
    try:
        # Import and run the test function
        if script_name == "test_ingestion":
            from test_ingestion import test_ingestion
            test_ingestion()
        elif script_name == "test_scene_analysis":
            from test_scene_analysis import test_scene_analysis
            test_scene_analysis()
        elif script_name == "test_sequencing":
            from test_sequencing import test_sequencing
            test_sequencing()
        elif script_name == "test_audio_generation":
            from test_audio_generation import test_audio_generation
            test_audio_generation()
        elif script_name == "test_video_composition":
            from test_video_composition import test_video_composition_auto
            test_video_composition_auto()
        
        return True
        
    except Exception as e:
        print(f"\n✗ ERROR in {step_name}:")
        print(f"  {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run complete video creation pipeline"""
    
    print("="*70)
    print("AI VIDEO CREATOR - COMPLETE PIPELINE")
    print("="*70)
    print("\nThis will run all steps:")
    print("  Step 2: Media Ingestion")
    print("  Step 3: Scene Analysis (uses AI, takes time)")
    print("  Step 4: Content Sequencing")
    print("  Step 5: Audio Generation (uses AI, takes time)")
    print("  Step 6: Video Composition (rendering, takes longest)")
    print("\nTotal estimated time: 15-30 minutes")
    print()
    
    # Check if input folder has files
    if not Config.INPUT_DIR.exists() or not list(Config.INPUT_DIR.iterdir()):
        print("✗ No files in input folder!")
        print(f"Please add images/videos to: {Config.INPUT_DIR}")
        return
    
    confirm = input("Continue with full pipeline? (y/n): ").lower().strip()
    if confirm != 'y':
        print("\nCancelled")
        return
    
    # Run all steps
    steps = [
        ("STEP 2: Media Ingestion & Metadata Extraction", "test_ingestion"),
        ("STEP 3: Scene Analysis with Vision AI", "test_scene_analysis"),
        ("STEP 4: Content Sequencing", "test_sequencing"),
        ("STEP 5: Audio Generation with Music AI", "test_audio_generation"),
        ("STEP 6: Video Composition & Rendering", "test_video_composition"),
    ]
    
    completed_steps = []
    
    for step_name, script_name in steps:
        success = run_step(step_name, script_name)
        
        if success:
            completed_steps.append(step_name)
        else:
            print(f"\n{'='*70}")
            print("PIPELINE STOPPED DUE TO ERROR")
            print(f"{'='*70}")
            print(f"\nCompleted steps:")
            for completed in completed_steps:
                print(f"  ✓ {completed}")
            print(f"\nFailed at: {step_name}")
            return
    
    # Success!
    print(f"\n{'='*70}")
    print("PIPELINE COMPLETE!")
    print(f"{'='*70}")
    
    output_video = Config.OUTPUT_DIR / "final_video.mp4"
    if output_video.exists():
        file_size = output_video.stat().st_size / (1024 * 1024)
        print(f"\n✓ Video created successfully!")
        print(f"  Location: {output_video}")
        print(f"  Size: {file_size:.2f} MB")
        print(f"\n  Ready to upload to YouTube!")
    else:
        print("\n⚠ Video file not found, but pipeline completed")


if __name__ == "__main__":
    main()