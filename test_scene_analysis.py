# ============================================================================
# AI Video Creator - Step 3: Test Scene Analysis
# ============================================================================

"""
Test script for scene analysis module
"""

from pathlib import Path
from media_ingester import MediaIngester
from scene_analyzer import SceneAnalyzer
from config import Config


def test_scene_analysis():
    """Test scene analysis functionality"""
    
    print("="*70)
    print("TESTING SCENE ANALYSIS MODULE")
    print("="*70)
    
    # Step 1: Ingest media
    print("\n[Step 1] Loading media files...")
    ingester = MediaIngester()
    media_items = ingester.process_folder(Config.INPUT_DIR)
    
    if not media_items:
        print("\n✗ No media files found")
        print("Please add images/videos to:", Config.INPUT_DIR)
        return
    
    # Sort by datetime
    sorted_items = ingester.sort_by_datetime(media_items)
    print(f"✓ Loaded {len(sorted_items)} media files")
    
    # Step 2: Analyze scenes
    print("\n[Step 2] Analyzing scenes...")
    print("This may take a few minutes depending on the number of files...\n")
    
    analyzer = SceneAnalyzer(model_key='blip2-flan-t5-xl')
    
    # Analyze first 5 items for testing
    test_items = sorted_items[:5]
    print(f"Testing with first {len(test_items)} items\n")
    
    analyses = analyzer.analyze_media_collection(test_items)
    
    # Display results
    print(f"\n{'='*70}")
    print("SCENE ANALYSIS RESULTS")
    print(f"{'='*70}\n")
    
    for i, (path, analysis) in enumerate(analyses.items(), 1):
        print(f"{i}. {path.name}")
        print(f"   Caption: {analysis.caption}")
        print(f"   Mood: {analysis.mood}")
        print(f"   Type: {analysis.scene_type}")
        print(f"   Lighting: {analysis.lighting}")
        
        if analysis.dominant_colors:
            print(f"   Colors: {', '.join(analysis.dominant_colors)}")
        
        if analysis.objects:
            print(f"   Objects: {', '.join(analysis.objects)}")
        
        if analysis.time_of_day:
            print(f"   Time: {analysis.time_of_day}")
        
        print(f"\n   Description: {analysis.detailed_description}")
        print()
    
    # Export results
    print(f"{'='*70}")
    print("EXPORTING RESULTS")
    print(f"{'='*70}")
    
    output_path = Config.OUTPUT_DIR / "scene_analysis.json"
    analyzer.export_analysis(analyses, output_path)
    
    # Unload model
    analyzer.unload_model()
    
    print(f"\n{'='*70}")
    print("✓ TEST COMPLETE")
    print(f"{'='*70}")
    print("\nResults saved to:", output_path)
    print("\nNext steps:")
    print("1. Review the scene analysis results")
    print("2. Adjust model or parameters if needed")
    print("3. Ready for Step 4: Content Sequencing")


if __name__ == "__main__":
    test_scene_analysis()