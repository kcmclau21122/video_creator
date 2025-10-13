"""
Main Script - Choose between Slideshow or Animated Movie
Integrates with your existing process_all_media and create_movie functions
"""

import os
import sys
import argparse
from pathlib import Path

# Import your existing modules (adjust paths as needed)
# from your_existing_module import process_all_media, create_movie

# Import new animation module
from animated_movie_creator import AnimatedMovieCreator


class VideoCreationPipeline:
    """
    Unified pipeline for creating both slideshows and animated videos
    """
    
    def __init__(self, input_dir, output_dir="output", music_dir="music"):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.music_dir = Path(music_dir)
        
        # Create directories
        self.output_dir.mkdir(exist_ok=True)
        
        # Paths for intermediate files
        self.sequence_file = self.output_dir / "video_sequence.json"
        self.soundtrack_file = self.output_dir / "soundtrack.wav"
    
    def create_slideshow(self, style="anime", **kwargs):
        """
        Create traditional slideshow video using existing pipeline
        
        This calls your existing process_all_media and create_movie functions
        """
        print("\n" + "="*70)
        print("CREATING SLIDESHOW VIDEO")
        print("="*70)
        
        # Call your existing functions here
        # Example (adjust to match your actual function signatures):
        """
        from your_module import process_all_media, create_movie
        
        # Process media and create sequence
        process_all_media(
            input_dir=str(self.input_dir),
            output_dir=str(self.output_dir),
            style=style,
            **kwargs
        )
        
        # Create slideshow movie
        output_path = create_movie(
            sequence_file=str(self.sequence_file),
            audio_file=str(self.soundtrack_file),
            output_dir=str(self.output_dir),
            style=style,
            **kwargs
        )
        """
        
        # Placeholder for demonstration
        print("📽️  Running existing slideshow pipeline...")
        print(f"   Input: {self.input_dir}")
        print(f"   Style: {style}")
        print(f"   Output: {self.output_dir}")
        print("\n⚠️  Replace this section with your actual process_all_media() and create_movie() calls")
        
        output_path = self.output_dir / f"slideshow_{style}.mp4"
        
        print(f"\n✓ Slideshow created: {output_path}")
        return str(output_path)
    
    def create_animated_movie(self, 
                            frames_per_image=50,
                            motion_intensity=127,
                            transition_frames=15,
                            fps=25,
                            device="cuda",
                            **kwargs):
        """
        Create fully animated video with AI-generated motion
        
        Uses the AnimatedMovieCreator for true animation
        """
        print("\n" + "="*70)
        print("CREATING ANIMATED MOVIE (AI-POWERED)")
        print("="*70)
        
        # First, we need the sequence and audio prepared
        # You can reuse your existing media processing for this:
        """
        from your_module import process_all_media
        
        process_all_media(
            input_dir=str(self.input_dir),
            output_dir=str(self.output_dir),
            create_video=False  # Only prepare sequence and audio
        )
        """
        
        print("\n📋 Using existing sequence and soundtrack...")
        print(f"   Sequence: {self.sequence_file}")
        print(f"   Audio: {self.soundtrack_file}")
        
        # Check if sequence exists
        if not self.sequence_file.exists():
            print(f"\n❌ Sequence file not found: {self.sequence_file}")
            print("   Run slideshow creation first to generate sequence, or")
            print("   Call your process_all_media() function to create it.")
            sys.exit(1)
        
        # Initialize animator
        animator = AnimatedMovieCreator(
            device=device,
            fps=fps,
            output_dir=str(self.output_dir)
        )
        
        # Generate animated movie
        output_path = animator.create_animated_movie(
            sequence_file=str(self.sequence_file),
            audio_file=str(self.soundtrack_file),
            output_filename="animated_movie.mp4",
            frames_per_image=frames_per_image,
            motion_intensity=motion_intensity,
            transition_frames=transition_frames
        )
        
        return output_path
    
    def create_both(self, **kwargs):
        """
        Create both slideshow and animated versions
        """
        print("\n" + "="*70)
        print("CREATING BOTH VERSIONS")
        print("="*70)
        
        # Create slideshow first
        slideshow_path = self.create_slideshow(**kwargs)
        
        # Then create animated version
        animated_path = self.create_animated_movie(**kwargs)
        
        return {
            'slideshow': slideshow_path,
            'animated': animated_path
        }


def main():
    parser = argparse.ArgumentParser(
        description='Create videos from images - Slideshow or Animated',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Create slideshow (traditional)
  python main_with_animation_option.py --input photos/ --mode slideshow
  
  # Create animated movie (AI-powered)
  python main_with_animation_option.py --input photos/ --mode animated
  
  # Create both versions
  python main_with_animation_option.py --input photos/ --mode both
  
  # Customize animation settings
  python main_with_animation_option.py --input photos/ --mode animated \\
      --frames-per-image 75 --motion 180 --transition-frames 20
        """
    )
    
    # Required arguments
    parser.add_argument('--input', required=True, 
                       help='Input directory with images/videos')
    parser.add_argument('--mode', required=True,
                       choices=['slideshow', 'animated', 'both'],
                       help='Video creation mode')
    
    # Optional arguments
    parser.add_argument('--output', default='output',
                       help='Output directory (default: output)')
    parser.add_argument('--music', default='music',
                       help='Music directory (default: music)')
    parser.add_argument('--style', default='anime',
                       help='Visual style for slideshow (default: anime)')
    
    # Animation-specific arguments
    parser.add_argument('--frames-per-image', type=int, default=50,
                       help='Frames per image for animation (default: 50 = 2s at 25fps)')
    parser.add_argument('--motion', type=int, default=127,
                       help='Motion intensity 0-255 (default: 127 = moderate)')
    parser.add_argument('--transition-frames', type=int, default=15,
                       help='Transition frames between scenes (default: 15)')
    parser.add_argument('--fps', type=int, default=25,
                       help='Frames per second (default: 25)')
    parser.add_argument('--device', default='cuda', choices=['cuda', 'cpu'],
                       help='Device for animation (default: cuda)')
    
    args = parser.parse_args()
    
    # Initialize pipeline
    pipeline = VideoCreationPipeline(
        input_dir=args.input,
        output_dir=args.output,
        music_dir=args.music
    )
    
    # Display configuration
    print("\n" + "="*70)
    print("VIDEO CREATION PIPELINE")
    print("="*70)
    print(f"Mode: {args.mode.upper()}")
    print(f"Input: {args.input}")
    print(f"Output: {args.output}")
    print(f"Music: {args.music}")
    
    if args.mode in ['animated', 'both']:
        print(f"\nAnimation Settings:")
        print(f"  Frames per image: {args.frames_per_image} ({args.frames_per_image/args.fps:.1f}s)")
        print(f"  Motion intensity: {args.motion}/255")
        print(f"  Transition frames: {args.transition_frames} ({args.transition_frames/args.fps:.1f}s)")
        print(f"  Device: {args.device.upper()}")
    
    print("="*70)
    
    # Create video(s) based on mode
    try:
        if args.mode == 'slideshow':
            output = pipeline.create_slideshow(
                style=args.style
            )
            print(f"\n🎬 Slideshow created: {output}")
            
        elif args.mode == 'animated':
            output = pipeline.create_animated_movie(
                frames_per_image=args.frames_per_image,
                motion_intensity=args.motion,
                transition_frames=args.transition_frames,
                fps=args.fps,
                device=args.device
            )
            print(f"\n🎬 Animated movie created: {output}")
            
        elif args.mode == 'both':
            outputs = pipeline.create_both(
                style=args.style,
                frames_per_image=args.frames_per_image,
                motion_intensity=args.motion,
                transition_frames=args.transition_frames,
                fps=args.fps,
                device=args.device
            )
            print(f"\n🎬 Slideshow: {outputs['slideshow']}")
            print(f"🎬 Animated: {outputs['animated']}")
        
        print("\n✓ Process complete!")
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Process interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()