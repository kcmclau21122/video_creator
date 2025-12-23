# ============================================================================
# AI Video Creator - Step 4: Content Sequencing Engine (FIXED)
# ============================================================================

"""
Content Sequencing Engine
Intelligently sequences media items based on chronology and scene coherence
UPDATED: Image durations now 6-10 seconds (was 2-8 seconds)
FIXED: Handle both Path and string file_path types
"""

from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import json

from media_ingester import MediaItem
from scene_analyzer import SceneAnalysis


@dataclass
class TransitionType:
    """Transition types between clips"""
    FADE = "fade"
    DISSOLVE = "dissolve"
    WIPE = "wipe"
    CUT = "cut"
    ZOOM = "zoom"


@dataclass
class SequenceItem:
    """Represents a sequenced media item with timing and transitions"""
    media_item: MediaItem
    scene_analysis: SceneAnalysis
    sequence_index: int
    start_time: float  # seconds in final video
    duration: float  # seconds
    transition_in: str  # transition type
    transition_duration: float  # seconds
    group_id: int  # scene grouping identifier
    
    def to_dict(self) -> dict:
        """Convert to dictionary"""
        return {
            'media_item_path': str(self.media_item.file_path),
            'sequence_index': self.sequence_index,
            'start_time': self.start_time,
            'duration': self.duration,
            'transition_in': self.transition_in,
            'transition_duration': self.transition_duration,
            'group_id': self.group_id,
            'caption': self.scene_analysis.caption,
            'scene_type': self.scene_analysis.scene_type,
            'mood': self.scene_analysis.mood
        }


class SequencingEngine:
    """Sequences media items intelligently"""

    # DEPRECATED: Now instance variables set in __init__
    # DEFAULT_IMAGE_DURATION = 7.0
    # MIN_IMAGE_DURATION = 6.0
    # MAX_IMAGE_DURATION = 10.0
    DEFAULT_TRANSITION_DURATION = 1.0  # Keep as fallback

    def __init__(
        self,
        base_image_duration: float = 7.0,
        fade_duration: float = 1.0
    ):
        """
        Initialize sequencing engine

        Args:
            base_image_duration: Base duration for images in seconds (default: 7.0)
            fade_duration: Duration for fade transitions in seconds (default: 1.0)
        """
        self.base_image_duration = base_image_duration
        self.fade_duration = fade_duration

        # Calculate min/max durations relative to base
        self.min_image_duration = max(base_image_duration - 2.0, 2.0)
        self.max_image_duration = base_image_duration + 3.0
    
    def _get_path_object(self, file_path) -> Path:
        """
        Convert file_path to Path object if it's a string
        
        Args:
            file_path: Either a Path object or string
            
        Returns:
            Path object
        """
        if isinstance(file_path, str):
            return Path(file_path)
        return file_path
    
    def create_sequence(
        self,
        media_items: List[MediaItem],
        scene_analyses: Dict,
        sort_chronologically: bool = True
    ) -> List[SequenceItem]:
        """
        Create optimized sequence from media items
        
        Args:
            media_items: List of MediaItem objects
            scene_analyses: Dictionary mapping paths to SceneAnalysis
            sort_chronologically: Whether to sort by datetime first
            
        Returns:
            List of SequenceItem objects
        """
        print(f"\n{'='*70}")
        print("CREATING SEQUENCE")
        print(f"{'='*70}")
        print(f"Base image duration: {self.base_image_duration}s")
        print(f"Image duration range: {self.min_image_duration}-{self.max_image_duration}s")
        print(f"Fade transition duration: {self.fade_duration}s")
        
        # Sort chronologically if requested
        if sort_chronologically:
            media_items = sorted(
                media_items,
                key=lambda x: x.datetime_original or datetime.min
            )
        
        # Create initial sequence
        sequence = []
        current_time = 0.0
        
        for idx, item in enumerate(media_items):
            # Convert file_path to Path object for consistent handling
            path_obj = self._get_path_object(item.file_path)
            
            # Get scene analysis - try both string and Path keys
            analysis = scene_analyses.get(item.file_path) or scene_analyses.get(path_obj) or scene_analyses.get(str(path_obj))
            
            if not analysis:
                print(f"⚠️  No analysis for {path_obj.name}, skipping")
                continue
            
            # Determine duration
            if item.file_type == 'image':
                duration = self._calculate_image_duration(analysis)
            else:
                duration = item.duration or 5.0
            
            # Determine transition
            if idx == 0:
                transition = TransitionType.FADE
                transition_duration = self.DEFAULT_TRANSITION_DURATION
            else:
                prev_item = media_items[idx-1]
                prev_path = self._get_path_object(prev_item.file_path)
                prev_analysis = scene_analyses.get(prev_item.file_path) or scene_analyses.get(prev_path) or scene_analyses.get(str(prev_path))
                transition, transition_duration = self._select_transition(
                    prev_analysis, analysis
                )
            
            # Determine group
            group_id = self._assign_group(idx, item, analysis, sequence)
            
            # Create sequence item
            seq_item = SequenceItem(
                media_item=item,
                scene_analysis=analysis,
                sequence_index=idx,
                start_time=current_time,
                duration=duration,
                transition_in=transition,
                transition_duration=transition_duration,
                group_id=group_id
            )
            
            sequence.append(seq_item)
            current_time += duration
        
        print(f"✓ Created sequence with {len(sequence)} items")
        print(f"Total duration: {current_time/60:.2f} minutes")
        
        return sequence
    
    def _calculate_image_duration(self, analysis: SceneAnalysis) -> float:
        """
        Calculate optimal duration for an image based on scene complexity
        Uses configurable base_image_duration with smart adjustments

        Args:
            analysis: Scene analysis for the image

        Returns:
            Duration in seconds
        """
        # Start with configurable base duration
        base_duration = self.base_image_duration

        # Adjust based on scene complexity
        num_objects = len(analysis.objects) if hasattr(analysis, 'objects') else 0

        if num_objects > 10:
            # Complex scenes need more time
            duration = min(base_duration + 2.0, self.max_image_duration)
        elif num_objects < 3:
            # Simple scenes need less time
            duration = max(base_duration - 1.0, self.min_image_duration)
        else:
            duration = base_duration

        # Adjust for mood
        if hasattr(analysis, 'mood'):
            if analysis.mood in ['serene', 'peaceful', 'calm']:
                duration = min(duration + 1.0, self.max_image_duration)
            elif analysis.mood in ['exciting', 'dynamic', 'energetic']:
                duration = max(duration - 0.5, self.min_image_duration)

        return duration
    
    def _select_transition(
        self,
        prev_analysis: Optional[SceneAnalysis],
        curr_analysis: SceneAnalysis
    ) -> Tuple[str, float]:
        """
        Select appropriate transition between scenes
        Uses configurable fade_duration for FADE transitions

        Args:
            prev_analysis: Analysis of previous scene
            curr_analysis: Analysis of current scene

        Returns:
            Tuple of (transition_type, duration)
        """
        if not prev_analysis:
            return TransitionType.FADE, self.fade_duration

        # Scene type changes
        if hasattr(prev_analysis, 'scene_type') and hasattr(curr_analysis, 'scene_type'):
            if prev_analysis.scene_type != curr_analysis.scene_type:
                # Different scene types - use dissolve (NOT affected by fade_duration)
                return TransitionType.DISSOLVE, 1.5

        # Mood changes
        if hasattr(prev_analysis, 'mood') and hasattr(curr_analysis, 'mood'):
            mood_contrast = self._calculate_mood_contrast(
                prev_analysis.mood,
                curr_analysis.mood
            )

            if mood_contrast > 0.7:
                # High contrast - use FADE (use configurable fade_duration)
                return TransitionType.FADE, self.fade_duration
            elif mood_contrast < 0.3:
                # Low contrast - use quick cut
                return TransitionType.CUT, 0.5

        # Default smooth transition (dissolve, not fade)
        return TransitionType.DISSOLVE, self.DEFAULT_TRANSITION_DURATION
    
    def _calculate_mood_contrast(self, mood1: str, mood2: str) -> float:
        """
        Calculate contrast between two moods
        
        Args:
            mood1: First mood
            mood2: Second mood
            
        Returns:
            Contrast value between 0 and 1
        """
        # Mood similarity matrix
        similar_moods = {
            ('happy', 'joyful'): 0.1,
            ('happy', 'exciting'): 0.3,
            ('serene', 'peaceful'): 0.1,
            ('serene', 'calm'): 0.1,
            ('dramatic', 'intense'): 0.2,
            ('melancholic', 'somber'): 0.1,
        }
        
        # Check if moods are the same
        if mood1 == mood2:
            return 0.0
        
        # Check similarity matrix
        pair = (mood1, mood2)
        reverse_pair = (mood2, mood1)
        
        if pair in similar_moods:
            return similar_moods[pair]
        if reverse_pair in similar_moods:
            return similar_moods[reverse_pair]
        
        # Default moderate contrast
        return 0.5
    
    def _assign_group(
        self,
        idx: int,
        item: MediaItem,
        analysis: SceneAnalysis,
        sequence: List[SequenceItem]
    ) -> int:
        """
        Assign scene group to item
        
        Args:
            idx: Current index
            item: Media item
            analysis: Scene analysis
            sequence: Existing sequence items
            
        Returns:
            Group ID
        """
        if idx == 0 or not sequence:
            return 0
        
        prev_item = sequence[-1]
        prev_analysis = prev_item.scene_analysis
        
        # Time-based grouping
        if item.datetime_original and prev_item.media_item.datetime_original:
            time_diff = abs(
                (item.datetime_original - prev_item.media_item.datetime_original).total_seconds()
            )
            
            # Same group if within 1 hour
            if time_diff < 3600:
                # Also check scene similarity
                if hasattr(analysis, 'scene_type') and hasattr(prev_analysis, 'scene_type'):
                    if analysis.scene_type == prev_analysis.scene_type:
                        return prev_item.group_id
        
        # Start new group
        return prev_item.group_id + 1
    
    def optimize_sequence(self, sequence: List[SequenceItem]) -> List[SequenceItem]:
        """
        Optimize sequence by adjusting timings and transitions
        
        Args:
            sequence: Input sequence
            
        Returns:
            Optimized sequence
        """
        # Recalculate start times
        current_time = 0.0
        for item in sequence:
            item.start_time = current_time
            current_time += item.duration
        
        return sequence
    
    def export_sequence(self, sequence: List[SequenceItem], output_path: Path):
        """
        Export sequence to JSON file
        
        Args:
            sequence: List of sequence items
            output_path: Output file path
        """
        sequence_data = {
            'created_at': datetime.now().isoformat(),
            'total_items': len(sequence),
            'total_duration': sequence[-1].start_time + sequence[-1].duration if sequence else 0,
            'num_groups': max(item.group_id for item in sequence) + 1 if sequence else 0,
            'items': [item.to_dict() for item in sequence]
        }
        
        with open(output_path, 'w') as f:
            json.dump(sequence_data, f, indent=2)
        
        print(f"\n✓ Sequence exported to: {output_path}")
    
    def get_sequence_statistics(self, sequence: List[SequenceItem]) -> dict:
        """
        Get statistics about the sequence
        
        Args:
            sequence: List of sequence items
            
        Returns:
            Dictionary with statistics
        """
        if not sequence:
            return {}
        
        total_duration = sequence[-1].start_time + sequence[-1].duration
        
        transition_counts = {}
        for item in sequence:
            trans = item.transition_in
            transition_counts[trans] = transition_counts.get(trans, 0) + 1
        
        avg_duration = sum(item.duration for item in sequence) / len(sequence)
        
        return {
            'total_items': len(sequence),
            'total_duration_seconds': total_duration,
            'total_duration_minutes': total_duration / 60,
            'num_groups': max(item.group_id for item in sequence) + 1,
            'avg_item_duration': avg_duration,
            'transition_counts': transition_counts,
            'shortest_item': min(item.duration for item in sequence),
            'longest_item': max(item.duration for item in sequence)
        }