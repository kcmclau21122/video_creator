# ============================================================================
# AI Video Creator - Step 4: Content Sequencing Engine (UPDATED)
# ============================================================================

"""
Content Sequencing Engine
Intelligently sequences media items based on chronology and scene coherence
UPDATED: Image durations now 6-10 seconds (was 2-8 seconds)
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
    
    # Updated durations (seconds) - now 6-10 seconds
    DEFAULT_IMAGE_DURATION = 7.0  # Changed from 4.0
    MIN_IMAGE_DURATION = 6.0      # Changed from 2.0
    MAX_IMAGE_DURATION = 10.0     # Changed from 8.0
    DEFAULT_TRANSITION_DURATION = 1.0
    
    def __init__(self):
        """Initialize sequencing engine"""
        pass
    
    def create_sequence(
        self,
        media_items: List[MediaItem],
        scene_analyses: Dict[Path, SceneAnalysis],
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
        print(f"Image duration range: {self.MIN_IMAGE_DURATION}-{self.MAX_IMAGE_DURATION} seconds")
        
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
            # Get scene analysis
            analysis = scene_analyses.get(item.file_path)
            if not analysis:
                print(f"⚠️  No analysis for {item.file_path.name}, skipping")
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
                prev_analysis = scene_analyses.get(media_items[idx-1].file_path)
                transition, transition_duration = self._select_transition(
                    prev_analysis, analysis
                )
            
            # Create sequence item
            seq_item = SequenceItem(
                media_item=item,
                scene_analysis=analysis,
                sequence_index=idx,
                start_time=current_time,
                duration=duration,
                transition_in=transition,
                transition_duration=transition_duration,
                group_id=0  # Will be assigned in grouping step
            )
            
            sequence.append(seq_item)
            current_time += duration
        
        # Apply scene grouping
        sequence = self._apply_scene_grouping(sequence)
        
        # Optimize pacing
        sequence = self._optimize_pacing(sequence)
        
        print(f"\n✓ Created sequence with {len(sequence)} items")
        print(f"✓ Total duration: {current_time:.2f} seconds ({current_time/60:.2f} minutes)")
        
        return sequence
    
    def _calculate_image_duration(self, analysis: SceneAnalysis) -> float:
        """
        Calculate optimal duration for an image based on content
        Now uses 6-10 second range instead of 2-8 seconds
        
        Args:
            analysis: SceneAnalysis object
            
        Returns:
            Duration in seconds (6-10 range)
        """
        # Base duration (7 seconds)
        duration = self.DEFAULT_IMAGE_DURATION
        
        # Adjust based on complexity (number of objects)
        num_objects = len(analysis.objects) if analysis.objects else 0
        if num_objects > 3:
            duration += 1.5  # More to look at, longer duration
        
        # Adjust based on caption length (more description = longer view)
        caption_words = len(analysis.caption.split())
        if caption_words > 10:
            duration += 1.0
        
        # Clamp to min/max (6-10 seconds)
        duration = max(self.MIN_IMAGE_DURATION, min(self.MAX_IMAGE_DURATION, duration))
        
        return duration
    
    def _select_transition(
        self,
        prev_analysis: Optional[SceneAnalysis],
        current_analysis: SceneAnalysis
    ) -> Tuple[str, float]:
        """
        Select appropriate transition based on scene change
        
        Args:
            prev_analysis: Previous scene analysis
            current_analysis: Current scene analysis
            
        Returns:
            Tuple of (transition_type, duration)
        """
        if not prev_analysis:
            return TransitionType.FADE, self.DEFAULT_TRANSITION_DURATION
        
        # Compare scenes
        scene_similarity = self._calculate_scene_similarity(
            prev_analysis, current_analysis
        )
        
        # Choose transition based on similarity
        if scene_similarity > 0.7:
            # Very similar scenes - quick cut
            return TransitionType.CUT, 0.0
        elif scene_similarity > 0.4:
            # Somewhat similar - dissolve
            return TransitionType.DISSOLVE, 0.8
        else:
            # Different scenes - fade
            return TransitionType.FADE, 1.0
    
    def _calculate_scene_similarity(
        self,
        scene1: SceneAnalysis,
        scene2: SceneAnalysis
    ) -> float:
        """
        Calculate similarity between two scenes (0.0 to 1.0)
        
        Args:
            scene1: First scene analysis
            scene2: Second scene analysis
            
        Returns:
            Similarity score
        """
        score = 0.0
        weights = 0.0
        
        # Compare scene type (indoor/outdoor)
        if scene1.scene_type and scene2.scene_type:
            if scene1.scene_type.lower() == scene2.scene_type.lower():
                score += 0.3
            weights += 0.3
        
        # Compare mood
        if scene1.mood and scene2.mood:
            mood1_words = set(scene1.mood.lower().split())
            mood2_words = set(scene2.mood.lower().split())
            mood_overlap = len(mood1_words & mood2_words) / max(len(mood1_words | mood2_words), 1)
            score += mood_overlap * 0.2
            weights += 0.2
        
        # Compare dominant colors
        if scene1.dominant_colors and scene2.dominant_colors:
            colors1 = set(scene1.dominant_colors)
            colors2 = set(scene2.dominant_colors)
            color_overlap = len(colors1 & colors2) / max(len(colors1 | colors2), 1)
            score += color_overlap * 0.2
            weights += 0.2
        
        # Compare objects
        if scene1.objects and scene2.objects:
            objects1 = set(scene1.objects)
            objects2 = set(scene2.objects)
            object_overlap = len(objects1 & objects2) / max(len(objects1 | objects2), 1)
            score += object_overlap * 0.3
            weights += 0.3
        
        # Normalize
        return score / weights if weights > 0 else 0.0
    
    def _apply_scene_grouping(
        self,
        sequence: List[SequenceItem]
    ) -> List[SequenceItem]:
        """
        Group similar scenes together
        
        Args:
            sequence: List of sequence items
            
        Returns:
            Updated sequence with group_id assigned
        """
        if not sequence:
            return sequence
        
        current_group = 0
        sequence[0].group_id = current_group
        
        for i in range(1, len(sequence)):
            similarity = self._calculate_scene_similarity(
                sequence[i-1].scene_analysis,
                sequence[i].scene_analysis
            )
            
            # Start new group if scenes are very different
            if similarity < 0.3:
                current_group += 1
            
            sequence[i].group_id = current_group
        
        num_groups = current_group + 1
        print(f"✓ Organized into {num_groups} scene groups")
        
        return sequence
    
    def _optimize_pacing(
        self,
        sequence: List[SequenceItem]
    ) -> List[SequenceItem]:
        """
        Optimize video pacing by adjusting durations
        
        Args:
            sequence: List of sequence items
            
        Returns:
            Optimized sequence
        """
        # Add variety in pacing
        for i, item in enumerate(sequence):
            # Alternate between slightly shorter and longer
            if i % 3 == 0:
                # Every third item slightly longer for rhythm
                item.duration = min(item.duration * 1.1, self.MAX_IMAGE_DURATION)
        
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