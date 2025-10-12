# ============================================================================
# AI Video Creator - Narrative Generator Module
# Generates movie-style narration and dialog from scene analysis
# ============================================================================

"""
Narrative Generator
Creates cohesive story narration from scene analyses
Uses LLM to generate engaging narrative and dialog
"""

from pathlib import Path
from typing import List, Dict, Optional
import json
from dataclasses import dataclass
from datetime import datetime
import torch

# For LLM-based narrative generation
from transformers import AutoTokenizer, AutoModelForCausalLM

from scene_analyzer import SceneAnalysis
from sequencing_engine import SequenceItem
from config import Config


@dataclass
class NarrativeSegment:
    """Represents a narrative segment for a scene"""
    sequence_index: int
    start_time: float
    duration: float
    narration_text: str
    speaker: str  # e.g., "narrator", "character1"
    emotion: str  # e.g., "cheerful", "nostalgic", "excited"
    audio_file: Optional[Path] = None


class NarrativeGenerator:
    """Generate movie-style narration from scene analysis"""
    
    # Available LLM models for narrative generation
    MODELS = {
        'mistral-7b': {
            'name': 'mistralai/Mistral-7B-Instruct-v0.2',
            'size': '~14 GB',
            'vram': '8-10 GB',
            'quality': 'Excellent for creative writing'
        },
        'llama-3.1-8b': {
            'name': 'meta-llama/Meta-Llama-3.1-8B-Instruct',
            'size': '~16 GB',
            'vram': '10-12 GB',
            'quality': 'Best for storytelling'
        },
        'phi-3-mini': {
            'name': 'microsoft/Phi-3-mini-128k-instruct',
            'size': '~7 GB',
            'vram': '6-8 GB',
            'quality': 'Fast, good quality'
        }
    }
    
    # Narrative styles
    STYLES = {
        'documentary': "Create a documentary-style narration that is informative and engaging.",
        'story': "Create a story-like narration with emotional depth and character.",
        'travel': "Create a travel vlog style narration that is enthusiastic and descriptive.",
        'family': "Create a warm, personal family memory narration.",
        'adventure': "Create an exciting adventure-style narration with dynamic energy."
    }
    
    def __init__(self, model_key: str = 'phi-3-mini', style: str = 'family'):
        """
        Initialize narrative generator
        
        Args:
            model_key: LLM model to use
            style: Narrative style
        """
        self.model_key = model_key
        self.style = style
        self.model = None
        self.tokenizer = None
        self.device = Config.DEVICE
        
    def load_model(self):
        """Load LLM for narrative generation"""
        if self.model is not None:
            return
        
        model_info = self.MODELS.get(self.model_key)
        if not model_info:
            raise ValueError(f"Unknown model: {self.model_key}")
        
        print(f"\n{'='*70}")
        print(f"LOADING NARRATIVE MODEL: {self.model_key}")
        print(f"{'='*70}")
        print(f"Model: {model_info['name']}")
        print(f"Size: {model_info['size']}")
        print("This may take a minute...\n")
        
        try:
            # Load tokenizer
            print("Loading tokenizer...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_info['name'],
                trust_remote_code=True
            )
            
            # Load model
            print("Loading model...")
            self.model = AutoModelForCausalLM.from_pretrained(
                model_info['name'],
                torch_dtype=torch.float16 if Config.USE_FP16 else torch.float32,
                device_map="auto",
                trust_remote_code=True
            )
            
            print(f"✅ Model loaded on {self.device}\n")
            
        except Exception as e:
            print(f"✗ Error loading model: {e}")
            raise
    
    def unload_model(self):
        """Unload model from memory"""
        if self.model:
            del self.model
            self.model = None
        if self.tokenizer:
            del self.tokenizer
            self.tokenizer = None
        torch.cuda.empty_cache()
        print("✅ Narrative model unloaded")
    
    def generate_narrative(
        self,
        sequence: List[SequenceItem],
        video_title: str = "My Video Story"
    ) -> List[NarrativeSegment]:
        """
        Generate narration for entire video sequence
        
        Args:
            sequence: List of sequence items
            video_title: Title of the video
            
        Returns:
            List of narrative segments
        """
        print(f"\n{'='*70}")
        print("GENERATING MOVIE NARRATION")
        print(f"{'='*70}")
        print(f"Style: {self.style}")
        print(f"Scenes: {len(sequence)}")
        
        # Ensure model is loaded
        if not self.model:
            self.load_model()
        
        # Step 1: Analyze overall theme and arc
        print("\n[1/3] Analyzing story arc...")
        story_context = self._analyze_story_arc(sequence, video_title)
        
        # Step 2: Generate narration for each scene group
        print("\n[2/3] Generating narration for scene groups...")
        narrative_segments = self._generate_scene_narrations(sequence, story_context)
        
        # Step 3: Refine timing and flow
        print("\n[3/3] Refining narrative timing...")
        narrative_segments = self._refine_timing(narrative_segments, sequence)
        
        print(f"\n✅ Generated {len(narrative_segments)} narrative segments")
        
        return narrative_segments
    
    def _analyze_story_arc(
        self,
        sequence: List[SequenceItem],
        video_title: str
    ) -> Dict:
        """Analyze scenes to create story arc"""
        
        # Collect scene information
        scenes_summary = []
        for i, item in enumerate(sequence[:20]):  # Limit to first 20 for context
            scenes_summary.append({
                'index': i,
                'caption': item.scene_analysis.caption,
                'mood': item.scene_analysis.mood,
                'scene_type': item.scene_analysis.scene_type,
                'time': item.media_item.datetime_original
            })
        
        # Create prompt for story arc
        prompt = f"""You are a creative storyteller. Analyze these scenes from a video titled "{video_title}" and create a story arc.

Style: {self.STYLES.get(self.style, self.STYLES['family'])}

Scenes:
{json.dumps(scenes_summary, indent=2, default=str)}

Create a brief story arc with:
1. Theme (one sentence)
2. Opening (how to start the narration)
3. Development (key story beats)
4. Conclusion (how to end)

Keep it concise and emotional. Format as JSON."""

        # Generate story arc
        response = self._generate_text(prompt, max_length=500)
        
        try:
            story_context = json.loads(response)
        except:
            # Fallback if JSON parsing fails
            story_context = {
                'theme': f"A journey through memories captured in {video_title}",
                'opening': "This is a story worth telling...",
                'development': ["Each moment reveals something special"],
                'conclusion': "And that's how this chapter of our story unfolds."
            }
        
        return story_context
    
    def _generate_scene_narrations(
        self,
        sequence: List[SequenceItem],
        story_context: Dict
    ) -> List[NarrativeSegment]:
        """Generate narration for each scene"""
        
        narrative_segments = []
        
        # Group scenes by scene groups
        scene_groups = {}
        for item in sequence:
            group_id = item.group_id
            if group_id not in scene_groups:
                scene_groups[group_id] = []
            scene_groups[group_id].append(item)
        
        # Generate narration for each group
        for group_id, group_items in scene_groups.items():
            # Create scene summary for this group
            scene_descriptions = []
            for item in group_items[:3]:  # Max 3 items per group for context
                scene_descriptions.append({
                    'caption': item.scene_analysis.caption,
                    'mood': item.scene_analysis.mood,
                    'setting': item.scene_analysis.scene_type,
                    'time_of_day': item.scene_analysis.time_of_day
                })
            
            # Create prompt
            prompt = f"""You are narrating a {self.style} video. 

Story theme: {story_context.get('theme', 'A memorable journey')}

Current scene group:
{json.dumps(scene_descriptions, indent=2)}

Generate a natural, conversational narration (2-3 sentences) that:
1. Describes what's happening
2. Conveys the emotion/mood
3. Connects to the overall story

Make it sound like a real person telling a story, not robotic. Be warm and engaging."""

            # Generate narration
            narration_text = self._generate_text(prompt, max_length=150)
            
            # Clean up the text
            narration_text = self._clean_narration(narration_text)
            
            # Create segment for first item in group (represents the group)
            first_item = group_items[0]
            segment = NarrativeSegment(
                sequence_index=first_item.sequence_index,
                start_time=first_item.start_time,
                duration=sum(item.duration for item in group_items),
                narration_text=narration_text,
                speaker="narrator",
                emotion=first_item.scene_analysis.mood or "neutral"
            )
            
            narrative_segments.append(segment)
        
        return narrative_segments
    
    def _generate_text(self, prompt: str, max_length: int = 200) -> str:
        """Generate text using LLM"""
        
        # Format prompt for instruction-following
        formatted_prompt = f"<|system|>You are a creative storyteller.</s><|user|>{prompt}</s><|assistant|>"
        
        # Tokenize
        inputs = self.tokenizer(
            formatted_prompt,
            return_tensors="pt",
            truncation=True,
            max_length=2048
        ).to(self.device)
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_length,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract only the assistant's response
        if "<|assistant|>" in generated_text:
            generated_text = generated_text.split("<|assistant|>")[-1].strip()
        
        return generated_text
    
    def _clean_narration(self, text: str) -> str:
        """Clean and format narration text"""
        
        # Remove common artifacts
        text = text.replace("<|assistant|>", "")
        text = text.replace("</s>", "")
        
        # Remove any JSON formatting if present
        if text.startswith('{') or text.startswith('['):
            try:
                parsed = json.loads(text)
                if isinstance(parsed, dict) and 'narration' in parsed:
                    text = parsed['narration']
                elif isinstance(parsed, str):
                    text = parsed
            except:
                pass
        
        # Clean up whitespace
        text = ' '.join(text.split())
        
        # Ensure it ends with proper punctuation
        if text and text[-1] not in '.!?':
            text += '.'
        
        return text
    
    def _refine_timing(
        self,
        narrative_segments: List[NarrativeSegment],
        sequence: List[SequenceItem]
    ) -> List[NarrativeSegment]:
        """Refine timing of narrative segments"""
        
        # Estimate speech duration (rough: 150 words per minute = 2.5 words per second)
        for segment in narrative_segments:
            word_count = len(segment.narration_text.split())
            estimated_speech_duration = word_count / 2.5
            
            # Adjust if narration is longer than scene duration
            if estimated_speech_duration > segment.duration * 0.8:
                # Trim narration to fit
                target_words = int(segment.duration * 0.8 * 2.5)
                words = segment.narration_text.split()[:target_words]
                segment.narration_text = ' '.join(words) + '...'
        
        return narrative_segments
    
    def export_script(
        self,
        narrative_segments: List[NarrativeSegment],
        output_path: Path
    ):
        """Export narrative script to file"""
        
        script_data = {
            'created_at': datetime.now().isoformat(),
            'style': self.style,
            'model_used': self.model_key,
            'total_segments': len(narrative_segments),
            'segments': [
                {
                    'sequence_index': seg.sequence_index,
                    'start_time': seg.start_time,
                    'duration': seg.duration,
                    'narration': seg.narration_text,
                    'speaker': seg.speaker,
                    'emotion': seg.emotion
                }
                for seg in narrative_segments
            ]
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(script_data, f, indent=2, ensure_ascii=False)
        
        print(f"\n✅ Narrative script exported to: {output_path}")
        
        # Also create a readable text version
        text_output = output_path.with_suffix('.txt')
        with open(text_output, 'w', encoding='utf-8') as f:
            f.write(f"NARRATION SCRIPT\n")
            f.write(f"Style: {self.style}\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("="*70 + "\n\n")
            
            for i, seg in enumerate(narrative_segments, 1):
                f.write(f"[Segment {i}] @ {seg.start_time:.1f}s\n")
                f.write(f"{seg.narration_text}\n\n")
        
        print(f"✅ Readable script exported to: {text_output}")


if __name__ == "__main__":
    # Test narrative generator
    print("Narrative Generator Module")
    print("Use this to generate movie-style narration for your videos")
    print("\nAvailable models:")
    for key, info in NarrativeGenerator.MODELS.items():
        print(f"  • {key}: {info['quality']}")