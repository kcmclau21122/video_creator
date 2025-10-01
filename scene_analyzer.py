# ============================================================================
# AI Video Creator - Step 3: Scene Analyzer
# ============================================================================

"""
Scene Analyzer
Analyzes images and videos to extract scene information, mood, themes, etc.
"""

from pathlib import Path
from typing import List, Dict, Optional
import json
from dataclasses import dataclass, asdict
from datetime import datetime
import cv2
from PIL import Image
from tqdm import tqdm

from vision_models import ModelFactory
from media_ingester import MediaItem
from config import Config


@dataclass
class SceneAnalysis:
    """Scene analysis result"""
    media_item_path: Path
    caption: str
    detailed_description: str
    mood: str
    dominant_colors: List[str]
    scene_type: str  # indoor, outdoor, nature, urban, etc.
    objects: List[str]
    lighting: str  # bright, dark, natural, artificial
    weather: Optional[str]  # for outdoor scenes
    time_of_day: Optional[str]  # morning, afternoon, evening, night
    
    def to_dict(self) -> dict:
        """Convert to dictionary"""
        result = asdict(self)
        result['media_item_path'] = str(self.media_item_path)
        return result
    
    @classmethod
    def from_dict(cls, data: dict) -> 'SceneAnalysis':
        """Create from dictionary"""
        if 'media_item_path' in data:
            data['media_item_path'] = Path(data['media_item_path'])
        return cls(**data)


class SceneAnalyzer:
    """Analyze scenes in images and videos"""
    
    # Questions for scene analysis
    ANALYSIS_QUESTIONS = {
        'mood': "What is the mood or emotional tone of this image?",
        'colors': "What are the dominant colors in this image?",
        'scene_type': "Is this indoors or outdoors?",
        'objects': "What are the main objects or subjects in this image?",
        'lighting': "Describe the lighting in this image.",
        'weather': "If outdoors, what is the weather like?",
        'time_of_day': "What time of day does this appear to be?"
    }
    
    def __init__(self, model_key: str = 'blip2-flan-t5-xl', cache_enabled: bool = True):
        """
        Initialize scene analyzer
        
        Args:
            model_key: Vision model to use
            cache_enabled: Whether to cache analysis results
        """
        self.model_key = model_key
        self.model = None
        self.cache_enabled = cache_enabled
        self.cache_dir = Config.CACHE_DIR / "scene_analysis"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
    def load_model(self):
        """Load vision model"""
        if self.model is None:
            print(f"Loading vision model: {self.model_key}")
            self.model = ModelFactory.create_model(self.model_key)
            self.model.load()
            print("✓ Model loaded")
    
    def unload_model(self):
        """Unload vision model"""
        if self.model:
            self.model.unload()
            self.model = None
            print("✓ Model unloaded")
    
    def analyze_image(self, image_path: Path, use_cache: bool = True) -> SceneAnalysis:
        """
        Analyze a single image
        
        Args:
            image_path: Path to image
            use_cache: Whether to use cached results
            
        Returns:
            SceneAnalysis object
        """
        # Check cache
        if use_cache and self.cache_enabled:
            cached = self._load_from_cache(image_path)
            if cached:
                return cached
        
        # Ensure model is loaded
        if not self.model:
            self.load_model()
        
        # Load image
        image = Image.open(image_path).convert('RGB')
        
        # Resize if too large
        max_size = Config.MAX_IMAGE_SIZE
        if image.size[0] > max_size[0] or image.size[1] > max_size[1]:
            image.thumbnail(max_size, Image.Resampling.LANCZOS)
        
        # Generate caption
        caption = self.model.generate_caption(image)
        
        # Generate detailed description
        detailed_description = self.model.answer_question(
            image, 
            "Describe this image in detail, including the setting, subjects, and atmosphere."
        )
        
        # Extract scene attributes
        mood = self.model.answer_question(image, self.ANALYSIS_QUESTIONS['mood'])
        colors_str = self.model.answer_question(image, self.ANALYSIS_QUESTIONS['colors'])
        scene_type = self.model.answer_question(image, self.ANALYSIS_QUESTIONS['scene_type'])
        objects_str = self.model.answer_question(image, self.ANALYSIS_QUESTIONS['objects'])
        lighting = self.model.answer_question(image, self.ANALYSIS_QUESTIONS['lighting'])
        weather = self.model.answer_question(image, self.ANALYSIS_QUESTIONS['weather'])
        time_of_day = self.model.answer_question(image, self.ANALYSIS_QUESTIONS['time_of_day'])
        
        # Parse colors and objects
        dominant_colors = self._parse_list(colors_str)
        objects = self._parse_list(objects_str)
        
        # Create analysis result
        analysis = SceneAnalysis(
            media_item_path=image_path,
            caption=caption,
            detailed_description=detailed_description,
            mood=mood,
            dominant_colors=dominant_colors,
            scene_type=scene_type,
            objects=objects,
            lighting=lighting,
            weather=weather if 'outdoor' in scene_type.lower() else None,
            time_of_day=time_of_day
        )
        
        # Cache result
        if self.cache_enabled:
            self._save_to_cache(image_path, analysis)
        
        return analysis
    
    def analyze_video(self, video_path: Path, sample_fps: float = 1.0, use_cache: bool = True) -> List[SceneAnalysis]:
        """
        Analyze video by sampling keyframes
        
        Args:
            video_path: Path to video
            sample_fps: Frames per second to sample
            use_cache: Whether to use cached results
            
        Returns:
            List of SceneAnalysis objects for keyframes
        """
        # Check cache
        cache_key = f"{video_path.stem}_fps{sample_fps}"
        if use_cache and self.cache_enabled:
            cached = self._load_video_cache(cache_key)
            if cached:
                return cached
        
        # Extract keyframes
        keyframes = self._extract_keyframes(video_path, sample_fps)
        
        if not keyframes:
            return []
        
        # Ensure model is loaded
        if not self.model:
            self.load_model()
        
        # Analyze each keyframe
        analyses = []
        for i, frame in enumerate(tqdm(keyframes, desc=f"Analyzing {video_path.name}")):
            # Convert to PIL Image
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(frame_rgb)
            
            # Generate caption
            caption = self.model.generate_caption(image)
            
            # Simplified analysis for video frames
            detailed_description = self.model.answer_question(
                image,
                "Describe this scene briefly."
            )
            
            mood = self.model.answer_question(image, self.ANALYSIS_QUESTIONS['mood'])
            scene_type = self.model.answer_question(image, self.ANALYSIS_QUESTIONS['scene_type'])
            
            # Create analysis
            analysis = SceneAnalysis(
                media_item_path=video_path,
                caption=f"Frame {i}: {caption}",
                detailed_description=detailed_description,
                mood=mood,
                dominant_colors=[],
                scene_type=scene_type,
                objects=[],
                lighting="",
                weather=None,
                time_of_day=None
            )
            
            analyses.append(analysis)
        
        # Cache results
        if self.cache_enabled:
            self._save_video_cache(cache_key, analyses)
        
        return analyses
    
    def analyze_media_collection(self, media_items: List[MediaItem]) -> Dict[Path, SceneAnalysis]:
        """
        Analyze a collection of media items
        
        Args:
            media_items: List of MediaItem objects
            
        Returns:
            Dictionary mapping file paths to SceneAnalysis
        """
        results = {}
        
        # Ensure model is loaded
        self.load_model()
        
        print(f"\n{'='*70}")
        print(f"ANALYZING {len(media_items)} MEDIA ITEMS")
        print(f"{'='*70}\n")
        
        for item in tqdm(media_items, desc="Analyzing media"):
            try:
                if item.file_type == 'image':
                    analysis = self.analyze_image(item.file_path)
                    results[item.file_path] = analysis
                elif item.file_type == 'video':
                    # For videos, use the first frame analysis as representative
                    analyses = self.analyze_video(item.file_path, sample_fps=Config.VIDEO_CLIP_SAMPLE_FPS)
                    if analyses:
                        results[item.file_path] = analyses[0]
            except Exception as e:
                print(f"\n✗ Error analyzing {item.file_path.name}: {e}")
        
        print(f"\n✓ Successfully analyzed {len(results)} items")
        
        return results
    
    def export_analysis(self, analyses: Dict[Path, SceneAnalysis], output_path: Path):
        """
        Export scene analyses to JSON
        
        Args:
            analyses: Dictionary of analyses
            output_path: Output file path
        """
        export_data = {
            'created_at': datetime.now().isoformat(),
            'model_used': self.model_key,
            'total_items': len(analyses),
            'analyses': {str(path): analysis.to_dict() for path, analysis in analyses.items()}
        }
        
        with open(output_path, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        print(f"\n✓ Analysis exported to: {output_path}")
    
    def _extract_keyframes(self, video_path: Path, sample_fps: float) -> List:
        """Extract keyframes from video"""
        keyframes = []
        
        cap = cv2.VideoCapture(str(video_path))
        
        if not cap.isOpened():
            return keyframes
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_interval = int(fps / sample_fps) if fps > 0 else 30
        
        frame_count = 0
        while True:
            ret, frame = cap.read()
            
            if not ret:
                break
            
            if frame_count % frame_interval == 0:
                keyframes.append(frame)
            
            frame_count += 1
            
            # Limit keyframes to avoid excessive processing
            if len(keyframes) >= 20:
                break
        
        cap.release()
        
        return keyframes
    
    def _parse_list(self, text: str) -> List[str]:
        """Parse comma-separated list from text"""
        if not text:
            return []
        
        # Split by common delimiters
        items = text.replace(' and ', ',').split(',')
        items = [item.strip().lower() for item in items if item.strip()]
        
        return items[:5]  # Limit to 5 items
    
    def _get_cache_key(self, file_path: Path) -> str:
        """Generate cache key"""
        stat = file_path.stat()
        return f"{file_path.stem}_{stat.st_size}_{stat.st_mtime}.json"
    
    def _load_from_cache(self, file_path: Path) -> Optional[SceneAnalysis]:
        """Load analysis from cache"""
        cache_file = self.cache_dir / self._get_cache_key(file_path)
        
        if cache_file.exists():
            try:
                with open(cache_file, 'r') as f:
                    data = json.load(f)
                return SceneAnalysis.from_dict(data)
            except Exception:
                pass
        
        return None
    
    def _save_to_cache(self, file_path: Path, analysis: SceneAnalysis):
        """Save analysis to cache"""
        cache_file = self.cache_dir / self._get_cache_key(file_path)
        
        try:
            with open(cache_file, 'w') as f:
                json.dump(analysis.to_dict(), f, indent=2)
        except Exception as e:
            print(f"⚠ Could not cache analysis: {e}")
    
    def _load_video_cache(self, cache_key: str) -> Optional[List[SceneAnalysis]]:
        """Load video analysis from cache"""
        cache_file = self.cache_dir / f"{cache_key}.json"
        
        if cache_file.exists():
            try:
                with open(cache_file, 'r') as f:
                    data = json.load(f)
                return [SceneAnalysis.from_dict(item) for item in data]
            except Exception:
                pass
        
        return None
    
    def _save_video_cache(self, cache_key: str, analyses: List[SceneAnalysis]):
        """Save video analysis to cache"""
        cache_file = self.cache_dir / f"{cache_key}.json"
        
        try:
            with open(cache_file, 'w') as f:
                json.dump([analysis.to_dict() for analysis in analyses], f, indent=2)
        except Exception as e:
            print(f"⚠ Could not cache video analysis: {e}")


if __name__ == "__main__":
    # Test scene analyzer
    from config import Config
    
    analyzer = SceneAnalyzer()
    
    # Find test image
    image_files = list(Config.INPUT_DIR.glob("*.jpg")) + list(Config.INPUT_DIR.glob("*.jpeg"))
    
    if image_files:
        print(f"Testing with: {image_files[0].name}")
        
        # Analyze image
        analysis = analyzer.analyze_image(image_files[0])
        
        print(f"\n{'='*70}")
        print("SCENE ANALYSIS RESULT")
        print(f"{'='*70}")
        print(f"Caption: {analysis.caption}")
        print(f"Description: {analysis.detailed_description}")
        print(f"Mood: {analysis.mood}")
        print(f"Scene Type: {analysis.scene_type}")
        print(f"Lighting: {analysis.lighting}")
        print(f"Colors: {', '.join(analysis.dominant_colors)}")
        print(f"Objects: {', '.join(analysis.objects)}")
        if analysis.weather:
            print(f"Weather: {analysis.weather}")
        if analysis.time_of_day:
            print(f"Time of Day: {analysis.time_of_day}")
        
        analyzer.unload_model()
    else:
        print("No images found in input folder")