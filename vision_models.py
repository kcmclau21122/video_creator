# ============================================================================
# AI Video Creator - Step 3: Vision Models Wrapper (FIXED)
# ============================================================================

"""
Vision Models Wrapper
Provides unified interface for different vision models
"""

from pathlib import Path
from typing import List
import torch
from PIL import Image
from transformers import (
    Blip2Processor,
    Blip2ForConditionalGeneration
)
from config import Config


class VisionModel:
    """Base class for vision models"""
    
    def __init__(self, model_name: str, cache_dir: Path):
        """Initialize vision model"""
        self.model_name = model_name
        self.cache_dir = cache_dir
        self.device = Config.DEVICE
        self.processor = None
        self.model = None
        
    def load(self):
        """Load model into memory"""
        raise NotImplementedError
    
    def unload(self):
        """Unload model from memory"""
        if self.model:
            del self.model
            self.model = None
        if self.processor:
            del self.processor
            self.processor = None
        torch.cuda.empty_cache()
    
    def generate_caption(self, image: Image.Image) -> str:
        """Generate caption for image"""
        raise NotImplementedError
    
    def answer_question(self, image: Image.Image, question: str) -> str:
        """Answer question about image"""
        raise NotImplementedError


class BLIP2Model(VisionModel):
    """BLIP-2 model wrapper"""
    
    def __init__(self, model_name: str, cache_dir: Path):
        """Initialize BLIP-2 model"""
        super().__init__(model_name, cache_dir)
        
    def load(self):
        """Load BLIP-2 model"""
        print(f"Loading BLIP-2 model: {self.model_name}")
        
        try:
            # Load directly from HuggingFace (bypasses corrupted cache)
            print("Loading processor from HuggingFace...")
            self.processor = Blip2Processor.from_pretrained(
                self.model_name
            )
            
            print("Loading model from HuggingFace (this may take a minute)...")
            self.model = Blip2ForConditionalGeneration.from_pretrained(
                self.model_name,
                dtype=torch.float16 if Config.USE_FP16 else torch.float32,
                device_map="auto"
            )
            
            self.model.eval()
            
            print(f"✓ Model loaded on {self.device}")
            
        except Exception as e:
            print(f"✗ Error loading model: {e}")
            raise
    
    def generate_caption(self, image: Image.Image, max_length: int = 50) -> str:
        """
        Generate caption for image
        
        Args:
            image: PIL Image
            max_length: Maximum caption length
            
        Returns:
            Generated caption
        """
        if not self.model or not self.processor:
            raise RuntimeError("Model not loaded. Call load() first.")
        
        # Preprocess image
        inputs = self.processor(image, return_tensors="pt").to(
            self.device, 
            torch.float16 if Config.USE_FP16 else torch.float32
        )
        
        # Generate caption
        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_length=max_length,
                num_beams=5,
                temperature=1.0
            )
        
        # Decode caption
        caption = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
        
        return caption
    
    def answer_question(self, image: Image.Image, question: str, max_length: int = 50) -> str:
        """
        Answer question about image
        
        Args:
            image: PIL Image
            question: Question to answer
            max_length: Maximum answer length
            
        Returns:
            Answer to question
        """
        if not self.model or not self.processor:
            raise RuntimeError("Model not loaded. Call load() first.")
        
        # Format prompt
        prompt = f"Question: {question} Answer:"
        
        # Preprocess
        inputs = self.processor(image, text=prompt, return_tensors="pt").to(
            self.device, 
            torch.float16 if Config.USE_FP16 else torch.float32
        )
        
        # Generate answer
        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_length=max_length,
                num_beams=5,
                temperature=1.0
            )
        
        # Decode answer
        answer = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
        
        return answer
    
    def batch_generate_captions(self, images: List[Image.Image], max_length: int = 50) -> List[str]:
        """
        Generate captions for multiple images
        
        Args:
            images: List of PIL Images
            max_length: Maximum caption length
            
        Returns:
            List of generated captions
        """
        if not self.model or not self.processor:
            raise RuntimeError("Model not loaded. Call load() first.")
        
        # Preprocess all images
        inputs = self.processor(images, return_tensors="pt").to(
            self.device,
            torch.float16 if Config.USE_FP16 else torch.float32
        )
        
        # Generate captions
        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_length=max_length,
                num_beams=5,
                temperature=1.0
            )
        
        # Decode all captions
        captions = self.processor.batch_decode(generated_ids, skip_special_tokens=True)
        captions = [caption.strip() for caption in captions]
        
        return captions


class ModelFactory:
    """Factory for creating vision models"""
    
    @staticmethod
    def create_model(model_key: str) -> VisionModel:
        """
        Create vision model instance
        
        Args:
            model_key: Model identifier
            
        Returns:
            VisionModel instance
        """
        from model_downloader import ModelDownloader
        
        downloader = ModelDownloader()
        
        # Get model info
        model_info = downloader.get_model_info(model_key)
        if not model_info:
            raise ValueError(f"Unknown model key: {model_key}")
        
        model_name = downloader.get_model_name(model_key)
        cache_dir = downloader.get_model_path(model_key)
        
        # Create appropriate model wrapper
        if model_info['type'] == 'blip2':
            return BLIP2Model(model_name, cache_dir)
        else:
            raise ValueError(f"Unsupported model type: {model_info['type']}")


if __name__ == "__main__":
    # Test model loading
    print("Testing vision model...")
    
    try:
        # Create model
        model = ModelFactory.create_model('blip2-flan-t5-xl')
        
        # Load model
        model.load()
        
        # Test with sample image
        from config import Config
        
        # Find first image in input folder
        image_files = list(Config.INPUT_DIR.glob("*.jpg")) + list(Config.INPUT_DIR.glob("*.jpeg"))
        
        if image_files:
            print(f"\nTesting with: {image_files[0].name}")
            
            # Load image
            image = Image.open(image_files[0])
            
            # Generate caption
            print("\nGenerating caption...")
            caption = model.generate_caption(image)
            print(f"Caption: {caption}")
            
            # Answer questions
            questions = [
                "What is in this image?",
                "What is the mood of this image?",
                "What colors are prominent?",
                "Is this indoors or outdoors?"
            ]
            
            print("\nAnswering questions...")
            for question in questions:
                answer = model.answer_question(image, question)
                print(f"Q: {question}")
                print(f"A: {answer}\n")
            
            print("✓ Model test successful")
        else:
            print("No images found in input folder")
        
        # Unload model
        model.unload()
        
    except Exception as e:
        print(f"✗ Error: {e}")