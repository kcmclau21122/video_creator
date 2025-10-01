# ============================================================================
# AI Video Creator - Step 3: Model Downloader (CORRECTED)
# ============================================================================

"""
Model Downloader
Downloads and caches AI models for scene analysis
"""

from pathlib import Path
from transformers import (
    Blip2Processor, 
    Blip2ForConditionalGeneration,
)
import torch
from config import Config


class ModelDownloader:
    """Download and cache AI models"""
    
    # Available models
    MODELS = {
        'blip2-opt-2.7b': {
            'name': 'Salesforce/blip2-opt-2.7b',
            'type': 'blip2',
            'size': '~5.4 GB',
            'description': 'Fast image captioning and Q&A',
            'vram': '6-8 GB',
            'processor': Blip2Processor,
            'model': Blip2ForConditionalGeneration
        },
        'blip2-opt-6.7b': {
            'name': 'Salesforce/blip2-opt-6.7b',
            'type': 'blip2',
            'size': '~13 GB',
            'description': 'Better quality image captioning',
            'vram': '12-14 GB',
            'processor': Blip2Processor,
            'model': Blip2ForConditionalGeneration
        },
        'blip2-flan-t5-xl': {
            'name': 'Salesforce/blip2-flan-t5-xl',
            'type': 'blip2',
            'size': '~10 GB',
            'description': 'Best for detailed descriptions',
            'vram': '10-12 GB',
            'processor': Blip2Processor,
            'model': Blip2ForConditionalGeneration
        }
    }
    
    def __init__(self):
        """Initialize model downloader"""
        self.models_dir = Config.MODELS_DIR
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
    def list_models(self):
        """List available models"""
        print("\n" + "="*70)
        print("AVAILABLE VISION MODELS")
        print("="*70 + "\n")
        
        for key, info in self.MODELS.items():
            print(f"Model: {key}")
            print(f"  Name: {info['name']}")
            print(f"  Description: {info['description']}")
            print(f"  Size: {info['size']}")
            print(f"  VRAM Required: {info['vram']}")
            print(f"  Downloaded: {self.is_downloaded(key)}")
            print()
    
    def is_downloaded(self, model_key: str) -> bool:
        """Check if model is already downloaded"""
        if model_key not in self.MODELS:
            return False
        
        model_info = self.MODELS[model_key]
        cache_dir = self.models_dir / model_key
        
        # Check HuggingFace cache structure
        # Format: models--{org}--{model_name}
        org, model_name = model_info['name'].split('/')
        hf_cache_dir = cache_dir / f"models--{org}--{model_name}"
        
        # Check if cache directory exists and has snapshots
        if hf_cache_dir.exists():
            snapshots_dir = hf_cache_dir / "snapshots"
            if snapshots_dir.exists() and any(snapshots_dir.iterdir()):
                return True
        
        return False
    
    def download_model(self, model_key: str, force: bool = False):
        """
        Download a specific model
        
        Args:
            model_key: Model identifier
            force: Force re-download if already exists
        """
        if model_key not in self.MODELS:
            print(f"✗ Unknown model: {model_key}")
            print(f"Available models: {list(self.MODELS.keys())}")
            return False
        
        model_info = self.MODELS[model_key]
        cache_dir = self.models_dir / model_key
        
        # Check if already downloaded
        if self.is_downloaded(model_key) and not force:
            print(f"✓ Model '{model_key}' already downloaded")
            return True
        
        print(f"\n{'='*70}")
        print(f"DOWNLOADING MODEL: {model_key}")
        print(f"{'='*70}")
        print(f"Name: {model_info['name']}")
        print(f"Size: {model_info['size']}")
        print(f"Description: {model_info['description']}")
        print(f"Destination: {cache_dir}")
        print()
        
        try:
            # Create directory
            cache_dir.mkdir(parents=True, exist_ok=True)
            
            # Download processor
            print("Downloading processor...")
            processor = model_info['processor'].from_pretrained(
                model_info['name'],
                cache_dir=cache_dir
            )
            print("✓ Processor downloaded")
            
            # Download model
            print("\nDownloading model (this may take several minutes)...")
            model = model_info['model'].from_pretrained(
                model_info['name'],
                cache_dir=cache_dir,
                dtype=torch.float16 if Config.USE_FP16 else torch.float32
            )
            print("✓ Model downloaded")
            
            print(f"\n{'='*70}")
            print(f"✓ MODEL DOWNLOADED SUCCESSFULLY")
            print(f"{'='*70}\n")
            
            return True
            
        except Exception as e:
            print(f"\n✗ Error downloading model: {e}")
            return False
    
    def download_recommended(self):
        """Download recommended model for user's hardware"""
        print("\n" + "="*70)
        print("DOWNLOADING RECOMMENDED MODEL")
        print("="*70)
        print(f"Your GPU: RTX 4080 ({Config.VRAM_GB} GB VRAM)")
        print("\nRecommended: blip2-flan-t5-xl")
        print("  - Best quality for your hardware")
        print("  - ~10 GB download")
        print("  - Excellent scene descriptions")
        print()
        
        confirm = input("Download now? (y/n): ").lower().strip()
        
        if confirm == 'y':
            return self.download_model('blip2-flan-t5-xl')
        else:
            print("Download cancelled")
            return False
    
    def get_model_info(self, model_key: str) -> dict:
        """Get information about a model"""
        return self.MODELS.get(model_key, {})
    
    def get_model_path(self, model_key: str) -> Path:
        """Get path to model cache directory"""
        return self.models_dir / model_key
    
    def get_model_name(self, model_key: str) -> str:
        """Get HuggingFace model name"""
        if model_key in self.MODELS:
            return self.MODELS[model_key]['name']
        return None
    
    def delete_model(self, model_key: str):
        """Delete a downloaded model"""
        model_path = self.get_model_path(model_key)
        
        if not model_path.exists():
            print(f"Model '{model_key}' not found")
            return
        
        print(f"Deleting model: {model_key}")
        
        import shutil
        shutil.rmtree(model_path)
        
        print(f"✓ Deleted {model_key}")
    
    def get_disk_usage(self) -> dict:
        """Get disk usage statistics"""
        stats = {
            'total_size_gb': 0,
            'models': {}
        }
        
        for model_key in self.MODELS.keys():
            if self.is_downloaded(model_key):
                model_path = self.get_model_path(model_key)
                
                # Calculate directory size
                total_size = sum(
                    f.stat().st_size 
                    for f in model_path.rglob('*') 
                    if f.is_file()
                )
                
                size_gb = total_size / (1024**3)
                stats['models'][model_key] = size_gb
                stats['total_size_gb'] += size_gb
        
        return stats


def main():
    """Main function for interactive model management"""
    downloader = ModelDownloader()
    
    while True:
        print("\n" + "="*70)
        print("MODEL DOWNLOADER - MAIN MENU")
        print("="*70)
        print("\n1. List available models")
        print("2. Download recommended model (blip2-flan-t5-xl)")
        print("3. Download specific model")
        print("4. Check disk usage")
        print("5. Delete model")
        print("6. Exit")
        
        choice = input("\nEnter choice (1-6): ").strip()
        
        if choice == '1':
            downloader.list_models()
            
        elif choice == '2':
            downloader.download_recommended()
            
        elif choice == '3':
            print("\nAvailable models:")
            for i, key in enumerate(downloader.MODELS.keys(), 1):
                print(f"  {i}. {key}")
            
            model_num = input("\nEnter model number: ").strip()
            try:
                model_key = list(downloader.MODELS.keys())[int(model_num) - 1]
                downloader.download_model(model_key)
            except (ValueError, IndexError):
                print("Invalid selection")
        
        elif choice == '4':
            stats = downloader.get_disk_usage()
            print(f"\n{'='*70}")
            print("DISK USAGE")
            print(f"{'='*70}")
            
            if stats['models']:
                for model_key, size_gb in stats['models'].items():
                    print(f"{model_key}: {size_gb:.2f} GB")
                print(f"\nTotal: {stats['total_size_gb']:.2f} GB")
            else:
                print("No models downloaded")
        
        elif choice == '5':
            downloader.list_models()
            model_key = input("\nEnter model name to delete: ").strip()
            confirm = input(f"Delete {model_key}? (y/n): ").lower().strip()
            if confirm == 'y':
                downloader.delete_model(model_key)
        
        elif choice == '6':
            print("\nExiting...")
            break
        
        else:
            print("Invalid choice")


if __name__ == "__main__":
    main()