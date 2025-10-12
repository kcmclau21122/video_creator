import shutil
from pathlib import Path

# Find and remove the cached model
cache_dir = Path.home() / '.cache' / 'huggingface' / 'hub'
blip2_cache = cache_dir / 'models--Salesforce--blip2-flan-t5-xl'

if blip2_cache.exists():
    shutil.rmtree(blip2_cache)
    print(f"Removed cached model at {blip2_cache}")