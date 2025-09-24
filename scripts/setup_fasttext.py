#!/usr/bin/env python3
"""
Setup script to download and configure FastText language identification model.

This script downloads the FastText language identification model (lid.176.bin)
which provides faster and more accurate language detection compared to the
default langdetect fallback.

Usage:
    python scripts/setup_fasttext.py
    
Or via make:
    make setup-fasttext
"""

import sys
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlretrieve

# FastText language identification model
FASTTEXT_MODEL_URL = "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin"
MODEL_FILENAME = "lid.176.bin"
MODEL_SIZE_MB = 125


def download_with_progress(url: str, filename: str) -> None:
    """Download file with progress indicator."""
    def progress_hook(block_num: int, block_size: int, total_size: int) -> None:
        downloaded = block_num * block_size
        if total_size > 0:
            percent = min(100, (downloaded * 100) // total_size)
            mb_downloaded = downloaded / (1024 * 1024)
            mb_total = total_size / (1024 * 1024)
            print(f"\rDownloading: {percent:3d}% ({mb_downloaded:.1f}/{mb_total:.1f} MB)", end="", flush=True)
    
    try:
        urlretrieve(url, filename, progress_hook)
        print()  # New line after progress
    except URLError as e:
        print(f"\nError downloading {url}: {e}")
        sys.exit(1)


def main() -> None:
    """Main setup function."""
    print("FastText Language Detection Model Setup")
    print("=" * 40)
    
    # Check if model already exists
    if Path(MODEL_FILENAME).exists():
        print(f"✓ FastText model '{MODEL_FILENAME}' already exists.")
        
        # Verify it's not corrupted by checking size
        size_mb = Path(MODEL_FILENAME).stat().st_size / (1024 * 1024)
        if size_mb < MODEL_SIZE_MB * 0.9:  # Allow 10% variance
            print(f"⚠ Model file seems incomplete ({size_mb:.1f} MB, expected ~{MODEL_SIZE_MB} MB)")
            print("Redownloading...")
        else:
            print("Model appears to be complete. Setup finished!")
            return
    
    print(f"Downloading FastText language identification model (~{MODEL_SIZE_MB} MB)...")
    print(f"URL: {FASTTEXT_MODEL_URL}")
    print()
    
    # Download the model
    download_with_progress(FASTTEXT_MODEL_URL, MODEL_FILENAME)
    
    # Verify download
    if not Path(MODEL_FILENAME).exists():
        print("❌ Download failed - model file not found")
        sys.exit(1)
    
    size_mb = Path(MODEL_FILENAME).stat().st_size / (1024 * 1024)
    print(f"✓ Download complete: {MODEL_FILENAME} ({size_mb:.1f} MB)")
    
    # Test the model
    print("\nTesting FastText model...")
    try:
        import fasttext
        model = fasttext.load_model(MODEL_FILENAME)
        
        # Test prediction
        test_text = "Hello world"
        labels, probs = model.predict(test_text, k=1)
        lang = labels[0].replace("__label__", "")
        conf = float(probs[0])
        
        print(f"✓ Model test successful: '{test_text}' -> {lang} ({conf:.3f} confidence)")
        
    except ImportError:
        print("⚠ FastText package not found. Install with: pip install fasttext-wheel")
    except Exception as e:
        print(f"⚠ Model test failed: {e}")
        print("The model was downloaded but may have compatibility issues.")
    
    print("\n" + "=" * 40)
    print("Setup complete!")
    print("\nThe ingestor will now use FastText for language detection,")
    print("which is significantly faster than the langdetect fallback.")
    print("\nTo use a custom model location, set the FASTTEXT_LID_PATH environment variable:")
    print(f"export FASTTEXT_LID_PATH=/path/to/your/{MODEL_FILENAME}")


if __name__ == "__main__":
    main()
