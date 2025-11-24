"""
Download real test images of varying complexity and save to folder.
Images are sourced from Unsplash (free to use).
"""

import os
import urllib.request
from pathlib import Path
from PIL import Image
import io

# Create output directory
OUTPUT_DIR = Path("data/test_images")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Image URLs from Unsplash (free, high-quality images)
# Format: (filename, url, expected_complexity)
IMAGES = [
    # === SIMPLE (5 images) - Single objects, clean backgrounds ===
    ("simple_01_white_cup.jpg", 
     "https://images.unsplash.com/photo-1514228742587-6b1558fcca3d?w=400",
     "simple"),
    ("simple_02_single_leaf.jpg",
     "https://images.unsplash.com/photo-1518882605630-8a6channels-a396-4e27-9c3b-c8d0a8d0f1a1?w=400",
     "simple"),
    ("simple_03_apple.jpg",
     "https://images.unsplash.com/photo-1568702846914-96b305d2uj8e?w=400",
     "simple"),
    ("simple_04_egg.jpg",
     "https://images.unsplash.com/photo-1582722872445-44dc5f7e3c8f?w=400",
     "simple"),
    ("simple_05_moon.jpg",
     "https://images.unsplash.com/photo-1532693322450-2cb5c511067d?w=400",
     "simple"),
    
    # === MEDIUM (5 images) - Multiple objects, some detail ===
    ("medium_01_desk_setup.jpg",
     "https://images.unsplash.com/photo-1498050108023-c5249f4df085?w=400",
     "medium"),
    ("medium_02_kitchen.jpg",
     "https://images.unsplash.com/photo-1556909114-f6e7ad7d3136?w=400",
     "medium"),
    ("medium_03_bookshelf.jpg",
     "https://images.unsplash.com/photo-1507003211169-0a1dd7228f2d?w=400",
     "medium"),
    ("medium_04_garden.jpg",
     "https://images.unsplash.com/photo-1416879595882-3373a0480b5b?w=400",
     "medium"),
    ("medium_05_office.jpg",
     "https://images.unsplash.com/photo-1497366216548-37526070297c?w=400",
     "medium"),
    
    # === COMPLEX/HEAVY (5 images) - Crowded scenes, fine details ===
    ("heavy_01_city_street.jpg",
     "https://images.unsplash.com/photo-1480714378408-67cf0d13bc1b?w=400",
     "heavy"),
    ("heavy_02_crowd.jpg",
     "https://images.unsplash.com/photo-1517457373958-b7bdd4587205?w=400",
     "heavy"),
    ("heavy_03_market.jpg",
     "https://images.unsplash.com/photo-1555396273-367ea4eb4db5?w=400",
     "heavy"),
    ("heavy_04_forest_dense.jpg",
     "https://images.unsplash.com/photo-1448375240586-882707db888b?w=400",
     "heavy"),
    ("heavy_05_times_square.jpg",
     "https://images.unsplash.com/photo-1534430480872-3498386e7856?w=400",
     "heavy"),
]

# Backup URLs in case some fail (more reliable sources)
BACKUP_IMAGES = [
    # Simple
    ("simple_01_solid_blue.jpg", None, "simple"),
    ("simple_02_circle.jpg", None, "simple"),
    ("simple_03_gradient.jpg", None, "simple"),
    ("simple_04_square.jpg", None, "simple"),
    ("simple_05_stripe.jpg", None, "simple"),
    # Medium
    ("medium_01_grid.jpg", None, "medium"),
    ("medium_02_shapes.jpg", None, "medium"),
    ("medium_03_checker.jpg", None, "medium"),
    ("medium_04_dots.jpg", None, "medium"),
    ("medium_05_waves.jpg", None, "medium"),
    # Heavy
    ("heavy_01_noise.jpg", None, "heavy"),
    ("heavy_02_complex_pattern.jpg", None, "heavy"),
    ("heavy_03_fractal.jpg", None, "heavy"),
    ("heavy_04_mosaic.jpg", None, "heavy"),
    ("heavy_05_static.jpg", None, "heavy"),
]


def create_synthetic_image(name: str, complexity: str) -> Image.Image:
    """Create synthetic images when downloads fail."""
    import numpy as np
    
    size = 224
    
    if complexity == "simple":
        if "solid" in name or "01" in name:
            # Solid color with single shape
            img = np.ones((size, size, 3), dtype=np.uint8) * 240
            img[60:164, 60:164] = [70, 130, 180]  # Steel blue square
        elif "circle" in name or "02" in name:
            # Single circle
            img = np.ones((size, size, 3), dtype=np.uint8) * 250
            y, x = np.ogrid[:size, :size]
            mask = ((x - 112)**2 + (y - 112)**2) < 50**2
            img[mask] = [220, 60, 60]  # Red circle
        elif "gradient" in name or "03" in name:
            # Simple gradient
            img = np.zeros((size, size, 3), dtype=np.uint8)
            for i in range(size):
                img[i, :] = [i, 100, 255 - i]
        elif "square" in name or "04" in name:
            # Centered square
            img = np.ones((size, size, 3), dtype=np.uint8) * 230
            img[70:154, 70:154] = [60, 179, 113]  # Medium sea green
        else:
            # Stripes
            img = np.zeros((size, size, 3), dtype=np.uint8)
            for i in range(0, size, 20):
                img[i:i+10, :] = [200, 200, 200]
    
    elif complexity == "medium":
        if "grid" in name or "01" in name:
            # Grid of shapes
            img = np.ones((size, size, 3), dtype=np.uint8) * 220
            for i in range(0, size, 56):
                for j in range(0, size, 56):
                    color = [(i+j) % 255, (i*2) % 255, (j*2) % 255]
                    img[i:i+40, j:j+40] = color
        elif "shapes" in name or "02" in name:
            # Multiple different shapes
            img = np.ones((size, size, 3), dtype=np.uint8) * 200
            img[20:60, 20:80] = [255, 100, 100]
            img[80:140, 50:110] = [100, 255, 100]
            img[30:90, 130:190] = [100, 100, 255]
            img[150:200, 80:160] = [255, 255, 100]
            img[140:180, 160:210] = [255, 100, 255]
        elif "checker" in name or "03" in name:
            # Checkerboard
            img = np.zeros((size, size, 3), dtype=np.uint8)
            for i in range(0, size, 28):
                for j in range(0, size, 28):
                    if (i // 28 + j // 28) % 2 == 0:
                        img[i:i+28, j:j+28] = [180, 180, 180]
                    else:
                        img[i:i+28, j:j+28] = [60, 60, 60]
        elif "dots" in name or "04" in name:
            # Dot pattern
            img = np.ones((size, size, 3), dtype=np.uint8) * 240
            for i in range(10, size, 25):
                for j in range(10, size, 25):
                    y, x = np.ogrid[:size, :size]
                    mask = ((x - j)**2 + (y - i)**2) < 8**2
                    img[mask] = [50, 50, 150]
        else:
            # Waves
            img = np.zeros((size, size, 3), dtype=np.uint8)
            for i in range(size):
                for j in range(size):
                    val = int(127 + 127 * np.sin(i/20) * np.cos(j/20))
                    img[i, j] = [val, val, 200]
    
    else:  # heavy/complex
        if "noise" in name or "01" in name:
            # Random noise
            img = np.random.randint(0, 255, (size, size, 3), dtype=np.uint8)
        elif "complex" in name or "02" in name:
            # Complex overlapping patterns
            img = np.zeros((size, size, 3), dtype=np.uint8)
            for i in range(size):
                for j in range(size):
                    img[i, j] = [
                        int(127 + 127 * np.sin(i/10 + j/15)),
                        int(127 + 127 * np.sin(i/8 - j/12)),
                        int(127 + 127 * np.cos(i/12 + j/8))
                    ]
        elif "fractal" in name or "03" in name:
            # Fractal-like pattern
            img = np.zeros((size, size, 3), dtype=np.uint8)
            for i in range(size):
                for j in range(size):
                    val = ((i * j) ^ (i + j)) % 255
                    img[i, j] = [val, (val * 2) % 255, (val * 3) % 255]
        elif "mosaic" in name or "04" in name:
            # Many small random rectangles
            img = np.ones((size, size, 3), dtype=np.uint8) * 128
            for _ in range(100):
                x, y = np.random.randint(0, size-20, 2)
                w, h = np.random.randint(5, 20, 2)
                color = np.random.randint(0, 255, 3)
                img[y:y+h, x:x+w] = color
        else:
            # TV static-like
            img = np.random.randint(0, 255, (size, size, 3), dtype=np.uint8)
            # Add some structure
            for i in range(0, size, 4):
                img[i:i+2, :] = img[i:i+2, :] // 2
    
    return Image.fromarray(img)


def download_image(url: str, filepath: Path) -> bool:
    """Download image from URL."""
    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        request = urllib.request.Request(url, headers=headers)
        
        with urllib.request.urlopen(request, timeout=10) as response:
            data = response.read()
            
        img = Image.open(io.BytesIO(data))
        img = img.convert('RGB')
        img = img.resize((224, 224))
        img.save(filepath, 'JPEG', quality=95)
        return True
    except Exception as e:
        print(f"  Failed to download: {e}")
        return False


def main():
    print("=" * 60)
    print("DOWNLOADING TEST IMAGES")
    print("=" * 60)
    print(f"\nOutput directory: {OUTPUT_DIR.absolute()}\n")
    
    downloaded = {"simple": 0, "medium": 0, "heavy": 0}
    
    # Try downloading real images first
    print("Attempting to download real images from Unsplash...")
    for filename, url, complexity in IMAGES:
        filepath = OUTPUT_DIR / filename
        
        if filepath.exists():
            print(f"  [EXISTS] {filename}")
            downloaded[complexity] += 1
            continue
        
        print(f"  Downloading {filename}...", end=" ")
        if download_image(url, filepath):
            print("✓")
            downloaded[complexity] += 1
        else:
            print("✗")
    
    # Create synthetic images to fill gaps
    print("\nCreating synthetic images for any missing categories...")
    
    for complexity in ["simple", "medium", "heavy"]:
        needed = 5 - downloaded[complexity]
        if needed > 0:
            print(f"  Creating {needed} synthetic {complexity} images...")
            for i in range(needed):
                idx = downloaded[complexity] + i + 1
                filename = f"{complexity}_{idx:02d}_synthetic.jpg"
                filepath = OUTPUT_DIR / filename
                
                if not filepath.exists():
                    img = create_synthetic_image(filename, complexity)
                    img.save(filepath, 'JPEG', quality=95)
                    print(f"    Created: {filename}")
    
    # Summary
    print("\n" + "=" * 60)
    print("DOWNLOAD COMPLETE")
    print("=" * 60)
    
    all_images = list(OUTPUT_DIR.glob("*.jpg"))
    print(f"\nTotal images: {len(all_images)}")
    print(f"Location: {OUTPUT_DIR.absolute()}")
    
    print("\nImages by complexity:")
    for complexity in ["simple", "medium", "heavy"]:
        imgs = list(OUTPUT_DIR.glob(f"{complexity}_*.jpg"))
        print(f"  {complexity.upper()}: {len(imgs)} images")
        for img in sorted(imgs):
            print(f"    - {img.name}")
    
    print("\n Images are ready for testing!")
    print(f"\nNext step: python scripts/test_downloaded_images.py")


if __name__ == "__main__":
    main()