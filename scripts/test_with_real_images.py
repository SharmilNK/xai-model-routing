"""
Test the router with synthetic images of varying complexity.
This simulates simple vs complex inputs.
"""

import torch
import numpy as np
from PIL import Image
from torchvision import transforms
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from routing.router import XAIModelRouter
from routing.complexity_predictor import generate_synthetic_training_data, ModelTier


def create_test_images():
    """Create synthetic images with different complexity levels."""
    images = {}
    
    # 1. SIMPLE: Single solid color with one shape
    simple = np.ones((224, 224, 3), dtype=np.uint8) * 240  # Light gray background
    simple[80:144, 80:144] = [65, 105, 225]  # Blue square in center
    images['simple_square'] = Image.fromarray(simple)
    
    # 2. SIMPLE: Gradient background
    gradient = np.zeros((224, 224, 3), dtype=np.uint8)
    for i in range(224):
        gradient[i, :] = [i, i, 255 - i]
    images['simple_gradient'] = Image.fromarray(gradient)
    
    # 3. MEDIUM: Multiple shapes
    medium = np.ones((224, 224, 3), dtype=np.uint8) * 220
    medium[20:70, 20:90] = [255, 100, 100]    # Red rectangle
    medium[100:180, 60:140] = [100, 255, 100]  # Green rectangle
    medium[40:100, 140:200] = [100, 100, 255]  # Blue rectangle
    medium[150:200, 150:210] = [255, 255, 100] # Yellow rectangle
    images['medium_shapes'] = Image.fromarray(medium)
    
    # 4. MEDIUM: Checkerboard pattern
    checker = np.zeros((224, 224, 3), dtype=np.uint8)
    for i in range(0, 224, 28):
        for j in range(0, 224, 28):
            if (i // 28 + j // 28) % 2 == 0:
                checker[i:i+28, j:j+28] = [200, 200, 200]
            else:
                checker[i:i+28, j:j+28] = [50, 50, 50]
    images['medium_checker'] = Image.fromarray(checker)
    
    # 5. COMPLEX: High-frequency noise
    noise = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    images['complex_noise'] = Image.fromarray(noise)
    
    # 6. COMPLEX: Many small random shapes
    complex_shapes = np.ones((224, 224, 3), dtype=np.uint8) * 128
    for _ in range(50):
        x, y = np.random.randint(0, 200, 2)
        w, h = np.random.randint(5, 25, 2)
        color = np.random.randint(0, 255, 3)
        complex_shapes[y:y+h, x:x+w] = color
    images['complex_many_shapes'] = Image.fromarray(complex_shapes)
    
    # 7. COMPLEX: Fine texture pattern
    texture = np.zeros((224, 224, 3), dtype=np.uint8)
    for i in range(224):
        for j in range(224):
            texture[i, j] = [(i * j) % 255, (i + j) % 255, (i - j) % 255]
    images['complex_texture'] = Image.fromarray(texture)
    
    return images


def main():
    print("=" * 60)
    print("TESTING XAI ROUTER WITH VARYING COMPLEXITY")
    print("=" * 60)
    
    # Initialize router
    print("\nInitializing router...")
    router = XAIModelRouter(device='cuda' if torch.cuda.is_available() else 'cpu')
    
    # Train predictor
    print("Training complexity predictor...")
    X, y = generate_synthetic_training_data(1000)
    router.train_predictor(X, y)
    
    # Preprocessing
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Create test images
    print("\nCreating test images...")
    test_images = create_test_images()
    
    # Test each image
    print("\n" + "-" * 60)
    print(f"{'Image Name':<25} {'Tier':<10} {'Latency':<12} {'Attn Entropy':<15}")
    print("-" * 60)
    
    results_summary = {'TINY': 0, 'LIGHT': 0, 'MEDIUM': 0, 'HEAVY': 0}
    
    for name, img in test_images.items():
        img_tensor = transform(img).unsqueeze(0)
        result = router.route_and_infer(img_tensor)
        
        tier = result.routing_decision.tier.name
        latency = result.actual_latency_ms
        attn_entropy = result.xai_features.attention_entropy
        
        results_summary[tier] += 1
        
        print(f"{name:<25} {tier:<10} {latency:>8.1f}ms   {attn_entropy:.4f}")
    
    # Summary
    print("\n" + "=" * 60)
    print("ROUTING SUMMARY")
    print("=" * 60)
    
    total = len(test_images)
    for tier, count in results_summary.items():
        pct = count / total * 100
        bar = "█" * int(pct / 5)
        print(f"  {tier:<8}: {count} ({pct:5.1f}%) {bar}")
    
    # Compare with baseline
    print("\n" + "=" * 60)
    print("EFFICIENCY COMPARISON (vs always using HEAVY model)")
    print("=" * 60)
    
    # Quick comparison on one image
    sample_img = transform(test_images['simple_square']).unsqueeze(0)
    comparison = router.compare_with_baseline(sample_img)
    
    print(f"\n  Sample image: simple_square")
    print(f"  Routed to: {comparison['routed'].routing_decision.tier.name}")
    print(f"  Latency savings: {comparison['latency_savings_pct']:.1f}%")
    print(f"  FLOPs savings: {comparison['flops_savings_pct']:.1f}%")
    print(f"  Same prediction: {comparison['same_prediction']}")


if __name__ == "__main__":
    main()