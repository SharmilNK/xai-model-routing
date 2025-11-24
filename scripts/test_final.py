"""
Final test script using the fixed XAI Feature Extractor V3.
Tests on downloaded images with proper complexity differentiation.
"""

import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))

from xai.feature_extractor_v3 import XAIFeatureExtractorV3, XAIFeatures
from routing.complexity_predictor import ComplexityPredictor, ModelTier


def create_balanced_training_data(n_per_tier: int = 500):
    """Create training data with clear feature separation per tier."""
    np.random.seed(42)
    features = []
    labels = []
    
    for tier in ModelTier:
        for _ in range(n_per_tier):
            if tier == ModelTier.TINY:
                # Simple: low entropy, low spatial, high confidence, high sparsity
                f = [
                    np.random.uniform(0.05, 0.20),   # attention_entropy
                    np.random.uniform(0.05, 0.15),   # saliency_sparsity
                    np.random.uniform(0.02, 0.08),   # gradient_magnitude
                    np.random.uniform(0.02, 0.08),   # feature_variance
                    np.random.uniform(0.60, 0.85),   # spatial_complexity (edges exist but simple)
                    np.random.uniform(0.05, 0.15),   # confidence_margin (low for synthetic)
                    np.random.uniform(0.80, 0.98),   # activation_sparsity (high)
                ]
            elif tier == ModelTier.LIGHT:
                # Light: moderate values
                f = [
                    np.random.uniform(0.15, 0.35),   # attention_entropy
                    np.random.uniform(0.12, 0.30),   # saliency_sparsity
                    np.random.uniform(0.05, 0.12),   # gradient_magnitude
                    np.random.uniform(0.04, 0.10),   # feature_variance
                    np.random.uniform(0.50, 0.75),   # spatial_complexity
                    np.random.uniform(0.10, 0.30),   # confidence_margin
                    np.random.uniform(0.60, 0.85),   # activation_sparsity
                ]
            elif tier == ModelTier.MEDIUM:
                # Medium: higher complexity
                f = [
                    np.random.uniform(0.30, 0.55),   # attention_entropy
                    np.random.uniform(0.08, 0.20),   # saliency_sparsity
                    np.random.uniform(0.03, 0.08),   # gradient_magnitude
                    np.random.uniform(0.03, 0.07),   # feature_variance
                    np.random.uniform(0.70, 0.90),   # spatial_complexity
                    np.random.uniform(0.02, 0.12),   # confidence_margin
                    np.random.uniform(0.35, 0.65),   # activation_sparsity
                ]
            else:  # HEAVY
                # Heavy: highest complexity, random noise-like
                f = [
                    np.random.uniform(0.50, 0.95),   # attention_entropy (high)
                    np.random.uniform(0.00, 0.10),   # saliency_sparsity (low - diffuse)
                    np.random.uniform(0.00, 0.05),   # gradient_magnitude (low)
                    np.random.uniform(0.01, 0.05),   # feature_variance
                    np.random.uniform(0.85, 1.00),   # spatial_complexity (high)
                    np.random.uniform(0.00, 0.08),   # confidence_margin (low)
                    np.random.uniform(0.02, 0.30),   # activation_sparsity (low)
                ]
            
            features.append(f)
            labels.append(int(tier))
    
    return np.array(features), np.array(labels)


def main():
    print("=" * 75)
    print("XAI MODEL ROUTING - FINAL TEST")
    print("=" * 75)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nDevice: {device}")
    
    # Initialize extractor
    print("Loading XAI Feature Extractor V3...")
    extractor = XAIFeatureExtractorV3(device=device)
    
    # Train complexity predictor with balanced data
    print("Training complexity predictor with balanced data...")
    X, y = create_balanced_training_data(500)
    
    predictor = ComplexityPredictor()
    metrics = predictor.train(X, y)
    print(f"  CV Accuracy: {metrics['cv_accuracy_mean']:.3f} ± {metrics['cv_accuracy_std']:.3f}")
    print(f"  Train Accuracy: {metrics['train_accuracy']:.3f}")
    
    # Feature importances
    print("\n  Feature Importances:")
    for name, imp in sorted(metrics['feature_importances'].items(), key=lambda x: -x[1])[:4]:
        print(f"    {name}: {imp:.3f}")
    
    # Preprocessing
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Check for downloaded images
    image_dir = Path("data/test_images")
    if image_dir.exists() and len(list(image_dir.glob("*.jpg"))) > 0:
        print(f"\n📁 Testing on downloaded images from: {image_dir}")
        image_paths = sorted(image_dir.glob("*.jpg"))
    else:
        print("\n⚠️ No downloaded images found. Creating synthetic test images...")
        image_dir.mkdir(parents=True, exist_ok=True)
        
        # Create synthetic images
        def save_synthetic(name, arr):
            Image.fromarray(arr).save(image_dir / f"{name}.jpg")
        
        # Simple images
        for i in range(5):
            arr = np.ones((224, 224, 3), dtype=np.uint8) * (200 + i * 10)
            size = 60 + i * 10
            offset = 80 - i * 5
            color = [70 + i*20, 130, 180 - i*20]
            arr[offset:offset+size, offset:offset+size] = color
            save_synthetic(f"simple_{i+1:02d}_square", arr)
        
        # Medium images
        for i in range(5):
            arr = np.ones((224, 224, 3), dtype=np.uint8) * 200
            for j in range(3 + i):
                x, y = np.random.randint(10, 150, 2)
                w, h = np.random.randint(30, 70, 2)
                color = np.random.randint(50, 255, 3).tolist()
                arr[y:y+h, x:x+w] = color
            save_synthetic(f"medium_{i+1:02d}_shapes", arr)
        
        # Complex/Heavy images
        for i in range(5):
            arr = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
            save_synthetic(f"heavy_{i+1:02d}_noise", arr)
        
        image_paths = sorted(image_dir.glob("*.jpg"))
        print(f"  Created {len(image_paths)} synthetic test images")
    
    # Test each image
    print("\n" + "-" * 75)
    print(f"{'Image':<35} {'Expected':<12} {'Predicted':<10} {'Confidence':<12} {'Match'}")
    print("-" * 75)
    
    results = []
    tier_counts = {tier.name: 0 for tier in ModelTier}
    correct = 0
    total = 0
    
    for img_path in image_paths:
        name = img_path.stem
        
        # Expected tier from filename
        if name.startswith("simple"):
            expected_tiers = ["TINY", "LIGHT"]
            category = "simple"
        elif name.startswith("medium"):
            expected_tiers = ["LIGHT", "MEDIUM"]
            category = "medium"
        else:
            expected_tiers = ["MEDIUM", "HEAVY"]
            category = "heavy"
        
        # Load and extract features
        img = Image.open(img_path).convert('RGB')
        img_tensor = transform(img).unsqueeze(0)
        features = extractor.extract(img_tensor)
        
        # Predict tier
        decision = predictor.predict(features.to_vector())
        tier = decision.tier.name
        conf = decision.confidence
        
        # Check match
        match = "✓" if tier in expected_tiers else "✗"
        if tier in expected_tiers:
            correct += 1
        total += 1
        
        tier_counts[tier] += 1
        
        results.append({
            'name': name,
            'category': category,
            'tier': tier,
            'confidence': conf,
            'features': features
        })
        
        exp_str = "/".join(expected_tiers)
        print(f"{name:<35} {exp_str:<12} {tier:<10} {conf:<12.2%} {match}")
    
    # Summary
    print("\n" + "=" * 75)
    print("RESULTS SUMMARY")
    print("=" * 75)
    
    print(f"\nRouting Accuracy: {correct}/{total} ({correct/total*100:.1f}%)")
    
    print("\nTier Distribution:")
    for tier in ["TINY", "LIGHT", "MEDIUM", "HEAVY"]:
        count = tier_counts[tier]
        pct = count / total * 100
        bar = "█" * int(pct / 2.5)
        print(f"  {tier:<8}: {count:>2} ({pct:5.1f}%) {bar}")
    
    # Per-category breakdown
    print("\nBy Category:")
    for cat in ["simple", "medium", "heavy"]:
        cat_results = [r for r in results if r['category'] == cat]
        if cat_results:
            tiers = [r['tier'] for r in cat_results]
            print(f"\n  {cat.upper()} ({len(cat_results)} images):")
            for tier in ["TINY", "LIGHT", "MEDIUM", "HEAVY"]:
                count = tiers.count(tier)
                if count > 0:
                    print(f"    → {tier}: {count}")
    
    # Feature analysis
    print("\n" + "=" * 75)
    print("FEATURE ANALYSIS BY CATEGORY")
    print("=" * 75)
    
    for cat in ["simple", "medium", "heavy"]:
        cat_results = [r for r in results if r['category'] == cat]
        if cat_results:
            print(f"\n{cat.upper()}:")
            
            # Average features
            avg_features = {}
            for key in cat_results[0]['features'].to_dict().keys():
                vals = [r['features'].to_dict()[key] for r in cat_results]
                avg_features[key] = np.mean(vals)
            
            for key in ['attention_entropy', 'spatial_complexity', 'activation_sparsity']:
                val = avg_features[key]
                bar = "█" * int(val * 20)
                print(f"  {key:<25}: {val:.3f} {bar}")
    
    print("\n" + "=" * 75)
    print("✓ Test complete!")


if __name__ == "__main__":
    main()