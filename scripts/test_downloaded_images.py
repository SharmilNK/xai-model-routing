"""
Test the XAI router with downloaded/synthetic test images.
Compares routing decisions across simple, medium, and heavy complexity images.
"""

import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))

from routing.router import XAIModelRouter
from routing.complexity_predictor import generate_synthetic_training_data, ModelTier


def main():
    print("=" * 70)
    print("TESTING XAI ROUTER WITH DOWNLOADED IMAGES")
    print("=" * 70)
    
    # Check for images
    image_dir = Path("data/test_images")
    if not image_dir.exists() or len(list(image_dir.glob("*.jpg"))) == 0:
        print("\n⚠️  No test images found!")
        print("Run this first: python scripts/download_test_images.py")
        return
    
    # Initialize router
    print("\nInitializing router...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    router = XAIModelRouter(device=device)
    
    # Train predictor with MORE balanced data for better MEDIUM/HEAVY routing
    print("Training complexity predictor with balanced data...")
    
    # Generate custom training data with more MEDIUM and HEAVY samples
    np.random.seed(42)
    n_samples = 2000
    features = []
    labels = []
    
    # Create balanced dataset
    samples_per_tier = n_samples // 4
    
    for tier in [ModelTier.TINY, ModelTier.LIGHT, ModelTier.MEDIUM, ModelTier.HEAVY]:
        for _ in range(samples_per_tier):
            if tier == ModelTier.TINY:
                # Simple: low entropy, high confidence, low sparsity
                f = np.array([
                    np.random.uniform(0.0, 0.2),   # attention_entropy (low)
                    np.random.uniform(0.0, 0.15),  # saliency_sparsity (low)
                    np.random.uniform(0.05, 0.2),  # gradient_magnitude
                    np.random.uniform(0.0, 0.1),   # feature_importance_variance
                    np.random.uniform(0.0, 0.2),   # spatial_complexity (low)
                    np.random.uniform(0.6, 0.95),  # confidence_margin (high)
                    np.random.uniform(0.6, 0.9),   # activation_sparsity (high)
                ])
            elif tier == ModelTier.LIGHT:
                # Light: moderate values
                f = np.array([
                    np.random.uniform(0.15, 0.4),  # attention_entropy
                    np.random.uniform(0.1, 0.3),   # saliency_sparsity
                    np.random.uniform(0.15, 0.3),  # gradient_magnitude
                    np.random.uniform(0.05, 0.15), # feature_importance_variance
                    np.random.uniform(0.15, 0.4),  # spatial_complexity
                    np.random.uniform(0.4, 0.7),   # confidence_margin
                    np.random.uniform(0.4, 0.7),   # activation_sparsity
                ])
            elif tier == ModelTier.MEDIUM:
                # Medium: higher complexity indicators
                f = np.array([
                    np.random.uniform(0.35, 0.6),  # attention_entropy (higher)
                    np.random.uniform(0.25, 0.5),  # saliency_sparsity (higher)
                    np.random.uniform(0.25, 0.4),  # gradient_magnitude
                    np.random.uniform(0.1, 0.25),  # feature_importance_variance
                    np.random.uniform(0.35, 0.6),  # spatial_complexity (higher)
                    np.random.uniform(0.2, 0.5),   # confidence_margin (lower)
                    np.random.uniform(0.25, 0.5),  # activation_sparsity (lower)
                ])
            else:  # HEAVY
                # Heavy: highest complexity, lowest confidence
                f = np.array([
                    np.random.uniform(0.5, 0.9),   # attention_entropy (high)
                    np.random.uniform(0.4, 0.7),   # saliency_sparsity (high)
                    np.random.uniform(0.3, 0.5),   # gradient_magnitude
                    np.random.uniform(0.15, 0.3),  # feature_importance_variance
                    np.random.uniform(0.5, 0.9),   # spatial_complexity (high)
                    np.random.uniform(0.05, 0.3),  # confidence_margin (low)
                    np.random.uniform(0.1, 0.4),   # activation_sparsity (low)
                ])
            
            features.append(f)
            labels.append(int(tier))
    
    X = np.array(features)
    y = np.array(labels)
    
    metrics = router.train_predictor(X, y)
    print(f"Training accuracy: {metrics['train_accuracy']:.3f}")
    print(f"CV accuracy: {metrics['cv_accuracy_mean']:.3f} ± {metrics['cv_accuracy_std']:.3f}")
    
    # Preprocessing
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Load and test images
    print("\n" + "-" * 70)
    print(f"{'Image':<30} {'Expected':<10} {'Routed':<10} {'Latency':<10} {'AttnEnt':<10}")
    print("-" * 70)
    
    results = {
        'simple': {'expected': [], 'actual': []},
        'medium': {'expected': [], 'actual': []},
        'heavy': {'expected': [], 'actual': []}
    }
    
    all_results = []
    
    for img_path in sorted(image_dir.glob("*.jpg")):
        # Determine expected complexity from filename
        name = img_path.stem
        if name.startswith("simple"):
            expected = "TINY/LIGHT"
            category = "simple"
        elif name.startswith("medium"):
            expected = "LIGHT/MED"
            category = "medium"
        else:
            expected = "MED/HEAVY"
            category = "heavy"
        
        # Load and process image
        img = Image.open(img_path).convert('RGB')
        img_tensor = transform(img).unsqueeze(0)
        
        # Run inference
        result = router.route_and_infer(img_tensor)
        tier = result.routing_decision.tier.name
        latency = result.actual_latency_ms
        attn_ent = result.xai_features.attention_entropy
        
        # Store results
        results[category]['actual'].append(tier)
        all_results.append({
            'name': name,
            'category': category,
            'tier': tier,
            'latency': latency,
            'attn_entropy': attn_ent,
            'features': result.xai_features
        })
        
        # Check if routing matches expectation
        match = "✓" if (
            (category == "simple" and tier in ["TINY", "LIGHT"]) or
            (category == "medium" and tier in ["LIGHT", "MEDIUM"]) or
            (category == "heavy" and tier in ["MEDIUM", "HEAVY"])
        ) else "✗"
        
        print(f"{name:<30} {expected:<10} {tier:<10} {latency:>6.1f}ms   {attn_ent:.4f}  {match}")
    
    # Summary statistics
    print("\n" + "=" * 70)
    print("ROUTING SUMMARY BY CATEGORY")
    print("=" * 70)
    
    for category in ["simple", "medium", "heavy"]:
        tiers = results[category]['actual']
        if tiers:
            print(f"\n{category.upper()} images ({len(tiers)} total):")
            for tier in ["TINY", "LIGHT", "MEDIUM", "HEAVY"]:
                count = tiers.count(tier)
                if count > 0:
                    pct = count / len(tiers) * 100
                    bar = "█" * int(pct / 5)
                    print(f"  {tier:<8}: {count} ({pct:5.1f}%) {bar}")
    
    # Overall tier distribution
    print("\n" + "=" * 70)
    print("OVERALL TIER DISTRIBUTION")
    print("=" * 70)
    
    all_tiers = [r['tier'] for r in all_results]
    total = len(all_tiers)
    
    for tier in ["TINY", "LIGHT", "MEDIUM", "HEAVY"]:
        count = all_tiers.count(tier)
        pct = count / total * 100
        bar = "█" * int(pct / 2.5)
        print(f"  {tier:<8}: {count:>2} ({pct:5.1f}%) {bar}")
    
    # XAI Feature Analysis
    print("\n" + "=" * 70)
    print("XAI FEATURE ANALYSIS BY CATEGORY")
    print("=" * 70)
    
    for category in ["simple", "medium", "heavy"]:
        cat_results = [r for r in all_results if r['category'] == category]
        if cat_results:
            avg_entropy = np.mean([r['attn_entropy'] for r in cat_results])
            avg_latency = np.mean([r['latency'] for r in cat_results])
            print(f"\n{category.upper()}:")
            print(f"  Avg Attention Entropy: {avg_entropy:.4f}")
            print(f"  Avg Latency: {avg_latency:.1f}ms")
    
    # Efficiency summary
    print("\n" + "=" * 70)
    print("EFFICIENCY SUMMARY")
    print("=" * 70)
    
    avg_latency = np.mean([r['latency'] for r in all_results])
    baseline_latency = 80  # Approximate ResNet50 latency
    savings = (1 - avg_latency / baseline_latency) * 100
    
    print(f"\n  Average latency (routed): {avg_latency:.1f}ms")
    print(f"  Baseline latency (heavy): ~{baseline_latency}ms")
    print(f"  Latency savings: {savings:.1f}%")
    
    # Count efficient routings
    efficient = sum(1 for t in all_tiers if t in ["TINY", "LIGHT"])
    print(f"\n  Images routed to efficient models (TINY/LIGHT): {efficient}/{total} ({efficient/total*100:.0f}%)")


if __name__ == "__main__":
    main()