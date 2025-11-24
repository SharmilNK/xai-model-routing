"""
Complete Training Pipeline for XAI Model Router

This script:
1. Loads complexity dataset (or generates synthetic)
2. Trains the complexity predictor
3. Evaluates on validation set
4. Saves the trained model
5. Runs quick benchmark

Usage:
    python scripts/train_router.py --dataset ./data/complexity_dataset.npz --output ./models/router
"""

import argparse
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
import json
import matplotlib.pyplot as plt

import sys
sys.path.append(str(Path(__file__).parent.parent))

from routing.complexity_predictor import ComplexityPredictor, ModelTier, generate_synthetic_training_data


def load_dataset(dataset_path: Path) -> tuple:
    """Load dataset from .npz file."""
    data = np.load(dataset_path)
    return data['features'], data['labels']


def analyze_dataset(features: np.ndarray, labels: np.ndarray):
    """Print dataset statistics."""
    print("\n" + "=" * 50)
    print("DATASET ANALYSIS")
    print("=" * 50)
    
    print(f"\nTotal samples: {len(labels)}")
    print(f"Feature dimensions: {features.shape[1]}")
    
    print("\nTier distribution:")
    for tier in ModelTier:
        count = (labels == tier).sum()
        pct = count / len(labels) * 100
        bar = "█" * int(pct / 2)
        print(f"  {tier.name:<8}: {count:>5} ({pct:5.1f}%) {bar}")
    
    print("\nFeature statistics:")
    feature_names = [
        'attention_entropy', 'saliency_sparsity', 'gradient_magnitude',
        'feature_variance', 'spatial_complexity', 'confidence_margin',
        'activation_sparsity'
    ]
    
    for i, name in enumerate(feature_names):
        vals = features[:, i]
        print(f"  {name:<25}: mean={vals.mean():.3f}, std={vals.std():.3f}, "
              f"min={vals.min():.3f}, max={vals.max():.3f}")


def train_and_evaluate(features: np.ndarray, 
                       labels: np.ndarray,
                       output_dir: Path,
                       test_size: float = 0.2) -> dict:
    """Train predictor and evaluate on held-out test set."""
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, test_size=test_size, random_state=42, stratify=labels
    )
    
    print(f"\nTraining set: {len(y_train)} samples")
    print(f"Test set: {len(y_test)} samples")
    
    # Initialize and train
    predictor = ComplexityPredictor()
    
    print("\nTraining complexity predictor...")
    train_metrics = predictor.train(X_train, y_train, validate=True)
    
    print(f"\n✓ Cross-validation accuracy: {train_metrics['cv_accuracy_mean']:.3f} "
          f"± {train_metrics['cv_accuracy_std']:.3f}")
    print(f"✓ Training accuracy: {train_metrics['train_accuracy']:.3f}")
    
    # Feature importances
    print("\nFeature Importances:")
    sorted_imp = sorted(train_metrics['feature_importances'].items(), 
                       key=lambda x: -x[1])
    for name, imp in sorted_imp:
        bar = "█" * int(imp * 50)
        print(f"  {name:<25}: {imp:.3f} {bar}")
    
    # Evaluate on test set
    print("\nEvaluating on test set...")
    test_preds = []
    for feat in X_test:
        decision = predictor.predict(feat)
        test_preds.append(decision.tier)
    
    test_preds = np.array(test_preds)
    test_accuracy = (test_preds == y_test).mean()
    
    print(f"✓ Test accuracy: {test_accuracy:.3f}")
    
    # Per-tier accuracy
    print("\nPer-tier accuracy:")
    for tier in ModelTier:
        mask = y_test == tier
        if mask.sum() > 0:
            tier_acc = (test_preds[mask] == y_test[mask]).mean()
            print(f"  {tier.name:<8}: {tier_acc:.3f} (n={mask.sum()})")
    
    # Confusion matrix
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_test, test_preds)
    
    # Save model
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    predictor.save(output_dir)
    print(f"\n✓ Model saved to: {output_dir}")
    
    # Save training metrics
    metrics = {
        'cv_accuracy_mean': train_metrics['cv_accuracy_mean'],
        'cv_accuracy_std': train_metrics['cv_accuracy_std'],
        'train_accuracy': train_metrics['train_accuracy'],
        'test_accuracy': float(test_accuracy),
        'feature_importances': train_metrics['feature_importances'],
        'confusion_matrix': cm.tolist()
    }
    
    with open(output_dir / 'training_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    
    # Plot confusion matrix
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, cmap='Blues')
    
    tier_names = [t.name for t in ModelTier]
    ax.set_xticks(range(len(tier_names)))
    ax.set_yticks(range(len(tier_names)))
    ax.set_xticklabels(tier_names)
    ax.set_yticklabels(tier_names)
    ax.set_xlabel('Predicted Tier')
    ax.set_ylabel('True Tier')
    ax.set_title('Confusion Matrix')
    
    # Add text annotations
    for i in range(len(tier_names)):
        for j in range(len(tier_names)):
            text = ax.text(j, i, cm[i, j], ha='center', va='center',
                          color='white' if cm[i, j] > cm.max()/2 else 'black')
    
    plt.colorbar(im)
    plt.tight_layout()
    plt.savefig(output_dir / 'confusion_matrix.png', dpi=150)
    print(f"✓ Confusion matrix saved to: {output_dir / 'confusion_matrix.png'}")
    
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Train XAI Model Router")
    parser.add_argument('--dataset', type=str, default=None,
                        help="Path to complexity dataset (.npz)")
    parser.add_argument('--synthetic_samples', type=int, default=2000,
                        help="Number of synthetic samples if no dataset provided")
    parser.add_argument('--output', type=str, default='./models/router',
                        help="Output directory for trained model")
    parser.add_argument('--test_size', type=float, default=0.2,
                        help="Fraction of data for testing")
    
    args = parser.parse_args()
    
    print("=" * 50)
    print("XAI MODEL ROUTER - TRAINING PIPELINE")
    print("=" * 50)
    
    # Load or generate data
    if args.dataset and Path(args.dataset).exists():
        print(f"\nLoading dataset from: {args.dataset}")
        features, labels = load_dataset(Path(args.dataset))
    else:
        print(f"\nGenerating {args.synthetic_samples} synthetic samples...")
        features, labels = generate_synthetic_training_data(args.synthetic_samples)
    
    # Analyze dataset
    analyze_dataset(features, labels)
    
    # Train and evaluate
    metrics = train_and_evaluate(
        features, labels,
        output_dir=Path(args.output),
        test_size=args.test_size
    )
    
    print("\n" + "=" * 50)
    print("TRAINING COMPLETE")
    print("=" * 50)
    print(f"\nFinal Test Accuracy: {metrics['test_accuracy']:.1%}")
    print(f"Model saved to: {args.output}")
    
    print("\nNext steps:")
    print("  1. Run: python scripts/evaluate_system.py --model_dir ./models/router")
    print("  2. Launch demo: streamlit run app/streamlit_app.py")


if __name__ == "__main__":
    main()