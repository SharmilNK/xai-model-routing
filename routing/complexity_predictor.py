"""
Complexity Predictor
Predicts which model tier is needed based on XAI features.
"""

import torch
import torch.nn as nn
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
import joblib
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import IntEnum


class ModelTier(IntEnum):
    """Model complexity tiers for routing."""
    TINY = 0      # MobileNetV3-Small, ~2M params
    LIGHT = 1     # MobileNetV3-Large, ~5M params  
    MEDIUM = 2    # EfficientNet-B0, ~5M params
    HEAVY = 3     # ResNet50 / EfficientNet-B4, ~25M params


@dataclass
class RoutingDecision:
    """Container for routing decision and metadata."""
    tier: ModelTier
    confidence: float
    predicted_latency_ms: float
    predicted_flops_m: float
    reasoning: str


class ComplexityPredictor:
    """
    Predicts the minimum model tier needed for accurate inference.
    Uses XAI features as input to a lightweight meta-model.
    """
    
    # Approximate compute costs per tier (relative to HEAVY)
    TIER_COSTS = {
        ModelTier.TINY: {'flops_m': 60, 'latency_ms': 5, 'energy_factor': 0.1},
        ModelTier.LIGHT: {'flops_m': 220, 'latency_ms': 12, 'energy_factor': 0.25},
        ModelTier.MEDIUM: {'flops_m': 400, 'latency_ms': 20, 'energy_factor': 0.4},
        ModelTier.HEAVY: {'flops_m': 4100, 'latency_ms': 80, 'energy_factor': 1.0},
    }
    
    def __init__(self, model_path: Optional[Path] = None):
        self.scaler = StandardScaler()
        self.classifier = GradientBoostingClassifier(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.1,
            random_state=42
        )
        self.is_fitted = False
        
        if model_path and model_path.exists():
            self.load(model_path)
    
    def _generate_reasoning(self, features: np.ndarray, tier: ModelTier) -> str:
        """Generate human-readable explanation for routing decision."""
        # Feature indices (matching XAIFeatures order)
        attn_entropy = features[0]
        saliency_sparsity = features[1]
        confidence_margin = features[5]
        
        reasons = []
        
        if attn_entropy > 0.7:
            reasons.append("scattered attention pattern")
        elif attn_entropy < 0.3:
            reasons.append("focused attention")
            
        if saliency_sparsity > 0.4:
            reasons.append("many salient regions")
        elif saliency_sparsity < 0.15:
            reasons.append("single salient object")
            
        if confidence_margin < 0.2:
            reasons.append("low prediction confidence")
        elif confidence_margin > 0.6:
            reasons.append("high prediction confidence")
        
        tier_names = {
            ModelTier.TINY: "tiny",
            ModelTier.LIGHT: "lightweight", 
            ModelTier.MEDIUM: "medium",
            ModelTier.HEAVY: "heavy"
        }
        
        if reasons:
            return f"Routing to {tier_names[tier]} model due to: {', '.join(reasons)}"
        return f"Routing to {tier_names[tier]} model based on complexity analysis"
    
    def train(self, 
              features: np.ndarray, 
              labels: np.ndarray,
              validate: bool = True) -> Dict[str, float]:
        """
        Train the complexity predictor.
        
        Args:
            features: XAI feature vectors [N, 7]
            labels: Ground truth tier labels [N]
            validate: Whether to run cross-validation
            
        Returns:
            Training metrics
        """
        # Scale features
        features_scaled = self.scaler.fit_transform(features)
        
        # Cross-validation
        metrics = {}
        if validate:
            cv_scores = cross_val_score(self.classifier, features_scaled, labels, cv=5)
            metrics['cv_accuracy_mean'] = cv_scores.mean()
            metrics['cv_accuracy_std'] = cv_scores.std()
        
        # Full training
        self.classifier.fit(features_scaled, labels)
        self.is_fitted = True
        
        # Training accuracy
        train_preds = self.classifier.predict(features_scaled)
        metrics['train_accuracy'] = (train_preds == labels).mean()
        
        # Feature importances
        feature_names = [
            'attention_entropy', 'saliency_sparsity', 'gradient_magnitude',
            'feature_importance_variance', 'spatial_complexity', 
            'confidence_margin', 'activation_sparsity'
        ]
        metrics['feature_importances'] = dict(zip(
            feature_names, 
            self.classifier.feature_importances_
        ))
        
        return metrics
    
    def predict(self, features: np.ndarray) -> RoutingDecision:
        """
        Predict the optimal model tier for given XAI features.
        
        Args:
            features: XAI feature vector [7] or [1, 7]
            
        Returns:
            RoutingDecision with tier and metadata
        """
        if not self.is_fitted:
            raise RuntimeError("Predictor must be trained before prediction")
        
        features = np.atleast_2d(features)
        features_scaled = self.scaler.transform(features)
        
        # Get prediction and probabilities
        tier_idx = self.classifier.predict(features_scaled)[0]
        tier = ModelTier(tier_idx)
        probs = self.classifier.predict_proba(features_scaled)[0]
        confidence = probs[tier_idx]
        
        # Get compute estimates
        costs = self.TIER_COSTS[tier]
        
        return RoutingDecision(
            tier=tier,
            confidence=confidence,
            predicted_latency_ms=costs['latency_ms'],
            predicted_flops_m=costs['flops_m'],
            reasoning=self._generate_reasoning(features[0], tier)
        )
    
    def predict_batch(self, features: np.ndarray) -> List[RoutingDecision]:
        """Predict tiers for a batch of feature vectors."""
        return [self.predict(f) for f in features]
    
    def save(self, path: Path):
        """Save trained model to disk."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        joblib.dump(self.classifier, path / "classifier.joblib")
        joblib.dump(self.scaler, path / "scaler.joblib")
        
    def load(self, path: Path):
        """Load trained model from disk."""
        path = Path(path)
        
        self.classifier = joblib.load(path / "classifier.joblib")
        self.scaler = joblib.load(path / "scaler.joblib")
        self.is_fitted = True


def generate_synthetic_training_data(n_samples: int = 1000) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic training data for initial experiments.
    In practice, you'd collect this from real model performance data.
    """
    np.random.seed(42)
    
    features = []
    labels = []
    
    for _ in range(n_samples):
        # Generate complexity-correlated features
        complexity = np.random.beta(2, 5)  # Skew toward simpler
        
        f = np.array([
            complexity * 0.8 + np.random.normal(0, 0.1),  # attention_entropy
            complexity * 0.5 + np.random.normal(0, 0.1),  # saliency_sparsity
            0.1 + complexity * 0.3 + np.random.normal(0, 0.05),  # gradient_magnitude
            complexity * 0.2 + np.random.normal(0, 0.05),  # feature_importance_variance
            complexity * 0.6 + np.random.normal(0, 0.1),  # spatial_complexity
            (1 - complexity) * 0.7 + 0.1 + np.random.normal(0, 0.1),  # confidence_margin
            (1 - complexity) * 0.5 + 0.3 + np.random.normal(0, 0.1),  # activation_sparsity
        ])
        f = np.clip(f, 0, 1)
        features.append(f)
        
        # Assign tier based on complexity
        if complexity < 0.25:
            tier = ModelTier.TINY
        elif complexity < 0.5:
            tier = ModelTier.LIGHT
        elif complexity < 0.75:
            tier = ModelTier.MEDIUM
        else:
            tier = ModelTier.HEAVY
            
        labels.append(tier)
    
    return np.array(features), np.array(labels)


if __name__ == "__main__":
    # Demo training and prediction
    print("Generating synthetic training data...")
    X, y = generate_synthetic_training_data(1000)
    
    print("Training complexity predictor...")
    predictor = ComplexityPredictor()
    metrics = predictor.train(X, y)
    
    print(f"\nTraining Results:")
    print(f"  CV Accuracy: {metrics['cv_accuracy_mean']:.3f} ± {metrics['cv_accuracy_std']:.3f}")
    print(f"  Train Accuracy: {metrics['train_accuracy']:.3f}")
    print(f"\nFeature Importances:")
    for name, imp in sorted(metrics['feature_importances'].items(), key=lambda x: -x[1]):
        print(f"  {name}: {imp:.3f}")
    
    # Test prediction
    print("\n--- Sample Predictions ---")
    test_simple = np.array([0.2, 0.1, 0.15, 0.05, 0.1, 0.8, 0.7])
    test_complex = np.array([0.8, 0.5, 0.4, 0.25, 0.7, 0.15, 0.3])
    
    for name, features in [("Simple image", test_simple), ("Complex image", test_complex)]:
        decision = predictor.predict(features)
        print(f"\n{name}:")
        print(f"  Tier: {decision.tier.name}")
        print(f"  Confidence: {decision.confidence:.2f}")
        print(f"  Predicted Latency: {decision.predicted_latency_ms}ms")
        print(f"  {decision.reasoning}")