"""
Dynamic Model Router
Routes inputs to the optimal model based on XAI complexity analysis.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import timm
import time
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from pathlib import Path
from thop import profile, clever_format

# Import our modules
from xai.feature_extractor import XAIFeatureExtractor, XAIFeatures
from routing.complexity_predictor import ComplexityPredictor, ModelTier, RoutingDecision


@dataclass
class InferenceResult:
    """Complete result from the routing system."""
    # Prediction
    predicted_class: int
    class_probabilities: np.ndarray
    top5_classes: List[Tuple[int, float]]
    
    # Routing info
    routing_decision: RoutingDecision
    model_used: str
    
    # Compute metrics
    actual_latency_ms: float
    flops: int
    params: int
    
    # Explainability
    xai_features: XAIFeatures
    attention_map: Optional[np.ndarray] = None
    saliency_map: Optional[np.ndarray] = None


class ModelPool:
    """
    Pool of models at different complexity tiers.
    Lazy-loads models on first use to save memory.
    """
    
    TIER_MODELS = {
        ModelTier.TINY: "mobilenetv3_small_100",
        ModelTier.LIGHT: "mobilenetv3_large_100",
        ModelTier.MEDIUM: "efficientnet_b0",
        ModelTier.HEAVY: "resnet50",
    }
    
    def __init__(self, device: str = "cuda", preload_all: bool = False):
        self.device = device if torch.cuda.is_available() else "cpu"
        self.models: Dict[ModelTier, nn.Module] = {}
        self.model_stats: Dict[ModelTier, Dict] = {}
        
        if preload_all:
            for tier in ModelTier:
                self._load_model(tier)
    
    def _load_model(self, tier: ModelTier) -> nn.Module:
        """Load a model for the given tier."""
        if tier not in self.models:
            model_name = self.TIER_MODELS[tier]
            print(f"Loading {model_name} for tier {tier.name}...")
            
            model = timm.create_model(model_name, pretrained=True)
            model.eval().to(self.device)
            self.models[tier] = model
            
            # Compute and cache model stats
            dummy_input = torch.randn(1, 3, 224, 224).to(self.device)
            flops, params = profile(model, inputs=(dummy_input,), verbose=False)
            self.model_stats[tier] = {
                'name': model_name,
                'flops': int(flops),
                'params': int(params),
                'flops_str': clever_format(flops, "%.2f")[0],
                'params_str': clever_format(params, "%.2f")[0]
            }
            
        return self.models[tier]
    
    def get_model(self, tier: ModelTier) -> nn.Module:
        """Get model for tier, loading if necessary."""
        return self._load_model(tier)
    
    def get_stats(self, tier: ModelTier) -> Dict:
        """Get compute stats for a tier."""
        if tier not in self.model_stats:
            self._load_model(tier)
        return self.model_stats[tier]


class XAIModelRouter:
    """
    Main routing system that:
    1. Analyzes input complexity using XAI features
    2. Routes to the optimal model tier
    3. Runs inference and returns results with explanations
    """
    
    def __init__(self, 
                 predictor_path: Optional[Path] = None,
                 device: str = "cuda",
                 preload_models: bool = False):
        
        self.device = device if torch.cuda.is_available() else "cpu"
        
        # Initialize components
        self.xai_extractor = XAIFeatureExtractor(device=self.device)
        self.complexity_predictor = ComplexityPredictor(model_path=predictor_path)
        self.model_pool = ModelPool(device=self.device, preload_all=preload_models)
        
        # Tracking for evaluation
        self.inference_history: List[InferenceResult] = []
    
    def train_predictor(self, features: np.ndarray, labels: np.ndarray) -> Dict:
        """Train the complexity predictor."""
        return self.complexity_predictor.train(features, labels)
    
    @torch.no_grad()
    def _run_inference(self, 
                       model: nn.Module, 
                       img_tensor: torch.Tensor) -> Tuple[int, np.ndarray, float]:
        """Run inference and measure latency."""
        img_tensor = img_tensor.to(self.device)
        
        # Warm-up run
        _ = model(img_tensor)
        
        # Timed run
        if self.device == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        
        logits = model(img_tensor)
        
        if self.device == "cuda":
            torch.cuda.synchronize()
        latency = (time.perf_counter() - start) * 1000  # ms
        
        probs = F.softmax(logits, dim=1).cpu().numpy()[0]
        pred_class = int(np.argmax(probs))
        
        return pred_class, probs, latency
    
    def route_and_infer(self, 
                        img_tensor: torch.Tensor,
                        return_explanations: bool = True,
                        force_tier: Optional[ModelTier] = None) -> InferenceResult:
        """
        Main routing pipeline:
        1. Extract XAI features
        2. Predict optimal tier
        3. Route to model
        4. Run inference
        5. Return results with explanations
        
        Args:
            img_tensor: Preprocessed image [1, 3, H, W]
            return_explanations: Whether to include saliency/attention maps
            force_tier: Force a specific tier (for baseline comparison)
        """
        img_tensor = img_tensor.to(self.device)
        
        # Step 1 & 2: Extract features and get routing decision
        xai_features = self.xai_extractor.extract(img_tensor)
        
        if force_tier is not None:
            routing = RoutingDecision(
                tier=force_tier,
                confidence=1.0,
                predicted_latency_ms=self.complexity_predictor.TIER_COSTS[force_tier]['latency_ms'],
                predicted_flops_m=self.complexity_predictor.TIER_COSTS[force_tier]['flops_m'],
                reasoning=f"Forced tier: {force_tier.name}"
            )
        else:
            routing = self.complexity_predictor.predict(xai_features.to_vector())
        
        # Step 3: Get the routed model
        model = self.model_pool.get_model(routing.tier)
        model_stats = self.model_pool.get_stats(routing.tier)
        
        # Step 4: Run inference
        pred_class, probs, latency = self._run_inference(model, img_tensor)
        
        # Get top-5 predictions
        top5_idx = np.argsort(probs)[-5:][::-1]
        top5 = [(int(idx), float(probs[idx])) for idx in top5_idx]
        
        # Step 5: Package results
        result = InferenceResult(
            predicted_class=pred_class,
            class_probabilities=probs,
            top5_classes=top5,
            routing_decision=routing,
            model_used=model_stats['name'],
            actual_latency_ms=latency,
            flops=model_stats['flops'],
            params=model_stats['params'],
            xai_features=xai_features,
        )
        
        self.inference_history.append(result)
        return result
    
    def compare_with_baseline(self, 
                              img_tensor: torch.Tensor,
                              baseline_tier: ModelTier = ModelTier.HEAVY) -> Dict:
        """
        Compare routed inference vs always using the baseline (heavy) model.
        """
        # Routed inference
        routed_result = self.route_and_infer(img_tensor.clone())
        
        # Baseline inference
        baseline_result = self.route_and_infer(img_tensor.clone(), force_tier=baseline_tier)
        
        # Compute savings
        latency_savings = 1 - (routed_result.actual_latency_ms / baseline_result.actual_latency_ms)
        flops_savings = 1 - (routed_result.flops / baseline_result.flops)
        
        return {
            'routed': routed_result,
            'baseline': baseline_result,
            'same_prediction': routed_result.predicted_class == baseline_result.predicted_class,
            'latency_savings_pct': latency_savings * 100,
            'flops_savings_pct': flops_savings * 100,
            'routed_tier': routed_result.routing_decision.tier.name,
        }
    
    def get_efficiency_summary(self) -> Dict:
        """Get summary statistics from inference history."""
        if not self.inference_history:
            return {}
        
        tier_counts = {}
        total_latency = 0
        baseline_latency = 0
        
        for result in self.inference_history:
            tier = result.routing_decision.tier.name
            tier_counts[tier] = tier_counts.get(tier, 0) + 1
            total_latency += result.actual_latency_ms
            # Estimate baseline latency (using HEAVY tier stats)
            baseline_latency += self.complexity_predictor.TIER_COSTS[ModelTier.HEAVY]['latency_ms']
        
        return {
            'total_inferences': len(self.inference_history),
            'tier_distribution': tier_counts,
            'avg_latency_ms': total_latency / len(self.inference_history),
            'estimated_baseline_latency_ms': baseline_latency / len(self.inference_history),
            'latency_reduction_pct': (1 - total_latency / baseline_latency) * 100
        }


if __name__ == "__main__":
    from routing.complexity_predictor import generate_synthetic_training_data
    
    print("Initializing XAI Model Router...")
    router = XAIModelRouter(device="cpu", preload_models=False)
    
    # Train predictor with synthetic data
    print("\nTraining complexity predictor...")
    X, y = generate_synthetic_training_data(500)
    metrics = router.train_predictor(X, y)
    print(f"Training accuracy: {metrics['train_accuracy']:.3f}")
    
    # Test with dummy image
    print("\nTesting routing pipeline...")
    dummy_img = torch.randn(1, 3, 224, 224)
    
    comparison = router.compare_with_baseline(dummy_img)
    
    print(f"\n--- Comparison Results ---")
    print(f"Routed to: {comparison['routed_tier']}")
    print(f"Same prediction as baseline: {comparison['same_prediction']}")
    print(f"Latency savings: {comparison['latency_savings_pct']:.1f}%")
    print(f"FLOPs savings: {comparison['flops_savings_pct']:.1f}%")
    print(f"\nRouting reasoning: {comparison['routed'].routing_decision.reasoning}")