"""
XAI Feature Extractor
Extracts explainability-based complexity indicators from images.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple, Optional
from dataclasses import dataclass
from captum.attr import Saliency, IntegratedGradients
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
import timm


@dataclass
class XAIFeatures:
    """Container for extracted XAI-based complexity features."""
    attention_entropy: float          # How scattered/focused attention is
    saliency_sparsity: float          # Fraction of image that's "important"
    gradient_magnitude: float          # Overall gradient strength
    feature_importance_variance: float # Stability of importance across regions
    spatial_complexity: float          # Edge density / texture complexity
    confidence_margin: float           # Top-1 vs Top-2 prediction gap
    activation_sparsity: float         # How many neurons are firing
    
    def to_vector(self) -> np.ndarray:
        return np.array([
            self.attention_entropy,
            self.saliency_sparsity,
            self.gradient_magnitude,
            self.feature_importance_variance,
            self.spatial_complexity,
            self.confidence_margin,
            self.activation_sparsity
        ])
    
    def to_dict(self) -> Dict[str, float]:
        return {
            'attention_entropy': self.attention_entropy,
            'saliency_sparsity': self.saliency_sparsity,
            'gradient_magnitude': self.gradient_magnitude,
            'feature_importance_variance': self.feature_importance_variance,
            'spatial_complexity': self.spatial_complexity,
            'confidence_margin': self.confidence_margin,
            'activation_sparsity': self.activation_sparsity
        }


class XAIFeatureExtractor:
    """
    Extracts explainability features using a lightweight probe model.
    These features indicate how "complex" an input is for the model.
    """
    
    def __init__(self, probe_model_name: str = "mobilenetv3_small_100", device: str = "cuda"):
        self.device = device if torch.cuda.is_available() else "cpu"
        
        # Use a lightweight model as the "probe" for initial analysis
        self.probe_model = timm.create_model(probe_model_name, pretrained=True)
        self.probe_model.eval().to(self.device)
        
        # Get the target layer for GradCAM (last conv layer)
        self.target_layer = self._get_target_layer()
        
        # Initialize explainers
        self.saliency = Saliency(self.probe_model)
        self.integrated_gradients = IntegratedGradients(self.probe_model)
        self.grad_cam = GradCAM(model=self.probe_model, target_layers=[self.target_layer])
        
        # Hook for capturing activations
        self.activations = {}
        self._register_hooks()
    
    def _get_target_layer(self):
        """Get the last convolutional layer for GradCAM."""
        for name, module in reversed(list(self.probe_model.named_modules())):
            if isinstance(module, nn.Conv2d):
                return module
        raise ValueError("No Conv2d layer found in model")
    
    def _register_hooks(self):
        """Register forward hooks to capture intermediate activations."""
        def hook_fn(name):
            def hook(module, input, output):
                self.activations[name] = output.detach()
            return hook
        
        for name, module in self.probe_model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                module.register_forward_hook(hook_fn(name))
    
    @torch.no_grad()
    def _compute_prediction_confidence(self, img_tensor: torch.Tensor) -> Tuple[int, float]:
        """Get predicted class and confidence margin."""
        logits = self.probe_model(img_tensor)
        probs = F.softmax(logits, dim=1)
        top2 = torch.topk(probs, 2, dim=1)
        
        predicted_class = top2.indices[0, 0].item()
        confidence_margin = (top2.values[0, 0] - top2.values[0, 1]).item()
        
        return predicted_class, confidence_margin
    
    def _compute_attention_entropy(self, attention_map: np.ndarray) -> float:
        """
        Compute entropy of attention distribution.
        High entropy = scattered attention = complex image
        Low entropy = focused attention = simple image
        """
        # Normalize to probability distribution
        attn_flat = attention_map.flatten()
        attn_flat = attn_flat / (attn_flat.sum() + 1e-8)
        attn_flat = attn_flat + 1e-8  # Avoid log(0)
        
        entropy = -np.sum(attn_flat * np.log(attn_flat))
        max_entropy = np.log(len(attn_flat))
        
        return entropy / max_entropy  # Normalize to [0, 1]
    
    def _compute_saliency_sparsity(self, saliency_map: np.ndarray, threshold: float = 0.1) -> float:
        """
        Compute what fraction of the image is "salient".
        Low sparsity (more salient regions) = complex image
        """
        normalized = saliency_map / (saliency_map.max() + 1e-8)
        salient_fraction = (normalized > threshold).mean()
        return salient_fraction
    
    def _compute_activation_sparsity(self) -> float:
        """
        Compute average activation sparsity across layers.
        Sparse activations = simpler input representation
        """
        sparsities = []
        for name, activation in self.activations.items():
            if activation.dim() >= 2:
                # Fraction of near-zero activations
                sparsity = (activation.abs() < 0.01).float().mean().item()
                sparsities.append(sparsity)
        
        return np.mean(sparsities) if sparsities else 0.5
    
    def _compute_spatial_complexity(self, img_tensor: torch.Tensor) -> float:
        """
        Compute spatial complexity using edge detection (Sobel).
        More edges = more complex scene
        """
        # Convert to grayscale
        gray = img_tensor.mean(dim=1, keepdim=True)
        
        # Sobel kernels
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], 
                               dtype=torch.float32, device=self.device).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], 
                               dtype=torch.float32, device=self.device).view(1, 1, 3, 3)
        
        edges_x = F.conv2d(gray, sobel_x, padding=1)
        edges_y = F.conv2d(gray, sobel_y, padding=1)
        edges = torch.sqrt(edges_x**2 + edges_y**2)
        
        # Normalize by image size and max possible gradient
        complexity = edges.mean().item()
        return min(complexity / 0.5, 1.0)  # Normalize to ~[0, 1]
    
    def extract(self, img_tensor: torch.Tensor) -> XAIFeatures:
        """
        Extract all XAI-based complexity features from an image.
        
        Args:
            img_tensor: Preprocessed image tensor [1, 3, H, W]
        
        Returns:
            XAIFeatures dataclass with all complexity indicators
        """
        img_tensor = img_tensor.to(self.device)
        img_tensor.requires_grad = True
        
        # Clear previous activations
        self.activations.clear()
        
        # 1. Get prediction and confidence
        pred_class, confidence_margin = self._compute_prediction_confidence(img_tensor.detach())
        
        # 2. Compute saliency map
        saliency_attr = self.saliency.attribute(img_tensor, target=pred_class)
        saliency_map = saliency_attr.abs().sum(dim=1).squeeze().cpu().numpy()
        
        # 3. Compute GradCAM attention
        targets = [ClassifierOutputTarget(pred_class)]
        cam_map = self.grad_cam(input_tensor=img_tensor.detach(), targets=targets)
        cam_map = cam_map[0]  # Remove batch dimension
        
        # 4. Extract features
        features = XAIFeatures(
            attention_entropy=self._compute_attention_entropy(cam_map),
            saliency_sparsity=self._compute_saliency_sparsity(saliency_map),
            gradient_magnitude=float(saliency_map.mean()),
            feature_importance_variance=float(saliency_map.std()),
            spatial_complexity=self._compute_spatial_complexity(img_tensor.detach()),
            confidence_margin=confidence_margin,
            activation_sparsity=self._compute_activation_sparsity()
        )
        
        return features
    
    def extract_batch(self, img_tensors: torch.Tensor) -> list:
        """Extract features for a batch of images."""
        return [self.extract(img.unsqueeze(0)) for img in img_tensors]


# Quick test
if __name__ == "__main__":
    from torchvision import transforms
    from PIL import Image
    
    # Create dummy image
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    extractor = XAIFeatureExtractor(device="cpu")
    dummy_img = torch.randn(1, 3, 224, 224)
    features = extractor.extract(dummy_img)
    
    print("Extracted XAI Features:")
    for k, v in features.to_dict().items():
        print(f"  {k}: {v:.4f}")