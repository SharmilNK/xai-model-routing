"""
XAI Feature Extractor V3 (Simplified & Robust)
No complex hooks - uses direct computation for all features.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple
from dataclasses import dataclass
import timm
from PIL import Image


@dataclass
class XAIFeatures:
    """Container for extracted XAI-based complexity features."""
    attention_entropy: float
    saliency_sparsity: float
    gradient_magnitude: float
    feature_importance_variance: float
    spatial_complexity: float
    confidence_margin: float
    activation_sparsity: float

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


class XAIFeatureExtractorV3:
    """
    Simplified XAI Feature Extractor - no complex hooks.
    Uses image-based features that reliably differentiate complexity.
    """

    def __init__(self, probe_model_name: str = "mobilenetv3_small_100", device: str = "cuda"):
        self.device = device if (device == "cuda" and torch.cuda.is_available()) else "cpu"
        self.probe_model = timm.create_model(probe_model_name, pretrained=True)
        self.probe_model.eval().to(self.device)

    def _compute_edge_density(self, img_tensor: torch.Tensor) -> float:
        """Compute edge density using Sobel operators."""
        gray = img_tensor.mean(dim=1, keepdim=True)
        
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                               dtype=torch.float32, device=self.device).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
                               dtype=torch.float32, device=self.device).view(1, 1, 3, 3)

        edges_x = F.conv2d(gray, sobel_x, padding=1)
        edges_y = F.conv2d(gray, sobel_y, padding=1)
        edge_mag = torch.sqrt(edges_x**2 + edges_y**2)
        
        return float(edge_mag.mean().item())

    def _compute_texture_complexity(self, img_tensor: torch.Tensor) -> float:
        """Compute local variance (texture complexity)."""
        gray = img_tensor.mean(dim=1, keepdim=True)
        
        # Local mean
        local_mean = F.avg_pool2d(gray, kernel_size=7, stride=1, padding=3)
        # Local variance
        local_var = F.avg_pool2d((gray - local_mean)**2, kernel_size=7, stride=1, padding=3)
        
        return float(local_var.mean().item())

    def _compute_frequency_content(self, img_tensor: torch.Tensor) -> float:
        """Compute high-frequency content using Laplacian."""
        gray = img_tensor.mean(dim=1, keepdim=True)
        
        laplacian = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]],
                                  dtype=torch.float32, device=self.device).view(1, 1, 3, 3)
        
        lap_response = F.conv2d(gray, laplacian, padding=1)
        return float(lap_response.abs().mean().item())

    def _compute_color_diversity(self, img_tensor: torch.Tensor) -> float:
        """Compute color channel diversity."""
        # Downsample
        small = F.interpolate(img_tensor, size=(32, 32), mode='bilinear', align_corners=False)
        
        # Channel differences
        r, g, b = small[:, 0], small[:, 1], small[:, 2]
        rg = (r - g).abs().mean().item()
        rb = (r - b).abs().mean().item()
        gb = (g - b).abs().mean().item()
        
        # Spatial variance per channel
        spatial_var = small.var(dim=[2, 3]).mean().item()
        
        return float((rg + rb + gb) / 3 + spatial_var)

    def _compute_entropy(self, img_tensor: torch.Tensor) -> float:
        """Compute image histogram entropy."""
        gray = img_tensor.mean(dim=1).squeeze().cpu().numpy()
        
        # Normalize to 0-255 range
        gray = ((gray - gray.min()) / (gray.max() - gray.min() + 1e-8) * 255).astype(np.uint8)
        
        # Compute histogram
        hist, _ = np.histogram(gray.flatten(), bins=256, range=(0, 256))
        hist = hist / hist.sum()
        hist = hist[hist > 0]
        
        entropy = -np.sum(hist * np.log2(hist))
        # Normalize by max entropy (8 bits)
        return float(entropy / 8.0)

    def _compute_object_count_proxy(self, img_tensor: torch.Tensor) -> float:
        """
        Proxy for number of distinct regions/objects.
        Uses connected components on edge map.
        """
        gray = img_tensor.mean(dim=1, keepdim=True)
        
        # Sobel edges
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                               dtype=torch.float32, device=self.device).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
                               dtype=torch.float32, device=self.device).view(1, 1, 3, 3)
        
        edges_x = F.conv2d(gray, sobel_x, padding=1)
        edges_y = F.conv2d(gray, sobel_y, padding=1)
        edges = torch.sqrt(edges_x**2 + edges_y**2)
        
        # Threshold
        threshold = edges.mean() + edges.std()
        edge_binary = (edges > threshold).float()
        
        # Count edge pixels as proxy for complexity
        edge_ratio = edge_binary.mean().item()
        
        return float(edge_ratio)

    @torch.no_grad()
    def _compute_prediction_confidence(self, img_tensor: torch.Tensor) -> Tuple[int, float, float]:
        """Get prediction class and confidence margin."""
        logits = self.probe_model(img_tensor)
        probs = F.softmax(logits, dim=1)
        top2 = torch.topk(probs, 2, dim=1)
        
        pred_class = int(top2.indices[0, 0].item())
        top1_conf = float(top2.values[0, 0].item())
        top2_conf = float(top2.values[0, 1].item())
        margin = top1_conf - top2_conf
        
        return pred_class, top1_conf, margin

    def _compute_saliency(self, img_tensor: torch.Tensor, pred_class: int) -> Tuple[float, float, float]:
        """Compute gradient-based saliency."""
        img = img_tensor.clone().requires_grad_(True)
        
        logits = self.probe_model(img)
        score = logits[0, pred_class]
        
        self.probe_model.zero_grad()
        score.backward()
        
        grad = img.grad
        if grad is None:
            return 0.3, 0.1, 0.1
        
        saliency = grad.abs().max(dim=1)[0].squeeze().detach().cpu().numpy()
        
        # Normalize
        sal_max = saliency.max()
        if sal_max > 0:
            saliency = saliency / sal_max
        
        sparsity = float((saliency > 0.1).mean())
        magnitude = float(saliency.mean())
        variance = float(saliency.std())
        
        return sparsity, magnitude, variance

    def extract(self, img_tensor: torch.Tensor) -> XAIFeatures:
        """Extract all XAI features."""
        img_tensor = img_tensor.to(self.device)

        # Image-based features (no model needed, very reliable)
        edge_density = self._compute_edge_density(img_tensor)
        texture = self._compute_texture_complexity(img_tensor)
        frequency = self._compute_frequency_content(img_tensor)
        color_div = self._compute_color_diversity(img_tensor)
        entropy = self._compute_entropy(img_tensor)
        object_proxy = self._compute_object_count_proxy(img_tensor)

        # Model-based features
        pred_class, top1_conf, conf_margin = self._compute_prediction_confidence(img_tensor)
        sal_sparsity, sal_mag, sal_var = self._compute_saliency(img_tensor.clone(), pred_class)

        # Normalize and combine features
        # Edge density: typical range 0-0.5
        edge_norm = min(edge_density / 0.3, 1.0)
        
        # Texture: typical range 0-0.05
        texture_norm = min(texture / 0.03, 1.0)
        
        # Frequency: typical range 0-0.3
        freq_norm = min(frequency / 0.2, 1.0)
        
        # Color diversity: typical range 0-0.5
        color_norm = min(color_div / 0.4, 1.0)
        
        # Combined spatial complexity
        spatial_complexity = (edge_norm * 0.3 + texture_norm * 0.25 + 
                              freq_norm * 0.25 + color_norm * 0.2)

        # Attention entropy proxy (using image entropy + object proxy)
        attention_entropy = (entropy * 0.5 + object_proxy * 5 * 0.5)
        attention_entropy = min(attention_entropy, 1.0)

        return XAIFeatures(
            attention_entropy=attention_entropy,
            saliency_sparsity=sal_sparsity,
            gradient_magnitude=sal_mag,
            feature_importance_variance=sal_var,
            spatial_complexity=spatial_complexity,
            confidence_margin=conf_margin,
            activation_sparsity=1.0 - entropy  # Low entropy = high sparsity
        )


if __name__ == "__main__":
    from torchvision import transforms

    print("Testing XAI Feature Extractor V3 (Simplified)...")
    print("=" * 70)

    extractor = XAIFeatureExtractorV3(device="cpu")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    def create_simple():
        """Single solid square on plain background."""
        arr = np.ones((224, 224, 3), dtype=np.uint8) * 240
        arr[70:154, 70:154] = [70, 130, 180]
        return Image.fromarray(arr)

    def create_medium():
        """Multiple shapes with different colors."""
        arr = np.ones((224, 224, 3), dtype=np.uint8) * 200
        arr[20:60, 20:80] = [255, 100, 100]
        arr[80:140, 50:120] = [100, 255, 100]
        arr[30:90, 130:200] = [100, 100, 255]
        arr[150:200, 80:160] = [255, 200, 100]
        arr[120:180, 10:50] = [200, 100, 200]
        arr[160:210, 170:220] = [100, 200, 200]
        return Image.fromarray(arr)

    def create_complex():
        """Random noise - maximum complexity."""
        return Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))

    test_cases = [
        ("SIMPLE (single square)", create_simple),
        ("MEDIUM (6 shapes)", create_medium),
        ("COMPLEX (random noise)", create_complex),
    ]

    print(f"\n{'Image Type':<28} {'AttnEnt':<10} {'SalSpar':<10} {'Spatial':<10} {'ConfMar':<10}")
    print("-" * 68)

    for name, fn in test_cases:
        img = fn()
        features = extractor.extract(transform(img).unsqueeze(0))
        f = features.to_dict()
        print(f"{name:<28} {f['attention_entropy']:<10.4f} {f['saliency_sparsity']:<10.4f} "
              f"{f['spatial_complexity']:<10.4f} {f['confidence_margin']:<10.4f}")
    
    print("\n" + "=" * 70)
    print("Detailed Feature Comparison:")
    print("=" * 70)
    
    for name, fn in test_cases:
        img = fn()
        features = extractor.extract(transform(img).unsqueeze(0))
        print(f"\n{name}:")
        for k, v in features.to_dict().items():
            bar_len = int(v * 30)
            bar = "█" * bar_len + "░" * (30 - bar_len)
            print(f"  {k:<28}: {v:.4f} |{bar}|")