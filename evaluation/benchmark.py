"""
Benchmark the XAI Routing System

Compares:
- Baseline (always heavy model)
- Oracle (perfect routing - cheating baseline)
- XAI Routing (our system)

Metrics:
- Accuracy
- Average latency
- Total FLOPs
- Energy estimate
"""

import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
from dataclasses import dataclass, field
from typing import Dict, List
import matplotlib.pyplot as plt
import seaborn as sns
from torchvision import datasets, transforms

import sys
sys.path.append(str(Path(__file__).parent.parent))

from routing.router import XAIModelRouter, InferenceResult
from routing.complexity_predictor import ModelTier, generate_synthetic_training_data


@dataclass
class BenchmarkResults:
    """Container for benchmark metrics."""
    name: str
    accuracy: float
    avg_latency_ms: float
    total_flops: int
    avg_flops: float
    energy_factor: float  # Relative to baseline
    tier_distribution: Dict[str, int] = field(default_factory=dict)
    per_sample_results: List[Dict] = field(default_factory=list)


class SystemBenchmark:
    """Benchmark harness for the XAI routing system."""
    
    # Energy factors per tier (relative)
    ENERGY_FACTORS = {
        ModelTier.TINY: 0.1,
        ModelTier.LIGHT: 0.25,
        ModelTier.MEDIUM: 0.4,
        ModelTier.HEAVY: 1.0,
    }
    
    def __init__(self, router: XAIModelRouter, device: str = "cuda"):
        self.router = router
        self.device = device if torch.cuda.is_available() else "cpu"
        
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    
    def run_baseline(self, 
                     dataloader, 
                     tier: ModelTier = ModelTier.HEAVY) -> BenchmarkResults:
        """Run baseline: always use the specified tier."""
        
        results = []
        correct = 0
        total = 0
        total_flops = 0
        total_latency = 0
        
        for images, labels in tqdm(dataloader, desc=f"Baseline ({tier.name})"):
            for img, label in zip(images, labels):
                img_tensor = img.unsqueeze(0)
                
                result = self.router.route_and_infer(img_tensor, force_tier=tier)
                
                is_correct = (result.predicted_class == label.item())
                correct += is_correct
                total += 1
                total_flops += result.flops
                total_latency += result.actual_latency_ms
                
                results.append({
                    'predicted': result.predicted_class,
                    'actual': label.item(),
                    'correct': is_correct,
                    'tier': tier.name,
                    'latency_ms': result.actual_latency_ms,
                    'flops': result.flops
                })
        
        return BenchmarkResults(
            name=f"Baseline ({tier.name})",
            accuracy=correct / total,
            avg_latency_ms=total_latency / total,
            total_flops=total_flops,
            avg_flops=total_flops / total,
            energy_factor=self.ENERGY_FACTORS[tier],
            tier_distribution={tier.name: total},
            per_sample_results=results
        )
    
    def run_xai_routing(self, dataloader) -> BenchmarkResults:
        """Run our XAI-guided routing system."""
        
        results = []
        correct = 0
        total = 0
        total_flops = 0
        total_latency = 0
        total_energy = 0
        tier_counts = {tier.name: 0 for tier in ModelTier}
        
        for images, labels in tqdm(dataloader, desc="XAI Routing"):
            for img, label in zip(images, labels):
                img_tensor = img.unsqueeze(0)
                
                result = self.router.route_and_infer(img_tensor)
                tier = result.routing_decision.tier
                
                is_correct = (result.predicted_class == label.item())
                correct += is_correct
                total += 1
                total_flops += result.flops
                total_latency += result.actual_latency_ms
                total_energy += self.ENERGY_FACTORS[tier]
                tier_counts[tier.name] += 1
                
                results.append({
                    'predicted': result.predicted_class,
                    'actual': label.item(),
                    'correct': is_correct,
                    'tier': tier.name,
                    'latency_ms': result.actual_latency_ms,
                    'flops': result.flops,
                    'routing_confidence': result.routing_decision.confidence
                })
        
        return BenchmarkResults(
            name="XAI Routing",
            accuracy=correct / total,
            avg_latency_ms=total_latency / total,
            total_flops=total_flops,
            avg_flops=total_flops / total,
            energy_factor=total_energy / total,
            tier_distribution=tier_counts,
            per_sample_results=results
        )
    
    def run_oracle(self, dataloader) -> BenchmarkResults:
        """
        Oracle baseline: for each sample, use the smallest tier that 
        gives the correct answer (requires knowing ground truth).
        This represents the theoretical best routing could achieve.
        """
        results = []
        total = 0
        total_flops = 0
        total_latency = 0
        total_energy = 0
        tier_counts = {tier.name: 0 for tier in ModelTier}
        
        for images, labels in tqdm(dataloader, desc="Oracle"):
            for img, label in zip(images, labels):
                img_tensor = img.unsqueeze(0)
                
                # Try each tier from smallest to largest
                best_result = None
                for tier in [ModelTier.TINY, ModelTier.LIGHT, ModelTier.MEDIUM, ModelTier.HEAVY]:
                    result = self.router.route_and_infer(img_tensor, force_tier=tier)
                    
                    if result.predicted_class == label.item():
                        best_result = result
                        break
                
                # If none correct, use heavy
                if best_result is None:
                    best_result = self.router.route_and_infer(img_tensor, force_tier=ModelTier.HEAVY)
                
                tier = best_result.routing_decision.tier
                total += 1
                total_flops += best_result.flops
                total_latency += best_result.actual_latency_ms
                total_energy += self.ENERGY_FACTORS[tier]
                tier_counts[tier.name] += 1
                
                results.append({
                    'predicted': best_result.predicted_class,
                    'actual': label.item(),
                    'correct': best_result.predicted_class == label.item(),
                    'tier': tier.name,
                    'latency_ms': best_result.actual_latency_ms,
                    'flops': best_result.flops
                })
        
        correct = sum(1 for r in results if r['correct'])
        
        return BenchmarkResults(
            name="Oracle (Upper Bound)",
            accuracy=correct / total,
            avg_latency_ms=total_latency / total,
            total_flops=total_flops,
            avg_flops=total_flops / total,
            energy_factor=total_energy / total,
            tier_distribution=tier_counts,
            per_sample_results=results
        )
    
    def compare_systems(self, dataloader) -> Dict[str, BenchmarkResults]:
        """Run all systems and return comparison."""
        return {
            'baseline_heavy': self.run_baseline(dataloader, ModelTier.HEAVY),
            'baseline_light': self.run_baseline(dataloader, ModelTier.LIGHT),
            'xai_routing': self.run_xai_routing(dataloader),
            'oracle': self.run_oracle(dataloader),
        }
    
    def generate_report(self, results: Dict[str, BenchmarkResults]) -> str:
        """Generate a text report comparing systems."""
        baseline = results['baseline_heavy']
        xai = results['xai_routing']
        oracle = results['oracle']
        
        report = []
        report.append("=" * 60)
        report.append("XAI-GUIDED MODEL ROUTING - BENCHMARK REPORT")
        report.append("=" * 60)
        
        report.append("\n📊 SUMMARY COMPARISON\n")
        report.append(f"{'Metric':<25} {'Baseline':<15} {'XAI Routing':<15} {'Oracle':<15}")
        report.append("-" * 70)
        
        report.append(f"{'Accuracy':<25} {baseline.accuracy*100:.1f}%{'':<10} {xai.accuracy*100:.1f}%{'':<10} {oracle.accuracy*100:.1f}%")
        report.append(f"{'Avg Latency (ms)':<25} {baseline.avg_latency_ms:.1f}{'':<12} {xai.avg_latency_ms:.1f}{'':<12} {oracle.avg_latency_ms:.1f}")
        report.append(f"{'Avg FLOPs (M)':<25} {baseline.avg_flops/1e6:.0f}{'':<12} {xai.avg_flops/1e6:.0f}{'':<12} {oracle.avg_flops/1e6:.0f}")
        report.append(f"{'Energy Factor':<25} {baseline.energy_factor:.2f}{'':<12} {xai.energy_factor:.2f}{'':<12} {oracle.energy_factor:.2f}")
        
        report.append("\n📈 EFFICIENCY GAINS (vs Baseline)\n")
        
        latency_save = (1 - xai.avg_latency_ms / baseline.avg_latency_ms) * 100
        flops_save = (1 - xai.avg_flops / baseline.avg_flops) * 100
        energy_save = (1 - xai.energy_factor / baseline.energy_factor) * 100
        acc_drop = (baseline.accuracy - xai.accuracy) * 100
        
        report.append(f"  Latency Reduction:  {latency_save:+.1f}%")
        report.append(f"  FLOPs Reduction:    {flops_save:+.1f}%")
        report.append(f"  Energy Reduction:   {energy_save:+.1f}%")
        report.append(f"  Accuracy Change:    {-acc_drop:+.1f}%")
        
        report.append("\n🎯 TIER DISTRIBUTION (XAI Routing)\n")
        total = sum(xai.tier_distribution.values())
        for tier, count in sorted(xai.tier_distribution.items()):
            pct = count / total * 100
            bar = "█" * int(pct / 2)
            report.append(f"  {tier:<8} {count:>5} ({pct:5.1f}%) {bar}")
        
        report.append("\n" + "=" * 60)
        
        return "\n".join(report)


def visualize_results(results: Dict[str, BenchmarkResults], save_path: Path = None):
    """Create visualization of benchmark results."""
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 1. Accuracy comparison
    ax = axes[0, 0]
    names = [r.name for r in results.values()]
    accuracies = [r.accuracy * 100 for r in results.values()]
    colors = ['#ff6b6b', '#ffa06b', '#4ecdc4', '#a8e6cf']
    ax.bar(names, accuracies, color=colors)
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Accuracy Comparison')
    ax.set_ylim(0, 100)
    for i, v in enumerate(accuracies):
        ax.text(i, v + 1, f'{v:.1f}%', ha='center')
    
    # 2. Latency comparison
    ax = axes[0, 1]
    latencies = [r.avg_latency_ms for r in results.values()]
    ax.bar(names, latencies, color=colors)
    ax.set_ylabel('Avg Latency (ms)')
    ax.set_title('Latency Comparison')
    for i, v in enumerate(latencies):
        ax.text(i, v + 1, f'{v:.1f}ms', ha='center')
    
    # 3. FLOPs comparison
    ax = axes[1, 0]
    flops = [r.avg_flops / 1e9 for r in results.values()]  # GFLOPs
    ax.bar(names, flops, color=colors)
    ax.set_ylabel('Avg GFLOPs')
    ax.set_title('Compute Comparison')
    for i, v in enumerate(flops):
        ax.text(i, v + 0.05, f'{v:.2f}G', ha='center')
    
    # 4. Tier distribution (XAI Routing)
    ax = axes[1, 1]
    xai_results = results['xai_routing']
    tiers = list(xai_results.tier_distribution.keys())
    counts = list(xai_results.tier_distribution.values())
    tier_colors = ['#a8e6cf', '#88d8b0', '#4ecdc4', '#2c8c99']
    ax.pie(counts, labels=tiers, colors=tier_colors, autopct='%1.1f%%', startangle=90)
    ax.set_title('XAI Routing - Tier Distribution')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to {save_path}")
    
    plt.show()


if __name__ == "__main__":
    # Quick demo with synthetic data
    print("Initializing system...")
    router = XAIModelRouter(device="cpu")
    
    # Train predictor
    X, y = generate_synthetic_training_data(500)
    router.train_predictor(X, y)
    
    # Create dummy dataloader
    from torch.utils.data import DataLoader, TensorDataset
    
    n_samples = 50
    dummy_images = torch.randn(n_samples, 3, 224, 224)
    dummy_labels = torch.randint(0, 1000, (n_samples,))
    dummy_dataset = TensorDataset(dummy_images, dummy_labels)
    dummy_loader = DataLoader(dummy_dataset, batch_size=10)
    
    print("\nRunning benchmark...")
    benchmark = SystemBenchmark(router, device="cpu")
    
    # Just run XAI routing for quick demo
    xai_results = benchmark.run_xai_routing(dummy_loader)
    
    print(f"\nXAI Routing Results:")
    print(f"  Avg Latency: {xai_results.avg_latency_ms:.1f}ms")
    print(f"  Tier Distribution: {xai_results.tier_distribution}")
    print(f"  Energy Factor: {xai_results.energy_factor:.2f}")