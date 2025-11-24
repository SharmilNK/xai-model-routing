"""
Generate Ground-Truth Complexity Labels

This script:
1. Runs all model tiers on a dataset
2. Finds the minimum tier that achieves correct prediction
3. Extracts XAI features for each image
4. Saves features + labels for training the router

Usage:
    python scripts/generate_complexity_dataset.py --data_dir ./data/images --output ./data/complexity_dataset.npz
"""

import argparse
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
from torchvision import transforms, datasets
import timm
import torch.nn.functional as F

import sys
sys.path.append(str(Path(__file__).parent.parent))

from xai.feature_extractor import XAIFeatureExtractor
from routing.complexity_predictor import ModelTier


# Models for each tier
TIER_MODELS = {
    ModelTier.TINY: "mobilenetv3_small_100",
    ModelTier.LIGHT: "mobilenetv3_large_100",
    ModelTier.MEDIUM: "efficientnet_b0",
    ModelTier.HEAVY: "resnet50",
}


class ComplexityDatasetGenerator:
    """Generates training data for the complexity predictor."""
    
    def __init__(self, device: str = "cuda"):
        self.device = device if torch.cuda.is_available() else "cpu"
        
        # Load all models
        print("Loading models...")
        self.models = {}
        for tier, model_name in TIER_MODELS.items():
            print(f"  Loading {model_name}...")
            model = timm.create_model(model_name, pretrained=True)
            model.eval().to(self.device)
            self.models[tier] = model
        
        # XAI feature extractor (uses lightweight probe)
        print("Initializing XAI extractor...")
        self.xai_extractor = XAIFeatureExtractor(device=self.device)
        
        # Preprocessing
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    
    @torch.no_grad()
    def _get_predictions(self, img_tensor: torch.Tensor) -> dict:
        """Get predictions from all model tiers."""
        img_tensor = img_tensor.to(self.device)
        predictions = {}
        
        for tier, model in self.models.items():
            logits = model(img_tensor)
            probs = F.softmax(logits, dim=1)
            pred_class = logits.argmax(dim=1).item()
            confidence = probs[0, pred_class].item()
            predictions[tier] = {'class': pred_class, 'confidence': confidence}
        
        return predictions
    
    def _find_minimum_tier(self, predictions: dict) -> ModelTier:
        """
        Find the smallest model tier that gives the same prediction
        as the heavy model (with sufficient confidence).
        """
        heavy_pred = predictions[ModelTier.HEAVY]['class']
        min_confidence = 0.5  # Minimum confidence threshold
        
        # Check tiers from smallest to largest
        for tier in [ModelTier.TINY, ModelTier.LIGHT, ModelTier.MEDIUM, ModelTier.HEAVY]:
            if (predictions[tier]['class'] == heavy_pred and 
                predictions[tier]['confidence'] >= min_confidence):
                return tier
        
        # Default to heavy if no match
        return ModelTier.HEAVY
    
    def process_image(self, img_path: Path) -> dict:
        """Process a single image and return features + label."""
        from PIL import Image
        
        img = Image.open(img_path).convert('RGB')
        img_tensor = self.transform(img).unsqueeze(0)
        
        # Get predictions from all tiers
        predictions = self._get_predictions(img_tensor)
        
        # Find minimum sufficient tier
        min_tier = self._find_minimum_tier(predictions)
        
        # Extract XAI features
        xai_features = self.xai_extractor.extract(img_tensor)
        
        return {
            'path': str(img_path),
            'features': xai_features.to_vector(),
            'label': int(min_tier),
            'predictions': predictions
        }
    
    def generate_from_imagefolder(self, 
                                   data_dir: Path, 
                                   max_samples: int = None) -> tuple:
        """
        Generate dataset from an ImageFolder-style directory.
        
        Returns:
            features: np.ndarray [N, 7]
            labels: np.ndarray [N]
            metadata: list of dicts
        """
        # Collect image paths
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
        image_paths = [
            p for p in Path(data_dir).rglob('*') 
            if p.suffix.lower() in image_extensions
        ]
        
        if max_samples:
            image_paths = image_paths[:max_samples]
        
        print(f"Processing {len(image_paths)} images...")
        
        all_features = []
        all_labels = []
        all_metadata = []
        
        for img_path in tqdm(image_paths, desc="Generating complexity labels"):
            try:
                result = self.process_image(img_path)
                all_features.append(result['features'])
                all_labels.append(result['label'])
                all_metadata.append({
                    'path': result['path'],
                    'predictions': result['predictions']
                })
            except Exception as e:
                print(f"Error processing {img_path}: {e}")
                continue
        
        return np.array(all_features), np.array(all_labels), all_metadata
    
    def generate_from_torchvision(self, 
                                   dataset_name: str = "CIFAR10",
                                   max_samples: int = 1000) -> tuple:
        """Generate dataset using a torchvision dataset."""
        
        # Load dataset
        if dataset_name == "CIFAR10":
            dataset = datasets.CIFAR10(
                root='./data', train=False, download=True, transform=self.transform
            )
        elif dataset_name == "ImageNet":
            # Requires ImageNet to be downloaded
            dataset = datasets.ImageNet(
                root='./data/imagenet', split='val', transform=self.transform
            )
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")
        
        # Limit samples
        indices = np.random.choice(len(dataset), min(max_samples, len(dataset)), replace=False)
        
        print(f"Processing {len(indices)} samples from {dataset_name}...")
        
        all_features = []
        all_labels = []
        
        for idx in tqdm(indices, desc="Generating complexity labels"):
            img_tensor, true_label = dataset[idx]
            img_tensor = img_tensor.unsqueeze(0)
            
            try:
                predictions = self._get_predictions(img_tensor)
                min_tier = self._find_minimum_tier(predictions)
                xai_features = self.xai_extractor.extract(img_tensor)
                
                all_features.append(xai_features.to_vector())
                all_labels.append(int(min_tier))
            except Exception as e:
                print(f"Error at index {idx}: {e}")
                continue
        
        return np.array(all_features), np.array(all_labels), None


def main():
    parser = argparse.ArgumentParser(description="Generate complexity dataset")
    parser.add_argument('--data_dir', type=str, help="Path to image directory")
    parser.add_argument('--dataset', type=str, default="CIFAR10", 
                        help="Torchvision dataset name (CIFAR10, ImageNet)")
    parser.add_argument('--output', type=str, default="./data/complexity_dataset.npz")
    parser.add_argument('--max_samples', type=int, default=1000)
    parser.add_argument('--device', type=str, default="cuda")
    
    args = parser.parse_args()
    
    generator = ComplexityDatasetGenerator(device=args.device)
    
    if args.data_dir:
        features, labels, metadata = generator.generate_from_imagefolder(
            Path(args.data_dir), max_samples=args.max_samples
        )
    else:
        features, labels, metadata = generator.generate_from_torchvision(
            args.dataset, max_samples=args.max_samples
        )
    
    # Save dataset
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    np.savez(
        output_path,
        features=features,
        labels=labels
    )
    
    # Print statistics
    print(f"\n--- Dataset Statistics ---")
    print(f"Total samples: {len(labels)}")
    print(f"Feature shape: {features.shape}")
    print(f"\nTier distribution:")
    for tier in ModelTier:
        count = (labels == tier).sum()
        pct = count / len(labels) * 100
        print(f"  {tier.name}: {count} ({pct:.1f}%)")
    
    print(f"\nSaved to: {output_path}")


if __name__ == "__main__":
    main()