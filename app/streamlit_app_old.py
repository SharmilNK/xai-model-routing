"""
Streamlit Demo: XAI-Guided Model Routing

Run with: streamlit run app/streamlit_app.py
"""

import streamlit as st
import torch
import numpy as np
from PIL import Image
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from torchvision import transforms
import time

import sys
sys.path.append(str(Path(__file__).parent.parent))

from routing.router import XAIModelRouter
from routing.complexity_predictor import ModelTier, generate_synthetic_training_data

# ImageNet class labels (abbreviated - you'd load full list in production)
IMAGENET_CLASSES = {
    0: "tench", 1: "goldfish", 2: "great_white_shark",
    # ... add full list or load from file
}


@st.cache_resource
def load_router():
    """Load and initialize the routing system."""
    router = XAIModelRouter(device="cuda" if torch.cuda.is_available() else "cpu")
    
    # Train with synthetic data (in production, load pre-trained)
    X, y = generate_synthetic_training_data(1000)
    router.train_predictor(X, y)
    
    return router


def get_transform():
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])


def visualize_xai_features(features, ax=None):
    """Create radar chart of XAI features."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    
    labels = [
        'Attention\nEntropy', 'Saliency\nSparsity', 'Gradient\nMagnitude',
        'Feature\nVariance', 'Spatial\nComplexity', 'Confidence\nMargin',
        'Activation\nSparsity'
    ]
    
    values = features.to_vector()
    values = np.concatenate([values, [values[0]]])  # Close the polygon
    
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False)
    angles = np.concatenate([angles, [angles[0]]])
    
    ax.plot(angles, values, 'o-', linewidth=2, color='#4ecdc4')
    ax.fill(angles, values, alpha=0.25, color='#4ecdc4')
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, size=8)
    ax.set_ylim(0, 1)
    
    return ax


def main():
    st.set_page_config(
        page_title="XAI Model Routing Demo",
        page_icon="🧠",
        layout="wide"
    )
    
    st.title("🧠 Explainability-Guided Model Routing")
    st.markdown("""
    This demo shows how **explainable AI features** can predict input complexity 
    and route to the most efficient model.
    
    **How it works:**
    1. Upload an image
    2. XAI features are extracted (attention, saliency, etc.)
    3. A meta-model predicts the minimum model needed
    4. The image is routed to the optimal model
    """)
    
    # Load router
    with st.spinner("Loading models..."):
        router = load_router()
    
    # Sidebar controls
    st.sidebar.header("⚙️ Settings")
    
    compare_baseline = st.sidebar.checkbox("Compare with baseline", value=True)
    show_xai_details = st.sidebar.checkbox("Show XAI feature details", value=True)
    
    # Main content
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("📤 Input")
        
        uploaded_file = st.file_uploader(
            "Upload an image", 
            type=['jpg', 'jpeg', 'png', 'webp']
        )
        
        # Sample images
        st.markdown("**Or try a sample:**")
        sample_col1, sample_col2, sample_col3 = st.columns(3)
        
        use_sample = None
        with sample_col1:
            if st.button("🐕 Simple (Dog)"):
                use_sample = "simple"
        with sample_col2:
            if st.button("🏙️ Medium (Street)"):
                use_sample = "medium"
        with sample_col3:
            if st.button("🔬 Complex (Medical)"):
                use_sample = "complex"
    
    # Process image
    image = None
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert('RGB')
    elif use_sample:
        # Generate synthetic sample images
        np.random.seed(42 if use_sample == "simple" else 123 if use_sample == "medium" else 456)
        if use_sample == "simple":
            # Simple: single object, clear background
            arr = np.ones((224, 224, 3), dtype=np.uint8) * 200
            arr[50:174, 50:174] = [100, 150, 200]
        elif use_sample == "medium":
            # Medium: some structure
            arr = np.random.randint(100, 200, (224, 224, 3), dtype=np.uint8)
        else:
            # Complex: high frequency noise
            arr = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        image = Image.fromarray(arr)
    
    if image is not None:
        with col1:
            st.image(image, caption="Input Image", use_container_width=True)
        
        # Process
        transform = get_transform()
        img_tensor = transform(image).unsqueeze(0)
        
        with st.spinner("Analyzing complexity and routing..."):
            if compare_baseline:
                comparison = router.compare_with_baseline(img_tensor)
                result = comparison['routed']
                baseline = comparison['baseline']
            else:
                result = router.route_and_infer(img_tensor)
                baseline = None
        
        with col2:
            st.header("📊 Results")
            
            # Routing decision
            tier = result.routing_decision.tier
            tier_colors = {
                ModelTier.TINY: "🟢",
                ModelTier.LIGHT: "🟡", 
                ModelTier.MEDIUM: "🟠",
                ModelTier.HEAVY: "🔴"
            }
            
            st.subheader(f"{tier_colors[tier]} Routed to: **{tier.name}** model")
            st.markdown(f"*{result.routing_decision.reasoning}*")
            
            # Metrics
            metrics_col1, metrics_col2, metrics_col3 = st.columns(3)
            
            with metrics_col1:
                st.metric(
                    "Model Used",
                    result.model_used.split('_')[0].title()
                )
            
            with metrics_col2:
                st.metric(
                    "Latency",
                    f"{result.actual_latency_ms:.1f}ms"
                )
            
            with metrics_col3:
                st.metric(
                    "FLOPs",
                    f"{result.flops/1e6:.0f}M"
                )
            
            # Comparison with baseline
            if baseline:
                st.subheader("📈 Efficiency Gains")
                
                gain_col1, gain_col2 = st.columns(2)
                
                latency_save = (1 - result.actual_latency_ms / baseline.actual_latency_ms) * 100
                flops_save = (1 - result.flops / baseline.flops) * 100
                
                with gain_col1:
                    st.metric(
                        "Latency Savings",
                        f"{latency_save:.0f}%",
                        delta=f"{-latency_save:.0f}%" if latency_save > 0 else None,
                        delta_color="inverse"
                    )
                
                with gain_col2:
                    st.metric(
                        "Compute Savings",
                        f"{flops_save:.0f}%",
                        delta=f"{-flops_save:.0f}%" if flops_save > 0 else None,
                        delta_color="inverse"
                    )
                
                # Same prediction check
                if comparison['same_prediction']:
                    st.success("✅ Same prediction as heavy model!")
                else:
                    st.warning("⚠️ Different prediction than heavy model")
            
            # Prediction
            st.subheader("🎯 Prediction")
            st.write(f"**Class ID:** {result.predicted_class}")
            
            # Top-5 predictions
            st.write("**Top-5 Predictions:**")
            for i, (cls_id, prob) in enumerate(result.top5_classes):
                st.progress(prob, text=f"Class {cls_id}: {prob*100:.1f}%")
        
        # XAI Features detail
        if show_xai_details:
            st.header("🔍 XAI Feature Analysis")
            
            feat_col1, feat_col2 = st.columns([1, 1])
            
            with feat_col1:
                st.subheader("Feature Values")
                features = result.xai_features
                
                feature_data = {
                    "Attention Entropy": features.attention_entropy,
                    "Saliency Sparsity": features.saliency_sparsity,
                    "Gradient Magnitude": features.gradient_magnitude,
                    "Feature Variance": features.feature_importance_variance,
                    "Spatial Complexity": features.spatial_complexity,
                    "Confidence Margin": features.confidence_margin,
                    "Activation Sparsity": features.activation_sparsity,
                }
                
                for name, value in feature_data.items():
                    st.progress(min(value, 1.0), text=f"{name}: {value:.3f}")
            
            with feat_col2:
                st.subheader("Complexity Profile")
                fig, ax = plt.subplots(figsize=(5, 5), subplot_kw=dict(polar=True))
                visualize_xai_features(features, ax)
                st.pyplot(fig)
                plt.close()
    
    # Footer
    st.markdown("---")
    st.markdown("""
    **About this project:** This system demonstrates how explainability signals 
    can make AI more efficient by routing simple inputs to lightweight models.
    
    *Built with PyTorch, Captum, and Streamlit*
    """)


if __name__ == "__main__":
    main()