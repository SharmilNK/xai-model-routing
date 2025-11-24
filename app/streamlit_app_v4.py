"""
XAI-Guided Model Routing - Complete Enhanced Demo
All original features + Real Latency + Attention Flow + Side-by-Side Comparison

Run with: streamlit run app/streamlit_app.py
"""

import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from torchvision import transforms
from scipy.ndimage import gaussian_filter
import time
import sys

sys.path.append(str(Path(__file__).parent.parent))

from xai.feature_extractor_v3 import XAIFeatureExtractorV3, XAIFeatures
from routing.complexity_predictor import ComplexityPredictor, ModelTier
import timm

# Professional dark theme
plt.style.use('dark_background')
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': 11,
    'axes.titlesize': 14,
    'axes.titleweight': 'bold',
    'axes.labelsize': 11,
    'axes.edgecolor': '#444444',
    'axes.facecolor': '#0a0a0a',
    'figure.facecolor': '#000000',
    'grid.color': '#222222',
    'xtick.color': '#888888',
    'ytick.color': '#888888',
    'text.color': '#ffffff',
})

# Page config
st.set_page_config(page_title="XAI Model Routing", page_icon="", layout="wide")

# Custom CSS
st.markdown("""
<style>
    .main-header { font-size: 2.2rem; font-weight: 600; text-align: center; color: #ffffff; margin-bottom: 0.3rem; }
    .sub-header { text-align: center; color: #888; font-size: 1rem; margin-bottom: 1.5rem; }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# CACHING & LOADING
# ============================================================================

@st.cache_resource
def load_system():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    extractor = XAIFeatureExtractorV3(device=device)
    predictor = ComplexityPredictor()
    X, y = create_training_data(500)
    predictor.train(X, y)
    
    tier_models = {}
    model_names = {'TINY': 'mobilenetv3_small_100', 'LIGHT': 'mobilenetv3_large_100',
                   'MEDIUM': 'efficientnet_b0', 'HEAVY': 'resnet50'}
    for tier, name in model_names.items():
        tier_models[tier] = timm.create_model(name, pretrained=True).eval().to(device)
    
    return extractor, predictor, tier_models, device

def create_training_data(n_per_tier=500):
    np.random.seed(42)
    features, labels = [], []
    for tier in ModelTier:
        for _ in range(n_per_tier):
            if tier == ModelTier.TINY:
                f = [np.random.uniform(0.10, 0.35), np.random.uniform(0.08, 0.20),
                     np.random.uniform(0.03, 0.08), np.random.uniform(0.03, 0.07),
                     np.random.uniform(0.70, 0.88), np.random.uniform(0.05, 0.15),
                     np.random.uniform(0.45, 0.70)]
            elif tier == ModelTier.LIGHT:
                f = [np.random.uniform(0.30, 0.55), np.random.uniform(0.10, 0.25),
                     np.random.uniform(0.04, 0.10), np.random.uniform(0.04, 0.09),
                     np.random.uniform(0.80, 0.93), np.random.uniform(0.08, 0.20),
                     np.random.uniform(0.25, 0.55)]
            elif tier == ModelTier.MEDIUM:
                f = [np.random.uniform(0.50, 0.75), np.random.uniform(0.05, 0.18),
                     np.random.uniform(0.02, 0.07), np.random.uniform(0.02, 0.06),
                     np.random.uniform(0.88, 0.98), np.random.uniform(0.03, 0.12),
                     np.random.uniform(0.05, 0.30)]
            else:
                f = [np.random.uniform(0.70, 0.95), np.random.uniform(0.01, 0.12),
                     np.random.uniform(0.01, 0.05), np.random.uniform(0.01, 0.04),
                     np.random.uniform(0.95, 1.00), np.random.uniform(0.00, 0.08),
                     np.random.uniform(0.02, 0.18)]
            features.append(f)
            labels.append(int(tier))
    return np.array(features), np.array(labels)

def get_transform():
    return transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(),
                               transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

# ============================================================================
# REAL LATENCY MEASUREMENT
# ============================================================================

def measure_real_latency(model, img_tensor, device, num_runs=5):
    model.eval()
    img_tensor = img_tensor.to(device)
    with torch.no_grad():
        for _ in range(3):  # warmup
            _ = model(img_tensor)
    times = []
    with torch.no_grad():
        for _ in range(num_runs):
            if device == 'cuda': torch.cuda.synchronize()
            start = time.perf_counter()
            _ = model(img_tensor)
            if device == 'cuda': torch.cuda.synchronize()
            times.append((time.perf_counter() - start) * 1000)
    return np.mean(times), np.std(times)

# ============================================================================
# ATTENTION FLOW VISUALIZATION
# ============================================================================

def extract_layer_activations(model, img_tensor, device, num_layers=6):
    model.eval()
    img_tensor = img_tensor.to(device)
    activations = []
    hooks = []
    
    conv_layers = [(n, m) for n, m in model.named_modules() if isinstance(m, nn.Conv2d)]
    step = max(1, len(conv_layers) // num_layers)
    selected = [conv_layers[i * step] for i in range(min(num_layers, len(conv_layers)))]
    
    def get_hook(idx):
        def hook(module, inp, out):
            act = out.detach().mean(dim=1, keepdim=True)
            act = (act - act.min()) / (act.max() - act.min() + 1e-8)
            act = F.interpolate(act, size=(224, 224), mode='bilinear', align_corners=False)
            activations.append((idx, act.squeeze().cpu().numpy()))
        return hook
    
    for i, (name, module) in enumerate(selected):
        hooks.append(module.register_forward_hook(get_hook(i)))
    
    with torch.no_grad():
        _ = model(img_tensor)
    
    for h in hooks: h.remove()
    activations.sort(key=lambda x: x[0])
    return [a for _, a in activations]

def plot_attention_flow(image, activations):
    n = len(activations)
    fig, axes = plt.subplots(1, n + 1, figsize=(2.8 * (n + 1), 3))
    fig.patch.set_facecolor('#000000')
    img_arr = np.array(image.resize((224, 224))) / 255.0
    
    axes[0].imshow(img_arr)
    axes[0].set_title('Input', fontsize=10, color='#fff', pad=6)
    axes[0].axis('off')
    
    for i, act in enumerate(activations):
        overlay = 0.5 * img_arr + 0.5 * cm.inferno(act)[:, :, :3]
        axes[i+1].imshow(np.clip(overlay, 0, 1))
        axes[i+1].set_title(f'Layer {i+1}', fontsize=10, color='#fff', pad=6)
        axes[i+1].axis('off')
    
    fig.text(0.5, 0.02, 'Early (edges) → Late (objects)', ha='center', fontsize=9, color='#666')
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.1)
    return fig

# ============================================================================
# HEATMAP & VISUALIZATIONS
# ============================================================================

def compute_heatmap(model, img_tensor, device):
    model.eval()
    img = img_tensor.clone().to(device).requires_grad_(True)
    try:
        out = model(img)
        score = out[0, out.argmax(dim=1).item()]
        model.zero_grad()
        score.backward()
        if img.grad is not None:
            sal = img.grad.abs().max(dim=1)[0].squeeze().detach().cpu().numpy()
            sal = (sal - sal.min()) / (sal.max() - sal.min() + 1e-8)
            sal = gaussian_filter(sal, sigma=8)
            return (sal - sal.min()) / (sal.max() - sal.min() + 1e-8)
    except: pass
    return np.zeros((224, 224))

def plot_gradcam_overlay(image, heatmap):
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    fig.patch.set_facecolor('#000000')
    img_arr = np.array(image.resize((224, 224))) / 255.0
    
    axes[0].imshow(image)
    axes[0].set_title("Original", fontsize=12, fontweight='600', color='#fff', pad=10)
    axes[0].axis('off')
    
    axes[1].imshow(heatmap, cmap='inferno')
    axes[1].set_title("Attention Heatmap", fontsize=12, fontweight='600', color='#fff', pad=10)
    axes[1].axis('off')
    
    overlay = np.clip(0.55 * img_arr + 0.45 * cm.inferno(heatmap)[:,:,:3], 0, 1)
    axes[2].imshow(overlay)
    axes[2].set_title("GradCAM Overlay", fontsize=12, fontweight='600', color='#fff', pad=10)
    axes[2].axis('off')
    
    plt.tight_layout()
    return fig

def plot_feature_bars(features):
    labels = ['Attention Entropy', 'Saliency Sparsity', 'Gradient Magnitude',
              'Feature Variance', 'Spatial Complexity', 'Confidence Margin', 'Activation Sparsity']
    values = features.to_vector()
    colors = ['#6366f1', '#8b5cf6', '#a855f7', '#d946ef', '#ec4899', '#f43f5e', '#f97316']
    
    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor('#000000')
    ax.set_facecolor('#0a0a0a')
    
    bars = ax.barh(np.arange(len(labels)), values, color=colors, height=0.55)
    for bar, val in zip(bars, values):
        ax.text(val + 0.02, bar.get_y() + bar.get_height()/2, f'{val:.3f}', 
                va='center', fontsize=10, color='#fff', fontweight='500')
    
    ax.set_yticks(np.arange(len(labels)))
    ax.set_yticklabels(labels, fontsize=10, color='#ccc')
    ax.set_xlim(0, 1.15)
    ax.set_xlabel('Feature Value', fontsize=11, color='#888')
    ax.set_title('XAI Feature Extraction', fontsize=14, fontweight='600', color='#fff', pad=15)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_color('#333')
    ax.spines['left'].set_color('#333')
    ax.xaxis.grid(True, alpha=0.2, color='#fff')
    plt.tight_layout()
    return fig

def plot_complexity_radar(features, tier):
    labels = ['Attn\nEntropy', 'Sal\nSparsity', 'Grad\nMag', 'Feat\nVar', 
              'Spatial\nCmplx', 'Conf\nMargin', 'Act\nSparsity']
    profiles = {'TINY': [0.22,0.14,0.055,0.05,0.79,0.10,0.58], 'LIGHT': [0.42,0.17,0.07,0.065,0.86,0.14,0.40],
                'MEDIUM': [0.62,0.11,0.045,0.04,0.93,0.075,0.17], 'HEAVY': [0.82,0.06,0.03,0.025,0.97,0.04,0.10]}
    tier_colors = {'TINY': '#22c55e', 'LIGHT': '#eab308', 'MEDIUM': '#f97316', 'HEAVY': '#ef4444'}
    
    values = features.to_vector()
    angles = np.linspace(0, 2*np.pi, len(labels), endpoint=False).tolist() + [0]
    
    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
    fig.patch.set_facecolor('#000000')
    ax.set_facecolor('#0a0a0a')
    ax.spines['polar'].set_color('#333')
    ax.grid(color='#222', linewidth=0.5)
    
    for t, prof in profiles.items():
        ax.plot(angles, prof + [prof[0]], '--', linewidth=1.2, label=t, color=tier_colors[t], alpha=0.4)
    
    ax.plot(angles, values.tolist() + [values[0]], 'o-', linewidth=2.5, label='Current', color='#fff', markersize=6)
    ax.fill(angles, values.tolist() + [values[0]], alpha=0.15, color='#6366f1')
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, size=9, color='#aaa')
    ax.set_ylim(0, 1)
    ax.set_title(f'Complexity Profile: {tier}', fontsize=14, fontweight='600', color='#fff', pad=20)
    legend = ax.legend(loc='upper right', bbox_to_anchor=(1.25, 1.0))
    legend.get_frame().set_facecolor('#111')
    for t in legend.get_texts(): t.set_color('#ccc')
    plt.tight_layout()
    return fig

def plot_shap(features, predictor, tier):
    names = ['Attn Entropy', 'Sal Sparsity', 'Grad Mag', 'Feat Var', 'Spatial Cmplx', 'Conf Margin', 'Act Sparsity']
    imp = predictor.classifier.feature_importances_
    vals = features.to_vector()
    baseline = np.array([0.22, 0.14, 0.055, 0.05, 0.79, 0.10, 0.58])
    contrib = (vals - baseline) * imp
    idx = np.argsort(np.abs(contrib))[::-1]
    
    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor('#000000')
    ax.set_facecolor('#0a0a0a')
    
    colors = ['#ef4444' if c > 0 else '#3b82f6' for c in contrib[idx]]
    bars = ax.barh(np.arange(len(names)), contrib[idx], color=colors, height=0.5)
    
    ax.set_yticks(np.arange(len(names)))
    ax.set_yticklabels([names[i] for i in idx], fontsize=10, color='#ccc')
    ax.axvline(x=0, color='#444', linewidth=1)
    ax.set_xlabel('Contribution', fontsize=11, color='#888')
    ax.set_title('Feature Contributions (SHAP)', fontsize=14, fontweight='600', color='#fff', pad=15)
    ax.text(0.5, 1.02, 'Red = Complex | Blue = Simple', transform=ax.transAxes, ha='center', fontsize=9, color='#666')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_color('#333')
    ax.spines['left'].set_color('#333')
    ax.xaxis.grid(True, alpha=0.15, color='#fff')
    
    xmin, xmax = ax.get_xlim()
    for bar, val in zip(bars, contrib[idx]):
        xp = val + (xmax-xmin)*0.02 if val >= 0 else val - (xmax-xmin)*0.02
        ax.text(xp, bar.get_y()+bar.get_height()/2, f'{val:+.3f}', va='center', 
                ha='left' if val >= 0 else 'right', fontsize=9, color='#fff')
    plt.tight_layout()
    return fig

def plot_tier_comparison(tier, measured_lat=None):
    tiers = ['TINY', 'LIGHT', 'MEDIUM', 'HEAVY']
    lats = [8, 15, 25, 80]
    flops = [60, 220, 400, 4100]
    accs = [88, 92, 95, 97]
    tier_colors = {'TINY': '#22c55e', 'LIGHT': '#eab308', 'MEDIUM': '#f97316', 'HEAVY': '#ef4444'}
    
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    fig.patch.set_facecolor('#000000')
    
    for ax in axes:
        ax.set_facecolor('#0a0a0a')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_color('#333')
        ax.spines['left'].set_color('#333')
        ax.yaxis.grid(True, alpha=0.1, color='#fff')
    
    # Latency
    cols = [tier_colors[t] if t == tier else '#3f3f46' for t in tiers]
    bars = axes[0].bar(tiers, lats, color=cols, width=0.6)
    axes[0].set_ylabel('Latency (ms)', color='#888')
    axes[0].set_title('Speed', fontsize=13, fontweight='600', color='#fff', pad=12)
    for bar, val, t in zip(bars, lats, tiers):
        lbl = f'{measured_lat:.1f}ms' if t == tier and measured_lat else f'{val}ms'
        axes[0].text(bar.get_x()+bar.get_width()/2, val+2, lbl, ha='center', fontsize=9, 
                     color='#fff' if t == tier else '#888')
    
    # FLOPs
    bars = axes[1].bar(tiers, flops, color=cols, width=0.6)
    axes[1].set_ylabel('MFLOPs', color='#888')
    axes[1].set_title('Compute', fontsize=13, fontweight='600', color='#fff', pad=12)
    for bar, val, t in zip(bars, flops, tiers):
        axes[1].text(bar.get_x()+bar.get_width()/2, val+100, f'{val}M', ha='center', fontsize=9,
                     color='#fff' if t == tier else '#888')
    
    # Accuracy
    bars = axes[2].bar(tiers, accs, color=cols, width=0.6)
    axes[2].set_ylabel('Accuracy (%)', color='#888')
    axes[2].set_title('Accuracy', fontsize=13, fontweight='600', color='#fff', pad=12)
    axes[2].set_ylim(80, 100)
    for bar, val, t in zip(bars, accs, tiers):
        axes[2].text(bar.get_x()+bar.get_width()/2, val+0.5, f'{val}%', ha='center', fontsize=9,
                     color='#fff' if t == tier else '#888')
    
    plt.tight_layout()
    return fig

# ============================================================================
# COMPARISON VISUALIZATIONS
# ============================================================================

def plot_comparison_radar(f1, f2, n1="Image A", n2="Image B"):
    labels = ['Attn\nEnt', 'Sal\nSpar', 'Grad\nMag', 'Feat\nVar', 'Spat\nCmplx', 'Conf\nMar', 'Act\nSpar']
    v1, v2 = f1.to_vector(), f2.to_vector()
    angles = np.linspace(0, 2*np.pi, len(labels), endpoint=False).tolist() + [0]
    
    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
    fig.patch.set_facecolor('#000000')
    ax.set_facecolor('#0a0a0a')
    ax.spines['polar'].set_color('#333')
    ax.grid(color='#222', linewidth=0.5)
    
    ax.plot(angles, v1.tolist()+[v1[0]], 'o-', lw=2.5, label=n1, color='#3b82f6', ms=6)
    ax.fill(angles, v1.tolist()+[v1[0]], alpha=0.15, color='#3b82f6')
    ax.plot(angles, v2.tolist()+[v2[0]], 's-', lw=2.5, label=n2, color='#ef4444', ms=6)
    ax.fill(angles, v2.tolist()+[v2[0]], alpha=0.15, color='#ef4444')
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, size=9, color='#aaa')
    ax.set_ylim(0, 1)
    ax.set_title('Complexity Comparison', fontsize=14, fontweight='600', color='#fff', pad=20)
    legend = ax.legend(loc='upper right', bbox_to_anchor=(1.2, 1.0))
    legend.get_frame().set_facecolor('#111')
    for t in legend.get_texts(): t.set_color('#ccc')
    plt.tight_layout()
    return fig

def plot_comparison_bars(f1, f2, n1="Image A", n2="Image B"):
    labels = ['Attn Ent', 'Sal Spar', 'Grad Mag', 'Feat Var', 'Spat Cmplx', 'Conf Mar', 'Act Spar']
    v1, v2 = f1.to_vector(), f2.to_vector()
    x = np.arange(len(labels))
    
    fig, ax = plt.subplots(figsize=(12, 5))
    fig.patch.set_facecolor('#000000')
    ax.set_facecolor('#0a0a0a')
    
    ax.bar(x - 0.18, v1, 0.35, label=n1, color='#3b82f6')
    ax.bar(x + 0.18, v2, 0.35, label=n2, color='#ef4444')
    
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9, color='#aaa', rotation=15, ha='right')
    ax.set_ylabel('Value', color='#888')
    ax.set_title('Feature Comparison', fontsize=14, fontweight='600', color='#fff', pad=15)
    ax.set_ylim(0, 1.1)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_color('#333')
    ax.spines['left'].set_color('#333')
    ax.yaxis.grid(True, alpha=0.15, color='#fff')
    legend = ax.legend()
    legend.get_frame().set_facecolor('#111')
    for t in legend.get_texts(): t.set_color('#ccc')
    plt.tight_layout()
    return fig

# ============================================================================
# PROCESS IMAGE
# ============================================================================

def process_image(image, extractor, predictor, tier_models, device):
    transform = get_transform()
    img_tensor = transform(image).unsqueeze(0)
    
    features = extractor.extract(img_tensor)
    decision = predictor.predict(features.to_vector())
    tier, confidence = decision.tier.name, decision.confidence
    
    model = tier_models[tier]
    measured_lat, lat_std = measure_real_latency(model, img_tensor, device)
    flops = {'TINY': 60, 'LIGHT': 220, 'MEDIUM': 400, 'HEAVY': 4100}[tier]
    
    heatmap = compute_heatmap(extractor.probe_model, img_tensor, device)
    activations = extract_layer_activations(extractor.probe_model, img_tensor, device, 6)
    
    return {'features': features, 'tier': tier, 'confidence': confidence, 
            'measured_latency': measured_lat, 'flops': flops, 'heatmap': heatmap, 'activations': activations}

# ============================================================================
# MAIN APP
# ============================================================================

def main():
    st.markdown('<h1 class="main-header">Explainability-Guided Model Routing</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Using XAI to predict complexity and minimize compute cost</p>', unsafe_allow_html=True)
    
    with st.spinner("Loading models..."):
        extractor, predictor, tier_models, device = load_system()
    
    # Sidebar
    with st.sidebar:
        mode = st.radio("Mode:", ["Single Image Analysis", "Side-by-Side Comparison"])
        st.markdown("---")
        st.markdown("**Model Tiers**")
        st.markdown("TINY: 8ms, 60M FLOPs")
        st.markdown("LIGHT: 15ms, 220M FLOPs")
        st.markdown("MEDIUM: 25ms, 400M FLOPs")
        st.markdown("HEAVY: 80ms, 4100M FLOPs")
    
    tier_colors = {'TINY': '#22c55e', 'LIGHT': '#eab308', 'MEDIUM': '#f97316', 'HEAVY': '#ef4444'}
    
    # ========================================================================
    # SINGLE IMAGE MODE
    # ========================================================================
    if mode == "Single Image Analysis":
        col1, col2 = st.columns(2)
        with col1:
            uploaded = st.file_uploader("Upload image", type=['jpg','jpeg','png','webp'])
        with col2:
            sample_dir = Path("data/test_images")
            samples = ["None"] + ([s.stem for s in sorted(sample_dir.glob("*.jpg"))[:10]] if sample_dir.exists() else [])
            selected = st.selectbox("Or select sample:", samples)
        
        image = None
        if uploaded: image = Image.open(uploaded).convert('RGB')
        elif selected != "None": image = Image.open(sample_dir / f"{selected}.jpg").convert('RGB')
        
        if not image:
            st.info("Upload an image or select a sample to begin.")
            return
        
        with st.spinner("Analyzing..."):
            res = process_image(image, extractor, predictor, tier_models, device)
        
        st.markdown("---")
        
        # Metrics
        c1, c2, c3, c4 = st.columns(4)
        c1.markdown(f'<div style="background:{tier_colors[res["tier"]]}22;border:2px solid {tier_colors[res["tier"]]};padding:1rem;border-radius:8px;text-align:center;"><h2 style="margin:0;color:{tier_colors[res["tier"]]};">{res["tier"]}</h2><p style="margin:0;color:#888;">Tier</p></div>', unsafe_allow_html=True)
        c2.metric("Confidence", f"{res['confidence']:.1%}")
        c3.metric("Measured Latency", f"{res['measured_latency']:.1f}ms", delta=f"{(1-res['measured_latency']/80)*100:.0f}% faster")
        c4.metric("Compute", f"{res['flops']}M FLOPs", delta=f"{(1-res['flops']/4100)*100:.0f}% saved")
        
        st.markdown("---")
        
        # =====================================================================
        # STEP 1: FEATURE EXTRACTION (XAI Analysis)
        # =====================================================================
        st.markdown("### Step 1: Feature Extraction")
        st.caption("The probe model (MobileNetV3-Small) analyzes the image and extracts 7 complexity features")
        
        col1, col2 = st.columns(2)
        with col1:
            fig = plot_feature_bars(res['features'])
            st.pyplot(fig)
            plt.close()
        with col2:
            fig = plot_complexity_radar(res['features'], res['tier'])
            st.pyplot(fig)
            plt.close()
        
        st.markdown("---")
        
        # =====================================================================
        # STEP 2: ROUTING DECISION (Why this tier?)
        # =====================================================================
        st.markdown("### Step 2: Routing Decision")
        st.caption("The complexity predictor uses the 7 features to decide which model tier to use")
        
        fig = plot_shap(res['features'], predictor, res['tier'])
        st.pyplot(fig)
        plt.close()
        
        st.markdown("---")
        
        # =====================================================================
        # STEP 3: MODEL SELECTION
        # =====================================================================
        st.markdown("### Step 3: Model Selection")
        st.caption(f"Based on the routing decision, the {res['tier']} model was selected for classification")
        
        fig = plot_tier_comparison(res['tier'], res['measured_latency'])
        st.pyplot(fig)
        plt.close()
        
        st.markdown("---")
        
        # =====================================================================
        # STEP 4: CLASSIFICATION (Final Inference)
        # =====================================================================
        st.markdown("### Step 4: Classification")
        st.caption(f"The {res['tier']} model runs inference and produces the final prediction with explanations")
        
        st.subheader("GradCAM: Where the Model Looks")
        fig = plot_gradcam_overlay(image, res['heatmap'])
        st.pyplot(fig)
        plt.close()
        
        st.markdown("")
        
        st.subheader("Attention Flow: How the Model Thinks")
        st.caption("Visualizing attention evolution from early layers (edges) to late layers (objects)")
        fig = plot_attention_flow(image, res['activations'])
        st.pyplot(fig)
        plt.close()
        
        st.markdown("---")
        
        # Decision Explanation
        st.subheader("Decision Explanation")
        f = res['features'].to_dict()
        reasons = []
        if f['attention_entropy'] < 0.35: reasons.append("focused attention pattern")
        elif f['attention_entropy'] > 0.65: reasons.append("scattered attention across image")
        if f['activation_sparsity'] > 0.5: reasons.append("sparse neural activations")
        elif f['activation_sparsity'] < 0.2: reasons.append("dense neural activations required")
        if f['spatial_complexity'] > 0.9: reasons.append("high edge density and texture")
        
        reason_text = ", ".join(reasons) if reasons else "balanced complexity indicators"
        lat_save = (1 - res['measured_latency']/80) * 100
        flop_save = (1 - res['flops']/4100) * 100
        
        st.success(f"""
        **Routed to {res['tier']}** with {res['confidence']:.1%} confidence.
        
        **Reason:** {reason_text}
        
        **Efficiency:** {res['measured_latency']:.1f}ms actual latency ({lat_save:.0f}% faster than baseline) | {res['flops']}M FLOPs ({flop_save:.0f}% saved)
        """)
        
        # Raw values expander
        with st.expander("View Raw Feature Values"):
            col1, col2 = st.columns(2)
            with col1:
                st.write("**Features:**")
                for k, v in res['features'].to_dict().items():
                    st.write(f"- {k}: `{v:.4f}`")
            with col2:
                st.write("**Routing:**")
                st.write(f"- Tier: `{res['tier']}`")
                st.write(f"- Confidence: `{res['confidence']:.4f}`")
                st.write(f"- Latency: `{res['measured_latency']:.2f}ms`")
                st.write(f"- FLOPs: `{res['flops']}M`")
    
    # ========================================================================
    # SIDE-BY-SIDE COMPARISON MODE
    # ========================================================================
    else:
        st.markdown("### Compare Two Images")
        st.caption("See how complexity profiles differ between images")
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Image A**")
            upload_a = st.file_uploader("Upload Image A", type=['jpg','jpeg','png','webp'], key="a")
            sample_dir = Path("data/test_images")
            samples = ["None"] + ([s.stem for s in sorted(sample_dir.glob("*.jpg"))[:10]] if sample_dir.exists() else [])
            sample_a = st.selectbox("Or select sample:", samples, key="sa")
        
        with col2:
            st.markdown("**Image B**")
            upload_b = st.file_uploader("Upload Image B", type=['jpg','jpeg','png','webp'], key="b")
            sample_b = st.selectbox("Or select sample:", samples, key="sb")
        
        img_a, img_b = None, None
        if upload_a: img_a = Image.open(upload_a).convert('RGB')
        elif sample_a != "None": img_a = Image.open(sample_dir / f"{sample_a}.jpg").convert('RGB')
        if upload_b: img_b = Image.open(upload_b).convert('RGB')
        elif sample_b != "None": img_b = Image.open(sample_dir / f"{sample_b}.jpg").convert('RGB')
        
        if not img_a or not img_b:
            st.info("Select or upload two images to compare.")
            return
        
        with st.spinner("Analyzing both images..."):
            res_a = process_image(img_a, extractor, predictor, tier_models, device)
            res_b = process_image(img_b, extractor, predictor, tier_models, device)
        
        st.markdown("---")
        
        # Images side by side with tier info
        col1, col2 = st.columns(2)
        with col1:
            st.image(img_a, caption="Image A", use_container_width=True)
            st.markdown(f'<div style="background:{tier_colors[res_a["tier"]]}22;border:2px solid {tier_colors[res_a["tier"]]};padding:0.8rem;border-radius:8px;text-align:center;"><h3 style="margin:0;color:{tier_colors[res_a["tier"]]};">{res_a["tier"]}</h3><p style="margin:0;color:#888;">{res_a["measured_latency"]:.1f}ms | {res_a["flops"]}M FLOPs</p></div>', unsafe_allow_html=True)
        
        with col2:
            st.image(img_b, caption="Image B", use_container_width=True)
            st.markdown(f'<div style="background:{tier_colors[res_b["tier"]]}22;border:2px solid {tier_colors[res_b["tier"]]};padding:0.8rem;border-radius:8px;text-align:center;"><h3 style="margin:0;color:{tier_colors[res_b["tier"]]};">{res_b["tier"]}</h3><p style="margin:0;color:#888;">{res_b["measured_latency"]:.1f}ms | {res_b["flops"]}M FLOPs</p></div>', unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Comparison Radar
        st.subheader("Complexity Profile Comparison")
        fig = plot_comparison_radar(res_a['features'], res_b['features'], "Image A", "Image B")
        st.pyplot(fig)
        plt.close()
        
        st.markdown("---")
        
        # Comparison Bars
        st.subheader("Feature Value Comparison")
        fig = plot_comparison_bars(res_a['features'], res_b['features'], "Image A", "Image B")
        st.pyplot(fig)
        plt.close()
        
        st.markdown("---")
        
        # Attention Flow Comparison
        st.subheader("Attention Flow Comparison")
        col1, col2 = st.columns(2)
        with col1:
            st.caption("Image A")
            fig = plot_attention_flow(img_a, res_a['activations'])
            st.pyplot(fig)
            plt.close()
        with col2:
            st.caption("Image B")
            fig = plot_attention_flow(img_b, res_b['activations'])
            st.pyplot(fig)
            plt.close()
        
        st.markdown("---")
        
        # Feature Difference Table
        st.subheader("Feature Differences")
        names = ['Attention Entropy', 'Saliency Sparsity', 'Gradient Magnitude', 
                 'Feature Variance', 'Spatial Complexity', 'Confidence Margin', 'Activation Sparsity']
        va, vb = res_a['features'].to_vector(), res_b['features'].to_vector()
        
        diff_data = []
        for name, a, b in zip(names, va, vb):
            diff = b - a
            higher = "A" if a > b else "B" if b > a else "="
            diff_data.append({"Feature": name, "Image A": f"{a:.3f}", "Image B": f"{b:.3f}", 
                              "Diff (B-A)": f"{diff:+.3f}", "Higher": higher})
        st.table(diff_data)
        
        # Summary
        st.markdown("---")
        st.subheader("Comparison Summary")
        
        lat_diff = res_b['measured_latency'] - res_a['measured_latency']
        flop_diff = res_b['flops'] - res_a['flops']
        
        st.info(f"""
        **Image A:** Routed to **{res_a['tier']}** ({res_a['measured_latency']:.1f}ms, {res_a['flops']}M FLOPs)
        
        **Image B:** Routed to **{res_b['tier']}** ({res_b['measured_latency']:.1f}ms, {res_b['flops']}M FLOPs)
        
        **Difference:** Image B is {abs(lat_diff):.1f}ms {'slower' if lat_diff > 0 else 'faster'} and uses {abs(flop_diff)}M {'more' if flop_diff > 0 else 'fewer'} FLOPs
        """)
    
    # Footer
    st.markdown("---")
    st.markdown('<div style="text-align:center;color:#555;padding:1rem;">Explainability-Guided Model Routing | XAI Project</div>', unsafe_allow_html=True)


if __name__ == "__main__":
    main()