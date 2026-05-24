"""I-JEPA — Streamlit dashboard for the video demo.

Run:  streamlit run scripts/demo_app.py

Loads the smoke-demo artifacts produced by scripts/run_demo_smoke.py and
lets you re-run masking + inference live. Designed to be filmed in the
20-minute video presentation.
"""
from __future__ import annotations

import csv
import io
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.masking import MaskingConfig, MultiBlockMaskCollator
from src.models import IJEPA
from src.models.ijepa import IJEPAConfig

ASSETS = ROOT / "assets"
CHECKPOINTS = ROOT / "checkpoints"


# ---------------------------------------------------------------------------
# Page config + global CSS
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="I-JEPA — ML2 Demo",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

CSS = """
<style>
    /* ---------- background + general ---------- */
    .stApp {
        background: linear-gradient(180deg, #0B1437 0%, #1E1B4B 50%, #0B1437 100%);
        color: #F8FAFC;
    }
    [data-testid="stHeader"] { background: transparent; }
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1E293B 0%, #0F172A 100%);
    }
    [data-testid="stSidebar"] * { color: #F8FAFC !important; }

    /* ---------- typography ---------- */
    h1, h2, h3, h4, h5 { color: #F8FAFC !important; font-weight: 700; }
    h1 { font-size: 3rem !important; letter-spacing: -0.02em; }
    p, span, label, li { color: #CBD5E1 !important; }

    /* ---------- hero banner ---------- */
    .hero {
        background: linear-gradient(135deg, #EC4899 0%, #8B5CF6 50%, #06B6D4 100%);
        padding: 2rem 2rem 1.5rem 2rem;
        border-radius: 16px;
        margin-bottom: 1.5rem;
        box-shadow: 0 10px 40px rgba(236, 72, 153, 0.25);
    }
    .hero h1, .hero p, .hero span { color: white !important; }
    .hero .eyebrow {
        font-size: 0.85rem; letter-spacing: 0.2em; font-weight: 700;
        text-transform: uppercase; opacity: 0.9;
    }
    .hero h1 { margin-top: 0.4rem; margin-bottom: 0.4rem; font-size: 3.4rem !important; }
    .hero p { font-size: 1.1rem; opacity: 0.95; }

    /* ---------- metric cards ---------- */
    [data-testid="stMetric"] {
        background: rgba(30, 41, 59, 0.6);
        border: 1px solid rgba(139, 92, 246, 0.3);
        padding: 1rem 1.3rem;
        border-radius: 12px;
        backdrop-filter: blur(10px);
    }
    [data-testid="stMetricLabel"] {
        color: #A3E635 !important;
        font-size: 0.75rem !important;
        font-weight: 600 !important;
        text-transform: uppercase;
        letter-spacing: 0.1em;
    }
    [data-testid="stMetricValue"] {
        color: #F8FAFC !important;
        font-size: 1.8rem !important;
        font-weight: 700 !important;
    }

    /* ---------- tabs ---------- */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0.5rem;
        background: rgba(15, 23, 42, 0.5);
        padding: 0.4rem;
        border-radius: 12px;
    }
    .stTabs [data-baseweb="tab"] {
        background: transparent;
        border-radius: 8px;
        color: #CBD5E1 !important;
        font-weight: 600;
        padding: 0.5rem 1.2rem;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(90deg, #EC4899, #8B5CF6) !important;
        color: white !important;
    }

    /* ---------- buttons ---------- */
    .stButton > button {
        background: linear-gradient(90deg, #EC4899, #8B5CF6);
        color: white !important;
        border: 0;
        border-radius: 8px;
        padding: 0.5rem 1.4rem;
        font-weight: 600;
        transition: transform 0.15s;
    }
    .stButton > button:hover { transform: translateY(-2px); }

    /* ---------- code blocks ---------- */
    pre, code {
        background: #0F172A !important;
        border: 1px solid #334155;
        border-radius: 8px;
        color: #A3E635 !important;
    }

    /* ---------- tables ---------- */
    [data-testid="stDataFrame"] {
        background: rgba(30, 41, 59, 0.6) !important;
        border-radius: 8px;
    }

    /* ---------- info / warning boxes ---------- */
    .stAlert {
        background: rgba(6, 182, 212, 0.1) !important;
        border-left: 4px solid #06B6D4 !important;
        border-radius: 8px;
    }

    /* ---------- pills ---------- */
    .pill {
        display: inline-block;
        padding: 0.25rem 0.8rem;
        border-radius: 999px;
        font-size: 0.8rem;
        font-weight: 600;
        margin: 0.15rem;
    }
    .pill-ok    { background: #16A34A; color: white !important; }
    .pill-warn  { background: #F59E0B; color: white !important; }
    .pill-info  { background: #06B6D4; color: white !important; }
    .pill-pink  { background: #EC4899; color: white !important; }
    .pill-violet{ background: #8B5CF6; color: white !important; }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def hero_banner():
    st.markdown("""
    <div class="hero">
        <div class="eyebrow">ML2 PROJECT · TUTORIAL + DEMO</div>
        <h1>I-JEPA</h1>
        <p><strong>Joint-Embedding Predictive Architecture</strong> · A self-supervised vision model built from scratch in PyTorch.</p>
        <span class="pill pill-info">PyTorch</span>
        <span class="pill pill-violet">Vision Transformer</span>
        <span class="pill pill-pink">Self-Supervised</span>
        <span class="pill pill-ok">CIFAR-10</span>
    </div>
    """, unsafe_allow_html=True)


def make_synthetic_image(seed: int = 0) -> torch.Tensor:
    rng = np.random.default_rng(seed)
    H = W = 32
    yy, xx = np.meshgrid(np.linspace(0, 1, H), np.linspace(0, 1, W), indexing="ij")
    r = 0.5 + 0.5 * np.sin(8 * xx + 1.0 * seed)
    g = 0.5 + 0.5 * np.sin(8 * yy + 1.7 * seed)
    b = 0.5 + 0.5 * np.cos(6 * (xx + yy) + 0.5 * seed)
    img = np.stack([r, g, b], axis=0).astype(np.float32)
    img += 0.08 * rng.standard_normal(img.shape).astype(np.float32)
    return torch.from_numpy(np.clip(img, 0, 1))


def patches_to_image(image: torch.Tensor, indices: torch.Tensor,
                     grid_size: int = 8, patch: int = 4,
                     mask_color: tuple = (0.06, 0.10, 0.20)) -> np.ndarray:
    H = grid_size * patch
    canvas = np.full((H, H, 3), mask_color, dtype=np.float32)
    img = image.permute(1, 2, 0).numpy()
    img = (img - img.min()) / (img.max() - img.min() + 1e-6)
    for idx in indices.tolist():
        r, c = idx // grid_size, idx % grid_size
        y0, x0 = r * patch, c * patch
        canvas[y0 : y0 + patch, x0 : x0 + patch] = img[y0 : y0 + patch, x0 : x0 + patch]
    return canvas


@st.cache_data(show_spinner=False)
def load_training_log(path: str):
    if not os.path.exists(path):
        return None
    steps, losses, lrs, momenta = [], [], [], []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            steps.append(int(row["step"]))
            losses.append(float(row["loss"]))
            lrs.append(float(row["lr"]))
            momenta.append(float(row["momentum"]))
    return {"steps": steps, "losses": losses, "lrs": lrs, "momenta": momenta}


@st.cache_resource(show_spinner=False)
def load_model(ckpt_path: str):
    if not os.path.exists(ckpt_path):
        return None
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = IJEPAConfig(**ckpt["cfg"])
    model = IJEPA(cfg)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------


hero_banner()

# ---------- sidebar ----------
with st.sidebar:
    st.markdown("### 📦 Project")
    st.markdown("""
- **Repo** · [github.com/mahmoud-elloumi/ML2-Projet](https://github.com/mahmoud-elloumi/ML2-Projet)
- **Author** · Mahmoud Elloumi
- **Paper** · Assran et al., CVPR 2023
- **Stack** · PyTorch · ViT-Tiny
""")
    st.markdown("---")
    st.markdown("### ⚙️ Model")
    st.code("""img_size      32
patch_size     4
embed_dim    192
encoder      6 layers
predictor    4 layers
heads          3
ema      0.996 → 1.0""", language="text")
    st.markdown("---")
    st.markdown("### 🛠️ AI tools used")
    for tool in ["Claude", "ChatGPT", "Copilot", "DALL·E 3", "Sora / Runway",
                 "Gamma.app", "ElevenLabs"]:
        st.markdown(f"• {tool}")


# ---------- top KPI row ----------
log = load_training_log(str(CHECKPOINTS / "train_log.csv"))
eval_path = CHECKPOINTS / "eval_results.txt"
eval_acc = None
if eval_path.exists():
    for line in eval_path.read_text(encoding="utf-8").splitlines():
        if line.startswith("linear_probing_accuracy="):
            eval_acc = float(line.split("=")[1])

cols = st.columns(4)
cols[0].metric("Encoder params", "≈ 1.4 M", "ViT-Tiny")
cols[1].metric("Predictor params", "≈ 250 k", "4 layers")
if log:
    cols[2].metric("Training steps", f"{log['steps'][-1]}", "smoke run")
    cols[3].metric("Final loss", f"{log['losses'][-1]:.4f}",
                   f"{(log['losses'][-1] - log['losses'][0]):.3f}",
                   delta_color="inverse")
else:
    cols[2].metric("Training steps", "—")
    cols[3].metric("Final loss", "—")

st.markdown("")

# ---------- tabs ----------
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🎭  Masking", "📉  Training", "🎯  Evaluation",
    "🌌  Embeddings", "🧪  Live inference",
])

# ----------------------------------------------------------------- Tab 1
with tab1:
    left, right = st.columns([2, 1])
    with right:
        st.markdown("#### Mask configuration")
        n_targets = st.slider("Target blocks", 1, 6, 4)
        ts_lo = st.slider("Target scale — low", 0.05, 0.4, 0.15, step=0.01)
        ts_hi = st.slider("Target scale — high", ts_lo, 0.5, 0.20, step=0.01)
        ar_lo = st.slider("Aspect ratio — low", 0.5, 1.0, 0.75, step=0.05)
        ar_hi = st.slider("Aspect ratio — high", 1.0, 2.0, 1.5, step=0.05)
        seed = st.number_input("Seed", value=42, step=1)
        regen = st.button("🎲  Resample masks", use_container_width=True)

    with left:
        cfg = MaskingConfig(
            grid_size=8,
            n_targets=n_targets,
            target_scale=(ts_lo, ts_hi),
            target_aspect=(ar_lo, ar_hi),
            context_scale=(0.85, 1.0),
        )
        collator = MultiBlockMaskCollator(cfg)
        import random; random.seed(int(seed))
        samples = [(make_synthetic_image(seed=int(seed) + k), 0) for k in range(3)]
        images, _, ctx_idx, tgt_idx = collator(samples)

        fig, axes = plt.subplots(3, 3, figsize=(7, 7), facecolor="#0B1437")
        fig.suptitle("Multi-block masking  ·  input | context | targets",
                     color="white", y=0.995, fontsize=12)
        for k in range(3):
            img = images[k]
            for ax, im, title in [
                (axes[k, 0], img.permute(1, 2, 0).numpy(), "input"),
                (axes[k, 1], patches_to_image(img, ctx_idx[k]), "context"),
                (axes[k, 2], patches_to_image(img, tgt_idx[k]), "targets"),
            ]:
                ax.imshow(im); ax.set_facecolor("#0B1437")
                ax.set_title(title, color="white", fontsize=10)
                ax.axis("off")
        fig.tight_layout()
        st.pyplot(fig, clear_figure=True)

        if (ASSETS / "masking_demo.png").exists():
            with st.expander("📷 Pre-generated masking figure (used in slide 44)"):
                st.image(str(ASSETS / "masking_demo.png"))

# ----------------------------------------------------------------- Tab 2
with tab2:
    if log is None:
        st.warning("No training log yet — run `python scripts/run_demo_smoke.py` first.")
    else:
        c1, c2 = st.columns([2, 1])
        with c1:
            fig, ax1 = plt.subplots(figsize=(8, 4), facecolor="#0B1437")
            ax1.set_facecolor("#1E1B4B")
            ax1.plot(log["steps"], log["losses"], color="#EC4899",
                     linewidth=1, alpha=0.55, label="loss")
            if len(log["losses"]) > 10:
                win = max(5, len(log["losses"]) // 20)
                kernel = np.ones(win) / win
                smooth = np.convolve(log["losses"], kernel, mode="valid")
                ax1.plot(log["steps"][win - 1:], smooth, color="#A3E635",
                         linewidth=2.5, label=f"loss (smoothed)")
            ax1.set_xlabel("step", color="white")
            ax1.set_ylabel("smooth-L1 loss", color="white")
            ax1.tick_params(colors="white")
            for spine in ax1.spines.values():
                spine.set_color("#475569")
            ax1.legend(facecolor="#1E293B", edgecolor="#475569",
                       labelcolor="white")
            ax1.set_title("Training loss", color="white")
            fig.tight_layout()
            st.pyplot(fig, clear_figure=True)
        with c2:
            st.markdown("#### Run stats")
            st.metric("Steps", len(log["steps"]))
            st.metric("Initial loss", f"{log['losses'][0]:.4f}")
            st.metric("Final loss", f"{log['losses'][-1]:.4f}")
            reduction = (1 - log["losses"][-1] / log["losses"][0]) * 100
            st.metric("Reduction", f"{reduction:.1f} %")

        with st.expander("📋 Last 10 steps from train_log.csv"):
            tail = list(zip(log["steps"][-10:], log["losses"][-10:],
                            log["lrs"][-10:], log["momenta"][-10:]))
            import pandas as pd
            df = pd.DataFrame(tail, columns=["step", "loss", "lr", "ema"])
            st.dataframe(df, use_container_width=True, hide_index=True)

# ----------------------------------------------------------------- Tab 3
with tab3:
    c1, c2 = st.columns([1, 2])
    with c1:
        st.markdown("#### Linear probe")
        if eval_acc is not None:
            st.metric("Test accuracy", f"{eval_acc:.2f} %",
                      delta=f"+{eval_acc - 10:.2f} vs chance",
                      delta_color="normal" if eval_acc > 10 else "off")
            st.caption("Synthetic smoke test — chance level ≈ 10 %.")
        else:
            st.warning("eval_results.txt not found.")
        st.markdown("---")
        st.markdown("#### Reference numbers (CIFAR-10)")
        st.markdown("""
- Random init · ~ **10 %**
- SimCLR · ~ **68 %**
- MAE · ~ **70 %**
- **I-JEPA · ~ 72 %**
- Supervised ViT-Tiny · ~ **68 %**
""")
    with c2:
        st.markdown("#### Comparison")
        ref = [
            ("Random",      10.0, "#64748B"),
            ("SimCLR",      68.0, "#06B6D4"),
            ("MAE",         70.0, "#8B5CF6"),
            ("I-JEPA",      72.0, "#EC4899"),
            ("Supervised",  68.0, "#FBBF24"),
        ]
        fig, ax = plt.subplots(figsize=(8, 4.5), facecolor="#0B1437")
        ax.set_facecolor("#1E1B4B")
        labels = [r[0] for r in ref]
        values = [r[1] for r in ref]
        colors = [r[2] for r in ref]
        bars = ax.barh(labels, values, color=colors)
        for bar, v in zip(bars, values):
            ax.text(v + 1.3, bar.get_y() + bar.get_height() / 2,
                    f"{v:.1f}%", va="center", color="white", fontsize=11,
                    fontweight="bold")
        ax.set_xlim(0, 90)
        ax.set_xlabel("Linear probe accuracy on CIFAR-10 (%)", color="white")
        ax.tick_params(colors="white")
        for spine in ax.spines.values():
            spine.set_color("#475569")
        ax.invert_yaxis()
        ax.set_title("Indicative CIFAR-10 numbers", color="white")
        fig.tight_layout()
        st.pyplot(fig, clear_figure=True)

# ----------------------------------------------------------------- Tab 4
with tab4:
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### t-SNE of the target encoder")
        path = ASSETS / "tsne.png"
        if path.exists():
            st.image(str(path), use_container_width=True)
        else:
            st.warning("Run the smoke demo to generate tsne.png.")
    with c2:
        st.markdown("#### Loss curve (saved)")
        path = ASSETS / "loss_curve.png"
        if path.exists():
            st.image(str(path), use_container_width=True)
        else:
            st.warning("Run the smoke demo to generate loss_curve.png.")

# ----------------------------------------------------------------- Tab 5
with tab5:
    st.markdown("#### Live inference on a custom image")
    st.caption("Upload any image — it will be resized to 32×32, masked, "
               "and encoded by the frozen target encoder.")

    up = st.file_uploader("Image", type=["png", "jpg", "jpeg"])
    use_synthetic = st.checkbox("Use a synthetic sample instead", value=(up is None))

    if up is not None and not use_synthetic:
        pil = Image.open(up).convert("RGB").resize((32, 32))
        arr = np.asarray(pil).astype(np.float32) / 255.0
        img = torch.from_numpy(arr).permute(2, 0, 1)
    else:
        img = make_synthetic_image(seed=int(np.random.randint(1, 9999)))

    cfg = MaskingConfig()
    collator = MultiBlockMaskCollator(cfg)
    images, _, ctx_idx, tgt_idx = collator([(img, 0)])

    model = load_model(str(CHECKPOINTS / "ijepa_smoke.pt"))
    if model is None:
        st.warning("No checkpoint found. Run `python scripts/run_demo_smoke.py`.")
    else:
        with torch.no_grad():
            embedding = model.encode(images, use_target=True)[0]
            embedding = embedding.cpu().numpy()

        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("**Input**")
            st.image(img.permute(1, 2, 0).numpy(), width=256)
        with c2:
            st.markdown("**Context patches**")
            ctx_img = patches_to_image(images[0], ctx_idx[0])
            st.image(ctx_img, width=256)
        with c3:
            st.markdown("**Target patches**")
            tgt_img = patches_to_image(images[0], tgt_idx[0])
            st.image(tgt_img, width=256)

        st.markdown("---")
        st.markdown(f"**Encoded representation** — shape {tuple(embedding.shape)}")
        fig, ax = plt.subplots(figsize=(10, 1.5), facecolor="#0B1437")
        ax.set_facecolor("#1E1B4B")
        ax.imshow(embedding.reshape(1, -1), aspect="auto", cmap="magma")
        ax.set_yticks([])
        ax.set_xlabel("embedding dim", color="white")
        ax.tick_params(colors="white")
        for spine in ax.spines.values():
            spine.set_color("#475569")
        fig.tight_layout()
        st.pyplot(fig, clear_figure=True)

        with st.expander("📊 First 20 components"):
            for i in range(0, min(20, embedding.size), 5):
                row = "  ".join(f"{v:+.3f}" for v in embedding[i:i + 5])
                st.code(f"[{i:3d}..{i+4:3d}]  {row}", language="text")
