"""
app.py — Interactive Heart Sound GAN Dashboard
Tabs:
  1. 🔍 Classifier        — upload WAV, live segment scrubber, confidence gauge
  2. 🎨 GAN Generator     — generate & compare MFCC / Delta / Delta-Delta sub-groups
  3. 📊 Model Performance — ROC, confusion matrices, training curves (auto-loaded)
  4. 📚 Educational Guide — interactive explainer with animated cards
"""

import os
import io
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import soundfile as sf
import librosa
import librosa.display
import streamlit as st

# ─────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Heart Sound AI Lab",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Inline CSS
st.markdown("""
<style>
    .metric-card {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 12px;
        padding: 18px 22px;
        text-align: center;
        margin: 6px 0;
    }
    .metric-card .value { font-size: 2rem; font-weight: 700; }
    .metric-card .label { font-size: 0.85rem; color: #aaa; margin-top: 4px; }
    .normal-card  { border-left: 4px solid #2eb85c !important; }
    .abnormal-card{ border-left: 4px solid #e55353 !important; }
    .info-card    { border-left: 4px solid #3d9df3 !important; }
    .stTabs [data-baseweb="tab-list"] { gap: 12px; }
    .stTabs [data-baseweb="tab"] { border-radius: 8px 8px 0 0; padding: 8px 20px; }
    div[data-testid="stExpander"] { border-radius: 10px; }
</style>
""", unsafe_allow_html=True)

base_dir = os.path.dirname(os.path.abspath(__file__))

from predict import predict
from preprocessing import preprocess_signal, extract_features


# ─────────────────────────────────────────────────────────
# Cached loaders
# ─────────────────────────────────────────────────────────
@st.cache_resource
def load_classifier(path):
    import tensorflow as tf
    return tf.keras.models.load_model(path) if os.path.exists(path) else None

@st.cache_resource
def load_generator(path):
    import tensorflow as tf
    return tf.keras.models.load_model(path) if os.path.exists(path) else None

@st.cache_data
def get_avg_bounds(base_dir):
    try:
        y_t = np.load(os.path.join(base_dir, "data", "processed", "y_train.npy"))
        b_t = np.load(os.path.join(base_dir, "data", "processed", "bounds_train.npy"))
        idx = np.where(y_t == 1)[0]
        if len(idx): return np.mean(b_t[idx], axis=0)
    except Exception:
        pass
    return np.array([-100.0, 100.0])

@st.cache_data
def load_train_data(base_dir):
    try:
        X = np.load(os.path.join(base_dir, "data", "processed", "X_train.npy"))
        y = np.load(os.path.join(base_dir, "data", "processed", "y_train.npy"))
        return X, y
    except Exception:
        return None, None

classifier = load_classifier(os.path.join(base_dir, "models", "cnn_classifier_gan.keras"))
generator  = load_generator(os.path.join(base_dir, "models", "gan_generator.keras"))
avg_bounds = get_avg_bounds(base_dir)
X_train, y_train = load_train_data(base_dir)


# ─────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🫀 Heart Sound AI Lab")
    st.markdown("---")

    # Model status indicators
    cls_ok = classifier is not None
    gen_ok = generator  is not None
    data_ok = X_train is not None

    st.markdown("### System Status")
    st.markdown(f"{'🟢' if cls_ok else '🔴'} **Classifier**  {'loaded' if cls_ok else 'not found'}")
    st.markdown(f"{'🟢' if gen_ok else '🔴'} **GAN Generator** {'loaded' if gen_ok else 'not found'}")
    st.markdown(f"{'🟢' if data_ok else '🔴'} **Training data** {'available' if data_ok else 'not found'}")

    st.markdown("---")
    st.markdown("### Pipeline")
    st.markdown("""
1. **Download** PhysioNet recordings
2. **Preprocess** → bandpass + MFCC
3. **Train GAN** on abnormal segments
4. **Augment** dataset → balanced
5. **Train Classifier** with Focal Loss
6. **Evaluate** & compare
    """)

    st.markdown("---")
    st.markdown("### About")
    st.caption("WGAN-GP · Multi-Scale Conv1D · SE Attention · MixUp · Focal Loss")


# ─────────────────────────────────────────────────────────
# Helper — feature sub-group plot (used in two tabs)
# ─────────────────────────────────────────────────────────

def plot_feature_subgroups(features_arr, title_prefix="Sample", figsize=(18, 3.5)):
    """
    features_arr: (N, 64, 39)
    Returns a matplotlib Figure showing MFCC / Delta / Delta-Delta in separate panels.
    """
    N = len(features_arr)
    slices = [slice(0, 13), slice(13, 26), slice(26, 39)]
    labels = ["MFCC (coeffs 0-12)", "Δ Delta (coeffs 13-25)", "ΔΔ Delta-Delta (coeffs 26-38)"]
    cmaps  = ["coolwarm", "PiYG", "RdYlBu"]

    fig, axes = plt.subplots(N, 3, figsize=(figsize[0], figsize[1] * N))
    if N == 1:
        axes = axes[np.newaxis, :]

    for i in range(N):
        for j in range(3):
            sub = features_arr[i, :, slices[j]].T   # (13, 64)
            im = axes[i, j].imshow(sub, cmap=cmaps[j], aspect="auto",
                                   origin="lower", vmin=-1.0, vmax=1.0)
            axes[i, j].set_title(f"{title_prefix} {i+1} — {labels[j]}", fontsize=10, fontweight="bold")
            axes[i, j].set_xlabel("Time frame")
            axes[i, j].set_ylabel("Coeff")
            fig.colorbar(im, ax=axes[i, j], fraction=0.046, pad=0.04)

    plt.tight_layout()
    return fig


def plot_full_heatmap(feature_39, title="MFCC+Δ+ΔΔ Heatmap"):
    """Single full 39-feature heatmap."""
    fig, ax = plt.subplots(figsize=(10, 3.5))
    im = ax.imshow(feature_39.T, cmap="coolwarm", aspect="auto", origin="lower")
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlabel("Time frames")
    ax.set_ylabel("Feature dim (0-38)")
    ax.axhline(12.5, color="white", lw=1.2, linestyle="--", alpha=0.6)
    ax.axhline(25.5, color="white", lw=1.2, linestyle="--", alpha=0.6)
    ax.text(1, 6,  "MFCC",    color="white", fontsize=8, va="center", fontweight="bold")
    ax.text(1, 19, "Delta",   color="white", fontsize=8, va="center", fontweight="bold")
    ax.text(1, 32, "Δ-Delta", color="white", fontsize=8, va="center", fontweight="bold")
    fig.colorbar(im, ax=ax)
    return fig


# ═══════════════════════════════════════════════════════════
# TABS
# ═══════════════════════════════════════════════════════════
tab1, tab2, tab3, tab4 = st.tabs([
    "🔍 Classifier",
    "🎨 GAN Generator",
    "📊 Model Performance",
    "📚 How It Works",
])


# ═══════════════════════════════════════════════════════════
# TAB 1 — Classifier
# ═══════════════════════════════════════════════════════════
with tab1:
    st.header("Heart Sound Classifier Sandbox")
    st.markdown("Upload a `.wav` heart sound recording to analyse it segment-by-segment and classify it.")

    if not cls_ok:
        st.error(f"Classifier model not found. Train it first with `python train_classifier.py`.")

    uploaded = st.file_uploader("Upload heart sound (.wav)", type=["wav"], key="cls_upload")

    if uploaded and cls_ok:
        temp_dir = os.path.join(base_dir, "temp")
        os.makedirs(temp_dir, exist_ok=True)
        temp_path = os.path.join(temp_dir, uploaded.name)

        with open(temp_path, "wb") as f:
            f.write(uploaded.getbuffer())

        try:
            y_raw, orig_fs = sf.read(temp_path)
            if len(y_raw.shape) > 1:
                y_raw = np.mean(y_raw, axis=1)
            duration = len(y_raw) / orig_fs

            # ── Audio player ──────────────────────────────────
            st.subheader("🔊 Playback")
            st.audio(uploaded, format="audio/wav")

            # ── Run inference ─────────────────────────────────
            with st.spinner("Running classifier…"):
                result = predict(temp_path, model=classifier)

            predicted_class = result["class"]
            confidence      = result["confidence"] * 100
            seg_probs       = np.array(result["segment_probs"])
            n_segs          = result["num_segments"]

            # ── Top result banner ─────────────────────────────
            is_abnormal = predicted_class == "Abnormal"
            banner_color = "#e55353" if is_abnormal else "#2eb85c"
            icon = "⚠️" if is_abnormal else "✅"

            st.markdown(f"""
            <div style="background:{banner_color}22; border:2px solid {banner_color};
                        border-radius:14px; padding:20px 28px; margin:12px 0;">
                <span style="font-size:2.2rem;">{icon}</span>
                <span style="font-size:1.8rem; font-weight:700; color:{banner_color};
                             margin-left:12px;">{predicted_class}</span>
                <span style="font-size:1.1rem; color:#ccc; margin-left:16px;">
                    Confidence: <b style="color:{banner_color}">{confidence:.1f}%</b>
                </span>
            </div>""", unsafe_allow_html=True)

            # ── KPI row ───────────────────────────────────────
            k1, k2, k3, k4 = st.columns(4)
            k1.metric("Confidence",   f"{confidence:.1f}%")
            k2.metric("Segments",     n_segs)
            k3.metric("Duration",     f"{duration:.2f}s")
            k4.metric("Sample Rate",  f"{orig_fs} Hz")

            # ── Waveform + full heatmap ───────────────────────
            col_w, col_h = st.columns(2)

            with col_w:
                st.subheader("Raw Waveform")
                fig_w, ax_w = plt.subplots(figsize=(10, 3.5))
                librosa.display.waveshow(y_raw, sr=orig_fs, ax=ax_w, color="#3d9df3")
                ax_w.set_title("Phonocardiogram Signal", fontweight="bold")
                ax_w.set_xlabel("Time (s)")
                ax_w.set_ylabel("Amplitude")
                ax_w.grid(True, alpha=0.3)
                st.pyplot(fig_w)
                plt.close(fig_w)

            with col_h:
                st.subheader("39-Feature Heatmap")
                y_res = librosa.resample(y_raw, orig_sr=orig_fs, target_sr=2000) if orig_fs != 2000 else y_raw
                y_filt = preprocess_signal(y_res, fs=2000)
                seg0 = y_filt[:5040] if len(y_filt) >= 5040 else np.pad(y_filt, (0, 5040 - len(y_filt)))
                feats0 = extract_features(seg0, fs=2000)["mfcc"]   # (64, 39)
                fig_h = plot_full_heatmap(feats0, "First Segment — MFCC + Δ + ΔΔ")
                st.pyplot(fig_h)
                plt.close(fig_h)

            # ── Feature Sub-group explorer ────────────────────
            st.markdown("---")
            st.subheader("🔬 Feature Sub-Group Explorer")
            st.markdown("Inspect MFCC, Delta, and Delta-Delta components **separately** for each segment.")

            all_segs = []
            from preprocessing import segment_signal
            segs = segment_signal(y_filt, segment_len=5040, overlap=1000)
            for s in segs:
                all_segs.append(extract_features(s, fs=2000)["mfcc"])

            if all_segs:
                seg_idx = st.slider(
                    "Select segment to inspect",
                    min_value=1, max_value=len(all_segs),
                    value=1, step=1,
                    key="seg_slider",
                )
                chosen_seg = all_segs[seg_idx - 1]
                seg_prob = float(seg_probs[seg_idx - 1]) if seg_idx - 1 < len(seg_probs) else 0.5
                seg_lbl  = "Abnormal" if seg_prob >= 0.5 else "Normal"
                seg_col  = "#e55353" if seg_prob >= 0.5 else "#2eb85c"

                st.markdown(
                    f"Segment **{seg_idx}/{len(all_segs)}** — "
                    f"<span style='color:{seg_col}; font-weight:700'>{seg_lbl} "
                    f"({seg_prob*100:.1f}%)</span>",
                    unsafe_allow_html=True,
                )
                fig_sg = plot_feature_subgroups(chosen_seg[np.newaxis], figsize=(18, 3.5))
                st.pyplot(fig_sg)
                plt.close(fig_sg)

            # ── Per-segment probability bar ───────────────────
            st.markdown("---")
            st.subheader("📈 Per-Segment Probability Timeline")
            fig_pb, ax_pb = plt.subplots(figsize=(max(10, n_segs * 0.5), 3))
            colors = ["#e55353" if p >= 0.5 else "#2eb85c" for p in seg_probs]
            ax_pb.bar(range(1, n_segs + 1), seg_probs * 100, color=colors, edgecolor="white", linewidth=0.4)
            ax_pb.axhline(50, color="orange", linestyle="--", lw=1.5, label="Decision threshold (50%)")
            ax_pb.set_xlabel("Segment index")
            ax_pb.set_ylabel("Abnormal probability (%)")
            ax_pb.set_title("Segment-level Predictions", fontweight="bold")
            ax_pb.set_ylim(0, 105)
            ax_pb.legend()
            ax_pb.grid(True, alpha=0.3)
            st.pyplot(fig_pb)
            plt.close(fig_pb)

        except Exception as e:
            st.error(f"Processing error: {e}")
        finally:
            if os.path.exists(temp_path):
                try: os.remove(temp_path)
                except: pass


# ═══════════════════════════════════════════════════════════
# TAB 2 — GAN Generator
# ═══════════════════════════════════════════════════════════
with tab2:
    st.header("🎨 GAN Synthetic Abnormal Murmur Generator")
    st.markdown("""
    The **Improved WGAN-GP Generator** maps random 100-dim noise → realistic `(64, 39)` MFCC features.
    Three **separate output heads** (MFCC / Delta / Delta-Delta) make each feature group independently
    controllable and visually distinct.
    """)

    if not gen_ok:
        st.error("GAN Generator not found. Train it first with `python train_gan.py`.")
    else:
        # ── Controls ───────────────────────────────────────────
        st.subheader("⚙️ Generation Controls")
        ctrl1, ctrl2, ctrl3 = st.columns(3)
        n_samples = ctrl1.slider("Number of samples", 1, 8, 4, key="n_gen")
        latent_seed = ctrl2.number_input("Random seed (−1 = random)", value=-1, step=1, key="seed")
        view_mode = ctrl3.radio("View mode", ["Sub-groups", "Full heatmap", "Both"], key="view_mode")

        gen_btn = st.button("🎲 Generate Synthetic Murmurs", key="gen_btn", type="primary")

        if gen_btn:
            import tensorflow as tf

            if latent_seed >= 0:
                tf.random.set_seed(int(latent_seed))
            noise = tf.random.normal([n_samples, 100])
            synth = generator(noise, training=False).numpy()  # (N, 64, 39)

            # ── Sub-group view ─────────────────────────────────
            if view_mode in ("Sub-groups", "Both"):
                st.markdown("---")
                st.subheader("📊 Feature Sub-Group Panels")
                st.caption("MFCC (coolwarm) · Delta (PiYG) · Delta-Delta (RdYlBu) — each group uses its own colour scale so differences are immediately visible.")
                fig_sg = plot_feature_subgroups(synth, title_prefix="Synthetic", figsize=(18, 3.5))
                st.pyplot(fig_sg)
                plt.close(fig_sg)

            # ── Full heatmap view ──────────────────────────────
            if view_mode in ("Full heatmap", "Both"):
                st.markdown("---")
                st.subheader("🗺️ Full 39-Feature Heatmaps")
                cols = st.columns(min(n_samples, 4))
                for i in range(n_samples):
                    with cols[i % 4]:
                        fig_fh = plot_full_heatmap(synth[i], f"Synthetic {i+1}")
                        st.pyplot(fig_fh)
                        plt.close(fig_fh)

            # ── Comparison with real abnormal ──────────────────
            if X_train is not None and y_train is not None:
                st.markdown("---")
                st.subheader("⚖️ Real vs Synthetic Comparison")
                col_r, col_s = st.columns(2)

                abnorm_idx = np.where(y_train == 1)[0]
                if len(abnorm_idx):
                    # Pick a random real sample
                    pick = int(np.random.choice(abnorm_idx))
                    real_feat = X_train[pick]  # (64, 39)

                    with col_r:
                        st.markdown("**🩺 Real Abnormal (PhysioNet)**")
                        fig_real_sg = plot_feature_subgroups(real_feat[np.newaxis], "Real", figsize=(18, 3.5))
                        st.pyplot(fig_real_sg)
                        plt.close(fig_real_sg)

                    with col_s:
                        st.markdown("**🤖 GAN Synthetic Abnormal**")
                        fig_synth_sg = plot_feature_subgroups(synth[:1], "Synthetic", figsize=(18, 3.5))
                        st.pyplot(fig_synth_sg)
                        plt.close(fig_synth_sg)

                # ── Feature stats ──────────────────────────────
                st.markdown("---")
                st.subheader("📐 Feature Statistics")
                if len(abnorm_idx):
                    r_mean = float(X_train[abnorm_idx].mean())
                    r_std  = float(X_train[abnorm_idx].std())
                else:
                    r_mean = r_std = 0.0

                s_mean = float(synth.mean())
                s_std  = float(synth.std())

                sc1, sc2, sc3, sc4 = st.columns(4)
                sc1.metric("Real Mean",      f"{r_mean:.4f}")
                sc2.metric("Real Std Dev",   f"{r_std:.4f}")
                sc3.metric("Synth Mean",     f"{s_mean:.4f}", delta=f"{s_mean-r_mean:+.4f}")
                sc4.metric("Synth Std Dev",  f"{s_std:.4f}",  delta=f"{s_std-r_std:+.4f}")

        # ── Audio reconstruction ───────────────────────────────
        st.markdown("---")
        st.subheader("🔊 Reconstruct & Listen to Synthetic Audio")
        st.caption("Griffin-Lim inversion converts synthetic MFCC → audio. Download it and test in Tab 1!")
        recon_seed = st.number_input("Audio seed (−1 = random)", value=-1, step=1, key="recon_seed")

        if st.button("🎵 Generate Audio", key="recon_btn"):
            import tensorflow as tf
            if recon_seed >= 0:
                tf.random.set_seed(int(recon_seed))
            noise_1 = tf.random.normal([1, 100])
            sf1 = generator(noise_1, training=False).numpy()[0]   # (64, 39)

            synth_mfcc = sf1[:, :13]
            mfcc_denorm = ((synth_mfcc + 1.0) / 2.0) * (avg_bounds[1] - avg_bounds[0]) + avg_bounds[0]

            with st.spinner("Griffin-Lim reconstruction…"):
                y_recon = librosa.feature.inverse.mfcc_to_audio(
                    mfcc_denorm.T, sr=2000, n_fft=256, hop_length=80, n_iter=200
                )
                mx = np.max(np.abs(y_recon))
                if mx > 0: y_recon /= mx

                wav_buf = io.BytesIO()
                sf.write(wav_buf, y_recon, 2000, format="WAV", subtype="PCM_16")
                wav_bytes = wav_buf.getvalue()

            st.audio(wav_bytes, format="audio/wav")

            col_dl, col_info = st.columns([1, 3])
            with col_dl:
                st.download_button("💾 Download WAV", wav_bytes,
                                   "synthetic_abnormal.wav", "audio/wav")
            with col_info:
                st.info("Upload this file in **Tab 1** to run it through the classifier!")

            # Show the sub-group plot for this specific audio sample
            fig_audio_sg = plot_feature_subgroups(sf1[np.newaxis], "Audio sample", figsize=(18, 3.5))
            st.pyplot(fig_audio_sg)
            plt.close(fig_audio_sg)


# ═══════════════════════════════════════════════════════════
# TAB 3 — Model Performance
# ═══════════════════════════════════════════════════════════
with tab3:
    st.header("📊 Model Performance Dashboard")

    outputs_dir = os.path.join(base_dir, "outputs")

    # ── Load CSV results ───────────────────────────────────
    csv_path = os.path.join(outputs_dir, "gan_vs_no_gan_results.csv")
    if os.path.exists(csv_path):
        import pandas as pd
        df = pd.read_csv(csv_path)
        st.subheader("Quantitative Comparison")

        # Highlight GAN row
        def highlight_gan(row):
            return ["background-color: #16213e" if "GAN" in str(row["Model"]) else "" for _ in row]

        st.dataframe(
            df.style.apply(highlight_gan, axis=1).format({
                "Accuracy": "{:.4f}", "Precision": "{:.4f}",
                "Recall": "{:.4f}", "F1-score": "{:.4f}", "AUC": "{:.4f}",
            }),
            use_container_width=True,
        )

        # Delta metrics
        if len(df) >= 2:
            st.markdown("#### Improvement (GAN vs Baseline)")
            row_bl  = df.iloc[0]
            row_gan = df.iloc[1]
            d1, d2, d3, d4, d5 = st.columns(5)
            for col_el, metric in zip([d1,d2,d3,d4,d5],
                                      ["Accuracy","Precision","Recall","F1-score","AUC"]):
                delta = float(row_gan[metric]) - float(row_bl[metric])
                col_el.metric(metric, f"{float(row_gan[metric]):.4f}", f"{delta:+.4f}")
    else:
        st.info("Run `python evaluation.py` to generate performance results.")

    st.markdown("---")

    # ── Plot images from outputs ───────────────────────────
    plot_map = {
        "ROC Curve Comparison":         os.path.join(outputs_dir, "plots", "performance_comparison.png"),
        "Confusion Matrices":            os.path.join(outputs_dir, "plots", "confusion_matrix_comparison.png"),
        "Class Distribution":            os.path.join(outputs_dir, "plots", "class_distribution_comparison.png"),
        "t-SNE Feature Space":           os.path.join(outputs_dir, "plots", "tsne_real_vs_generated.png"),
        "GAN Training Losses":           os.path.join(outputs_dir, "plots", "gan_loss.png"),
        "MFCC Heatmap Comparison":       os.path.join(outputs_dir, "plots", "spectrogram_comparisons.png"),
    }

    available = {k: v for k, v in plot_map.items() if os.path.exists(v)}

    if available:
        st.subheader("Visualisations")
        names = list(available.keys())
        # Show 2-column grid
        for i in range(0, len(names), 2):
            cols = st.columns(2)
            for j in range(2):
                if i + j < len(names):
                    n = names[i + j]
                    with cols[j]:
                        st.markdown(f"**{n}**")
                        st.image(available[n], use_container_width=True)
    else:
        st.info("Plots appear here after running `python evaluation.py`.")

    # ── Training history ───────────────────────────────────
    st.markdown("---")
    st.subheader("Training History")
    hist_nogan_path = os.path.join(base_dir, "data", "processed", "history_nogan.npy")
    hist_gan_path   = os.path.join(base_dir, "data", "processed", "history_gan.npy")

    has_hist = os.path.exists(hist_nogan_path) and os.path.exists(hist_gan_path)
    if has_hist:
        h_no  = np.load(hist_nogan_path, allow_pickle=True).item()
        h_gan = np.load(hist_gan_path,   allow_pickle=True).item()

        metric_opts = [k for k in h_no if not k.startswith("val_")]
        sel_metric = st.selectbox("Select metric", metric_opts, key="hist_metric")

        fig_hist, ax_hist = plt.subplots(1, 2, figsize=(14, 4))
        for ax_h, (h, title) in zip(ax_hist, [(h_no, "Baseline (No GAN)"), (h_gan, "GAN-Augmented")]):
            epochs = range(1, len(h[sel_metric]) + 1)
            ax_h.plot(epochs, h[sel_metric], lw=2, label=f"Train {sel_metric}")
            val_key = f"val_{sel_metric}"
            if val_key in h:
                ax_h.plot(epochs, h[val_key], lw=2, linestyle="--", label=f"Val {sel_metric}")
            ax_h.set_title(title, fontweight="bold")
            ax_h.set_xlabel("Epoch")
            ax_h.legend()
            ax_h.grid(True, alpha=0.3)

        st.pyplot(fig_hist)
        plt.close(fig_hist)
    else:
        st.info("Training history appears here after training.")


# ═══════════════════════════════════════════════════════════
# TAB 4 — Educational Guide
# ═══════════════════════════════════════════════════════════
with tab4:
    st.header("📚 How It Works — Interactive Guide")

    with st.expander("❤️ What are Heart Sounds?", expanded=True):
        c1, c2 = st.columns([2, 1])
        with c1:
            st.markdown("""
**Normal** hearts produce two sounds per beat:
- **S1 ("lub")** — closure of mitral & tricuspid valves
- **S2 ("dub")** — closure of aortic & pulmonary valves

**Abnormal** sounds (murmurs) arise from turbulent blood flow through diseased valves.
These are captured as **Phonocardiogram (PCG)** `.wav` recordings.
            """)
        with c2:
            # Mini waveform illustration
            fig_illus, ax_illus = plt.subplots(figsize=(5, 2))
            t = np.linspace(0, 1, 500)
            wave  = 0.9 * np.exp(-((t - 0.15)**2) / 0.001) * np.sin(2*np.pi*60*t)
            wave += 0.7 * np.exp(-((t - 0.55)**2) / 0.001) * np.sin(2*np.pi*90*t)
            ax_illus.plot(t, wave, color="#e55353", lw=1.5)
            ax_illus.text(0.15, 0.75, "S1", transform=ax_illus.transAxes, fontweight="bold", color="#2eb85c")
            ax_illus.text(0.55, 0.75, "S2", transform=ax_illus.transAxes, fontweight="bold", color="#3d9df3")
            ax_illus.set_axis_off()
            ax_illus.set_title("Cardiac Cycle", fontsize=9)
            st.pyplot(fig_illus)
            plt.close(fig_illus)

    with st.expander("🎵 Feature Extraction: MFCC + Delta + Delta-Delta"):
        st.markdown("""
| Feature Group | Coefficients | What it captures |
|:---|:---:|:---|
| **MFCC** | 0–12 | Static spectral shape — *what the heart sounds like* |
| **Delta (Δ)** | 13–25 | Velocity — *how quickly the spectrum changes* |
| **Delta-Delta (ΔΔ)** | 26–38 | Acceleration — *rate of change of change* |

Together these form a `(64, 39)` matrix: 64 time frames × 39 features.

The three groups look **visually different** (different colour scales above) because murmurs
cause sharp high-frequency transitions → large Delta values but moderate MFCC values.
        """)

    with st.expander("🤖 Why WGAN-GP instead of a plain GAN?"):
        st.markdown("""
| Problem | Plain GAN | WGAN-GP |
|:---|:---|:---|
| **Mode collapse** | Generator repeats the same pattern | Wasserstein distance prevents this |
| **Training instability** | Loss oscillates wildly | Gradient penalty stabilises both networks |
| **No useful loss signal** | Generator/discriminator loss doesn't track quality | Wasserstein distance tracks generation quality |
| **Diverse outputs** | Limited variation | High diversity score maintained throughout |

The **diversity score** (mean pairwise L2 distance between 64 generated samples)
is logged every epoch — if it drops below a threshold, mode collapse is occurring.
        """)

    with st.expander("🧠 Classifier Architecture — Multi-Scale SE-BiGRU"):
        st.markdown("""
```
Input (64, 39)
  ↓
MultiScaleConvBlock — parallel kernels: 3, 5, 7
  ┌───────┬─────────┬─────────┐
  K=3     K=5       K=7       → Concatenate → SE Channel Attention
  └───────┴─────────┴─────────┘
  MaxPool(2) → (32, 192)  + SpatialDropout
  ↓
MultiScaleConvBlock × 2  (128 filters)
  MaxPool(2) → (16, 384)  + SpatialDropout
  ↓
MultiScaleConvBlock × 3  (64 filters)
  → (16, 192)
  ↓
Bidirectional GRU(64) → (128,)  + Dropout
  ↓
Dense(64, relu) → Dropout → Dense(1, sigmoid)
```

**Why Focal Loss?** In medical datasets, false negatives (missed abnormalities) are far
more costly than false positives. Focal loss with `alpha=0.75` up-weights the abnormal
class and `gamma=2.0` focuses training on hard misclassifications.
        """)

    with st.expander("🔬 Three-Head Generator — why separate groups matter"):
        st.markdown("""
The improved generator outputs **three independent heads**, one per feature group:

- **MFCC head** (kernel=1, tanh): models static spectral shape of murmurs
- **Delta head** (kernel=1, tanh): models velocity — murmurs create rapid transitions
- **Delta-Delta head** (kernel=1, tanh): models acceleration — captures abrupt onset/offset

Without separate heads, the generator must learn a single combined distribution,
which often results in Delta features that don't match the MFCC features in timing.
Separate heads + feature-matching loss force each group to independently resemble
the corresponding group in real abnormal segments.
        """)

    with st.expander("🛠️ Step-by-Step Execution"):
        st.code("""
# 1. Install dependencies
pip install -r requirements.txt

# 2. Download 197 PhysioNet recordings
python download_physionet.py

# 3. Preprocess + extract features (leak-proof splitting)
python preprocessing.py

# 4. Train improved WGAN-GP (100 epochs recommended)
python train_gan.py --epochs 100 --disc_steps 5

# 5. Train both classifiers (Focal Loss + MixUp)
python train_classifier.py --epochs 25

# 6. Evaluate & generate report
python evaluation.py

# 7. Launch this dashboard
streamlit run app.py
        """, language="bash")
