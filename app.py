import os
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf
import librosa
import librosa.display

# Configure Streamlit page layout and aesthetics
st.set_page_config(
    page_title="Heart Sound GAN Classifier",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Resolve absolute base directory of the script
base_dir = os.path.dirname(os.path.abspath(__file__))

# Import predict functions locally
from predict import predict
from preprocessing import preprocess_signal, extract_features

# Cache classifier model loading to avoid loading it from scratch on every rerun
@st.cache_resource
def load_cached_classifier(model_path):
    import tensorflow as tf
    if not os.path.exists(model_path):
        return None
    return tf.keras.models.load_model(model_path)

# Cache GAN generator model loading
@st.cache_resource
def load_cached_generator(model_path):
    import tensorflow as tf
    if not os.path.exists(model_path):
        return None
    return tf.keras.models.load_model(model_path)

# Cache bounds calculation for denormalization
@st.cache_data
def get_average_abnormal_bounds(base_dir):
    try:
        X_train_path = os.path.join(base_dir, 'data', 'processed', 'X_train.npy')
        y_train_path = os.path.join(base_dir, 'data', 'processed', 'y_train.npy')
        bounds_train_path = os.path.join(base_dir, 'data', 'processed', 'bounds_train.npy')
        
        if os.path.exists(bounds_train_path) and os.path.exists(y_train_path):
            bounds_train = np.load(bounds_train_path)
            y_train = np.load(y_train_path)
            abnormal_idx = np.where(y_train == 1)[0]
            if len(abnormal_idx) > 0:
                return np.mean(bounds_train[abnormal_idx], axis=0)
    except Exception:
        pass
    return np.array([-100.0, 100.0]) # fallback typical bounds

# App header and description
st.title("❤️ Heart Sound Classification & GAN Dashboard")
st.markdown("""
This educational dashboard demonstrates the application of Deep Learning to classify Phonocardiogram (PCG) heart sound recordings and generate synthetic anomalies using a GAN.
""")

# Sidebar documentation
st.sidebar.header("About the Project")
st.sidebar.markdown("""
### Pipeline Steps:
1. **Butterworth Filter**: 25Hz - 400Hz bandpass filter to remove noise.
2. **Feature Extraction**: 13 MFCC + Delta + Delta-Delta (39 total features).
3. **GAN Data Augmentation**: Generator synthesizes features to balance dataset.
4. **Classification**: Hybrid Conv1D + BiLSTM network predicts.
""")

# Pre-load classifier
classifier_path = os.path.join(base_dir, 'models', 'cnn_classifier_gan.keras')
classifier = load_cached_classifier(classifier_path)

# Pre-load generator
generator_path = os.path.join(base_dir, 'models', 'gan_generator.keras')
generator = load_cached_generator(generator_path)

# Set up Tab layout
tab1, tab2 = st.tabs(["🔍 Heart Sound Classifier", "🎨 GAN Abnormalities Generator"])

# ==========================================
# TAB 1: Heart Sound Classifier Sandbox
# ==========================================
with tab1:
    st.header("Classifier Inference Sandbox")
    st.markdown("Upload any raw heart sound recording (`.wav`) to visualze its waveform and classify it in real-time.")
    
    if classifier is None:
        st.error(f"Trained classifier model could not be loaded at path: `{classifier_path}`. Please verify that the model has been trained.")

    # File Uploader widget
    uploaded_file = st.file_uploader("Upload a heart sound recording (.wav)", type=["wav"], key="classifier_upload")

    if uploaded_file is not None:
        # Save the uploaded file temporarily on disk
        temp_dir = os.path.join(base_dir, "temp")
        os.makedirs(temp_dir, exist_ok=True)
        temp_path = os.path.join(temp_dir, uploaded_file.name)
        try:
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
                
            # Prominent Audio Player section
            st.subheader("🔊 Listen to Heart Sound")
            st.audio(uploaded_file, format="audio/wav")
            
            # Load audio details
            y, fs = sf.read(temp_path)
            
            # Convert stereo to mono by averaging channels if necessary
            if len(y.shape) > 1:
                y = np.mean(y, axis=1)
                
            duration = len(y) / fs
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Audio Waveform")
                try:
                    fig, ax = plt.subplots(figsize=(10, 3.5))
                    librosa.display.waveshow(y, sr=fs, ax=ax, color='#1f77b4')
                    ax.set_title("Raw Phonocardiogram Signal", fontsize=12, fontweight='bold')
                    ax.set_xlabel("Time (seconds)")
                    ax.set_ylabel("Amplitude")
                    ax.grid(True, alpha=0.3)
                    st.pyplot(fig)
                    plt.close()
                except Exception as e:
                    st.error(f"Error plotting audio waveform: {e}")
                
            with col2:
                st.subheader("39-Feature MFCC Heatmap")
                try:
                    # Resample and filter signal for feature display
                    if fs != 2000:
                        y_res = librosa.resample(y, orig_sr=fs, target_sr=2000)
                        y_filt = preprocess_signal(y_res, fs=2000)
                    else:
                        y_filt = preprocess_signal(y, fs=2000)
                        
                    # Get first segment (or pad if short)
                    if len(y_filt) >= 5040:
                        display_seg = y_filt[:5040]
                    else:
                        display_seg = np.pad(y_filt, (0, max(0, 5040 - len(y_filt))))
                        
                    features = extract_features(display_seg, fs=2000)
                    mfcc_display = features['mfcc'].T # shape (39, 64)
                    
                    # Display Feature Shape
                    st.markdown("**Feature Shape**: `64 × 39` (64 time frames × 39 features)")
                    
                    fig, ax = plt.subplots(figsize=(10, 3.5))
                    im = ax.imshow(mfcc_display, cmap='coolwarm', aspect='auto', origin='lower')
                    ax.set_title("MFCC + Delta + Delta-Delta Features", fontsize=12, fontweight='bold')
                    ax.set_xlabel("Time frames")
                    ax.set_ylabel("Feature Dimension (0-38)")
                    fig.colorbar(im, ax=ax)
                    st.pyplot(fig)
                    plt.close()
                except Exception as e:
                    st.error(f"Error plotting MFCC heatmap: {e}")
                
            # Run Inference
            st.subheader("Model Inference")
            if classifier is not None:
                with st.spinner("Analyzing heart sound..."):
                    try:
                        # Predict using pre-loaded cached model
                        result = predict(temp_path, model_path=classifier_path, fs=2000, model=classifier)
                        
                        # Calculate details
                        segment_probs = np.array(result['segment_probs'])
                        abnormal_segments = np.sum(segment_probs >= 0.5)
                        normal_segments = np.sum(segment_probs < 0.5)
                        
                        avg_prob = np.mean(segment_probs)
                        abnormal_prob = avg_prob * 100
                        normal_prob = (1.0 - avg_prob) * 100
                        
                        # Determine prediction confidence and status
                        confidence = max(normal_prob, abnormal_prob)
                        
                        if confidence < 60.0:
                            pred_status = "uncertain"
                        elif abnormal_prob >= 50.0:
                            pred_status = "abnormal"
                        else:
                            pred_status = "normal"
                            
                        # Prediction Status Banner
                        if pred_status == "normal":
                            st.markdown(
                                f'<div style="background-color: rgba(40, 167, 69, 0.15); border-left: 6px solid #28a745; padding: 15px; border-radius: 4px; color: #2eb85c; font-size: 18px; font-weight: bold; margin-bottom: 20px; font-family: monospace; white-space: pre-wrap;">'
                                f'✅ Normal Heart Sound Classified\n'
                                f'Normal Probability   : {normal_prob:.2f}%\n'
                                f'Abnormal Probability : {abnormal_prob:.2f}%'
                                '</div>',
                                unsafe_allow_html=True
                            )
                        elif pred_status == "abnormal":
                            st.markdown(
                                f'<div style="background-color: rgba(220, 53, 69, 0.15); border-left: 6px solid #dc3545; padding: 15px; border-radius: 4px; color: #e55353; font-size: 18px; font-weight: bold; margin-bottom: 20px; font-family: monospace; white-space: pre-wrap;">'
                                f'🚨 Abnormal Heart Sound Detected\n'
                                f'Normal Probability   : {normal_prob:.2f}%\n'
                                f'Abnormal Probability : {abnormal_prob:.2f}%'
                                '</div>',
                                unsafe_allow_html=True
                            )
                        else:
                            st.markdown(
                                f'<div style="background-color: rgba(255, 193, 7, 0.15); border-left: 6px solid #ffc107; padding: 15px; border-radius: 4px; color: #f9b115; font-size: 18px; font-weight: bold; margin-bottom: 20px; font-family: monospace; white-space: pre-wrap;">'
                                f'⚠️ Prediction Uncertain\n'
                                f'Normal Probability   : {normal_prob:.2f}%\n'
                                f'Abnormal Probability : {abnormal_prob:.2f}%'
                                '</div>',
                                unsafe_allow_html=True
                            )
                            
                        # Metrics Columns
                        col_metrics1, col_metrics2 = st.columns(2)
                        
                        with col_metrics1:
                            st.markdown("#### 📊 Probability Breakdown")
                            st.markdown(
                                f"""
                                <div style="background-color: rgba(255,255,255,0.05); padding: 15px; border-radius: 6px; border: 1px solid rgba(255,255,255,0.1);">
                                    <table style="width: 100%; border-collapse: collapse;">
                                        <tr>
                                            <td style="padding: 8px 0; font-size: 15px; font-weight: bold; color: #2eb85c;">Normal Probability</td>
                                            <td style="padding: 8px 0; text-align: right; font-size: 16px; font-weight: bold; color: #2eb85c;">{normal_prob:.2f}%</td>
                                        </tr>
                                        <tr>
                                            <td style="padding: 8px 0; font-size: 15px; font-weight: bold; color: #e55353;">Abnormal Probability</td>
                                            <td style="padding: 8px 0; text-align: right; font-size: 16px; font-weight: bold; color: #e55353;">{abnormal_prob:.2f}%</td>
                                        </tr>
                                    </table>
                                </div>
                                """,
                                unsafe_allow_html=True
                            )
                            
                        with col_metrics2:
                            st.markdown("#### 🗳️ Segment Voting")
                            st.markdown(
                                f"""
                                <div style="background-color: rgba(255,255,255,0.05); padding: 15px; border-radius: 6px; border: 1px solid rgba(255,255,255,0.1);">
                                    <table style="width: 100%; border-collapse: collapse;">
                                        <tr>
                                            <td style="padding: 8px 0; font-size: 15px; font-weight: bold;">Normal Segments</td>
                                            <td style="padding: 8px 0; text-align: right; font-size: 16px; font-weight: bold; color: #2eb85c;">{normal_segments}</td>
                                        </tr>
                                        <tr>
                                            <td style="padding: 8px 0; font-size: 15px; font-weight: bold;">Abnormal Segments</td>
                                            <td style="padding: 8px 0; text-align: right; font-size: 16px; font-weight: bold; color: #e55353;">{abnormal_segments}</td>
                                        </tr>
                                    </table>
                                </div>
                                """,
                                unsafe_allow_html=True
                            )
                            
                        # Informational metrics
                        st.markdown("<br>", unsafe_allow_html=True)
                        col_inf1, col_inf2, col_inf3 = st.columns(3)
                        col_inf1.metric(label="Global Confidence", value=f"{confidence:.2f}%")
                        col_inf2.metric(label="Analyzed Segments", value=result['num_segments'])
                        col_inf3.metric(label="Audio Duration", value=f"{duration:.2f}s")
                        
                    except Exception as e:
                        st.error(f"Error during model classification: {e}")
            else:
                st.error("Classifier model not loaded. Cannot run inference.")
                
        except Exception as e:
            st.error(f"Error loading or processing audio file: {e}")
        finally:
            # Cleanup temp file
            if os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except Exception:
                    pass

# ==========================================
# TAB 2: GAN Abnormalities Generator Sandbox
# ==========================================
with tab2:
    st.header("🎨 GAN Abnormalities Generator Sandbox")
    st.markdown("""
    This tab demonstrates how the trained **1D DCGAN Generator** can synthesize new, unique abnormal heart sound features from random noise.
    The generator maps a 100-dimensional random noise vector $z \sim \mathcal{N}(0, I)$ to a realistic 39-feature sequence (MFCC + Delta + Delta-Delta) of shape `(64, 39)`.
    """)
    
    if generator is None:
        st.error(f"Trained GAN generator model could not be loaded at path: `{generator_path}`. Please verify that the model has been trained.")
    else:
        st.subheader("Generate Synthetic Abnormal Murmurs")
        st.write("Click the button below to generate a batch of 4 unique synthetic abnormal signals.")
        
        if st.button("Generate Synthetic Abnormalities", key="gan_generate_button"):
            import tensorflow as tf
            # Generate random noise vectors from a normal distribution
            noise = tf.random.normal([4, 100])
            
            # Predict using the cached generator network
            synthetic_features = generator(noise, training=False).numpy() # shape (4, 64, 39)
            
            # Plot the 4 generated samples in a 2x2 grid
            fig, axes = plt.subplots(2, 2, figsize=(14, 8))
            for i in range(4):
                row = i // 2
                col = i % 2
                # Transpose to shape (39, 64) for correct time-on-x axis plotting
                mfcc_display = synthetic_features[i].T
                
                im = axes[row, col].imshow(mfcc_display, cmap='coolwarm', aspect='auto', origin='lower')
                axes[row, col].set_title(f"Synthesized Murmur Sample {i+1}", fontsize=11, fontweight='bold')
                axes[row, col].set_xlabel('Time frames')
                axes[row, col].set_ylabel('MFCC Coeffs')
                fig.colorbar(im, ax=axes[row, col])
                
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
            
            # Show a real abnormal sample for direct comparison
            try:
                X_train_path = os.path.join(base_dir, 'data', 'processed', 'X_train.npy')
                y_train_path = os.path.join(base_dir, 'data', 'processed', 'y_train.npy')
                
                if os.path.exists(X_train_path) and os.path.exists(y_train_path):
                    X_train = np.load(X_train_path)
                    y_train = np.load(y_train_path)
                    
                    abnormal_idx = np.where(y_train == 1)[0]
                    if len(abnormal_idx) > 0:
                        real_abnormal = X_train[abnormal_idx[0]].T # shape (39, 64)
                        
                        st.markdown("---")
                        st.subheader("Comparison: Real Abnormal Heart Sound Features")
                        st.write("This is a real abnormal segment extracted from the PhysioNet dataset. Note the similarities in patterns and frequency bands compared to the synthesized murmurs above:")
                        
                        fig2, ax2 = plt.subplots(figsize=(8, 4))
                        im2 = ax2.imshow(real_abnormal, cmap='coolwarm', aspect='auto', origin='lower')
                        ax2.set_title("Real Abnormal MFCC + Delta + Delta-Delta Heatmap", fontsize=12, fontweight='bold')
                        ax2.set_xlabel('Time frames')
                        ax2.set_ylabel('MFCC Coeffs')
                        fig2.colorbar(im2, ax=ax2)
                        st.pyplot(fig2)
                        plt.close()
            except Exception as e:
                st.warning(f"Could not load real abnormal sample for comparison: {e}")
                
        # Interactive Audio Generation Section
        st.markdown("---")
        st.subheader("🔊 Generate & Listen to Synthetic Abnormal Audio")
        st.write("Generate a synthetic abnormal heartbeat wave, listen to it, and download it to test the classifier in Tab 1.")
        
        if st.button("Generate & Reconstruct Audio", key="gan_reconstruct_button"):
            import tensorflow as tf
            import io
            
            # 1. Generate random noise vector
            noise = tf.random.normal([1, 100])
            
            # 2. Synthesize features
            # Predictions shape: (1, 64, 39) -> take first sample (64, 39)
            synth_features = generator(noise, training=False).numpy()[0]
            
            # 3. Discard derivative features, keeping the 13 base MFCCs -> shape (64, 13)
            synth_mfcc = synth_features[:, :13]
            
            # 4. Denormalize to raw scale
            avg_bounds = get_average_abnormal_bounds(base_dir)
            mfcc_denorm = ((synth_mfcc + 1.0) / 2.0) * (avg_bounds[1] - avg_bounds[0]) + avg_bounds[0]
            
            # Transpose to (13, 64) for librosa inverse function
            mfcc_denorm_t = mfcc_denorm.T
            
            # 5. Invert MFCCs to raw audio signal
            try:
                with st.spinner("Reconstructing audio signal using Griffin-Lim..."):
                    # mfcc_to_audio converts spectral coefficients back to raw time-domain audio
                    y_recon = librosa.feature.inverse.mfcc_to_audio(
                        mfcc_denorm_t,
                        sr=2000,
                        n_fft=256,
                        hop_length=80,
                        n_iter=150
                    )
                    
                    # Normalize audio to prevent clipping
                    max_val = np.max(np.abs(y_recon))
                    if max_val > 0:
                        y_recon = y_recon / max_val
                        
                    # 6. Write audio to an in-memory WAV buffer
                    wav_buf = io.BytesIO()
                    sf.write(wav_buf, y_recon, 2000, format='WAV', subtype='PCM_16')
                    wav_bytes = wav_buf.getvalue()
                    
                    # 7. Render audio player and download button
                    st.audio(wav_bytes, format="audio/wav")
                    st.download_button(
                        label="💾 Download Reconstructed WAV File",
                        data=wav_bytes,
                        file_name="synthetic_abnormal.wav",
                        mime="audio/wav"
                    )
                    st.success("Audio synthesized successfully! Download the file above, then upload it in the 'Classifier' tab to run inference.")
            except Exception as e:
                st.error(f"Error reconstructing audio: {e}")
