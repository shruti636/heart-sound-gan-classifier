import os
import numpy as np
import tensorflow as tf
import librosa
from preprocessing import preprocess_signal, extract_features

# Load generator
gen_path = 'models/gan_generator.keras'
generator = tf.keras.models.load_model(gen_path)

# Generate one sample
noise = tf.random.normal([1, 100])
synth_features = generator(noise, training=False).numpy()[0] # shape (64, 39)

print("Original synthetic features shape:", synth_features.shape)

# Load training data bounds for denormalization
processed_dir = 'data/processed'
X_train = np.load(os.path.join(processed_dir, 'X_train.npy'))
y_train = np.load(os.path.join(processed_dir, 'y_train.npy'))
bounds_train = np.load(os.path.join(processed_dir, 'bounds_train.npy'))

abnormal_idx = np.where(y_train == 1)[0]
avg_bounds = np.mean(bounds_train[abnormal_idx], axis=0)

# Extract base MFCCs (first 13)
synth_mfcc = synth_features[:, :13]

# Denormalize
mfcc_denorm = ((synth_mfcc + 1.0) / 2.0) * (avg_bounds[1] - avg_bounds[0]) + avg_bounds[0]
mfcc_denorm_t = mfcc_denorm.T

# Reconstruct audio
y_recon = librosa.feature.inverse.mfcc_to_audio(
    mfcc_denorm_t,
    sr=2000,
    n_fft=256,
    hop_length=80,
    n_iter=150
)

# Normalize audio
max_val = np.max(np.abs(y_recon))
if max_val > 0:
    y_recon = y_recon / max_val

# Preprocess and re-extract features
y_filt = preprocess_signal(y_recon, fs=2000)

if len(y_filt) >= 5040:
    seg = y_filt[:5040]
else:
    seg = np.pad(y_filt, (0, max(0, 5040 - len(y_filt))))

features = extract_features(seg, fs=2000)
re_extracted = features['mfcc']

print("Re-extracted features shape:", re_extracted.shape)
print("Original synth features mean:", synth_features.mean(), "std:", synth_features.std())
print("Re-extracted features mean:", re_extracted.mean(), "std:", re_extracted.std())
print("Success!")
