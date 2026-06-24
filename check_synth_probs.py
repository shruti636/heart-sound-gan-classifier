import os
import numpy as np
import tensorflow as tf

processed_dir = 'data/processed'
X_train = np.load(os.path.join(processed_dir, 'X_train.npy'))
y_train = np.load(os.path.join(processed_dir, 'y_train.npy'))

X_real_normal = X_train[y_train == 0]
X_real_abnormal = X_train[y_train == 1]

# Recreate synthetic data exactly as in train_classifier.py
gen_path = 'models/gan_generator.keras'
generator = tf.keras.models.load_model(gen_path)
diff = len(X_real_normal) - len(X_real_abnormal)
latent_dim = 100
noise = tf.random.normal([diff, latent_dim], seed=42)
X_synth = generator(noise, training=False).numpy()

model_gan = tf.keras.models.load_model('models/cnn_classifier_gan.keras')

probs_normal = model_gan.predict(X_real_normal).flatten()
probs_abnormal = model_gan.predict(X_real_abnormal).flatten()
probs_synth = model_gan.predict(X_synth).flatten()

print("--- GAN Model Predictions on Train Split ---")
print(f"Real Normal (N={len(X_real_normal)}): Mean prob = {probs_normal.mean():.4f}, Std = {probs_normal.std():.4f}, >= 0.5 count = {np.sum(probs_normal >= 0.5)}")
print(f"Real Abnormal (N={len(X_real_abnormal)}): Mean prob = {probs_abnormal.mean():.4f}, Std = {probs_abnormal.std():.4f}, >= 0.5 count = {np.sum(probs_abnormal >= 0.5)}")
print(f"Synthetic Abnormal (N={len(X_synth)}): Mean prob = {probs_synth.mean():.4f}, Std = {probs_synth.std():.4f}, >= 0.5 count = {np.sum(probs_synth >= 0.5)}")
