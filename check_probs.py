import os
import numpy as np
import tensorflow as tf

processed_dir = 'data/processed'
X_val = np.load(os.path.join(processed_dir, 'X_val.npy'))
y_val = np.load(os.path.join(processed_dir, 'y_val.npy'))

print("Val shape:", X_val.shape)
print("Val labels distribution: Normal:", np.sum(y_val == 0), "Abnormal:", np.sum(y_val == 1))

model_nogan = tf.keras.models.load_model('models/cnn_classifier_nogan.keras')
model_gan = tf.keras.models.load_model('models/cnn_classifier_gan.keras')

probs_nogan = model_nogan.predict(X_val).flatten()
probs_gan = model_gan.predict(X_val).flatten()

print("\n--- Baseline Model (No GAN) ---")
print("Min prob:", probs_nogan.min())
print("Max prob:", probs_nogan.max())
print("Mean prob:", probs_nogan.mean())
print("Std prob:", probs_nogan.std())
print("Number of predictions >= 0.5:", np.sum(probs_nogan >= 0.5))

print("\n--- GAN-Augmented Model ---")
print("Min prob:", probs_gan.min())
print("Max prob:", probs_gan.max())
print("Mean prob:", probs_gan.mean())
print("Std prob:", probs_gan.std())
print("Number of predictions >= 0.5:", np.sum(probs_gan >= 0.5))

# Let's check predictions on some training data too
X_train = np.load(os.path.join(processed_dir, 'X_train.npy'))
y_train = np.load(os.path.join(processed_dir, 'y_train.npy'))

train_probs_gan = model_gan.predict(X_train).flatten()
print("\n--- GAN Model on Real Train ---")
print("Min prob:", train_probs_gan.min())
print("Max prob:", train_probs_gan.max())
print("Mean prob:", train_probs_gan.mean())
print("Number of predictions >= 0.5:", np.sum(train_probs_gan >= 0.5))
print("Actual Abnormal in Train:", np.sum(y_train == 1))
