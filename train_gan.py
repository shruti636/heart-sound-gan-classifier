"""Train a small GAN on abnormal heart-sound feature matrices.

This is a plain DCGAN-style training loop:
1. The discriminator learns real abnormal features vs generated features.
2. The generator learns to fool the discriminator.
3. The trained generator is saved for classifier augmentation.
"""

import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam

from gan_model import build_discriminator, build_generator


SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)


class SimpleGAN:
    """Small wrapper that keeps generator/discriminator training together."""

    def __init__(self, input_shape=(64, 71), latent_dim=100, learning_rate=2e-4):
        self.latent_dim = latent_dim
        self.discriminator = build_discriminator(input_shape)
        self.generator = build_generator(latent_dim, feature_dim=input_shape[-1])
        self.loss_fn = tf.keras.losses.BinaryCrossentropy()
        self.gen_opt = Adam(learning_rate, beta_1=0.5)
        self.disc_opt = Adam(learning_rate, beta_1=0.5)

    @tf.function
    def train_step(self, real_batch):
        batch_size = tf.shape(real_batch)[0]
        noise = tf.random.normal((batch_size, self.latent_dim))

        with tf.GradientTape() as disc_tape:
            fake_batch = self.generator(noise, training=True)
            real_pred = self.discriminator(real_batch, training=True)
            fake_pred = self.discriminator(fake_batch, training=True)

            real_loss = self.loss_fn(tf.ones_like(real_pred) * 0.9, real_pred)
            fake_loss = self.loss_fn(tf.zeros_like(fake_pred), fake_pred)
            disc_loss = real_loss + fake_loss

        disc_grads = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)
        self.disc_opt.apply_gradients(zip(disc_grads, self.discriminator.trainable_variables))

        noise = tf.random.normal((batch_size, self.latent_dim))
        with tf.GradientTape() as gen_tape:
            fake_batch = self.generator(noise, training=True)
            fake_pred = self.discriminator(fake_batch, training=False)
            gen_loss = self.loss_fn(tf.ones_like(fake_pred), fake_pred)

        gen_grads = gen_tape.gradient(gen_loss, self.generator.trainable_variables)
        self.gen_opt.apply_gradients(zip(gen_grads, self.generator.trainable_variables))

        return gen_loss, disc_loss

    def generate(self, n_samples):
        noise = tf.random.normal((n_samples, self.latent_dim))
        return self.generator(noise, training=False).numpy()


def save_sample_plot(samples, path):
    """Save a small grid of generated feature heatmaps."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    n = min(len(samples), 6)
    fig, axes = plt.subplots(1, n, figsize=(3 * n, 3))
    axes = np.atleast_1d(axes)
    for i, ax in enumerate(axes):
        ax.imshow(samples[i].T, aspect="auto", origin="lower", cmap="coolwarm", vmin=-1, vmax=1)
        ax.set_title(f"Sample {i + 1}")
        ax.set_xlabel("Time")
        ax.set_ylabel("Feature")
    plt.tight_layout()
    plt.savefig(path, dpi=140)
    plt.close()


def plot_losses(gen_losses, disc_losses, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.figure(figsize=(7, 4))
    plt.plot(gen_losses, label="Generator")
    plt.plot(disc_losses, label="Discriminator")
    plt.title("GAN training loss")
    plt.xlabel("Epoch")
    plt.ylabel("Binary cross-entropy")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=140)
    plt.close()


def train_gan_pipeline(
    epochs=50,
    batch_size=32,
    latent_dim=100,
    processed_dir="data/processed",
    model_dir="models",
    outputs_dir="outputs",
):
    X_train = np.load(os.path.join(processed_dir, "X_train.npy")).astype(np.float32)
    y_train = np.load(os.path.join(processed_dir, "y_train.npy"))
    X_abnormal = X_train[y_train == 1]

    if len(X_abnormal) == 0:
        raise ValueError("No abnormal samples found. The GAN needs minority-class examples.")

    dataset = (
        tf.data.Dataset.from_tensor_slices(X_abnormal)
        .shuffle(len(X_abnormal), seed=SEED, reshuffle_each_iteration=True)
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )

    gan = SimpleGAN(input_shape=X_train.shape[1:], latent_dim=latent_dim)
    gen_losses, disc_losses = [], []

    print(f"Training GAN on {len(X_abnormal)} abnormal feature segments.")
    for epoch in range(1, epochs + 1):
        epoch_g, epoch_d = [], []
        for real_batch in dataset:
            g_loss, d_loss = gan.train_step(real_batch)
            epoch_g.append(float(g_loss))
            epoch_d.append(float(d_loss))

        gen_losses.append(float(np.mean(epoch_g)))
        disc_losses.append(float(np.mean(epoch_d)))

        if epoch == 1 or epoch % 10 == 0 or epoch == epochs:
            print(f"Epoch {epoch:03d}/{epochs} | G: {gen_losses[-1]:.4f} | D: {disc_losses[-1]:.4f}")
            save_sample_plot(
                gan.generate(6),
                os.path.join(outputs_dir, "mfcc_plots", f"epoch_{epoch:03d}.png"),
            )

    os.makedirs(model_dir, exist_ok=True)
    gan.generator.save(os.path.join(model_dir, "gan_generator.keras"))
    gan.discriminator.save(os.path.join(model_dir, "gan_discriminator.keras"))

    synth = gan.generate(50)
    synth_dir = os.path.join(outputs_dir, "synthetic_features")
    os.makedirs(synth_dir, exist_ok=True)
    np.save(os.path.join(synth_dir, "synth_abnormal_mfcc.npy"), synth)

    plot_losses(gen_losses, disc_losses, os.path.join(outputs_dir, "plots", "gan_loss.png"))
    print("GAN training complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a beginner-friendly DCGAN.")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--latent_dim", type=int, default=100)
    args = parser.parse_args()
    train_gan_pipeline(args.epochs, args.batch_size, args.latent_dim)
