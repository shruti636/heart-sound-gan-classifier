"""Train a Progressive Wasserstein GAN (PWGAN) on abnormal heart-sound feature matrices.

The pipeline implements:
1. WGAN-GP loss function and gradient penalty.
2. Progressive training schedule: grows resolution from 16 to 32 to 64.
3. Final weight transfer to non-progressive models for seamless serialization.
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam

from gan_model import (
    build_generator,
    build_discriminator,
    build_final_generator,
    build_final_discriminator,
    copy_matching_weights,
)

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)


def get_resolution_and_alpha(epoch, total_epochs=100):
    """Calculates target resolution and fade-in coefficient alpha for progressive growth."""
    stage_len = total_epochs // 5
    if stage_len < 1:
        stage_len = 1
        
    if epoch <= stage_len:
        return 16, 1.0
    elif epoch <= 2 * stage_len:
        alpha = (epoch - stage_len) / float(stage_len)
        return 32, alpha
    elif epoch <= 3 * stage_len:
        return 32, 1.0
    elif epoch <= 4 * stage_len:
        alpha = (epoch - 3 * stage_len) / float(stage_len)
        return 64, alpha
    else:
        return 64, 1.0


def gradient_penalty(critic, real_samples, fake_samples, resolution, alpha):
    """Computes gradient penalty for WGAN-GP enforcing the 1-Lipschitz constraint."""
    batch_size = tf.shape(real_samples)[0]
    epsilon = tf.random.uniform([batch_size, 1, 1], 0.0, 1.0)
    interpolated = real_samples * epsilon + fake_samples * (1.0 - epsilon)
    
    with tf.GradientTape() as tape:
        tape.watch(interpolated)
        pred = critic(interpolated, resolution=resolution, alpha=alpha, training=True)
        
    grads = tape.gradient(pred, [interpolated])[0]
    norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=[1, 2]) + 1e-12)
    gp = tf.reduce_mean((norm - 1.0) ** 2)
    return gp


def train_critic_step(critic, generator, real_batch, resolution, alpha, latent_dim, opt_critic, gp_weight=10.0):
    """Performs one training step for the critic (WGAN-GP)."""
    batch_size = tf.shape(real_batch)[0]
    # Downsample real batch to current resolution using Average Pooling
    if resolution == 16:
        real_batch_down = tf.keras.layers.AveragePooling1D(pool_size=4)(real_batch)
    elif resolution == 32:
        real_batch_down = tf.keras.layers.AveragePooling1D(pool_size=2)(real_batch)
    else:
        real_batch_down = real_batch
        
    noise = tf.random.normal((batch_size, latent_dim))
    
    with tf.GradientTape() as tape:
        fake_batch = generator(noise, resolution=resolution, alpha=alpha, training=True)
        real_pred = critic(real_batch_down, resolution=resolution, alpha=alpha, training=True)
        fake_pred = critic(fake_batch, resolution=resolution, alpha=alpha, training=True)
        
        c_loss = tf.reduce_mean(fake_pred) - tf.reduce_mean(real_pred)
        gp = gradient_penalty(critic, real_batch_down, fake_batch, resolution, alpha)
        total_loss = c_loss + gp_weight * gp
        
    grads = tape.gradient(total_loss, critic.trainable_variables)
    grads_and_vars = [(g, v) for g, v in zip(grads, critic.trainable_variables) if g is not None]
    opt_critic.apply_gradients(grads_and_vars)
    return c_loss, gp


def train_generator_step(critic, generator, batch_size, resolution, alpha, latent_dim, opt_gen):
    """Performs one training step for the generator (WGAN-GP)."""
    noise = tf.random.normal((batch_size, latent_dim))
    with tf.GradientTape() as tape:
        fake_batch = generator(noise, resolution=resolution, alpha=alpha, training=True)
        fake_pred = critic(fake_batch, resolution=resolution, alpha=alpha, training=True)
        g_loss = -tf.reduce_mean(fake_pred)
        
    grads = tape.gradient(g_loss, generator.trainable_variables)
    grads_and_vars = [(g, v) for g, v in zip(grads, generator.trainable_variables) if g is not None]
    opt_gen.apply_gradients(grads_and_vars)
    return g_loss


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
    plt.plot(disc_losses, label="Critic (WGAN)")
    plt.title("Progressive GAN training loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss value")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=140)
    plt.close()


def train_gan_pipeline(
    epochs=100,
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

    feature_dim = X_train.shape[-1]
    dataset = (
        tf.data.Dataset.from_tensor_slices(X_abnormal)
        .shuffle(len(X_abnormal), seed=SEED, reshuffle_each_iteration=True)
        .batch(batch_size, drop_remainder=True)
        .prefetch(tf.data.AUTOTUNE)
    )

    # Instantiate PWGAN models
    generator = build_generator(latent_dim, feature_dim)
    critic = build_discriminator(X_train.shape[1:])

    # Optimizers adapted for progressive WGAN-GP training
    gen_opt = None
    critic_opt = None
    prev_res = None

    gen_losses, critic_losses = [], []
    print(f"Training PWGAN on {len(X_abnormal)} abnormal feature segments.")

    for epoch in range(1, epochs + 1):
        res, alpha = get_resolution_and_alpha(epoch, epochs)
        if res != prev_res:
            print(f"\n--- Transitioning to Resolution {res} (Epoch {epoch}) | Recreating Optimizers ---")
            gen_opt = Adam(learning_rate=1e-4, beta_1=0.0, beta_2=0.9)
            critic_opt = Adam(learning_rate=1e-4, beta_1=0.0, beta_2=0.9)
            prev_res = res
            
        epoch_g, epoch_c = [], []
        
        for real_batch in dataset:
            # WGAN-GP: train critic multiple times?
            # 1 to 5 steps of critic updates per generator update
            c_loss, gp = train_critic_step(
                critic, generator, real_batch, res, alpha, latent_dim, critic_opt, gp_weight=10.0
            )
            g_loss = train_generator_step(
                critic, generator, batch_size, res, alpha, latent_dim, gen_opt
            )
            epoch_g.append(float(g_loss))
            epoch_c.append(float(c_loss))

        gen_losses.append(float(np.mean(epoch_g)))
        critic_losses.append(float(np.mean(epoch_c)))

        if epoch == 1 or epoch % 10 == 0 or epoch == epochs:
            print(f"Epoch {epoch:03d}/{epochs} (Res={res}, Alpha={alpha:.2f}) | G: {gen_losses[-1]:.4f} | C: {critic_losses[-1]:.4f}")
            # Generate and save a sample plot of current resolution
            noise = tf.random.normal((6, latent_dim))
            synth_samples = generator(noise, resolution=res, alpha=alpha, training=False).numpy()
            save_sample_plot(
                synth_samples,
                os.path.join(outputs_dir, "mfcc_plots", f"epoch_{epoch:03d}.png"),
            )

    # Post-processing: transfer weights to final non-progressive Functional models for saving
    print("\nCopying weights to final non-progressive Keras models...")
    final_generator = build_final_generator(latent_dim, feature_dim)
    final_critic = build_final_discriminator(X_train.shape[1:])
    
    copy_matching_weights(generator, final_generator)
    copy_matching_weights(critic, final_critic)

    os.makedirs(model_dir, exist_ok=True)
    final_generator.save(os.path.join(model_dir, "gan_generator.keras"))
    final_critic.save(os.path.join(model_dir, "gan_discriminator.keras"))
    print("Functional models saved successfully in models/ directory.")

    # Generate test set of final samples
    noise = tf.random.normal((50, latent_dim))
    synth = final_generator(noise, training=False).numpy()
    synth_dir = os.path.join(outputs_dir, "synthetic_features")
    os.makedirs(synth_dir, exist_ok=True)
    np.save(os.path.join(synth_dir, "synth_abnormal_mfcc.npy"), synth)

    plot_losses(gen_losses, critic_losses, os.path.join(outputs_dir, "plots", "gan_loss.png"))
    print("PWGAN training complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a Progressive Wasserstein GAN (PWGAN).")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--latent_dim", type=int, default=100)
    args = parser.parse_args()
    train_gan_pipeline(args.epochs, args.batch_size, args.latent_dim)
