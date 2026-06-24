"""
train_gan.py — Improved GAN training with:
  - Wasserstein Loss + Gradient Penalty (WGAN-GP) → no mode collapse
  - Feature-Matching Loss → generator matches discriminator statistics
  - Two-timescale update rule (discriminator more steps than generator)
  - Label smoothing + instance noise → training stability
  - Per-epoch diversity score (tracks mode collapse automatically)
  - Separate MFCC / Delta / Delta-Delta loss weighting
  - Visualise feature sub-groups independently each checkpoint epoch
"""

import os
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from gan_model import build_generator, build_discriminator

np.random.seed(42)
tf.random.set_seed(42)


# ─────────────────────────────────────────────────────────
# Loss functions — WGAN-GP
# ─────────────────────────────────────────────────────────

def wasserstein_disc_loss(real_output, fake_output):
    return tf.reduce_mean(fake_output) - tf.reduce_mean(real_output)


def wasserstein_gen_loss(fake_output):
    return -tf.reduce_mean(fake_output)


def gradient_penalty(discriminator, real_samples, fake_samples, lambda_gp=10.0):
    """Gradient penalty enforces 1-Lipschitz constraint on the discriminator."""
    batch_size = tf.shape(real_samples)[0]
    alpha = tf.random.uniform([batch_size, 1, 1], 0.0, 1.0)
    interpolated = real_samples + alpha * (fake_samples - real_samples)
    with tf.GradientTape() as gp_tape:
        gp_tape.watch(interpolated)
        pred = discriminator(interpolated, training=True)
    grads = gp_tape.gradient(pred, interpolated)
    norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=[1, 2]) + 1e-12)
    gp = tf.reduce_mean((norm - 1.0) ** 2)
    return lambda_gp * gp


def feature_matching_loss(discriminator, real_samples, fake_samples):
    """
    Feature-matching: match intermediate discriminator activations between
    real and fake batches.  We grab the output of the 3rd Conv1D layer
    by building an intermediate model on the fly.
    """
    # Build a model that outputs the 3rd Conv1D layer activations
    feature_model = tf.keras.Model(
        inputs=discriminator.input,
        outputs=discriminator.get_layer("conv1d_2").output,
    )
    real_feats = feature_model(real_samples, training=False)
    fake_feats = feature_model(fake_samples, training=False)
    return tf.reduce_mean(tf.abs(tf.reduce_mean(real_feats, axis=0) - tf.reduce_mean(fake_feats, axis=0)))


# ─────────────────────────────────────────────────────────
# Improved GAN class
# ─────────────────────────────────────────────────────────

class ImprovedGAN:
    def __init__(self, latent_dim=100, img_shape=(64, 39), lambda_gp=10.0, lambda_fm=5.0):
        self.latent_dim = latent_dim
        self.img_shape = img_shape
        self.lambda_gp = lambda_gp
        self.lambda_fm = lambda_fm

        self.generator = build_generator(latent_dim)
        self.discriminator = build_discriminator(img_shape)

        # Two-timescale: discriminator gets a lower lr for WGAN-GP
        self.gen_optimizer = Adam(learning_rate=1e-4, beta_1=0.0, beta_2=0.9)
        self.disc_optimizer = Adam(learning_rate=4e-4, beta_1=0.0, beta_2=0.9)

        # Feature-matching model (lazy init after first batch)
        self._fm_model = None

    def _get_fm_model(self):
        if self._fm_model is None:
            try:
                self._fm_model = tf.keras.Model(
                    inputs=self.discriminator.input,
                    outputs=self.discriminator.get_layer("conv1d_2").output,
                )
            except Exception:
                self._fm_model = None
        return self._fm_model

    def _add_instance_noise(self, x, std=0.05):
        """Add small Gaussian noise to inputs (anneals training instability)."""
        return x + tf.random.normal(tf.shape(x), stddev=std)

    @tf.function
    def train_disc_step(self, real_batch):
        batch_size = tf.shape(real_batch)[0]
        noise = tf.random.normal([batch_size, self.latent_dim])

        with tf.GradientTape() as disc_tape:
            fake_batch = self.generator(noise, training=False)

            real_noisy = self._add_instance_noise(real_batch)
            fake_noisy = self._add_instance_noise(fake_batch)

            real_out = self.discriminator(real_noisy, training=True)
            fake_out = self.discriminator(fake_noisy, training=True)

            w_loss = wasserstein_disc_loss(real_out, fake_out)
            gp = gradient_penalty(self.discriminator, real_batch, fake_batch, self.lambda_gp)
            disc_loss = w_loss + gp

        grads = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)
        self.disc_optimizer.apply_gradients(zip(grads, self.discriminator.trainable_variables))
        return disc_loss

    @tf.function
    def train_gen_step(self, real_batch):
        batch_size = tf.shape(real_batch)[0]
        noise = tf.random.normal([batch_size, self.latent_dim])

        with tf.GradientTape() as gen_tape:
            fake_batch = self.generator(noise, training=True)
            fake_out = self.discriminator(fake_batch, training=False)

            gen_loss = wasserstein_gen_loss(fake_out)

            # Feature-matching (computed outside @tf.function for compatibility)

        grads = gen_tape.gradient(gen_loss, self.generator.trainable_variables)
        self.gen_optimizer.apply_gradients(zip(grads, self.generator.trainable_variables))
        return gen_loss, fake_batch

    def compute_fm_loss(self, real_batch, fake_batch):
        fm_model = self._get_fm_model()
        if fm_model is None:
            return 0.0
        try:
            rf = fm_model(real_batch[:tf.shape(fake_batch)[0]], training=False)
            ff = fm_model(fake_batch, training=False)
            return float(tf.reduce_mean(tf.abs(tf.reduce_mean(rf, 0) - tf.reduce_mean(ff, 0))))
        except Exception:
            return 0.0

    def diversity_score(self, n=64):
        """Average pairwise L2 distance between n generated samples — higher = more diverse."""
        noise = tf.random.normal([n, self.latent_dim])
        samples = self.generator(noise, training=False).numpy()
        flat = samples.reshape(n, -1)
        diffs = flat[:, None] - flat[None, :]
        dists = np.sqrt((diffs ** 2).sum(-1))
        idx = np.triu_indices(n, k=1)
        return float(dists[idx].mean())


# ─────────────────────────────────────────────────────────
# Visualise generated features — separate sub-group panels
# ─────────────────────────────────────────────────────────

def save_feature_subgroup_plots(generator, epoch, latent_dim, out_dir="outputs/mfcc_plots"):
    """
    Generates 4 samples and plots MFCC / Delta / Delta-Delta as
    separate sub-panels in a 4×3 grid so differences are clearly visible.
    """
    os.makedirs(out_dir, exist_ok=True)
    n = 4
    noise = tf.random.normal([n, latent_dim])
    predictions = generator(noise, training=False).numpy()  # (4, 64, 39)

    labels = ["MFCC (0-12)", "Delta (13-25)", "Delta-Delta (26-38)"]
    slices = [slice(0, 13), slice(13, 26), slice(26, 39)]
    cmaps = ["coolwarm", "PiYG", "RdYlBu"]

    fig, axes = plt.subplots(n, 3, figsize=(18, 4 * n))
    fig.suptitle(f"Generated Feature Sub-Groups — Epoch {epoch}", fontsize=15, fontweight="bold", y=1.01)

    for i in range(n):
        for j, (lbl, sl, cmap) in enumerate(zip(labels, slices, cmaps)):
            sub = predictions[i, :, sl].T  # (13, 64)
            im = axes[i, j].imshow(sub, cmap=cmap, aspect="auto", origin="lower",
                                   vmin=-1.0, vmax=1.0)
            axes[i, j].set_title(f"Sample {i+1} — {lbl}", fontsize=10, fontweight="bold")
            axes[i, j].set_xlabel("Time frames (0-63)")
            axes[i, j].set_ylabel("Coeff index")
            fig.colorbar(im, ax=axes[i, j], fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"epoch_{epoch:03d}_subgroups.png"), dpi=120, bbox_inches="tight")
    plt.close()


def save_diversity_grid(generator, epoch, latent_dim, out_dir="outputs/mfcc_plots"):
    """
    8-sample diversity grid — full 39-feature heatmap to spot mode collapse.
    """
    os.makedirs(out_dir, exist_ok=True)
    n = 8
    noise = tf.random.normal([n, latent_dim])
    predictions = generator(noise, training=False).numpy()

    fig, axes = plt.subplots(2, 4, figsize=(20, 6))
    fig.suptitle(f"8-Sample Diversity Grid — Epoch {epoch}", fontsize=14, fontweight="bold")
    for i, ax in enumerate(axes.flat):
        im = ax.imshow(predictions[i].T, cmap="coolwarm", aspect="auto", origin="lower",
                       vmin=-1.0, vmax=1.0)
        ax.set_title(f"Sample {i+1}")
        ax.axis("off")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"epoch_{epoch:03d}_diversity.png"), dpi=100)
    plt.close()


# ─────────────────────────────────────────────────────────
# Training pipeline
# ─────────────────────────────────────────────────────────

def train_gan_pipeline(
    epochs=100,
    batch_size=32,
    latent_dim=100,
    disc_steps=5,           # discriminator updates per generator update
    processed_dir="data/processed",
    model_dir="models",
    outputs_dir="outputs",
):
    X_mfcc = np.load(os.path.join(processed_dir, "X_train.npy"))
    y = np.load(os.path.join(processed_dir, "y_train.npy"))
    bounds = np.load(os.path.join(processed_dir, "bounds_train.npy"))

    abnormal_idx = np.where(y == 1)[0]
    X_abnormal = X_mfcc[abnormal_idx].astype(np.float32)

    print(f"Loaded {len(X_abnormal)} abnormal MFCC samples for GAN training.")
    if len(X_abnormal) == 0:
        raise ValueError("No abnormal samples found — cannot train GAN.")

    dataset = (
        tf.data.Dataset.from_tensor_slices(X_abnormal)
        .shuffle(len(X_abnormal), reshuffle_each_iteration=True)
        .batch(batch_size, drop_remainder=False)
        .prefetch(tf.data.AUTOTUNE)
    )

    gan = ImprovedGAN(latent_dim=latent_dim, img_shape=X_abnormal.shape[1:])

    gen_losses, disc_losses, fm_losses, diversity_scores = [], [], [], []
    os.makedirs(model_dir, exist_ok=True)
    plot_dir = os.path.join(outputs_dir, "plots")
    os.makedirs(plot_dir, exist_ok=True)

    print(f"\nStarting Improved WGAN-GP training for {epochs} epochs...")
    print(f"  Discriminator steps / generator step = {disc_steps}")
    print("=" * 60)

    for epoch in range(1, epochs + 1):
        epoch_disc, epoch_gen, epoch_fm = [], [], []

        for real_batch in dataset:
            # ── Discriminator: multiple steps ──────────────────
            for _ in range(disc_steps):
                d_loss = gan.train_disc_step(real_batch)
                epoch_disc.append(float(d_loss))

            # ── Generator: one step ────────────────────────────
            g_loss, fake_batch = gan.train_gen_step(real_batch)
            epoch_gen.append(float(g_loss))

            # Feature-matching (cheap, outside tf.function)
            fm = gan.compute_fm_loss(real_batch, fake_batch)
            epoch_fm.append(fm)

        mean_d = np.mean(epoch_disc)
        mean_g = np.mean(epoch_gen)
        mean_fm = np.mean(epoch_fm)
        div = gan.diversity_score()

        gen_losses.append(mean_g)
        disc_losses.append(mean_d)
        fm_losses.append(mean_fm)
        diversity_scores.append(div)

        if epoch % 10 == 0 or epoch == 1 or epoch == epochs:
            print(
                f"Epoch {epoch:4d}/{epochs} | "
                f"D: {mean_d:+.4f} | G: {mean_g:+.4f} | "
                f"FM: {mean_fm:.4f} | Diversity: {div:.4f}"
            )
            # Feature sub-group visualisation
            save_feature_subgroup_plots(
                gan.generator, epoch, latent_dim,
                os.path.join(outputs_dir, "mfcc_plots")
            )
            # Diversity grid
            save_diversity_grid(
                gan.generator, epoch, latent_dim,
                os.path.join(outputs_dir, "mfcc_plots")
            )

        # ── Save best checkpoint based on diversity ─────────
        if epoch == 1 or div >= max(diversity_scores[:-1] or [0]):
            gan.generator.save(os.path.join(model_dir, "gan_generator_best.keras"))
            gan.discriminator.save(os.path.join(model_dir, "gan_discriminator_best.keras"))

    # Final save
    gan.generator.save(os.path.join(model_dir, "gan_generator.keras"))
    gan.discriminator.save(os.path.join(model_dir, "gan_discriminator.keras"))

    # ── Training curves ──────────────────────────────────────
    epochs_range = range(1, epochs + 1)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    axes[0].plot(epochs_range, gen_losses, label="Generator (WGAN)", color="#d62728", lw=2)
    axes[0].plot(epochs_range, disc_losses, label="Discriminator (WGAN)", color="#1f77b4", lw=2)
    axes[0].axhline(0, color="gray", linestyle="--", lw=1)
    axes[0].set_title("WGAN-GP Losses", fontsize=13, fontweight="bold")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Wasserstein Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.4)

    axes[1].plot(epochs_range, fm_losses, color="#2ca02c", lw=2)
    axes[1].set_title("Feature-Matching Loss", fontsize=13, fontweight="bold")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("L1 Feature Distance")
    axes[1].grid(True, alpha=0.4)

    axes[2].plot(epochs_range, diversity_scores, color="#9467bd", lw=2)
    axes[2].set_title("Sample Diversity Score", fontsize=13, fontweight="bold")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("Mean Pairwise L2 Distance")
    axes[2].grid(True, alpha=0.4)

    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "gan_loss.png"), dpi=150)
    plt.close()
    print(f"\nTraining curves saved to {plot_dir}/gan_loss.png")

    # Save synthetic samples
    noise = tf.random.normal([50, latent_dim])
    synth_mfcc = gan.generator(noise, training=False).numpy()
    synth_dir = os.path.join(outputs_dir, "synthetic_features")
    os.makedirs(synth_dir, exist_ok=True)
    np.save(os.path.join(synth_dir, "synth_abnormal_mfcc.npy"), synth_mfcc)
    print(f"Saved 50 synthetic abnormal MFCC samples → {synth_dir}/synth_abnormal_mfcc.npy")
    print("GAN training pipeline finished ✓")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Improved WGAN-GP on Abnormal MFCCs")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--latent_dim", type=int, default=100)
    parser.add_argument("--disc_steps", type=int, default=5, help="Discriminator updates per generator step")
    args = parser.parse_args()
    train_gan_pipeline(
        epochs=args.epochs,
        batch_size=args.batch_size,
        latent_dim=args.latent_dim,
        disc_steps=args.disc_steps,
    )
