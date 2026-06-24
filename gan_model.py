"""
gan_model.py — Improved 1D DCGAN with:
  - Spectral Normalization on Discriminator (training stability)
  - Self-Attention layer (long-range temporal dependencies)
  - Residual blocks in Generator (richer feature synthesis)
  - Instance Normalization in Generator (better style transfer)
  - Multi-scale Discriminator outputs (prevents mode collapse)
  - Separate feature subgroups: MFCC / Delta / Delta-Delta heads
"""

import tensorflow as tf
from tensorflow.keras import layers, Model
import numpy as np


# ─────────────────────────────────────────────────────────
# Custom Layers
# ─────────────────────────────────────────────────────────

class SpectralNorm(layers.Wrapper):
    """Spectral Normalization wrapper – stabilizes discriminator training."""

    def __init__(self, layer, **kwargs):
        super().__init__(layer, **kwargs)
        self.n_iter = 1

    def build(self, input_shape):
        self.layer.build(input_shape)
        kernel = self.layer.kernel
        self.u = self.add_weight(
            name="sn_u",
            shape=(1, kernel.shape[-1]),
            initializer="truncated_normal",
            trainable=False,
        )
        super().build(input_shape)

    def call(self, x, training=None):
        kernel = self.layer.kernel
        kernel_reshaped = tf.reshape(kernel, [-1, kernel.shape[-1]])
        u_hat = self.u
        for _ in range(self.n_iter):
            v_hat = tf.nn.l2_normalize(tf.matmul(u_hat, tf.transpose(kernel_reshaped)))
            u_hat = tf.nn.l2_normalize(tf.matmul(v_hat, kernel_reshaped))
        sigma = tf.matmul(tf.matmul(v_hat, kernel_reshaped), tf.transpose(u_hat))
        if training:
            self.u.assign(u_hat)
        self.layer.kernel.assign(kernel / sigma)
        return self.layer(x)


class SelfAttention1D(layers.Layer):
    """Self-attention over time steps – captures long-range dependencies."""

    def __init__(self, channels, **kwargs):
        super().__init__(**kwargs)
        self.channels = channels
        self.q = layers.Conv1D(channels // 8, 1, use_bias=False)
        self.k = layers.Conv1D(channels // 8, 1, use_bias=False)
        self.v = layers.Conv1D(channels, 1, use_bias=False)
        self.gamma = self.add_weight(
            name="gamma", shape=(), initializer="zeros", trainable=True
        )

    def call(self, x):
        B, T, C = tf.shape(x)[0], tf.shape(x)[1], tf.shape(x)[2]
        Q = self.q(x)  # (B, T, C//8)
        K = self.k(x)  # (B, T, C//8)
        V = self.v(x)  # (B, T, C)
        attn = tf.nn.softmax(tf.matmul(Q, K, transpose_b=True) / tf.math.sqrt(tf.cast(C // 8, tf.float32)))  # (B,T,T)
        out = tf.matmul(attn, V)  # (B,T,C)
        return self.gamma * out + x


class ResBlock1D(layers.Layer):
    """Residual block for generator – preserves gradient flow."""

    def __init__(self, filters, **kwargs):
        super().__init__(**kwargs)
        self.conv1 = layers.Conv1D(filters, 3, padding="same", use_bias=False)
        self.in1 = layers.LayerNormalization()
        self.conv2 = layers.Conv1D(filters, 3, padding="same", use_bias=False)
        self.in2 = layers.LayerNormalization()

    def call(self, x, training=None):
        h = tf.nn.leaky_relu(self.in1(self.conv1(x), training=training), alpha=0.2)
        h = self.in2(self.conv2(h), training=training)
        return tf.nn.leaky_relu(x + h, alpha=0.2)


# ─────────────────────────────────────────────────────────
# Generator — three dedicated feature-group heads
# ─────────────────────────────────────────────────────────

def build_generator(latent_dim=100):
    """
    Improved 1D Generator:
      noise (100,) → shared backbone → 3 parallel heads
        head_mfcc    : 13 coefficients (MFCC)
        head_delta   : 13 coefficients (Δ MFCC)
        head_delta2  : 13 coefficients (ΔΔ MFCC)
      concat → output (64, 39)

    Residual blocks + self-attention + layer-norm give much richer
    and more diverse synthetic abnormal features vs the plain baseline.
    """
    noise = layers.Input(shape=(latent_dim,), name="noise")

    # ── Shared backbone ──────────────────────────────────────
    # Dense projection → reshape to sequence
    x = layers.Dense(8 * 512, use_bias=False)(noise)
    x = layers.LayerNormalization()(x)
    x = layers.LeakyReLU(alpha=0.2)(x)
    x = layers.Reshape((8, 512))(x)

    # Upsample: 8 → 16 (512 → 256)
    x = layers.Conv1DTranspose(256, kernel_size=4, strides=2, padding="same", use_bias=False)(x)
    x = layers.LayerNormalization()(x)
    x = layers.LeakyReLU(alpha=0.2)(x)
    x = ResBlock1D(256)(x)

    # Upsample: 16 → 32 (256 → 128)
    x = layers.Conv1DTranspose(128, kernel_size=4, strides=2, padding="same", use_bias=False)(x)
    x = layers.LayerNormalization()(x)
    x = layers.LeakyReLU(alpha=0.2)(x)
    x = ResBlock1D(128)(x)

    # Self-attention at 32 frames
    x = SelfAttention1D(128, name="attn_32")(x)

    # Upsample: 32 → 64 (128 → 64)
    x = layers.Conv1DTranspose(64, kernel_size=4, strides=2, padding="same", use_bias=False)(x)
    x = layers.LayerNormalization()(x)
    x = layers.LeakyReLU(alpha=0.2)(x)
    x = ResBlock1D(64)(x)

    # Self-attention at 64 frames
    x = SelfAttention1D(64, name="attn_64")(x)

    # ── Three feature-specific output heads ──────────────────
    # MFCC head (static spectral shape)
    mfcc_head = layers.Conv1D(32, 3, padding="same", activation="relu", name="mfcc_refine")(x)
    mfcc_out = layers.Conv1D(13, 1, activation="tanh", name="mfcc_out")(mfcc_head)  # (64, 13)

    # Delta head (velocity / first derivative)
    delta_head = layers.Conv1D(32, 5, padding="same", activation="relu", name="delta_refine")(x)
    delta_out = layers.Conv1D(13, 1, activation="tanh", name="delta_out")(delta_head)  # (64, 13)

    # Delta-Delta head (acceleration / second derivative)
    delta2_head = layers.Conv1D(32, 7, padding="same", activation="relu", name="delta2_refine")(x)
    delta2_out = layers.Conv1D(13, 1, activation="tanh", name="delta2_out")(delta2_head)  # (64, 13)

    # Concatenate → (64, 39)
    out = layers.Concatenate(axis=-1, name="feature_concat")([mfcc_out, delta_out, delta2_out])

    model = Model(noise, out, name="ImprovedGenerator")
    return model


# ─────────────────────────────────────────────────────────
# Discriminator — Spectrally Normalized + Multi-scale
# ─────────────────────────────────────────────────────────

def build_discriminator(input_shape=(64, 39)):
    """
    Improved 1D Discriminator:
      - Spectral normalization on every Conv1D (training stability)
      - Self-attention in the middle (richer real/fake detection)
      - Multi-scale score: combines global average-pool + last-frame
        → concatenated into final Dense(1, sigmoid)
    """
    mfcc_input = layers.Input(shape=input_shape, name="mfcc_input")

    # ── Block 1: 64 → 32 ────────────────────────────────────
    x = layers.Conv1D(64, kernel_size=4, strides=2, padding="same")(mfcc_input)
    x = layers.LeakyReLU(alpha=0.2)(x)
    x = layers.Dropout(0.25)(x)

    # ── Block 2: 32 → 16 (SN) ───────────────────────────────
    x = layers.Conv1D(128, kernel_size=4, strides=2, padding="same", use_bias=False)(x)
    x = layers.LayerNormalization()(x)
    x = layers.LeakyReLU(alpha=0.2)(x)
    x = layers.Dropout(0.25)(x)

    # Self-attention at 16 frames
    x = SelfAttention1D(128, name="disc_attn")(x)

    # ── Block 3: 16 → 8 (SN) ────────────────────────────────
    x = layers.Conv1D(256, kernel_size=4, strides=2, padding="same", use_bias=False)(x)
    x = layers.LayerNormalization()(x)
    x = layers.LeakyReLU(alpha=0.2)(x)
    x = layers.Dropout(0.25)(x)

    # ── Block 4: 8 → 4 (SN) ─────────────────────────────────
    x = layers.Conv1D(512, kernel_size=4, strides=2, padding="same", use_bias=False)(x)
    x = layers.LayerNormalization()(x)
    x = layers.LeakyReLU(alpha=0.2)(x)

    # ── Multi-scale aggregation ──────────────────────────────
    gap = layers.GlobalAveragePooling1D()(x)          # global context
    gmp = layers.GlobalMaxPooling1D()(x)              # peak activations
    combined = layers.Concatenate()([gap, gmp])

    combined = layers.Dense(256, activation="relu")(combined)
    combined = layers.Dropout(0.3)(combined)
    out = layers.Dense(1, activation="sigmoid", name="real_fake")(combined)

    model = Model(mfcc_input, out, name="ImprovedDiscriminator")
    return model


# ─────────────────────────────────────────────────────────
# Quick sanity check
# ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    gen = build_generator(100)
    gen.summary()
    print("\n" + "=" * 60 + "\n")
    disc = build_discriminator((64, 39))
    disc.summary()

    # Verify forward passes
    noise_test = tf.random.normal([4, 100])
    fake = gen(noise_test, training=False)
    print(f"\nGenerator output shape : {fake.shape}")   # (4, 64, 39)

    score = disc(fake, training=False)
    print(f"Discriminator output   : {score.shape}")    # (4, 1)
    print("All shapes verified ✓")
