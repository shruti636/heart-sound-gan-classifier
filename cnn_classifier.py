"""
cnn_classifier.py — Improved hybrid classifier with:
  - Multi-scale Conv1D blocks (kernel sizes 3, 5, 7 in parallel)
  - Squeeze-and-Excitation channel attention after each conv block
  - Bidirectional GRU (faster convergence than LSTM, same capacity)
  - Focal Loss support (built-in function returned alongside model)
  - Deeper network with better regularisation (SpatialDropout1D)
"""

import tensorflow as tf
from tensorflow.keras import layers, Model
import numpy as np


# ─────────────────────────────────────────────────────────
# Custom blocks
# ─────────────────────────────────────────────────────────

class SqueezeExcite(layers.Layer):
    """Channel-wise attention — focus on the most informative feature maps."""

    def __init__(self, filters, ratio=8, **kwargs):
        super().__init__(**kwargs)
        self.gap = layers.GlobalAveragePooling1D()
        self.d1 = layers.Dense(max(filters // ratio, 1), activation="relu")
        self.d2 = layers.Dense(filters, activation="sigmoid")

    def call(self, x):
        # x: (B, T, C)
        scale = self.gap(x)           # (B, C)
        scale = self.d1(scale)
        scale = self.d2(scale)        # (B, C)
        scale = tf.expand_dims(scale, 1)  # (B, 1, C)
        return x * scale


class MultiScaleConvBlock(layers.Layer):
    """
    Parallel Conv1D with kernel sizes 3, 5, 7 — captures both fine
    (high-freq murmur bursts) and coarse (cardiac cycle) temporal patterns.
    """

    def __init__(self, filters, **kwargs):
        super().__init__(**kwargs)
        per = max(filters // 3, 1)
        self.c3 = layers.Conv1D(per, 3, padding="same", activation="relu", use_bias=False)
        self.c5 = layers.Conv1D(per, 5, padding="same", activation="relu", use_bias=False)
        self.c7 = layers.Conv1D(per, 7, padding="same", activation="relu", use_bias=False)
        self.bn = layers.BatchNormalization()
        self.se = SqueezeExcite(per * 3)

    def call(self, x, training=None):
        x3 = self.c3(x)
        x5 = self.c5(x)
        x7 = self.c7(x)
        out = layers.concatenate([x3, x5, x7])
        out = self.bn(out, training=training)
        out = self.se(out)
        return out


# ─────────────────────────────────────────────────────────
# Focal Loss — addresses class imbalance during training
# ─────────────────────────────────────────────────────────

def focal_loss(gamma=2.0, alpha=0.75):
    """
    Binary focal loss.
    alpha > 0.5 up-weights the minority (Abnormal) class.
    gamma > 0 down-weights easy negatives.
    """
    def loss_fn(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
        bce = -y_true * tf.math.log(y_pred) - (1 - y_true) * tf.math.log(1 - y_pred)
        p_t = y_true * y_pred + (1 - y_true) * (1 - y_pred)
        alpha_t = y_true * alpha + (1 - y_true) * (1 - alpha)
        fl = alpha_t * tf.pow(1 - p_t, gamma) * bce
        return tf.reduce_mean(fl)

    loss_fn.__name__ = "focal_loss"
    return loss_fn


# ─────────────────────────────────────────────────────────
# Main classifier
# ─────────────────────────────────────────────────────────

def build_cnn_classifier(input_shape=(64, 39)):
    """
    Improved hybrid Multi-Scale Conv1D + SE + Bidirectional GRU classifier.

    Architecture:
      Input (64, 39)
        ↓
      MultiScaleConvBlock(64)  + MaxPool → (32, 192)
        ↓ SpatialDropout1D
      MultiScaleConvBlock(128) + MaxPool → (16, 384)
        ↓ SpatialDropout1D
      MultiScaleConvBlock(64)            → (16, 192)
        ↓
      Bidirectional GRU(64) → (128,)
        ↓ Dropout
      Dense(64, relu) → Dense(1, sigmoid)
    """
    inputs = layers.Input(shape=input_shape, name="mfcc_input")

    # ── Block 1 ─────────────────────────────────────────────
    x = MultiScaleConvBlock(64, name="ms_conv1")(inputs)
    x = layers.MaxPooling1D(pool_size=2)(x)
    x = layers.SpatialDropout1D(0.1)(x)

    # ── Block 2 ─────────────────────────────────────────────
    x = MultiScaleConvBlock(128, name="ms_conv2")(x)
    x = layers.MaxPooling1D(pool_size=2)(x)
    x = layers.SpatialDropout1D(0.15)(x)

    # ── Block 3 (no pooling — keep temporal resolution for RNN) ──
    x = MultiScaleConvBlock(64, name="ms_conv3")(x)
    x = layers.SpatialDropout1D(0.1)(x)

    # ── Bidirectional GRU ────────────────────────────────────
    x = layers.Bidirectional(layers.GRU(64, return_sequences=False), name="bi_gru")(x)
    x = layers.Dropout(0.3)(x)

    # ── Classification head ──────────────────────────────────
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(1, activation="sigmoid", name="prediction")(x)

    model = Model(inputs, outputs, name="MultiScale_SE_BiGRU_Classifier")
    return model


if __name__ == "__main__":
    model = build_cnn_classifier((64, 39))
    model.summary()

    fl = focal_loss(gamma=2.0, alpha=0.75)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss=fl,
        metrics=["accuracy"],
    )
    print("\nImproved Multi-Scale SE-BiGRU Classifier compiled successfully ✓")
