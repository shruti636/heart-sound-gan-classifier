"""Small DCGAN models for heart-sound feature matrices.

The GAN learns to create abnormal feature matrices:
64 time frames and however many features preprocessing.py extracts.
This file is intentionally short so beginners can trace every layer.
"""

import tensorflow as tf
from tensorflow.keras import Model, layers


def conv_block(x, filters, kernel_size=4, strides=2, dropout=0.25):
    """One discriminator block: Conv1D -> LeakyReLU -> Dropout."""
    x = layers.Conv1D(filters, kernel_size, strides=strides, padding="same")(x)
    x = layers.LeakyReLU(negative_slope=0.2)(x)
    return layers.Dropout(dropout)(x)


def upsample_block(x, filters):
    """One generator block: Conv1DTranspose -> BatchNorm -> LeakyReLU."""
    x = layers.Conv1DTranspose(filters, 4, strides=2, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    return layers.LeakyReLU(negative_slope=0.2)(x)


def build_generator(latent_dim=100, feature_dim=71):
    """Build a compact generator: random noise -> synthetic feature matrix."""
    noise = layers.Input(shape=(latent_dim,), name="noise")

    x = layers.Dense(8 * 128, use_bias=False)(noise)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(negative_slope=0.2)(x)
    x = layers.Reshape((8, 128))(x)

    x = upsample_block(x, 128)  # 8 -> 16
    x = upsample_block(x, 64)   # 16 -> 32
    x = upsample_block(x, 32)   # 32 -> 64

    out = layers.Conv1D(feature_dim, 3, padding="same", activation="tanh", name="features")(x)
    return Model(noise, out, name="Generator")


def build_discriminator(input_shape=(64, 71)):
    """Build a compact discriminator: feature matrix -> real/fake probability."""
    features = layers.Input(shape=input_shape, name="features")

    x = conv_block(features, 32)
    x = conv_block(x, 64)
    x = conv_block(x, 128)
    x = layers.Flatten()(x)
    x = layers.Dense(64)(x)
    x = layers.LeakyReLU(negative_slope=0.2)(x)
    out = layers.Dense(1, activation="sigmoid", name="real_fake")(x)

    return Model(features, out, name="Discriminator")


if __name__ == "__main__":
    gen = build_generator()
    disc = build_discriminator()
    gen.summary()
    disc.summary()

    fake = gen(tf.random.normal((2, 100)))
    score = disc(fake)
    print("Generator output:", fake.shape)
    print("Discriminator output:", score.shape)
