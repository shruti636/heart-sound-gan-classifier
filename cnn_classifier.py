"""Beginner-friendly CNN + BiGRU classifier for heart-sound features."""

import tensorflow as tf
from tensorflow.keras import Model, layers


def focal_loss(gamma=2.0, alpha=0.75):
    """Optional focal loss kept for older experiments and notebooks.

    The simplified training script now uses binary cross-entropy by default,
    because it is easier to understand and loads without custom objects.
    """

    def loss_fn(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
        bce = -y_true * tf.math.log(y_pred) - (1.0 - y_true) * tf.math.log(1.0 - y_pred)
        p_t = y_true * y_pred + (1.0 - y_true) * (1.0 - y_pred)
        alpha_t = y_true * alpha + (1.0 - y_true) * (1.0 - alpha)
        return tf.reduce_mean(alpha_t * tf.pow(1.0 - p_t, gamma) * bce)

    loss_fn.__name__ = "focal_loss"
    return loss_fn


def build_cnn_classifier(input_shape=(64, 71)):
    """Build a compact CNN + BiGRU classifier.

    Input shape:
        (64, 71) = 64 time frames, 39 MFCC features + 32 log-mel features.

    Output:
        One probability. Values >= 0.5 mean Abnormal.
    """
    inputs = layers.Input(shape=input_shape, name="heart_sound_features")

    x = layers.Conv1D(48, 5, padding="same", activation="relu")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(2)(x)
    x = layers.Dropout(0.20)(x)

    x = layers.Conv1D(96, 3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(2)(x)
    x = layers.Dropout(0.25)(x)

    x = layers.Bidirectional(layers.GRU(48, return_sequences=False), name="bigru")(x)
    x = layers.Dropout(0.30)(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(0.30)(x)
    outputs = layers.Dense(1, activation="sigmoid", name="prediction")(x)

    return Model(inputs, outputs, name="HeartSound_CNN_BiGRU")


if __name__ == "__main__":
    model = build_cnn_classifier()
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    model.summary()
