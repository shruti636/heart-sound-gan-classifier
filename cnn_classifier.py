"""Beginner-friendly CNN + BiGRU classifier for heart-sound features."""

import tensorflow as tf
from tensorflow.keras import Model, layers
from tensorflow.keras import regularizers


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


def residual_conv_block(x, filters, kernel_size, dropout):
    """Small residual Conv1D block.

    Residual connections help deeper models train without making the code hard
    to follow: the block learns a correction on top of a simple shortcut.
    """
    shortcut = layers.Conv1D(filters, 1, padding="same")(x)

    x = layers.SeparableConv1D(
        filters,
        kernel_size,
        padding="same",
        depthwise_regularizer=regularizers.l2(1e-4),
        pointwise_regularizer=regularizers.l2(1e-4),
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    x = layers.SeparableConv1D(
        filters,
        kernel_size,
        padding="same",
        depthwise_regularizer=regularizers.l2(1e-4),
        pointwise_regularizer=regularizers.l2(1e-4),
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.Add()([shortcut, x])
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling1D(2)(x)
    return layers.SpatialDropout1D(dropout)(x)


def attention_pooling(x):
    """Let the model learn which time frames matter most."""
    weights = layers.Dense(1, activation="tanh")(x)
    weights = layers.Softmax(axis=1, name="time_attention")(weights)
    weighted = layers.Multiply()([x, weights])
    return layers.GlobalAveragePooling1D(name="attention_pool")(weighted)


def build_cnn_classifier(input_shape=(64, 71)):
    """Build a regularized CNN + BiGRU + attention classifier.

    Input shape:
        (64, 71) = 64 time frames, 39 MFCC features + 32 log-mel features.

    Output:
        One probability. Values >= 0.5 mean Abnormal.
    """
    inputs = layers.Input(shape=input_shape, name="heart_sound_features")

    x = residual_conv_block(inputs, filters=64, kernel_size=5, dropout=0.20)
    x = residual_conv_block(x, filters=96, kernel_size=3, dropout=0.25)

    x = layers.Bidirectional(layers.GRU(64, return_sequences=True), name="bigru")(x)
    attention_context = attention_pooling(x)
    average_context = layers.GlobalAveragePooling1D()(x)
    x = layers.Concatenate()([attention_context, average_context])

    x = layers.Dense(96, activation="relu", kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.Dropout(0.35)(x)
    x = layers.Dense(32, activation="relu", kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.Dropout(0.20)(x)
    outputs = layers.Dense(1, activation="sigmoid", name="prediction")(x)

    return Model(inputs, outputs, name="HeartSound_ResidualCNN_BiGRU_Attention")


def build_light_cnn_classifier(input_shape=(64, 71)):
    """
    Build a lightweight Conv1D classifier to prevent overfitting on small datasets.
    No BiGRU, no attention pooling.
    """
    inputs = layers.Input(shape=input_shape, name="heart_sound_features")
    
    # Conv Block 1
    x = layers.Conv1D(32, 5, padding="same", kernel_regularizer=regularizers.l2(1e-4))(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling1D(2)(x)
    x = layers.Dropout(0.25)(x)
    
    # Conv Block 2
    x = layers.Conv1D(64, 3, padding="same", kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling1D(2)(x)
    x = layers.Dropout(0.30)(x)
    
    # Flatten and Dense
    x = layers.Flatten()(x)
    x = layers.Dense(32, activation="relu", kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.Dropout(0.35)(x)
    outputs = layers.Dense(1, activation="sigmoid", name="prediction")(x)
    
    return Model(inputs, outputs, name="HeartSound_LightweightCNN")


if __name__ == "__main__":
    model = build_cnn_classifier()
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    model.summary()
