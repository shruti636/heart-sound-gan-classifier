"""Train baseline and GAN-augmented heart-sound classifiers.

The classifier compares two settings:
1. Baseline: train only on real data.
2. GAN-augmented: add generated abnormal samples to balance the training set.
"""

import argparse
import os

import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

from cnn_classifier import build_cnn_classifier


SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)


def class_weights(y):
    """Return simple inverse-frequency class weights."""
    normal = max(np.sum(y == 0), 1)
    abnormal = max(np.sum(y == 1), 1)
    total = normal + abnormal
    return {
        0: float(total / (2 * normal)),
        1: float(total / (2 * abnormal)),
    }


def compile_classifier(model):
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
            tf.keras.metrics.AUC(name="auc"),
        ],
    )
    return model


def fit_and_save(model, X_train, y_train, X_val, y_val, checkpoint_path, epochs, batch_size):
    callbacks = [
        ModelCheckpoint(checkpoint_path, monitor="val_auc", mode="max", save_best_only=True, verbose=1),
        EarlyStopping(monitor="val_auc", mode="max", patience=6, restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-5),
    ]
    return model.fit(
        X_train,
        y_train.astype(np.float32),
        validation_data=(X_val, y_val.astype(np.float32)),
        epochs=epochs,
        batch_size=batch_size,
        class_weight=class_weights(y_train),
        callbacks=callbacks,
        verbose=1,
    )


def check_recording_leakage(processed_dir):
    """Verify that no recording appears in more than one split."""
    split_files = {
        "train": os.path.join(processed_dir, "sources_train.npy"),
        "val": os.path.join(processed_dir, "sources_val.npy"),
        "test": os.path.join(processed_dir, "sources_test.npy"),
    }
    if not all(os.path.exists(path) for path in split_files.values()):
        print("Source arrays not found. Run preprocessing.py again for strict leakage checks.")
        return

    sources = {name: set(np.load(path).astype(str)) for name, path in split_files.items()}
    assert not (sources["train"] & sources["val"]), "Leakage: train and val share recordings."
    assert not (sources["train"] & sources["test"]), "Leakage: train and test share recordings."
    assert not (sources["val"] & sources["test"]), "Leakage: val and test share recordings."
    print("Recording leakage check passed.")


def make_gan_augmented_training_set(X_train, y_train, model_dir, latent_dim=100, augmentation_ratio=0.5):
    """Generate abnormal samples for part of the class gap.

    Full balancing can make the classifier over-predict Abnormal. A ratio of
    0.5 is a safer beginner default: it helps imbalance but reduces false
    positives compared with forcing a perfectly balanced set.
    """
    normal_count = int(np.sum(y_train == 0))
    abnormal_count = int(np.sum(y_train == 1))
    full_gap = max(normal_count - abnormal_count, 0)
    needed = int(full_gap * augmentation_ratio)

    if needed == 0:
        print("Training set is already balanced. No GAN samples added.")
        return X_train, y_train.astype(np.float32)

    gen_path = os.path.join(model_dir, "gan_generator.keras")
    if not os.path.exists(gen_path):
        raise FileNotFoundError("GAN generator not found. Run train_gan.py before GAN augmentation.")

    print(f"Generating {needed} abnormal feature samples ({augmentation_ratio:.2f} of the class gap).")
    generator = tf.keras.models.load_model(gen_path, compile=False)
    if generator.output_shape[-1] != X_train.shape[-1]:
        raise ValueError(
            "GAN generator feature size does not match X_train. "
            "Run preprocessing.py and train_gan.py again before train_classifier.py."
        )
    noise = tf.random.normal((needed, latent_dim))
    X_synth = generator(noise, training=False).numpy().astype(np.float32)
    y_synth = np.ones(needed, dtype=np.float32)

    X_aug = np.concatenate([X_train, X_synth], axis=0)
    y_aug = np.concatenate([y_train.astype(np.float32), y_synth], axis=0)
    order = np.random.permutation(len(X_aug))
    return X_aug[order], y_aug[order]


def train_classifier_pipeline(
    epochs=25,
    batch_size=16,
    processed_dir="data/processed",
    model_dir="models",
    latent_dim=100,
    augmentation_ratio=0.5,
):
    os.makedirs(model_dir, exist_ok=True)

    X_train = np.load(os.path.join(processed_dir, "X_train.npy")).astype(np.float32)
    y_train = np.load(os.path.join(processed_dir, "y_train.npy"))
    X_val = np.load(os.path.join(processed_dir, "X_val.npy")).astype(np.float32)
    y_val = np.load(os.path.join(processed_dir, "y_val.npy"))

    check_recording_leakage(processed_dir)

    print(f"Train: {X_train.shape} | normal={np.sum(y_train == 0)} abnormal={np.sum(y_train == 1)}")
    print(f"Val:   {X_val.shape} | normal={np.sum(y_val == 0)} abnormal={np.sum(y_val == 1)}")

    input_shape = X_train.shape[1:]

    print("\nStage 1: baseline classifier")
    baseline = compile_classifier(build_cnn_classifier(input_shape))
    hist_base = fit_and_save(
        baseline,
        X_train,
        y_train,
        X_val,
        y_val,
        os.path.join(model_dir, "cnn_classifier_nogan.keras"),
        epochs,
        batch_size,
    )
    np.save(os.path.join(processed_dir, "history_nogan.npy"), hist_base.history)

    print("\nStage 2: GAN-augmented classifier")
    X_aug, y_aug = make_gan_augmented_training_set(
        X_train,
        y_train,
        model_dir,
        latent_dim,
        augmentation_ratio=augmentation_ratio,
    )
    np.save(
        os.path.join(processed_dir, "split_counts.npy"),
        np.array([np.sum(y_train == 0), np.sum(y_train == 1), np.sum(y_aug == 0), np.sum(y_aug == 1)]),
    )

    gan_model = compile_classifier(build_cnn_classifier(input_shape))
    hist_gan = fit_and_save(
        gan_model,
        X_aug,
        y_aug,
        X_val,
        y_val,
        os.path.join(model_dir, "cnn_classifier_gan.keras"),
        epochs,
        batch_size,
    )
    np.save(os.path.join(processed_dir, "history_gan.npy"), hist_gan.history)
    print("Classifier training complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train baseline and GAN-augmented classifiers.")
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--latent_dim", type=int, default=100)
    parser.add_argument("--augmentation_ratio", type=float, default=0.5)
    args = parser.parse_args()
    train_classifier_pipeline(
        args.epochs,
        args.batch_size,
        latent_dim=args.latent_dim,
        augmentation_ratio=args.augmentation_ratio,
    )
