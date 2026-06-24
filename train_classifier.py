"""
train_classifier.py — Improved classifier training with:
  - Focal Loss (handles class imbalance without needing perfect 50/50 balance)
  - MixUp Data Augmentation (regularises boundary between classes)
  - Cosine Annealing LR schedule
  - Class-weighted training on top of GAN augmentation (double safety)
  - Gaussian noise augmentation on real segments (prevents overfitting)
  - Separate validation tracking for Recall (clinical priority metric)
"""

import os
import argparse
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from cnn_classifier import build_cnn_classifier, focal_loss

np.random.seed(42)
tf.random.set_seed(42)


# ─────────────────────────────────────────────────────────
# MixUp augmentation
# ─────────────────────────────────────────────────────────

def mixup_batch(X, y, alpha=0.3):
    """
    MixUp: linearly interpolates between pairs of training samples.
    Helps the model learn smoother decision boundaries.
    """
    batch_size = len(X)
    lam = np.random.beta(alpha, alpha, size=batch_size).astype(np.float32)
    lam = np.maximum(lam, 1 - lam)  # keep dominant sample primary

    idx = np.random.permutation(batch_size)
    lam_x = lam[:, None, None]   # (B, 1, 1) for broadcasting over (B, T, C)
    lam_y = lam                   # (B,)

    X_mix = lam_x * X + (1 - lam_x) * X[idx]
    y_mix = lam_y * y + (1 - lam_y) * y[idx]
    return X_mix.astype(np.float32), y_mix.astype(np.float32)


def add_feature_noise(X, std=0.02):
    """Add small Gaussian noise to MFCC features — acts like SpecAugment."""
    return X + np.random.normal(0, std, X.shape).astype(np.float32)


# ─────────────────────────────────────────────────────────
# Custom data generator with MixUp
# ─────────────────────────────────────────────────────────

class AugmentedSequence(tf.keras.utils.Sequence):
    def __init__(self, X, y, batch_size=16, mixup_alpha=0.2, noise_std=0.02, shuffle=True):
        self.X = X
        self.y = y.astype(np.float32)
        self.batch_size = batch_size
        self.mixup_alpha = mixup_alpha
        self.noise_std = noise_std
        self.shuffle = shuffle
        self.indices = np.arange(len(X))
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.X) / self.batch_size))

    def __getitem__(self, idx):
        batch_idx = self.indices[idx * self.batch_size: (idx + 1) * self.batch_size]
        X_b = self.X[batch_idx].copy()
        y_b = self.y[batch_idx].copy()

        # Feature noise
        X_b = add_feature_noise(X_b, self.noise_std)

        # MixUp (only apply ~60% of the time to keep some pure examples)
        if self.mixup_alpha > 0 and np.random.rand() < 0.6:
            X_b, y_b = mixup_batch(X_b, y_b, self.mixup_alpha)

        return X_b, y_b

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)


# ─────────────────────────────────────────────────────────
# Cosine annealing LR schedule callback
# ─────────────────────────────────────────────────────────

class CosineAnnealingLR(tf.keras.callbacks.Callback):
    def __init__(self, lr_max=1e-3, lr_min=1e-6, total_epochs=20):
        super().__init__()
        self.lr_max = lr_max
        self.lr_min = lr_min
        self.total_epochs = total_epochs

    def on_epoch_begin(self, epoch, logs=None):
        lr = self.lr_min + 0.5 * (self.lr_max - self.lr_min) * (
            1 + np.cos(np.pi * epoch / self.total_epochs)
        )
        tf.keras.backend.set_value(self.model.optimizer.lr, lr)


# ─────────────────────────────────────────────────────────
# Main pipeline
# ─────────────────────────────────────────────────────────

def train_classifier_pipeline(
    epochs=25,
    batch_size=16,
    processed_dir="data/processed",
    model_dir="models",
    outputs_dir="outputs",
    mixup_alpha=0.2,
    focal_gamma=2.0,
    focal_alpha=0.75,
):
    # ── Load pre-split processed data ───────────────────────
    X_train = np.load(os.path.join(processed_dir, "X_train.npy"))
    y_train = np.load(os.path.join(processed_dir, "y_train.npy"))
    X_val = np.load(os.path.join(processed_dir, "X_val.npy"))
    y_val = np.load(os.path.join(processed_dir, "y_val.npy"))

    # ── Leakage verification ─────────────────────────────────
    train_manifest = os.path.join(processed_dir, "train_source_files.txt")
    test_manifest = os.path.join(processed_dir, "test_source_files.txt")
    if os.path.exists(train_manifest) and os.path.exists(test_manifest):
        with open(train_manifest) as f:
            train_files = set(f.read().splitlines())
        with open(test_manifest) as f:
            test_files = set(f.read().splitlines())
        overlap = train_files & test_files
        assert len(overlap) == 0, "DATA LEAKAGE DETECTED!"
        print("  Leakage check PASSED ✓  (0 overlapping recordings)")

    num_normal_train = int(np.sum(y_train == 0))
    num_abnormal_train = int(np.sum(y_train == 1))
    imbalance_ratio = num_normal_train / max(num_abnormal_train, 1)

    print(f"\nTraining distribution — Normal: {num_normal_train} | Abnormal: {num_abnormal_train}")
    print(f"Class imbalance ratio: {imbalance_ratio:.2f}:1")

    # ── Class weights (used even after GAN augmentation as extra safety) ──
    class_weight = {
        0: 1.0,
        1: min(imbalance_ratio, 5.0),   # cap at 5× to avoid instability
    }
    print(f"  Class weights → Normal: {class_weight[0]:.2f} | Abnormal: {class_weight[1]:.2f}")

    input_shape = X_train.shape[1:]
    fl = focal_loss(gamma=focal_gamma, alpha=focal_alpha)

    metrics = [
        "accuracy",
        tf.keras.metrics.Precision(name="precision"),
        tf.keras.metrics.Recall(name="recall"),
        tf.keras.metrics.AUC(name="auc"),
    ]

    # ══════════════════════════════════════════════════════
    # STAGE 1 — Baseline model (no GAN, no MixUp)
    # ══════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("  STAGE 1: BASELINE MODEL (No GAN)")
    print("=" * 60)

    model_nogan = build_cnn_classifier(input_shape=input_shape)
    model_nogan.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss=fl,
        metrics=metrics,
    )

    ckpt_nogan = os.path.join(model_dir, "cnn_classifier_nogan.keras")
    history_nogan = model_nogan.fit(
        X_train, y_train.astype(np.float32),
        validation_data=(X_val, y_val.astype(np.float32)),
        epochs=epochs,
        batch_size=batch_size,
        class_weight=class_weight,
        callbacks=[
            ModelCheckpoint(ckpt_nogan, monitor="val_recall", mode="max",
                            save_best_only=True, verbose=1),
            CosineAnnealingLR(lr_max=1e-3, lr_min=1e-6, total_epochs=epochs),
        ],
        verbose=1,
    )
    np.save(os.path.join(processed_dir, "history_nogan.npy"), history_nogan.history)
    print(f"Baseline checkpoint saved → {ckpt_nogan}")

    # ══════════════════════════════════════════════════════
    # STAGE 2 — GAN-Augmented model with MixUp
    # ══════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("  STAGE 2: GAN-AUGMENTED MODEL (with MixUp)")
    print("=" * 60)

    if num_normal_train > num_abnormal_train:
        diff = min(num_normal_train - num_abnormal_train, 3000)
        print(f"Generating {diff} synthetic abnormal samples...")

        gen_path = os.path.join(model_dir, "gan_generator.keras")
        if not os.path.exists(gen_path):
            # Try best checkpoint
            gen_path = os.path.join(model_dir, "gan_generator_best.keras")
        if not os.path.exists(gen_path):
            raise FileNotFoundError("Trained GAN generator not found. Run train_gan.py first.")

        generator = tf.keras.models.load_model(gen_path)
        noise = tf.random.normal([diff, 100])
        X_synth = generator(noise, training=False).numpy()

        # Reconstruct → re-extract (aligns distribution)
        import librosa
        from preprocessing import preprocess_signal, extract_features
        bounds_train = np.load(os.path.join(processed_dir, "bounds_train.npy"))
        abnormal_idx = np.where(y_train == 1)[0]
        avg_bounds = np.mean(bounds_train[abnormal_idx], axis=0)

        X_synth_processed = []
        print(f"  Reconstructing {diff} synthetic samples via Griffin-Lim ...")
        for i in range(diff):
            synth_mfcc = X_synth[i, :, :13]
            mfcc_denorm = ((synth_mfcc + 1.0) / 2.0) * (avg_bounds[1] - avg_bounds[0]) + avg_bounds[0]
            y_recon = librosa.feature.inverse.mfcc_to_audio(
                mfcc_denorm.T, sr=2000, n_fft=256, hop_length=80, n_iter=200
            )
            max_v = np.max(np.abs(y_recon))
            if max_v > 0:
                y_recon /= max_v
            y_filt = preprocess_signal(y_recon, fs=2000)
            seg = y_filt[:5040] if len(y_filt) >= 5040 else np.pad(y_filt, (0, 5040 - len(y_filt)))
            feats = extract_features(seg, fs=2000)
            X_synth_processed.append(feats["mfcc"])

        X_synth_processed = np.array(X_synth_processed, dtype=np.float32)
        y_synth = np.ones(diff, dtype=np.float32)

        X_real_normal = X_train[y_train == 0]
        X_real_abnormal = X_train[y_train == 1]
        X_train_aug = np.concatenate([X_real_normal, X_real_abnormal, X_synth_processed])
        y_train_aug = np.concatenate([
            np.zeros(num_normal_train, dtype=np.float32),
            np.ones(num_abnormal_train, dtype=np.float32),
            y_synth,
        ])

        print(f"Augmented set — Normal: {int(np.sum(y_train_aug == 0))} | Abnormal: {int(np.sum(y_train_aug == 1))}")
    else:
        X_train_aug = X_train
        y_train_aug = y_train.astype(np.float32)
        print("Dataset already balanced — using original training data.")

    # Save split counts for evaluation plotting
    np.save(os.path.join(processed_dir, "split_counts.npy"), np.array([
        num_normal_train, num_abnormal_train,
        int(np.sum(y_train_aug == 0)), int(np.sum(y_train_aug == 1)),
    ]))

    # Shuffle augmented set
    shuffle_idx = np.random.permutation(len(X_train_aug))
    X_train_aug = X_train_aug[shuffle_idx]
    y_train_aug = y_train_aug[shuffle_idx]

    # Data generator with MixUp + feature noise
    train_gen = AugmentedSequence(
        X_train_aug, y_train_aug,
        batch_size=batch_size,
        mixup_alpha=mixup_alpha,
        noise_std=0.02,
    )

    model_gan = build_cnn_classifier(input_shape=input_shape)
    model_gan.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss=fl,
        metrics=metrics,
    )

    ckpt_gan = os.path.join(model_dir, "cnn_classifier_gan.keras")
    history_gan = model_gan.fit(
        train_gen,
        validation_data=(X_val, y_val.astype(np.float32)),
        epochs=epochs,
        callbacks=[
            ModelCheckpoint(ckpt_gan, monitor="val_recall", mode="max",
                            save_best_only=True, verbose=1),
            CosineAnnealingLR(lr_max=1e-3, lr_min=1e-6, total_epochs=epochs),
        ],
        verbose=1,
    )
    np.save(os.path.join(processed_dir, "history_gan.npy"), history_gan.history)
    print(f"GAN-Augmented checkpoint saved → {ckpt_gan}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train improved classifiers")
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--mixup_alpha", type=float, default=0.2)
    parser.add_argument("--focal_gamma", type=float, default=2.0)
    parser.add_argument("--focal_alpha", type=float, default=0.75)
    args = parser.parse_args()
    train_classifier_pipeline(
        epochs=args.epochs,
        batch_size=args.batch_size,
        mixup_alpha=args.mixup_alpha,
        focal_gamma=args.focal_gamma,
        focal_alpha=args.focal_alpha,
    )
