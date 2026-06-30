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

from cnn_classifier import build_cnn_classifier, build_light_cnn_classifier


SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)


def spec_augment(x, num_time_masks=2, time_mask_max=8, num_freq_masks=2, freq_mask_max=6):
    """
    Apply SpecAugment to a batch of feature matrices of shape (N, time_frames, feature_dim).
    """
    augmented = x.copy()
    N, time_frames, feature_dim = x.shape
    for i in range(N):
        # Time masks
        for _ in range(num_time_masks):
            t = np.random.randint(1, time_mask_max + 1)
            t0 = np.random.randint(0, time_frames - t)
            augmented[i, t0 : t0 + t, :] = -1.0  # -1.0 acts as a mask value
        # Frequency masks
        for _ in range(num_freq_masks):
            f = np.random.randint(1, freq_mask_max + 1)
            f0 = np.random.randint(0, feature_dim - f)
            augmented[i, :, f0 : f0 + f] = -1.0
    return augmented


def class_weights(y):
    """Return simple inverse-frequency class weights."""
    normal = max(np.sum(y == 0), 1)
    abnormal = max(np.sum(y == 1), 1)
    total = normal + abnormal
    return {
        0: float(total / (2 * normal)),
        1: float(total / (2 * abnormal)),
    }


def compile_classifier(model, use_focal_loss=False):
    try:
        optimizer = tf.keras.optimizers.AdamW(learning_rate=7e-4, weight_decay=1e-4)
    except AttributeError:
        optimizer = tf.keras.optimizers.Adam(learning_rate=7e-4)

    if use_focal_loss:
        from cnn_classifier import focal_loss
        loss = focal_loss(gamma=2.0, alpha=0.75)
        print("Using Focal Loss for model compilation.")
    else:
        loss = tf.keras.losses.BinaryCrossentropy(label_smoothing=0.03)

    model.compile(
        optimizer=optimizer,
        loss=loss,
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
        EarlyStopping(monitor="val_auc", mode="max", patience=8, restore_best_weights=True),
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


def make_gan_augmented_training_set(
    X_train,
    y_train,
    model_dir,
    latent_dim=100,
    augmentation_ratio=0.5,
    filter_synthetic=True,
    use_wavegan=False,
    filter_method="classifier",
    pool_multiplier=4,
):
    """Generate abnormal samples for part of the class gap using vanilla GAN or WaveGAN."""
    normal_count = int(np.sum(y_train == 0))
    abnormal_count = int(np.sum(y_train == 1))
    full_gap = max(normal_count - abnormal_count, 0)
    needed = int(full_gap * augmentation_ratio)

    if needed == 0:
        print("Training set is already balanced. No GAN samples added.")
        return X_train, y_train.astype(np.float32)

    pool_size = max(needed * pool_multiplier, needed) if filter_synthetic else needed

    from preprocessing import extract_features, FEATURE_KEY

    if use_wavegan:
        gen_path = os.path.join(model_dir, "wavegan_generator.keras")
        if not os.path.exists(gen_path):
            raise FileNotFoundError("WaveGAN generator not found. Run train_wavegan.py before WaveGAN augmentation.")
        
        print(f"Generating {pool_size} candidate abnormal waveforms using WaveGAN...")
        generator = tf.keras.models.load_model(gen_path, compile=False)
        
        # Generate in batches to prevent OOM on CPU
        gen_batch_size = 512
        synth_waves_list = []
        for start_idx in range(0, pool_size, gen_batch_size):
            batch_pool = min(gen_batch_size, pool_size - start_idx)
            noise = tf.random.normal((batch_pool, latent_dim))
            batch_waves = generator(noise, training=False).numpy()
            synth_waves_list.append(batch_waves)
        synth_waves = np.concatenate(synth_waves_list, axis=0)
        
        # Crop waves to segment length of 5040 or 2000 dynamically based on X_train frames
        segment_len = 2000 if X_train.shape[1] == 25 else 5040
        synth_waves_cropped = synth_waves[:, :segment_len, 0]
        
        # If filtering is required using the discriminator (critic)
        if filter_synthetic and filter_method in ["discriminator", "both"]:
            disc_path = os.path.join(model_dir, "wavegan_critic.keras")
            if os.path.exists(disc_path):
                print("Scoring waveforms using WaveGAN Critic...")
                critic = tf.keras.models.load_model(disc_path, compile=False)
                scores = critic.predict(synth_waves, verbose=0).reshape(-1)
                keep_idx = np.argsort(scores)[::-1]
                synth_waves_cropped = synth_waves_cropped[keep_idx]
            else:
                print("Warning: WaveGAN critic not found. Skipping critic filtering.")
                
        # Now convert the candidate waveforms to features
        print("Extracting features from generated waveforms...")
        X_candidates = []
        for i in range(len(synth_waves_cropped)):
            wave_out = synth_waves_cropped[i]
            # Normalize to [-1, 1] range
            max_val = np.max(np.abs(wave_out))
            if max_val > 0:
                wave_out = wave_out / max_val
            features = extract_features(wave_out, fs=2000)
            X_candidates.append(features[FEATURE_KEY])
        X_candidates = np.array(X_candidates, dtype=np.float32)
        
    else:
        # Vanilla GAN
        gen_path = os.path.join(model_dir, "gan_generator.keras")
        if not os.path.exists(gen_path):
            raise FileNotFoundError("GAN generator not found. Run train_gan.py before GAN augmentation.")
        
        print(f"Generating {pool_size} candidate abnormal features using Vanilla GAN...")
        generator = tf.keras.models.load_model(gen_path, compile=False)
        
        # Generate in batches to prevent OOM
        gen_batch_size = 512
        X_candidates_list = []
        for start_idx in range(0, pool_size, gen_batch_size):
            batch_pool = min(gen_batch_size, pool_size - start_idx)
            noise = tf.random.normal((batch_pool, latent_dim))
            batch_candidates = generator(noise, training=False).numpy()
            X_candidates_list.append(batch_candidates)
        X_candidates = np.concatenate(X_candidates_list, axis=0).astype(np.float32)
        
        if filter_synthetic and filter_method in ["discriminator", "both"]:
            disc_path = os.path.join(model_dir, "gan_discriminator.keras")
            if os.path.exists(disc_path):
                print("Scoring features using GAN Discriminator...")
                discriminator = tf.keras.models.load_model(disc_path, compile=False)
                scores = discriminator.predict(X_candidates, verbose=0).reshape(-1)
                keep_idx = np.argsort(scores)[::-1]
                X_candidates = X_candidates[keep_idx]
            else:
                print("Warning: GAN discriminator not found. Skipping discriminator filtering.")
                
    # Filter with baseline classifier if specified
    if filter_synthetic and filter_method in ["classifier", "both"]:
        clf_path = os.path.join(model_dir, "cnn_classifier_nogan.keras")
        if os.path.exists(clf_path):
            print("Filtering synthetic samples using downstream held-out classifier boundary...")
            classifier = tf.keras.models.load_model(clf_path, compile=False)
            probs = classifier.predict(X_candidates, verbose=0).reshape(-1)
            keep_idx = np.argsort(probs)[::-1]
            X_candidates = X_candidates[keep_idx]
        else:
            print("Warning: cnn_classifier_nogan.keras not found. Skipping classifier filtering.")
            
    # Select the top needed samples
    X_synth = X_candidates[:needed]
    y_synth = np.ones(needed, dtype=np.float32)
    
    # Save the selected synthetic features for evaluation.py validation
    synth_dir = os.path.join("outputs", "synthetic_features")
    os.makedirs(synth_dir, exist_ok=True)
    np.save(os.path.join(synth_dir, "synth_abnormal_mfcc.npy"), X_synth)
    
    print(f"Augmentation complete: Added {len(X_synth)} synthetic abnormal samples.")
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
    filter_synthetic=True,
    spec_augment_enabled=False,
    use_wavegan=False,
    filter_method="classifier",
    pool_multiplier=4,
    use_focal_loss=False,
    lightweight=False,
):
    os.makedirs(model_dir, exist_ok=True)

    X_train = np.load(os.path.join(processed_dir, "X_train.npy")).astype(np.float32)
    y_train = np.load(os.path.join(processed_dir, "y_train.npy"))
    X_val = np.load(os.path.join(processed_dir, "X_val.npy")).astype(np.float32)
    y_val = np.load(os.path.join(processed_dir, "y_val.npy"))

    check_recording_leakage(processed_dir)

    # Apply SpecAugment if enabled
    if spec_augment_enabled:
        print("Applying SpecAugment classic data augmentation to real training split...")
        X_train_aug = spec_augment(X_train)
        X_train = np.concatenate([X_train, X_train_aug], axis=0)
        y_train = np.concatenate([y_train, y_train], axis=0)
        indices = np.random.permutation(len(X_train))
        X_train = X_train[indices]
        y_train = y_train[indices]
        print(f"SpecAugment complete. New train shape: {X_train.shape}")

    print(f"Train: {X_train.shape} | normal={np.sum(y_train == 0)} abnormal={np.sum(y_train == 1)}")
    print(f"Val:   {X_val.shape} | normal={np.sum(y_val == 0)} abnormal={np.sum(y_val == 1)}")

    input_shape = X_train.shape[1:]

    # Select model builder function
    build_fn = build_light_cnn_classifier if lightweight else build_cnn_classifier
    model_name = "lightweight 1D CNN" if lightweight else "residual CNN-BiGRU-Attention"
    print(f"Using {model_name} architecture.")

    print("\nStage 1: baseline classifier")
    baseline = compile_classifier(build_fn(input_shape), use_focal_loss=use_focal_loss)
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
        filter_synthetic=filter_synthetic,
        use_wavegan=use_wavegan,
        filter_method=filter_method,
        pool_multiplier=pool_multiplier,
    )
    np.save(
        os.path.join(processed_dir, "split_counts.npy"),
        np.array([np.sum(y_train == 0), np.sum(y_train == 1), np.sum(y_aug == 0), np.sum(y_aug == 1)]),
    )

    gan_model = compile_classifier(build_fn(input_shape), use_focal_loss=use_focal_loss)
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
    parser.add_argument("--no_filter_synthetic", action="store_true")
    parser.add_argument("--spec_augment", action="store_true", help="Enable SpecAugment on real training features")
    parser.add_argument("--use_wavegan", action="store_true", help="Use WaveGAN raw waveforms instead of feature-space GAN")
    parser.add_argument("--filter_method", choices=["discriminator", "classifier", "both", "none"], default="classifier", help="Method used to filter synthetic samples")
    parser.add_argument("--pool_multiplier", type=int, default=4, help="Pool size multiplier for synthetic filtering")
    parser.add_argument("--use_focal_loss", action="store_true", help="Compile model using Focal Loss instead of BCE")
    parser.add_argument("--lightweight", action="store_true", help="Train a lightweight CNN instead of residual CNN-BiGRU-Attention")
    args = parser.parse_args()
    train_classifier_pipeline(
        args.epochs,
        args.batch_size,
        latent_dim=args.latent_dim,
        augmentation_ratio=args.augmentation_ratio,
        filter_synthetic=not args.no_filter_synthetic,
        spec_augment_enabled=args.spec_augment,
        use_wavegan=args.use_wavegan,
        filter_method=args.filter_method,
        pool_multiplier=args.pool_multiplier,
        use_focal_loss=args.use_focal_loss,
        lightweight=args.lightweight,
    )
