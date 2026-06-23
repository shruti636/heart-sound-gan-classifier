import os
import argparse
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from cnn_classifier import build_cnn_classifier

# Set random seed for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

def train_classifier_pipeline(epochs=20, batch_size=16, processed_dir='data/processed', model_dir='models', outputs_dir='outputs'):
    # 1. Load pre-split processed data
    X_train = np.load(os.path.join(processed_dir, 'X_train.npy'))
    y_train = np.load(os.path.join(processed_dir, 'y_train.npy'))
    X_val = np.load(os.path.join(processed_dir, 'X_val.npy'))
    y_val = np.load(os.path.join(processed_dir, 'y_val.npy'))
    
    # 2. Strict overlap check (leakage verification)
    train_manifest_path = os.path.join(processed_dir, 'train_source_files.txt')
    test_manifest_path = os.path.join(processed_dir, 'test_source_files.txt')
    if os.path.exists(train_manifest_path) and os.path.exists(test_manifest_path):
        with open(train_manifest_path, 'r') as f:
            train_files = set(f.read().splitlines())
        with open(test_manifest_path, 'r') as f:
            test_files = set(f.read().splitlines())
        overlap = train_files.intersection(test_files)
        print("="*60)
        print("  LEAKAGE VERIFICATION BEFORE TRAINING:")
        print(f"  Overlap count: {len(overlap)} files")
        assert len(overlap) == 0, "DATA LEAKAGE DETECTED: Overlap exists between Train and Test recordings!"
        print("  Verification PASSED: Train and Test splits have 100% separate recording sources.")
        print("="*60)
        
    num_normal_train = np.sum(y_train == 0)
    num_abnormal_train = np.sum(y_train == 1)
    
    print(f"Original real training set distribution:")
    print(f"  Normal (0): {num_normal_train}")
    print(f"  Abnormal (1): {num_abnormal_train}")
    
    input_shape = X_train.shape[1:]
    
    # -------------------------------------------------------------
    # STAGE 1: Train Baseline Model (No GAN)
    # -------------------------------------------------------------
    print("\n" + "="*50)
    print("  STAGE 1: TRAINING BASELINE MODEL (NO GAN)")
    print("="*50)
    
    model_nogan = build_cnn_classifier(input_shape=input_shape)
    model_nogan.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.Precision(name='precision'), tf.keras.metrics.Recall(name='recall')]
    )
    
    checkpoint_nogan = os.path.join(model_dir, 'cnn_classifier_nogan.keras')
    callbacks_nogan = [
        EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True, verbose=1),
        ModelCheckpoint(filepath=checkpoint_nogan, monitor='val_loss', save_best_only=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4, min_lr=1e-6, verbose=1)
    ]
    
    history_nogan = model_nogan.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks_nogan,
        verbose=1
    )
    np.save(os.path.join(processed_dir, 'history_nogan.npy'), history_nogan.history)
    print(f"Baseline model training complete. Checkpoint: {checkpoint_nogan}")
    
    # -------------------------------------------------------------
    # STAGE 2: Train GAN-Augmented Model
    # -------------------------------------------------------------
    print("\n" + "="*50)
    print("  STAGE 2: TRAINING GAN-AUGMENTED MODEL")
    print("="*50)
    
    # Generate synthetic abnormal MFCCs to balance the training set
    if num_normal_train > num_abnormal_train:
        diff = num_normal_train - num_abnormal_train
        print(f"Generating {diff} synthetic abnormal samples to balance train set...")
        
        gen_path = os.path.join(model_dir, 'gan_generator.keras')
        if not os.path.exists(gen_path):
            raise FileNotFoundError(f"Trained GAN generator not found at {gen_path}! Run train_gan.py first.")
            
        generator = tf.keras.models.load_model(gen_path)
        
        # Latent space vectors
        latent_dim = 100
        noise = tf.random.normal([diff, latent_dim])
        X_synth = generator(noise, training=False).numpy()
        y_synth = np.ones(diff, dtype=np.int32)
        
        # Merge real train and synthetic abnormal
        X_real_abnormal = X_train[y_train == 1]
        X_real_normal = X_train[y_train == 0]
        
        X_train_balanced = np.concatenate([X_real_normal, X_real_abnormal, X_synth], axis=0)
        y_train_balanced = np.concatenate([np.zeros(num_normal_train), np.ones(num_abnormal_train), y_synth], axis=0)
        
        print(f"Augmented balanced training set distribution:")
        print(f"  Normal (0): {np.sum(y_train_balanced == 0)}")
        print(f"  Abnormal (1): {np.sum(y_train_balanced == 1)}")
    else:
        print("Training set is already balanced. Using original training data.")
        X_train_balanced = X_train
        y_train_balanced = y_train
        
    model_gan = build_cnn_classifier(input_shape=input_shape)
    model_gan.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.Precision(name='precision'), tf.keras.metrics.Recall(name='recall')]
    )
    
    checkpoint_gan = os.path.join(model_dir, 'cnn_classifier_gan.keras')
    callbacks_gan = [
        EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True, verbose=1),
        ModelCheckpoint(filepath=checkpoint_gan, monitor='val_loss', save_best_only=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4, min_lr=1e-6, verbose=1)
    ]
    
    history_gan = model_gan.fit(
        X_train_balanced, y_train_balanced,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks_gan,
        verbose=1
    )
    np.save(os.path.join(processed_dir, 'history_gan.npy'), history_gan.history)
    print(f"GAN-Augmented model training complete. Checkpoint: {checkpoint_gan}")
    
    # Save the split counts for class distribution comparison
    np.save(os.path.join(processed_dir, 'split_counts.npy'), np.array([
        num_normal_train, num_abnormal_train,
        np.sum(y_train_balanced == 0), np.sum(y_train_balanced == 1)
    ]))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train Baseline and GAN-augmented classifiers")
    parser.add_argument('--epochs', type=int, default=15, help='Number of epochs to train')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for training')
    
    args = parser.parse_args()
    train_classifier_pipeline(epochs=args.epochs, batch_size=args.batch_size)
