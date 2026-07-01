"""Recording-level Stratified 10-fold cross-validation for the heart-sound classifier.

Preserves recording-level groups using StratifiedGroupKFold to prevent data leakage.
Tunes decision thresholds on validation sub-splits to optimize test-fold predictions.
"""

import argparse
import os

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import StratifiedGroupKFold

from cnn_classifier import build_cnn_classifier
from evaluation import choose_threshold, aggregate_recording_scores, metrics_at_threshold
from train_classifier import class_weights, compile_classifier

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)


def load_all_processed_segments(processed_dir):
    """Loads and concatenates train, validation, and test splits into a single dataset."""
    X_parts, y_parts, source_parts = [], [], []
    for split in ["train", "val", "test"]:
        X_parts.append(np.load(os.path.join(processed_dir, f"X_{split}.npy")).astype(np.float32))
        y_parts.append(np.load(os.path.join(processed_dir, f"y_{split}.npy")))
        source_path = os.path.join(processed_dir, f"sources_{split}.npy")
        if not os.path.exists(source_path):
            raise FileNotFoundError("Run preprocessing.py again to create source recording arrays.")
        source_parts.append(np.load(source_path).astype(str))

    return np.concatenate(X_parts), np.concatenate(y_parts), np.concatenate(source_parts)


def split_groups_train_val(X_train, y_train, groups_train, val_ratio=0.15):
    """Splits a training fold's group indices into sub-train and sub-val to avoid leakage."""
    unique_groups = np.unique(groups_train)
    np.random.shuffle(unique_groups)
    split_point = int(len(unique_groups) * (1.0 - val_ratio))
    
    train_groups = unique_groups[:split_point]
    val_groups = unique_groups[split_point:]
    
    train_idx = np.isin(groups_train, train_groups)
    val_idx = np.isin(groups_train, val_groups)
    
    return np.where(train_idx)[0], np.where(val_idx)[0]


def run_cross_validation(processed_dir="data/processed", outputs_dir="outputs", folds=10, epochs=15, batch_size=16):
    os.makedirs(outputs_dir, exist_ok=True)
    X, y, groups = load_all_processed_segments(processed_dir)
    
    # StratifiedGroupKFold ensures fold distribution is stratified and patient leak-proof
    splitter = StratifiedGroupKFold(n_splits=folds)
    rows = []

    print(f"\n--- Starting Stratified {folds}-Fold Cross-Validation ---")
    for fold, (train_idx, test_idx) in enumerate(splitter.split(X, y, groups), start=1):
        print(f"\nFold {fold}/{folds}:")
        
        # Sub-split training fold into sub-train and sub-val for early stopping and threshold tuning
        X_fold_train, y_fold_train, groups_fold_train = X[train_idx], y[train_idx], groups[train_idx]
        sub_train_idx, sub_val_idx = split_groups_train_val(X_fold_train, y_fold_train, groups_fold_train, val_ratio=0.15)
        
        X_sub_train, y_sub_train = X_fold_train[sub_train_idx], y_fold_train[sub_train_idx]
        X_sub_val, y_sub_val, groups_sub_val = X_fold_train[sub_val_idx], y_fold_train[sub_val_idx], groups_fold_train[sub_val_idx]
        
        # Determine dynamic class weights for Weighted BCE loss
        normal_count = max(np.sum(y_sub_train == 0), 1)
        abnormal_count = max(np.sum(y_sub_train == 1), 1)
        pos_weight = float(normal_count / abnormal_count)

        # Build and compile model
        model = compile_classifier(
            build_cnn_classifier(X.shape[1:]),
            use_focal_loss=False,
            use_weighted_bce=True,
            pos_weight=pos_weight
        )
        
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor="val_auc",
                mode="max",
                patience=5,
                restore_best_weights=True,
            )
        ]
        
        # Train on sub-train, validate on sub-val
        model.fit(
            X_sub_train,
            y_sub_train.astype(np.float32),
            validation_data=(X_sub_val, y_sub_val.astype(np.float32)),
            epochs=epochs,
            batch_size=batch_size,
            class_weight=class_weights(y_sub_train),
            callbacks=callbacks,
            verbose=0,  # Keep output clean during cross validation
        )
        
        # Predict on sub-val segments and aggregate to tune the threshold
        probs_sub_val = model.predict(X_sub_val, verbose=0).flatten()
        y_val_rec, probs_val_rec = aggregate_recording_scores(
            probs_sub_val, y_sub_val, groups_sub_val, method="mean"
        )
        
        # Tune threshold on sub-val set (maximizing balanced accuracy)
        tuned_threshold, _ = choose_threshold(y_val_rec, probs_val_rec, min_recall=None)
        print(f"  Tuned decision threshold on validation sub-split: {tuned_threshold:.3f}")
        
        # Predict on the unseen fold's test set
        probs_test = model.predict(X[test_idx], verbose=0).flatten()
        y_test_rec, probs_test_rec = aggregate_recording_scores(
            probs_test, y[test_idx], groups[test_idx], method="mean"
        )
        
        # Evaluate using both standard 0.5 threshold and the tuned threshold
        metrics_default = metrics_at_threshold(y_test_rec, probs_test_rec, threshold=0.5)
        metrics_tuned = metrics_at_threshold(y_test_rec, probs_test_rec, threshold=tuned_threshold)
        
        rows.append({
            "Fold": fold,
            "TunedThreshold": tuned_threshold,
            "Accuracy_0.5": metrics_default["accuracy"],
            "BalancedAccuracy_0.5": metrics_default["balanced_accuracy"],
            "Precision_0.5": metrics_default["precision"],
            "Recall_0.5": metrics_default["recall"],
            "Specificity_0.5": metrics_default["specificity"],
            "F1_0.5": metrics_default["f1"],
            "Accuracy_Tuned": metrics_tuned["accuracy"],
            "BalancedAccuracy_Tuned": metrics_tuned["balanced_accuracy"],
            "Precision_Tuned": metrics_tuned["precision"],
            "Recall_Tuned": metrics_tuned["recall"],
            "Specificity_Tuned": metrics_tuned["specificity"],
            "F1_Tuned": metrics_tuned["f1"],
        })
        
        print(f"  Test metrics (0.5)   - Acc: {metrics_default['accuracy']:.3f} | Rec: {metrics_default['recall']:.3f} | Spec: {metrics_default['specificity']:.3f} | F1: {metrics_default['f1']:.3f}")
        print(f"  Test metrics (Tuned) - Acc: {metrics_tuned['accuracy']:.3f} | Rec: {metrics_tuned['recall']:.3f} | Spec: {metrics_tuned['specificity']:.3f} | F1: {metrics_tuned['f1']:.3f}")

    df = pd.DataFrame(rows)
    df.loc["mean"] = df.mean(numeric_only=True)
    out_path = os.path.join(outputs_dir, "cross_validation_results.csv")
    df.to_csv(out_path, index=False)
    
    print("\n--- Cross-Validation Results Summary ---")
    print(df)
    print(f"\nSaved detailed cross-validation results to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run stratified recording-level cross-validation.")
    parser.add_argument("--folds", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=16)
    args = parser.parse_args()
    run_cross_validation(folds=args.folds, epochs=args.epochs, batch_size=args.batch_size)
