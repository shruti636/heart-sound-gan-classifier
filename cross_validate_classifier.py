"""Simple recording-level cross-validation for the heart-sound classifier.

Run this after preprocessing.py. It uses the saved source recording names so
segments from the same recording never appear in both train and validation.
"""

import argparse
import os

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import GroupKFold

from cnn_classifier import build_cnn_classifier
from evaluation import metrics_at_threshold
from train_classifier import class_weights, compile_classifier


SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)


def load_all_processed_segments(processed_dir):
    X_parts, y_parts, source_parts = [], [], []
    for split in ["train", "val", "test"]:
        X_parts.append(np.load(os.path.join(processed_dir, f"X_{split}.npy")).astype(np.float32))
        y_parts.append(np.load(os.path.join(processed_dir, f"y_{split}.npy")))
        source_path = os.path.join(processed_dir, f"sources_{split}.npy")
        if not os.path.exists(source_path):
            raise FileNotFoundError("Run preprocessing.py again to create source recording arrays.")
        source_parts.append(np.load(source_path).astype(str))

    return np.concatenate(X_parts), np.concatenate(y_parts), np.concatenate(source_parts)


def run_cross_validation(processed_dir="data/processed", outputs_dir="outputs", folds=3, epochs=10, batch_size=16):
    os.makedirs(outputs_dir, exist_ok=True)
    X, y, groups = load_all_processed_segments(processed_dir)
    splitter = GroupKFold(n_splits=folds)
    rows = []

    for fold, (train_idx, val_idx) in enumerate(splitter.split(X, y, groups), start=1):
        model = compile_classifier(build_cnn_classifier(X.shape[1:]))
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor="val_auc",
                mode="max",
                patience=4,
                restore_best_weights=True,
            )
        ]
        model.fit(
            X[train_idx],
            y[train_idx].astype(np.float32),
            validation_data=(X[val_idx], y[val_idx].astype(np.float32)),
            epochs=epochs,
            batch_size=batch_size,
            class_weight=class_weights(y[train_idx]),
            callbacks=callbacks,
            verbose=1,
        )
        probs = model.predict(X[val_idx], verbose=0).flatten()
        metrics = metrics_at_threshold(y[val_idx], probs, threshold=0.5)
        rows.append({
            "Fold": fold,
            "Recordings": len(set(groups[val_idx])),
            "Accuracy": metrics["accuracy"],
            "BalancedAccuracy": metrics["balanced_accuracy"],
            "Precision": metrics["precision"],
            "Recall": metrics["recall"],
            "Specificity": metrics["specificity"],
            "F1-score": metrics["f1"],
        })

    df = pd.DataFrame(rows)
    df.loc["mean"] = df.mean(numeric_only=True)
    out_path = os.path.join(outputs_dir, "cross_validation_results.csv")
    df.to_csv(out_path, index=False)
    print(df)
    print(f"Saved cross-validation results to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run recording-level cross-validation.")
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=16)
    args = parser.parse_args()
    run_cross_validation(folds=args.folds, epochs=args.epochs, batch_size=args.batch_size)
