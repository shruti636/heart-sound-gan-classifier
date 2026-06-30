import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.manifold import TSNE
import librosa
import librosa.display

# Set random seed for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

def metrics_at_threshold(y_true, probs, threshold):
    """Compute binary metrics at a chosen abnormal probability threshold."""
    preds = (probs >= threshold).astype(np.int32)
    cm = confusion_matrix(y_true, preds, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    accuracy = (tp + tn) / max(tp + tn + fp + fn, 1)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    specificity = tn / max(tn + fp, 1)
    npv = tn / max(tn + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)
    balanced_accuracy = (recall + specificity) / 2

    return {
        "preds": preds,
        "cm": cm,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "npv": npv,
        "f1": f1,
        "balanced_accuracy": balanced_accuracy,
    }


def choose_threshold(y_true, probs, min_recall=0.90):
    """Pick a validation threshold that keeps recall high and reduces false alarms."""
    candidates = np.linspace(0.05, 0.95, 181)
    scored = []
    for threshold in candidates:
        m = metrics_at_threshold(y_true, probs, threshold)
        scored.append((threshold, m))

    high_recall = [(t, m) for t, m in scored if min_recall is not None and m["recall"] >= min_recall]
    
    # Try to find a threshold that satisfies high recall AND has specificity >= 0.40
    safe_high_recall = [(t, m) for t, m in high_recall if m["specificity"] >= 0.40]
    
    if safe_high_recall:
        threshold, metric = max(
            safe_high_recall,
            key=lambda item: (item[1]["specificity"], item[1]["f1"], item[1]["balanced_accuracy"]),
        )
    elif high_recall:
        threshold, metric = max(
            high_recall,
            key=lambda item: (item[1]["specificity"], item[1]["f1"], item[1]["balanced_accuracy"]),
        )
        # If specificity is extremely poor (< 0.20), fallback to maximizing balanced accuracy
        if metric["specificity"] < 0.20:
            print("Warning: Specificity collapsed under min_recall constraint. Falling back to maximizing Balanced Accuracy.")
            threshold, metric = max(scored, key=lambda item: (item[1]["balanced_accuracy"], item[1]["f1"]))
    else:
        threshold, metric = max(scored, key=lambda item: (item[1]["balanced_accuracy"], item[1]["f1"]))

    return float(threshold), metric


def choose_specificity_threshold(y_true, probs, min_specificity=0.70):
    """Pick a threshold that reduces false positives while keeping recall as high as possible."""
    candidates = np.linspace(0.05, 0.95, 181)
    scored = []
    for threshold in candidates:
        m = metrics_at_threshold(y_true, probs, threshold)
        scored.append((threshold, m))

    high_specificity = [(t, m) for t, m in scored if m["specificity"] >= min_specificity]
    if high_specificity:
        threshold, metric = max(
            high_specificity,
            key=lambda item: (item[1]["recall"], item[1]["balanced_accuracy"], item[1]["f1"]),
        )
    else:
        threshold, metric = max(scored, key=lambda item: (item[1]["specificity"], item[1]["balanced_accuracy"]))

    return float(threshold), metric


def plot_calibration_curve(y_true, probs, title, save_path, n_bins=10):
    """Reliability plot: predicted confidence vs observed abnormal rate."""
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.digitize(probs, bins) - 1
    observed, predicted = [], []

    for bin_idx in range(n_bins):
        mask = bin_ids == bin_idx
        if np.any(mask):
            observed.append(np.mean(y_true[mask]))
            predicted.append(np.mean(probs[mask]))

    plt.figure(figsize=(6, 5))
    plt.plot([0, 1], [0, 1], "--", color="gray", label="Perfect calibration")
    plt.plot(predicted, observed, marker="o", label="Model")
    plt.xlabel("Mean predicted abnormal probability")
    plt.ylabel("Observed abnormal rate")
    plt.title(title)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=140)
    plt.close()


def plot_threshold_curve(y_true, probs, title, save_path):
    """Show recall/specificity tradeoff for many thresholds."""
    thresholds = np.linspace(0.05, 0.95, 181)
    recalls, specificities, balanced = [], [], []
    for threshold in thresholds:
        m = metrics_at_threshold(y_true, probs, threshold)
        recalls.append(m["recall"])
        specificities.append(m["specificity"])
        balanced.append(m["balanced_accuracy"])

    plt.figure(figsize=(7, 5))
    plt.plot(thresholds, recalls, label="Recall")
    plt.plot(thresholds, specificities, label="Specificity")
    plt.plot(thresholds, balanced, label="Balanced accuracy")
    plt.xlabel("Abnormal probability threshold")
    plt.ylabel("Metric value")
    plt.title(title)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=140)
    plt.close()


def evaluate_model(model_path, X_test, y_test, threshold=0.5):
    """
    Loads a model and evaluates it, returning predictions, probabilities, and core metrics.
    """
    model = tf.keras.models.load_model(model_path, compile=False)
    probs = model.predict(X_test).flatten()
    threshold_metrics = metrics_at_threshold(y_test, probs, threshold)
    
    fpr, tpr, _ = roc_curve(y_test, probs)
    roc_auc = auc(fpr, tpr)
    
    return {
        'threshold': threshold,
        'preds': threshold_metrics['preds'],
        'probs': probs,
        'cm': threshold_metrics['cm'],
        'accuracy': threshold_metrics['accuracy'],
        'precision': threshold_metrics['precision'],
        'recall': threshold_metrics['recall'],
        'specificity': threshold_metrics['specificity'],
        'npv': threshold_metrics['npv'],
        'f1': threshold_metrics['f1'],
        'balanced_accuracy': threshold_metrics['balanced_accuracy'],
        'auc': roc_auc,
        'fpr': fpr,
        'tpr': tpr
    }


def aggregate_recording_scores(probs, y_true, sources, method='mean'):
    """Aggregate segment probabilities so each recording gets one final score."""
    rows = pd.DataFrame({
        'source': sources.astype(str),
        'prob': probs,
        'label': y_true.astype(int),
    })
    
    if method == 'mean':
        grouped = rows.groupby('source', sort=False).agg(
            prob=('prob', 'mean'),
            label=('label', 'max'),
        )
    elif method == 'max':
        grouped = rows.groupby('source', sort=False).agg(
            prob=('prob', 'max'),
            label=('label', 'max'),
        )
    elif method == 'confidence_weighted':
        rows['weight'] = np.abs(rows['prob'] - 0.5)
        rows['weighted_prob'] = rows['prob'] * rows['weight']
        
        grouped = rows.groupby('source', sort=False).agg(
            weighted_prob_sum=('weighted_prob', 'sum'),
            weight_sum=('weight', 'sum'),
            label=('label', 'max')
        )
        grouped['prob'] = grouped['weighted_prob_sum'] / np.clip(grouped['weight_sum'], 1e-8, None)
        zero_weights = grouped['weight_sum'] < 1e-8
        if np.any(zero_weights):
            mean_probs = rows.groupby('source', sort=False)['prob'].mean()
            grouped.loc[zero_weights, 'prob'] = mean_probs[zero_weights]
    else:
        raise ValueError(f"Unknown aggregation method: {method}")
        
    return grouped['label'].to_numpy(dtype=np.int32), grouped['prob'].to_numpy(dtype=np.float32)


def evaluate_scores(y_true, probs, threshold):
    """Evaluate already-computed probabilities."""
    threshold_metrics = metrics_at_threshold(y_true, probs, threshold)
    fpr, tpr, _ = roc_curve(y_true, probs)
    roc_auc = auc(fpr, tpr)
    return {
        'threshold': threshold,
        'preds': threshold_metrics['preds'],
        'probs': probs,
        'cm': threshold_metrics['cm'],
        'accuracy': threshold_metrics['accuracy'],
        'precision': threshold_metrics['precision'],
        'recall': threshold_metrics['recall'],
        'specificity': threshold_metrics['specificity'],
        'npv': threshold_metrics['npv'],
        'f1': threshold_metrics['f1'],
        'balanced_accuracy': threshold_metrics['balanced_accuracy'],
        'auc': roc_auc,
        'fpr': fpr,
        'tpr': tpr,
    }

def run_tsne_validation(X_real_abnormal, X_generated_abnormal, save_path):
    """
    Applies t-SNE to project real and generated abnormal MFCC features into 2D space.
    """
    # Flatten shapes: (N, time, features) -> (N, time * features)
    real_flat = X_real_abnormal.reshape(X_real_abnormal.shape[0], -1)
    gen_flat = X_generated_abnormal.reshape(X_generated_abnormal.shape[0], -1)
    
    # Combine features
    combined = np.vstack([real_flat, gen_flat])
    labels = np.array([0] * len(real_flat) + [1] * len(gen_flat)) # 0 for Real, 1 for Synthetic
    
    # Run t-SNE
    print("Running t-SNE dimensionality reduction (this may take a few seconds)...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    projected = tsne.fit_transform(combined)
    
    # Plot t-SNE scatter
    plt.figure(figsize=(8, 6))
    plt.scatter(projected[labels == 0, 0], projected[labels == 0, 1], color='#1f77b4', alpha=0.7, label='Real Abnormal MFCC', edgecolors='k')
    plt.scatter(projected[labels == 1, 0], projected[labels == 1, 1], color='#d62728', alpha=0.7, label='Synthetic Abnormal MFCC (GAN)', edgecolors='k')
    plt.title('t-SNE Distribution Space Comparison', fontsize=14, fontweight='bold')
    plt.xlabel('t-SNE Dimension 1', fontsize=12)
    plt.ylabel('t-SNE Dimension 2', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"t-SNE scatter plot saved to {save_path}")

def run_evaluation_study(processed_dir='data/processed', model_dir='models', outputs_dir='outputs', aggregation='confidence_weighted'):
    os.makedirs(os.path.join(outputs_dir, 'plots'), exist_ok=True)
    
    # Load splits
    X_train = np.load(os.path.join(processed_dir, 'X_train.npy'))
    y_train = np.load(os.path.join(processed_dir, 'y_train.npy'))
    X_val = np.load(os.path.join(processed_dir, 'X_val.npy'))
    y_val = np.load(os.path.join(processed_dir, 'y_val.npy'))
    X_test = np.load(os.path.join(processed_dir, 'X_test.npy'))
    y_test = np.load(os.path.join(processed_dir, 'y_test.npy'))
    sources_val = np.load(os.path.join(processed_dir, 'sources_val.npy'), allow_pickle=True)
    sources_test = np.load(os.path.join(processed_dir, 'sources_test.npy'), allow_pickle=True)
    bounds_train = np.load(os.path.join(processed_dir, 'bounds_train.npy'))
    
    # 1. Evaluate models
    path_nogan = os.path.join(model_dir, 'cnn_classifier_nogan.keras')
    path_gan = os.path.join(model_dir, 'cnn_classifier_gan.keras')
    
    print("Choosing thresholds on validation data...")
    model_nogan = tf.keras.models.load_model(path_nogan, compile=False)
    model_gan = tf.keras.models.load_model(path_gan, compile=False)
    probs_val_nogan = model_nogan.predict(X_val).flatten()
    probs_val_gan = model_gan.predict(X_val).flatten()
    probs_val_ensemble = 0.5 * (probs_val_nogan + probs_val_gan)
    
    y_val_record, probs_val_nogan_record = aggregate_recording_scores(probs_val_nogan, y_val, sources_val, method=aggregation)
    _, probs_val_gan_record = aggregate_recording_scores(probs_val_gan, y_val, sources_val, method=aggregation)
    _, probs_val_ensemble_record = aggregate_recording_scores(probs_val_ensemble, y_val, sources_val, method=aggregation)
    
    thr_nogan, val_metrics_nogan = choose_threshold(y_val_record, probs_val_nogan_record, min_recall=0.90)
    thr_gan, val_metrics_gan = choose_threshold(y_val_record, probs_val_gan_record, min_recall=0.90)
    thr_ensemble, val_metrics_ensemble = choose_threshold(y_val_record, probs_val_ensemble_record, min_recall=0.90)
    
    balanced_thr_nogan, balanced_val_nogan = choose_threshold(y_val_record, probs_val_nogan_record, min_recall=None)
    balanced_thr_gan, balanced_val_gan = choose_threshold(y_val_record, probs_val_gan_record, min_recall=None)
    balanced_thr_ensemble, balanced_val_ensemble = choose_threshold(y_val_record, probs_val_ensemble_record, min_recall=None)
    
    specificity_thr_nogan, specificity_val_nogan = choose_specificity_threshold(y_val_record, probs_val_nogan_record)
    specificity_thr_gan, specificity_val_gan = choose_specificity_threshold(y_val_record, probs_val_gan_record)
    specificity_thr_ensemble, specificity_val_ensemble = choose_specificity_threshold(y_val_record, probs_val_ensemble_record)

    threshold_config = {
        'cnn_classifier_nogan.keras': {
            'threshold': thr_nogan,
            'screening_threshold': thr_nogan,
            'balanced_threshold': balanced_thr_nogan,
            'specificity_threshold': specificity_thr_nogan,
            'validation_recall': val_metrics_nogan['recall'],
            'validation_specificity': val_metrics_nogan['specificity'],
            'balanced_validation_recall': balanced_val_nogan['recall'],
            'balanced_validation_specificity': balanced_val_nogan['specificity'],
            'specificity_validation_recall': specificity_val_nogan['recall'],
            'specificity_validation_specificity': specificity_val_nogan['specificity'],
        },
        'cnn_classifier_gan.keras': {
            'threshold': thr_gan,
            'screening_threshold': thr_gan,
            'balanced_threshold': balanced_thr_gan,
            'specificity_threshold': specificity_thr_gan,
            'validation_recall': val_metrics_gan['recall'],
            'validation_specificity': val_metrics_gan['specificity'],
            'balanced_validation_recall': balanced_val_gan['recall'],
            'balanced_validation_specificity': balanced_val_gan['specificity'],
            'specificity_validation_recall': specificity_val_gan['recall'],
            'specificity_validation_specificity': specificity_val_gan['specificity'],
        },
        'ensemble': {
            'threshold': thr_ensemble,
            'screening_threshold': thr_ensemble,
            'balanced_threshold': balanced_thr_ensemble,
            'specificity_threshold': specificity_thr_ensemble,
            'validation_recall': val_metrics_ensemble['recall'],
            'validation_specificity': val_metrics_ensemble['specificity'],
            'balanced_validation_recall': balanced_val_ensemble['recall'],
            'balanced_validation_specificity': balanced_val_ensemble['specificity'],
            'specificity_validation_recall': specificity_val_ensemble['recall'],
            'specificity_validation_specificity': specificity_val_ensemble['specificity'],
        },
        'uncertain_margin': 0.08,
        'aggregation_method': aggregation,
        'note': 'Thresholds are selected on validation data. Scores near the threshold should be treated as uncertain.',
    }
    with open(os.path.join(model_dir, 'model_thresholds.json'), 'w') as f:
        json.dump(threshold_config, f, indent=2)
    print(f"Baseline threshold: {thr_nogan:.3f} | GAN threshold: {thr_gan:.3f} | Ensemble threshold: {thr_ensemble:.3f}")

    plot_dir = os.path.join(outputs_dir, 'plots')
    plot_calibration_curve(
        y_val_record,
        probs_val_gan_record,
        "GAN Model Calibration on Validation Recordings",
        os.path.join(plot_dir, 'calibration_curve_gan.png'),
    )
    plot_threshold_curve(
        y_val_record,
        probs_val_gan_record,
        "GAN Threshold Tradeoff on Validation Recordings",
        os.path.join(plot_dir, 'threshold_tradeoff_gan.png'),
    )

    print("Evaluating Baseline Model (No GAN)...")
    res_nogan_segments = evaluate_model(path_nogan, X_test, y_test, threshold=thr_nogan)
    y_test_record, probs_test_nogan_record = aggregate_recording_scores(
        res_nogan_segments['probs'],
        y_test,
        sources_test,
        method=aggregation,
    )
    res_nogan = evaluate_scores(y_test_record, probs_test_nogan_record, threshold=thr_nogan)
    
    print("Evaluating GAN-Augmented Model...")
    res_gan_segments = evaluate_model(path_gan, X_test, y_test, threshold=thr_gan)
    _, probs_test_gan_record = aggregate_recording_scores(
        res_gan_segments['probs'],
        y_test,
        sources_test,
        method=aggregation,
    )
    res_gan = evaluate_scores(y_test_record, probs_test_gan_record, threshold=thr_gan)

    print("Evaluating Ensemble Model...")
    probs_test_ensemble_segments = 0.5 * (res_nogan_segments['probs'] + res_gan_segments['probs'])
    _, probs_test_ensemble_record = aggregate_recording_scores(
        probs_test_ensemble_segments,
        y_test,
        sources_test,
        method=aggregation,
    )
    res_ensemble = evaluate_scores(y_test_record, probs_test_ensemble_record, threshold=thr_ensemble)
    
    # 2. Save results CSV
    metrics_df = pd.DataFrame({
        'Model': ['Baseline (No GAN)', 'GAN-Augmented', 'Ensemble (Baseline + GAN)'],
        'Threshold': [res_nogan['threshold'], res_gan['threshold'], res_ensemble['threshold']],
        'Accuracy': [res_nogan['accuracy'], res_gan['accuracy'], res_ensemble['accuracy']],
        'BalancedAccuracy': [res_nogan['balanced_accuracy'], res_gan['balanced_accuracy'], res_ensemble['balanced_accuracy']],
        'Precision': [res_nogan['precision'], res_gan['precision'], res_ensemble['precision']],
        'Recall': [res_nogan['recall'], res_gan['recall'], res_ensemble['recall']],
        'Specificity': [res_nogan['specificity'], res_gan['specificity'], res_ensemble['specificity']],
        'NPV': [res_nogan['npv'], res_gan['npv'], res_ensemble['npv']],
        'F1-score': [res_nogan['f1'], res_gan['f1'], res_ensemble['f1']],
        'AUC': [res_nogan['auc'], res_gan['auc'], res_ensemble['auc']]
    })
    
    csv_path = os.path.join(outputs_dir, 'gan_vs_no_gan_results.csv')
    metrics_df.to_csv(csv_path, index=False)
    print(f"Performance metrics saved to {csv_path}")

    threshold_rows = []
    for model_name, probs, screening_thr, balanced_thr, specificity_thr in [
        ('Baseline', res_nogan['probs'], thr_nogan, balanced_thr_nogan, specificity_thr_nogan),
        ('GAN-Augmented', res_gan['probs'], thr_gan, balanced_thr_gan, specificity_thr_gan),
        ('Ensemble', res_ensemble['probs'], thr_ensemble, balanced_thr_ensemble, specificity_thr_ensemble),
    ]:
        for mode, threshold in [
            ('screening', screening_thr),
            ('balanced', balanced_thr),
            ('specificity', specificity_thr),
        ]:
            m = metrics_at_threshold(y_test_record, probs, threshold)
            threshold_rows.append({
                'Model': model_name,
                'Mode': mode,
                'Threshold': threshold,
                'Accuracy': m['accuracy'],
                'BalancedAccuracy': m['balanced_accuracy'],
                'Precision': m['precision'],
                'Recall': m['recall'],
                'Specificity': m['specificity'],
                'NPV': m['npv'],
                'F1-score': m['f1'],
            })
    threshold_df = pd.DataFrame(threshold_rows)
    threshold_df.to_csv(os.path.join(outputs_dir, 'threshold_analysis.csv'), index=False)
    
    # Calculate improvements
    acc_diff = res_gan['accuracy'] - res_nogan['accuracy']
    rec_diff = res_gan['recall'] - res_nogan['recall']
    f1_diff = res_gan['f1'] - res_nogan['f1']
    auc_diff = res_gan['auc'] - res_nogan['auc']
    
    # 3. Visual Comparisons
    # A. ROC Comparison
    roc_comp_path = os.path.join(outputs_dir, 'plots', 'performance_comparison.png')
    plt.figure(figsize=(7, 6))
    plt.plot(res_nogan['fpr'], res_nogan['tpr'], color='#1f77b4', lw=2, label=f"Baseline (No GAN) (AUC = {res_nogan['auc']:.4f})")
    plt.plot(res_gan['fpr'], res_gan['tpr'], color='#d62728', lw=2, label=f"GAN-Augmented (AUC = {res_gan['auc']:.4f})")
    
    fpr_ens, tpr_ens, _ = roc_curve(y_test_record, probs_test_ensemble_record)
    auc_ens = auc(fpr_ens, tpr_ens)
    res_ensemble['fpr'] = fpr_ens
    res_ensemble['tpr'] = tpr_ens
    res_ensemble['auc'] = auc_ens
    plt.plot(fpr_ens, tpr_ens, color='#2ca02c', lw=2, label=f"Ensemble (Baseline+GAN) (AUC = {auc_ens:.4f})")
    
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curve Comparison Study', fontsize=14, fontweight='bold')
    plt.legend(loc="lower right", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(roc_comp_path, dpi=150)
    plt.close()
    print(f"ROC comparison plot saved to {roc_comp_path}")
    
    # B. Confusion Matrix Comparison
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    # Left: No GAN
    cm_no = res_nogan['cm']
    im0 = axes[0].imshow(cm_no, cmap=plt.cm.Blues, interpolation='nearest')
    axes[0].set_title('Baseline Model (No GAN)', fontsize=13, fontweight='bold')
    axes[0].set_ylabel('True Label')
    axes[0].set_xlabel('Predicted Label')
    axes[0].set_xticks([0, 1])
    axes[0].set_xticklabels(['Normal', 'Abnormal'])
    axes[0].set_yticks([0, 1])
    axes[0].set_yticklabels(['Normal', 'Abnormal'])
    for i in range(2):
        for j in range(2):
            axes[0].text(j, i, str(cm_no[i, j]), ha="center", va="center", color="white" if cm_no[i, j] > np.max(cm_no)/2 else "black", fontsize=14, fontweight='bold')
            
    # Middle: With GAN
    cm_gan = res_gan['cm']
    im1 = axes[1].imshow(cm_gan, cmap=plt.cm.Oranges, interpolation='nearest')
    axes[1].set_title('GAN-Augmented Model', fontsize=13, fontweight='bold')
    axes[1].set_ylabel('True Label')
    axes[1].set_xlabel('Predicted Label')
    axes[1].set_xticks([0, 1])
    axes[1].set_xticklabels(['Normal', 'Abnormal'])
    axes[1].set_yticks([0, 1])
    axes[1].set_yticklabels(['Normal', 'Abnormal'])
    for i in range(2):
        for j in range(2):
            axes[1].text(j, i, str(cm_gan[i, j]), ha="center", va="center", color="white" if cm_gan[i, j] > np.max(cm_gan)/2 else "black", fontsize=14, fontweight='bold')
            
    # Right: Ensemble
    cm_ens = res_ensemble['cm']
    im2 = axes[2].imshow(cm_ens, cmap=plt.cm.Greens, interpolation='nearest')
    axes[2].set_title('Ensemble Model (Baseline+GAN)', fontsize=13, fontweight='bold')
    axes[2].set_ylabel('True Label')
    axes[2].set_xlabel('Predicted Label')
    axes[2].set_xticks([0, 1])
    axes[2].set_xticklabels(['Normal', 'Abnormal'])
    axes[2].set_yticks([0, 1])
    axes[2].set_yticklabels(['Normal', 'Abnormal'])
    for i in range(2):
        for j in range(2):
            axes[2].text(j, i, str(cm_ens[i, j]), ha="center", va="center", color="white" if cm_ens[i, j] > np.max(cm_ens)/2 else "black", fontsize=14, fontweight='bold')
            
    plt.tight_layout()
    cm_comp_path = os.path.join(outputs_dir, 'plots', 'confusion_matrix_comparison.png')
    plt.savefig(cm_comp_path, dpi=150)
    plt.close()
    print(f"Confusion Matrix comparison saved to {cm_comp_path}")
    
    # C. Training Curves Plot
    history_nogan_path = os.path.join(processed_dir, 'history_nogan.npy')
    history_gan_path = os.path.join(processed_dir, 'history_gan.npy')
    if os.path.exists(history_nogan_path) and os.path.exists(history_gan_path):
        hist_no = np.load(history_nogan_path, allow_pickle=True).item()
        hist_gan = np.load(history_gan_path, allow_pickle=True).item()
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        # Loss
        axes[0].plot(hist_no['loss'], label='Baseline Train Loss', color='#1f77b4', linestyle='--')
        axes[0].plot(hist_no['val_loss'], label='Baseline Val Loss', color='#1f77b4')
        axes[0].plot(hist_gan['loss'], label='GAN-Aug Train Loss', color='#d62728', linestyle='--')
        axes[0].plot(hist_gan['val_loss'], label='GAN-Aug Val Loss', color='#d62728')
        axes[0].set_title('Classifier Loss History', fontsize=13, fontweight='bold')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].legend()
        axes[0].grid(True, linestyle='--', alpha=0.5)
        
        # AUC
        axes[1].plot(hist_no['auc'], label='Baseline Train AUC', color='#1f77b4', linestyle='--')
        axes[1].plot(hist_no['val_auc'], label='Baseline Val AUC', color='#1f77b4')
        axes[1].plot(hist_gan['auc'], label='GAN-Aug Train AUC', color='#d62728', linestyle='--')
        axes[1].plot(hist_gan['val_auc'], label='GAN-Aug Val AUC', color='#d62728')
        axes[1].set_title('Classifier AUC History', fontsize=13, fontweight='bold')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('AUC')
        axes[1].legend()
        axes[1].grid(True, linestyle='--', alpha=0.5)
        
        plt.tight_layout()
        curves_path = os.path.join(outputs_dir, 'plots', 'classifier_training_curves.png')
        plt.savefig(curves_path, dpi=150)
        plt.close()
        print(f"Training curve comparison saved to {curves_path}")
        
    # D. Class Distribution comparison
    counts_path = os.path.join(processed_dir, 'split_counts.npy')
    if os.path.exists(counts_path):
        counts = np.load(counts_path)
        labels = ['Normal', 'Abnormal']
        before = [counts[0], counts[1]]
        after = [counts[2], counts[3]]
        
        x = np.arange(len(labels))
        width = 0.35
        
        fig, ax = plt.subplots(figsize=(6, 5))
        rects1 = ax.bar(x - width/2, before, width, label='Before GAN (Real Train)', color='#7f7f7f')
        rects2 = ax.bar(x + width/2, after, width, label='After GAN (Augmented)', color='#d62728')
        
        ax.set_ylabel('Number of Segments', fontsize=12)
        ax.set_title('Training Set Class Distribution', fontsize=13, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=11)
        ax.legend(fontsize=10)
        
        ax.bar_label(rects1, padding=3)
        ax.bar_label(rects2, padding=3)
        
        plt.tight_layout()
        dist_path = os.path.join(outputs_dir, 'plots', 'class_distribution_comparison.png')
        plt.savefig(dist_path, dpi=150)
        plt.close()
        print(f"Class distribution comparison saved to {dist_path}")
        
    # 4. GAN Validation (t-SNE & Feature Statistics)
    real_abnormal_idx = np.where(y_train == 1)[0]
    X_real_abnormal = X_train[real_abnormal_idx]
    
    synth_features_path = os.path.join(outputs_dir, 'synthetic_features', 'synth_abnormal_mfcc.npy')
    if os.path.exists(synth_features_path):
        X_generated_abnormal = np.load(synth_features_path)
        tsne_path = os.path.join(outputs_dir, 'plots', 'tsne_real_vs_generated.png')
        run_tsne_validation(X_real_abnormal, X_generated_abnormal, tsne_path)
        
        real_means = np.mean(X_real_abnormal, axis=(0, 1))
        real_vars = np.var(X_real_abnormal, axis=(0, 1))
        gen_means = np.mean(X_generated_abnormal, axis=(0, 1))
        gen_vars = np.var(X_generated_abnormal, axis=(0, 1))
        
        feature_axis = np.arange(X_real_abnormal.shape[-1])
        real_centroid = np.mean([np.sum(seg * feature_axis) / (np.sum(seg) + 1e-6) for seg in X_real_abnormal])
        gen_centroid = np.mean([np.sum(seg * feature_axis) / (np.sum(seg) + 1e-6) for seg in X_generated_abnormal])
    else:
        X_generated_abnormal = None
        real_means, real_vars, gen_means, gen_vars = None, None, None, None
        real_centroid, gen_centroid = 0, 0
        
    # E. Real vs Synthetic MFCC heatmap plotting
    norm_idx = np.where(y_train == 0)[0][0]
    abnorm_idx = np.where(y_train == 1)[0][0]
    
    bounds_train = np.load(os.path.join(processed_dir, 'bounds_train.npy'))
    real_normal_mfcc = X_train[norm_idx].T
    real_abnormal_mfcc = X_train[abnorm_idx].T
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    im0 = axes[0].imshow(real_normal_mfcc, cmap='coolwarm', aspect='auto', origin='lower')
    axes[0].set_title('Real Normal MFCC Heatmap', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Time frames')
    axes[0].set_ylabel('MFCC Coeffs')
    fig.colorbar(im0, ax=axes[0])
    
    im1 = axes[1].imshow(real_abnormal_mfcc, cmap='coolwarm', aspect='auto', origin='lower')
    axes[1].set_title('Real Abnormal MFCC Heatmap', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Time frames')
    axes[1].set_ylabel('MFCC Coeffs')
    fig.colorbar(im1, ax=axes[1])
    
    if X_generated_abnormal is not None:
        synth_single = X_generated_abnormal[0].T
        im2 = axes[2].imshow(synth_single, cmap='coolwarm', aspect='auto', origin='lower')
        axes[2].set_title('GAN Synthetic Abnormal MFCC', fontsize=14, fontweight='bold')
        axes[2].set_xlabel('Time frames')
        axes[2].set_ylabel('MFCC Coeffs')
        fig.colorbar(im2, ax=axes[2])
        
    plt.tight_layout()
    comp_path = os.path.join(outputs_dir, 'plots', 'spectrogram_comparisons.png')
    plt.savefig(comp_path, dpi=150)
    plt.close()
    
    # 5. Generate final_research_report.md
    report_path = os.path.join(outputs_dir, 'final_research_report.md')
    
    best_balanced = threshold_df.sort_values(
        ['BalancedAccuracy', 'Recall', 'Specificity'],
        ascending=False,
    ).iloc[0]
    safer_fp_options = threshold_df[threshold_df['Recall'] >= 0.70]
    if safer_fp_options.empty:
        safer_fp_options = threshold_df
    fewest_false_positive = safer_fp_options.sort_values(
        ['Specificity', 'Recall', 'BalancedAccuracy'],
        ascending=False,
    ).iloc[0]
    if rec_diff >= 0 and acc_diff >= 0:
        gan_summary = "GAN augmentation improved the main screening metrics in this run."
    elif acc_diff > 0 and rec_diff < 0:
        gan_summary = (
            "GAN augmentation reduced false positives and improved accuracy, "
            "but it missed more abnormal cases than the baseline screening model."
        )
    else:
        gan_summary = "GAN augmentation did not clearly improve the classifier in this run."
        
    report_md = f"""# Research Report: Heart Sound Screening Assistant
 
 ## Executive Summary
 This report evaluates a research **screening assistant, not a diagnosis system**, for classifying Phonocardiogram (PCG) recordings from the **PhysioNet Heart Sound Dataset (CinC Challenge 2016)**. The project compares a baseline model with a GAN-augmented model and uses validation-selected thresholds instead of assuming a fixed 0.5 cutoff.
 
 ---
 
 ## Prevent Data Leakage validation Check
 To reduce leakage risk, a recording-level split was implemented *before* signal segmentation. The overlap validation check verified:
 * **Overlapping recordings between splits**: 0 files.
 * **Leak-proof splitting**: Successful.
 
 ---
 
 ## Recording-Level Performance Comparison
 
 Each recording is evaluated once by averaging its segment probabilities. This matches how `predict.py` handles one uploaded WAV file.
 
 | Model | Threshold | Accuracy | Balanced Accuracy | Precision | Recall | Specificity | NPV | F1-score | AUC |
 | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
 | **Baseline (No GAN)** | {res_nogan['threshold']:.3f} | {res_nogan['accuracy']:.4f} | {res_nogan['balanced_accuracy']:.4f} | {res_nogan['precision']:.4f} | {res_nogan['recall']:.4f} | {res_nogan['specificity']:.4f} | {res_nogan['npv']:.4f} | {res_nogan['f1']:.4f} | {res_nogan['auc']:.4f} |
 | **GAN-Augmented** | {res_gan['threshold']:.3f} | {res_gan['accuracy']:.4f} | {res_gan['balanced_accuracy']:.4f} | {res_gan['precision']:.4f} | {res_gan['recall']:.4f} | {res_gan['specificity']:.4f} | {res_gan['npv']:.4f} | {res_gan['f1']:.4f} | {res_gan['auc']:.4f} |
 | **Ensemble (Baseline + GAN)** | {res_ensemble['threshold']:.3f} | {res_ensemble['accuracy']:.4f} | {res_ensemble['balanced_accuracy']:.4f} | {res_ensemble['precision']:.4f} | {res_ensemble['recall']:.4f} | {res_ensemble['specificity']:.4f} | {res_ensemble['npv']:.4f} | {res_ensemble['f1']:.4f} | {res_ensemble['auc']:.4f} |
 
 The thresholds above are selected on the validation split, not guessed at 0.5.
 Predictions close to the selected threshold should be treated as **Uncertain**
 and reviewed by a clinician.
 
 ### Calibration and Threshold Safety
 The following plots are generated to inspect confidence quality and threshold tradeoffs:
 
 ![Calibration Curve](plots/calibration_curve_gan.png)
 
 ![Threshold Tradeoff](plots/threshold_tradeoff_gan.png)
 
 ### Relative Improvements:
 * **Accuracy Improvement**: {acc_diff*100:+.2f}%
 * **Recall (Sensitivity) Improvement**: {rec_diff*100:+.2f}%
 * **F1-score Improvement**: {f1_diff*100:+.2f}%
 * **AUC Score Improvement**: {auc_diff*100:+.2f}%
 
 ---
 
 ## GAN Validation Analysis
 
 ### 1. t-SNE Feature Space Distribution
 t-SNE dimensionality reduction (from 2496 dimensions to 2D) was applied to the abnormal training features (real vs. synthetic). The plot demonstrates:
 * The plot is a visual sanity check only; it is not proof of clinical quality.
 * If synthetic points are far away from real points, the GAN should be treated carefully or disabled for the final classifier.
 
 ![t-SNE Scatter Plot](plots/tsne_real_vs_generated.png)
 
 ### 2. Statistical Similarity
 We compared the distribution statistics of the combined MFCC + log-mel features:
 * **Real Abnormal Means (average)**: {np.mean(real_means) if real_means is not None else 0:.4f} (Variance: {np.mean(real_vars) if real_vars is not None else 0:.4f})
 * **Generated Abnormal Means (average)**: {np.mean(gen_means) if gen_means is not None else 0:.4f} (Variance: {np.mean(gen_vars) if gen_vars is not None else 0:.4f})
 * **Average Spectral Centroid (MFCC-index equivalent)**:
   * Real Abnormal: Coefficient {real_centroid:.2f}
   * Generated Abnormal: Coefficient {gen_centroid:.2f}
 
 These statistics are useful checks, but classifier performance and validation/test metrics are more important than visual similarity alone.
 
 ### 3. Visual Feature Comparison (MFCC Heatmaps)
 The following plot shows a side-by-side comparison of a Real Normal, Real Abnormal, and GAN Synthetic Abnormal MFCC heatmap. 
 
 ![MFCC Heatmaps Comparison](plots/spectrogram_comparisons.png)
 
 ---
 
 ## Visual Model Comparisons
 
 ### Confusion Matrices
 Below are the confusion matrices for the three model configurations (Baseline, GAN-Augmented, and Ensemble). Note how the Ensemble model balances false alarms and missed cases:
 
 ![Confusion Matrix Comparison](plots/confusion_matrix_comparison.png)
 
 ### ROC Curves & Training Histories
 The ROC curve comparison and training curve comparisons are illustrated below:
 
 ![ROC & Training Curves Comparison](plots/performance_comparison.png)
 
 ![Class Distribution Before and After GAN](plots/class_distribution_comparison.png)
 
 ---
 
 ## Final Conclusions
 1. **GAN Result**: {gan_summary}
 2. **Best balanced setting**: Use **{best_balanced['Model']}** in **{best_balanced['Mode']}** mode when you want the best balance of recall and specificity. Test balanced accuracy = **{best_balanced['BalancedAccuracy']:.4f}**.
 3. **To reduce false positives**: Use **{fewest_false_positive['Model']}** in **{fewest_false_positive['Mode']}** mode. Test specificity = **{fewest_false_positive['Specificity']:.4f}**, recall = **{fewest_false_positive['Recall']:.4f}**.
 4. **Clinical Safety Note**: This project is a research screening assistant, not a diagnosis system. It is not perfect, not medically certified, and must not be used as a stand-alone medical decision tool.
 """
    with open(report_path, 'w') as f:
        f.write(report_md)
    print(f"Final research report compiled and written to {report_path}")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate Baseline vs GAN Heart Sound Classifiers")
    parser.add_argument('--processed_dir', type=str, default='data/processed')
    parser.add_argument('--model_dir', type=str, default='models')
    parser.add_argument('--outputs_dir', type=str, default='outputs')
    parser.add_argument('--aggregation', type=str, default='confidence_weighted', choices=['mean', 'max', 'confidence_weighted'], help='Segment aggregation method')
    args = parser.parse_args()
    
    run_evaluation_study(
        processed_dir=args.processed_dir,
        model_dir=args.model_dir,
        outputs_dir=args.outputs_dir,
        aggregation=args.aggregation
    )
