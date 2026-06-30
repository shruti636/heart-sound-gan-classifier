import os
import argparse
import json
import numpy as np
import tensorflow as tf
import soundfile as sf
import librosa
from preprocessing import preprocess_signal, segment_signal, extract_features

# Resolve the default model path relative to the directory containing predict.py
DEFAULT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MODEL_PATH = os.path.join(DEFAULT_DIR, 'models', 'cnn_classifier_gan.keras')


def load_threshold(model_path, default_threshold=0.5, default_margin=0.08, mode="screening", is_ensemble=False):
    """Load validation-selected threshold if evaluation.py has created one."""
    model_dir = os.path.dirname(os.path.abspath(model_path))
    config_path = os.path.join(model_dir, 'model_thresholds.json')
    model_name = 'ensemble' if is_ensemble else os.path.basename(model_path)

    if not os.path.exists(config_path):
        return default_threshold, default_margin

    with open(config_path, 'r') as f:
        config = json.load(f)

    model_config = config.get(model_name, {})
    if mode == 'specificity':
        key = 'specificity_threshold'
    elif mode == 'balanced':
        key = 'balanced_threshold'
    else:
        key = 'screening_threshold'
    threshold = model_config.get(key, model_config.get('threshold', default_threshold))
    margin = config.get('uncertain_margin', default_margin)
    return float(threshold), float(margin)


def aggregate_probs(probs, method='mean'):
    """Aggregate segment probabilities using mean, max, or confidence-weighted averaging."""
    if len(probs) == 0:
        return 0.0
    if method == 'mean':
        return np.mean(probs)
    elif method == 'max':
        return np.max(probs)
    elif method == 'confidence_weighted':
        weights = np.abs(probs - 0.5)
        weight_sum = np.sum(weights)
        if weight_sum < 1e-8:
            return np.mean(probs)
        return np.sum(probs * weights) / weight_sum
    else:
        raise ValueError(f"Unknown aggregation method: {method}")


def predict(
    wav_path,
    model_path=DEFAULT_MODEL_PATH,
    fs=2000,
    model=None,
    threshold=None,
    uncertain_margin=None,
    mode="screening",
    aggregation=None,
    ensemble=False,
):
    """
    Predicts whether a heart sound recording is Normal or Abnormal.
    Pipeline: WAV -> Preprocessing -> Feature Extraction -> Classifier -> Inference.
    """
    if not os.path.exists(wav_path):
        raise FileNotFoundError(f"Audio file not found at: {wav_path}")
        
    model_dir = os.path.dirname(os.path.abspath(model_path))
    
    # 1. Load trained classifier model(s) if not pre-loaded
    if model is None:
        if ensemble:
            path_nogan = os.path.join(model_dir, 'cnn_classifier_nogan.keras')
            path_gan = os.path.join(model_dir, 'cnn_classifier_gan.keras')
            if not os.path.exists(path_nogan) or not os.path.exists(path_gan):
                raise FileNotFoundError("Both Baseline and GAN-augmented classifiers must be trained to run Ensemble prediction.")
            model_nogan = tf.keras.models.load_model(path_nogan, compile=False)
            model_gan = tf.keras.models.load_model(path_gan, compile=False)
            expected_features = model_nogan.input_shape[-1]
        else:
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Trained classifier model not found at: {model_path}. Please train the classifier first.")
            model = tf.keras.models.load_model(model_path, compile=False)
            expected_features = model.input_shape[-1]
    else:
        expected_features = model.input_shape[-1]
    
    # 2. Load raw .wav audio
    y, orig_fs = sf.read(wav_path)
    
    # Convert stereo to mono by averaging channels if necessary
    if len(y.shape) > 1:
        y = np.mean(y, axis=1)
        
    # Resample to 2000Hz if necessary
    if orig_fs != fs:
        y = librosa.resample(y, orig_sr=orig_fs, target_sr=fs)
        
    # 3. Apply Butterworth Bandpass Filter (25Hz-400Hz) and Normalize
    y_preprocessed = preprocess_signal(y, fs)
    
    # 4. Segment into windows dynamically based on model expected frames
    expected_frames = model_nogan.input_shape[1] if ensemble else model.input_shape[1]
    if expected_frames == 25:
        segments = segment_signal(y_preprocessed, segment_len=2000, overlap=0, use_cardiac_cycles=True, fs=fs)
    else:
        segments = segment_signal(y_preprocessed, segment_len=5040, overlap=1000, use_cardiac_cycles=False, fs=fs)
    
    # 5. Extract combined MFCC + log-mel features for each segment
    mfcc_segments = []
    for seg in segments:
        features = extract_features(seg, fs=fs)
        if expected_features == features['mfcc_only'].shape[-1]:
            mfcc_segments.append(features['mfcc_only'])
        else:
            mfcc_segments.append(features['features'])
        
    # Form batch array of shape: (num_segments, time_frames, feature_dim)
    X = np.array(mfcc_segments, dtype=np.float32)
    
    # 6. Run model prediction (returns probabilities)
    if ensemble:
        probs_nogan = model_nogan.predict(X, verbose=0).flatten()
        probs_gan = model_gan.predict(X, verbose=0).flatten()
        probs = 0.5 * (probs_nogan + probs_gan)
    else:
        probs = model.predict(X, verbose=0).flatten()
    
    # Determine segment aggregation method
    if aggregation is None:
        # Load from config if available
        config_path = os.path.join(model_dir, 'model_thresholds.json')
        if os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    config = json.load(f)
                aggregation = config.get('aggregation_method', 'mean')
            except Exception:
                aggregation = 'mean'
        else:
            aggregation = 'mean'
            
    # Aggregate segment-level probabilities to get the global classification.
    avg_prob = aggregate_probs(probs, method=aggregation)

    if threshold is None or uncertain_margin is None:
        loaded_threshold, loaded_margin = load_threshold(model_path, mode=mode, is_ensemble=ensemble)
        threshold = loaded_threshold if threshold is None else threshold
        uncertain_margin = loaded_margin if uncertain_margin is None else uncertain_margin

    lower_bound = threshold - uncertain_margin
    upper_bound = threshold + uncertain_margin

    if lower_bound < avg_prob < upper_bound:
        predicted_class = "Uncertain"
        confidence = 1.0 - abs(avg_prob - threshold) / max(uncertain_margin, 1e-6)
        recommendation = "Model score is close to the decision threshold. Ask a clinician to review the recording."
    elif avg_prob >= upper_bound:
        predicted_class = "Abnormal"
        confidence = min((avg_prob - threshold) / max(1.0 - threshold, 1e-6), 1.0)
        recommendation = "Possible abnormal heart sound. This is a screening result, not a diagnosis."
    else:
        predicted_class = "Normal"
        confidence = min((threshold - avg_prob) / max(threshold, 1e-6), 1.0)
        recommendation = "No abnormal pattern detected by the model. Seek clinical review if symptoms exist."
        
    return {
        'class': predicted_class,
        'confidence': float(confidence),
        'abnormal_probability': float(avg_prob),
        'threshold': float(threshold),
        'uncertain_margin': float(uncertain_margin),
        'mode': mode,
        'recommendation': recommendation,
        'segment_probs': probs.tolist(),
        'num_segments': len(segments)
    }

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Predict Heart Sound Classification (Normal/Abnormal)")
    parser.add_argument('--wav', type=str, required=True, help='Path to the input WAV file')
    parser.add_argument('--model', type=str, default='models/cnn_classifier_gan.keras', help='Path to the Keras model checkpoint')
    parser.add_argument('--threshold', type=float, default=None, help='Optional manual abnormal threshold')
    parser.add_argument('--uncertain_margin', type=float, default=None, help='Optional uncertain zone around the threshold')
    parser.add_argument(
        '--mode',
        choices=['screening', 'balanced', 'specificity'],
        default='screening',
        help='screening favors recall; balanced is even; specificity reduces false positives',
    )
    parser.add_argument(
        '--aggregation',
        choices=['mean', 'max', 'confidence_weighted'],
        default=None,
        help='Override saved aggregation method',
    )
    parser.add_argument('--ensemble', action='store_true', help='Use Ensemble of Baseline and GAN models')
    
    args = parser.parse_args()
    
    try:
        result = predict(
            args.wav,
            args.model,
            threshold=args.threshold,
            uncertain_margin=args.uncertain_margin,
            mode=args.mode,
            aggregation=args.aggregation,
            ensemble=args.ensemble,
        )
        print("\n" + "="*50)
        print("         HEART SOUND INFERENCE RESULT")
        print("="*50)
        print(f"File Path:  {args.wav}")
        print(f"Prediction: {result['class']}")
        print(f"Model Confidence: {result['confidence'] * 100:.2f}%")
        print(f"Abnormal Probability: {result['abnormal_probability']:.4f}")
        print(f"Decision Threshold:   {result['threshold']:.4f}")
        print(f"Operating Mode:       {result['mode']}")
        print(f"Segments:   {result['num_segments']}")
        print(f"Recommendation: {result['recommendation']}")
        print("="*50 + "\n")
    except Exception as e:
        print(f"Error during inference: {e}")
