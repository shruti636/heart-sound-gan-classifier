import os
import argparse
import numpy as np
import tensorflow as tf
import soundfile as sf
import librosa
from preprocessing import preprocess_signal, segment_signal, extract_features

# Resolve the default model path relative to the directory containing predict.py
DEFAULT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MODEL_PATH = os.path.join(DEFAULT_DIR, 'models', 'cnn_classifier_gan.keras')

def predict(wav_path, model_path=DEFAULT_MODEL_PATH, fs=2000, model=None):
    """
    Predicts whether a heart sound recording is Normal or Abnormal.
    Pipeline: WAV -> Preprocessing -> Feature Extraction -> Classifier -> Inference.
    """
    if not os.path.exists(wav_path):
        raise FileNotFoundError(f"Audio file not found at: {wav_path}")
        
    # 1. Load trained classifier model if not pre-loaded
    if model is None:
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Trained classifier model not found at: {model_path}. Please train the classifier first.")
        model = tf.keras.models.load_model(model_path)
    
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
    
    # 4. Segment into fixed-length windows (2.52 seconds = 5040 samples at 2000Hz)
    segments = segment_signal(y_preprocessed, segment_len=5040, overlap=1000)
    
    # 5. Extract 39 features (13 MFCC + 13 Delta + 13 Delta-Delta) for each segment
    mfcc_segments = []
    for seg in segments:
        features = extract_features(seg, fs=fs)
        mfcc_segments.append(features['mfcc'])
        
    # Form batch array of shape: (num_segments, 64, 39)
    X = np.array(mfcc_segments, dtype=np.float32)
    
    # 6. Run model prediction (returns probabilities)
    probs = model.predict(X, verbose=0).flatten()
    
    # Aggregate segment-level probabilities to get the global classification
    avg_prob = np.mean(probs)
    
    if avg_prob >= 0.5:
        predicted_class = "Abnormal"
        confidence = avg_prob
    else:
        predicted_class = "Normal"
        confidence = 1.0 - avg_prob
        
    return {
        'class': predicted_class,
        'confidence': float(confidence),
        'segment_probs': probs.tolist(),
        'num_segments': len(segments)
    }

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Predict Heart Sound Classification (Normal/Abnormal)")
    parser.add_argument('--wav', type=str, required=True, help='Path to the input WAV file')
    parser.add_argument('--model', type=str, default='models/cnn_classifier_gan.keras', help='Path to the Keras model checkpoint')
    
    args = parser.parse_args()
    
    try:
        result = predict(args.wav, args.model)
        print("\n" + "="*50)
        print("         HEART SOUND INFERENCE RESULT")
        print("="*50)
        print(f"File Path:  {args.wav}")
        print(f"Prediction: {result['class']}")
        print(f"Confidence: {result['confidence'] * 100:.2f}%")
        print(f"Segments:   {result['num_segments']}")
        print("="*50 + "\n")
    except Exception as e:
        print(f"Error during inference: {e}")
