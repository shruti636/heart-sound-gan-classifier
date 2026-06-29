import os
import argparse
import numpy as np
import scipy.signal as signal
import soundfile as sf
import librosa
import pandas as pd
from sklearn.model_selection import train_test_split


FEATURE_KEY = "features"

def butter_bandpass(lowcut, highcut, fs, order=4):
    """
    Design a Butterworth bandpass filter.
    """
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = signal.butter(order, [low, high], btype='band')
    return b, a

def butter_bandpass_filter(data, lowcut, highcut, fs, order=4):
    """
    Apply a Butterworth bandpass filter to a signal.
    """
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = signal.lfilter(b, a, data)
    return y

def preprocess_signal(y, fs, lowcut=25, highcut=400):
    """
    Filters and normalizes the signal.
    """
    y_filtered = butter_bandpass_filter(y, lowcut, highcut, fs, order=4)
    max_val = np.max(np.abs(y_filtered))
    if max_val > 0:
        y_filtered = y_filtered / max_val
    return y_filtered

def segment_signal(y, segment_len=5040, overlap=0):
    """
    Segments the signal into fixed-length windows.
    Default segment_len=5040 corresponds to 2.52 seconds at fs=2000Hz.
    """
    step = segment_len - overlap
    segments = []
    
    if len(y) < segment_len:
        padded = np.zeros(segment_len)
        padded[:len(y)] = y
        segments.append(padded)
        return segments
        
    for start in range(0, len(y) - segment_len + 1, step):
        segments.append(y[start : start + segment_len])
        
    return segments

def normalize_feature_block(block):
    """Scale one feature block to [-1, 1] for stable neural-network training."""
    block_min, block_max = block.min(), block.max()
    if block_max - block_min <= 0:
        return np.zeros_like(block), np.array([block_min, block_max])
    normalized = 2.0 * (block - block_min) / (block_max - block_min) - 1.0
    return normalized, np.array([block_min, block_max])


def fix_time_frames(features, target_frames=64):
    """Pad or crop feature matrices to a fixed number of time frames."""
    if features.shape[1] < target_frames:
        pad_width = target_frames - features.shape[1]
        features = np.pad(features, ((0, 0), (0, pad_width)), mode="constant")
    return features[:, :target_frames]


def extract_features(y, fs=2000, n_fft=256, hop_length=80, n_mfcc=13, n_mels=32):
    """
    Extract MFCC + log-mel features from one audio segment.

    Output shape is (64, 71):
      - 39 MFCC-style features: MFCC + delta + delta-delta
      - 32 log-mel spectrogram features
    """
    mfcc = librosa.feature.mfcc(y=y, sr=fs, n_fft=n_fft, hop_length=hop_length, n_mfcc=n_mfcc, center=True)
    delta = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)

    mel = librosa.feature.melspectrogram(
        y=y,
        sr=fs,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        fmin=25,
        fmax=400,
        center=True,
    )
    log_mel = librosa.power_to_db(mel, ref=np.max)

    mfcc_features = fix_time_frames(np.vstack([mfcc, delta, delta2]))
    log_mel = fix_time_frames(log_mel)

    mfcc_norm, mfcc_bounds = normalize_feature_block(mfcc_features)
    log_mel_norm, mel_bounds = normalize_feature_block(log_mel)
    combined = np.vstack([mfcc_norm, log_mel_norm]).T
    
    feature_dict = {
        FEATURE_KEY: combined,
        'mfcc': combined,
        'mfcc_only': mfcc_norm.T,
        'log_mel': log_mel_norm.T,
        'mfcc_bounds': mfcc_bounds,
        'mel_bounds': mel_bounds,
    }
    
    return feature_dict

def preprocess_subset(df_subset, raw_dir, fs=2000):
    """
    Loops over a subset of recordings, preprocesses, segments, and extracts MFCCs.
    """
    mfcc_list = []
    labels_list = []
    bounds_list = []
    source_list = []
    
    for idx, row in df_subset.iterrows():
        filepath = os.path.join(raw_dir, row['filename'])
        if not os.path.exists(filepath):
            print(f"Warning: File {filepath} not found. Skipping.")
            continue
            
        label_str = row['label']
        label = 1 if label_str.lower() in ['abnormal', '1', 1] else 0
        
        try:
            y, file_fs = sf.read(filepath)
            
            if file_fs != fs:
                y = librosa.resample(y, orig_sr=file_fs, target_sr=fs)
                
            y_preprocessed = preprocess_signal(y, fs)
            # Use overlap=1000 for data segment extraction
            segments = segment_signal(y_preprocessed, segment_len=5040, overlap=1000)
            
            for seg in segments:
                features = extract_features(seg, fs=fs)
                mfcc_list.append(features[FEATURE_KEY])
                labels_list.append(label)
                bounds_list.append(features['mfcc_bounds'])
                source_list.append(row['filename'])
                
        except Exception as e:
            print(f"Error processing file {row['filename']}: {e}")
            
    X = np.array(mfcc_list, dtype=np.float32)
    y_arr = np.array(labels_list, dtype=np.int32)
    bounds = np.array(bounds_list, dtype=np.float32)
    
    sources = np.array(source_list)
    return X, y_arr, bounds, sources

def process_dataset(raw_dir='data/raw', processed_dir='data/processed', fs=2000):
    """
    Processes all wav files in raw_dir by performing a recording-level train/val/test split
    to prevent data leakage, then processes each split separately and saves the outputs.
    """
    os.makedirs(processed_dir, exist_ok=True)
    csv_path = os.path.join(raw_dir, 'reference.csv')
    
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Dataset reference file not found at {csv_path}. Please run downloader first.")
        
    df = pd.read_csv(csv_path)
    print(f"Loaded database mapping: found {len(df)} total recordings.")
    
    # 1. Stratified Recording-Level split: 70% Train, 15% Val, 15% Test
    # Split into 85% Train/Val and 15% Test
    df_train_val, df_test = train_test_split(
        df, test_size=0.15, random_state=42, stratify=df['label']
    )
    # Split Train/Val into 70% Train and 15% Val
    df_train, df_val = train_test_split(
        df_train_val, test_size=0.1765, random_state=42, stratify=df_train_val['label']
    )
    
    print("\nRecording-level splits (independent audio files):")
    print(f"  Train recordings: {len(df_train)} (Normal: {sum(df_train['label'] == 'normal')}, Abnormal: {sum(df_train['label'] == 'abnormal')})")
    print(f"  Val recordings: {len(df_val)} (Normal: {sum(df_val['label'] == 'normal')}, Abnormal: {sum(df_val['label'] == 'abnormal')})")
    print(f"  Test recordings: {len(df_test)} (Normal: {sum(df_test['label'] == 'normal')}, Abnormal: {sum(df_test['label'] == 'abnormal')})")
    
    # 2. Strict Overlap Validation Checks (Prevent Data Leakage)
    train_recordings = set(df_train['filename'])
    val_recordings = set(df_val['filename'])
    test_recordings = set(df_test['filename'])
    
    overlap_train_test = train_recordings.intersection(test_recordings)
    overlap_train_val = train_recordings.intersection(val_recordings)
    overlap_val_test = val_recordings.intersection(test_recordings)
    
    print("\n--- Leakage Validation Check ---")
    print(f"  Overlap between Train and Test: {len(overlap_train_test)} files")
    print(f"  Overlap between Train and Val: {len(overlap_train_val)} files")
    print(f"  Overlap between Val and Test: {len(overlap_val_test)} files")
    
    # Assert absolutely no overlaps
    assert len(overlap_train_test) == 0, "CRITICAL ERROR: Train and Test recordings overlap!"
    assert len(overlap_train_val) == 0, "CRITICAL ERROR: Train and Val recordings overlap!"
    assert len(overlap_val_test) == 0, "CRITICAL ERROR: Val and Test recordings overlap!"
    print("  Validation successful: 0 overlapping recordings detected. Splitting is leak-proof!")
    
    # Write splitting manifests to verify
    with open(os.path.join(processed_dir, 'train_source_files.txt'), 'w') as f:
        f.write("\n".join(sorted(list(train_recordings))))
    with open(os.path.join(processed_dir, 'val_source_files.txt'), 'w') as f:
        f.write("\n".join(sorted(list(val_recordings))))
    with open(os.path.join(processed_dir, 'test_source_files.txt'), 'w') as f:
        f.write("\n".join(sorted(list(test_recordings))))
    print("  Split source filename logs saved to processed dir.")
    
    # 3. Preprocess subsets separately
    print("\nSegmenting and extracting features for Train split...")
    X_train, y_train, bounds_train, sources_train = preprocess_subset(df_train, raw_dir, fs)
    
    print("Segmenting and extracting features for Val split...")
    X_val, y_val, bounds_val, sources_val = preprocess_subset(df_val, raw_dir, fs)
    
    print("Segmenting and extracting features for Test split...")
    X_test, y_test, bounds_test, sources_test = preprocess_subset(df_test, raw_dir, fs)
    
    # Segment distribution summary
    print("\nPreprocessed segment-level counts (after windowing):")
    print(f"  Train segments: {X_train.shape[0]} (Normal: {sum(y_train == 0)}, Abnormal: {sum(y_train == 1)})")
    print(f"  Val segments: {X_val.shape[0]} (Normal: {sum(y_val == 0)}, Abnormal: {sum(y_val == 1)})")
    print(f"  Test segments: {X_test.shape[0]} (Normal: {sum(y_test == 0)}, Abnormal: {sum(y_test == 1)})")
    
    # Save arrays
    np.save(os.path.join(processed_dir, 'X_train.npy'), X_train)
    np.save(os.path.join(processed_dir, 'y_train.npy'), y_train)
    np.save(os.path.join(processed_dir, 'bounds_train.npy'), bounds_train)
    np.save(os.path.join(processed_dir, 'sources_train.npy'), sources_train)
    
    np.save(os.path.join(processed_dir, 'X_val.npy'), X_val)
    np.save(os.path.join(processed_dir, 'y_val.npy'), y_val)
    np.save(os.path.join(processed_dir, 'bounds_val.npy'), bounds_val)
    np.save(os.path.join(processed_dir, 'sources_val.npy'), sources_val)
    
    np.save(os.path.join(processed_dir, 'X_test.npy'), X_test)
    np.save(os.path.join(processed_dir, 'y_test.npy'), y_test)
    np.save(os.path.join(processed_dir, 'bounds_test.npy'), bounds_test)
    np.save(os.path.join(processed_dir, 'sources_test.npy'), sources_test)
    
    print(f"\nAll processed splits saved successfully in {processed_dir}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Preprocess Real Heart Sound Dataset")
    parser.add_argument('--raw_dir', type=str, default='data/raw', help='Directory containing raw .wav files')
    parser.add_argument('--processed_dir', type=str, default='data/processed', help='Directory to save processed features')
    parser.add_argument('--fs', type=int, default=2000, help='Sampling rate for processing')
    
    args = parser.parse_args()
    process_dataset(raw_dir=args.raw_dir, processed_dir=args.processed_dir, fs=args.fs)
