import os
import numpy as np
import scipy.signal as signal
import soundfile as sf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import librosa
import librosa.display

def generate_synthetic_heart_beat(duration=5.0, fs=2000, heart_rate=70, abnormal=False, murmur_type='systolic'):
    """
    Generates a synthetic heart sound signal (PCG) simulating S1 and S2 sounds.
    
    Parameters:
        duration (float): Duration of the signal in seconds.
        fs (int): Sampling rate in Hz.
        heart_rate (int): Heart rate in beats per minute.
        abnormal (bool): If True, injects murmur sounds representing abnormality.
        murmur_type (str): 'systolic' or 'diastolic' murmur if abnormal is True.
        
    Returns:
        numpy.ndarray: Simulated heart sound waveform.
    """
    t = np.arange(0, duration, 1.0 / fs)
    y = np.zeros_like(t)
    
    # Calculate heartbeat period in seconds
    beat_period = 60.0 / heart_rate
    num_beats = int(duration / beat_period)
    
    # Base characteristics
    s1_freq = 50.0   # Hz
    s1_dur = 0.1     # seconds
    s2_freq = 80.0   # Hz
    s2_dur = 0.08    # seconds
    
    # Systolic interval is typically around 1/3 of the cardiac cycle
    systolic_interval = 0.3 * beat_period
    
    for i in range(num_beats + 1):
        beat_start_time = i * beat_period
        
        # S1 time and signal
        s1_start = beat_start_time
        s1_idx = (t >= s1_start) & (t < s1_start + s1_dur)
        if np.any(s1_idx):
            t_s1 = t[s1_idx] - s1_start
            env_s1 = np.sin(np.pi * t_s1 / s1_dur)
            s1_wave = env_s1 * np.sin(2 * np.pi * s1_freq * t_s1)
            y[s1_idx] += s1_wave
            
        # S2 time and signal
        s2_start = beat_start_time + systolic_interval
        s2_idx = (t >= s2_start) & (t < s2_start + s2_dur)
        if np.any(s2_idx):
            t_s2 = t[s2_idx] - s2_start
            env_s2 = np.sin(np.pi * t_s2 / s2_dur)
            s2_wave = 0.8 * env_s2 * np.sin(2 * np.pi * s2_freq * t_s2)
            y[s2_idx] += s2_wave
            
        # Add murmur if abnormal
        if abnormal:
            if murmur_type == 'systolic':
                # Murmur between S1 and S2
                mur_start = s1_start + s1_dur
                mur_dur = systolic_interval - s1_dur - 0.02
                mur_idx = (t >= mur_start) & (t < mur_start + mur_dur)
                if np.any(mur_idx) and mur_dur > 0:
                    t_mur = t[mur_idx] - mur_start
                    mur_env = np.sin(np.pi * t_mur / mur_dur) ** 2
                    noise = np.random.normal(0, 0.25, size=len(t_mur))
                    b, a = signal.butter(4, [150 / (fs/2), 350 / (fs/2)], btype='bandpass')
                    filtered_noise = signal.lfilter(b, a, noise)
                    y[mur_idx] += mur_env * filtered_noise
                    
            elif murmur_type == 'diastolic':
                # Murmur between S2 and next S1
                mur_start = s2_start + s2_dur
                mur_dur = (beat_period - systolic_interval - s2_dur - 0.05)
                mur_idx = (t >= mur_start) & (t < mur_start + mur_dur)
                if np.any(mur_idx) and mur_dur > 0:
                    t_mur = t[mur_idx] - mur_start
                    mur_env = np.exp(-4 * t_mur / mur_dur)
                    noise = np.random.normal(0, 0.2, size=len(t_mur))
                    b, a = signal.butter(4, [100 / (fs/2), 300 / (fs/2)], btype='bandpass')
                    filtered_noise = signal.lfilter(b, a, noise)
                    y[mur_idx] += mur_env * filtered_noise
                    
    # Add minor background environment noise
    y += np.random.normal(0, 0.05, size=len(y))
    
    # Normalize to [-1.0, 1.0]
    max_val = np.max(np.abs(y))
    if max_val > 0:
        y = y / max_val
        
    return y

def generate_synthetic_raw_dataset(dest_dir='data/raw', num_samples=100, fs=2000):
    """
    Creates a set of synthetic .wav heart sound files for normal and abnormal cases.
    Generates a CSV reference metadata file.
    """
    os.makedirs(dest_dir, exist_ok=True)
    
    csv_path = os.path.join(dest_dir, 'reference.csv')
    with open(csv_path, 'w') as f:
        f.write("filename,label\n")
        
        num_normal = int(num_samples * 0.6)
        num_abnormal = num_samples - num_normal
        
        print(f"Generating {num_normal} normal and {num_abnormal} abnormal synthetic heart sound wav files...")
        
        for idx in range(num_normal):
            filename = f"normal_{idx+1:04d}.wav"
            filepath = os.path.join(dest_dir, filename)
            
            hr = np.random.randint(60, 85)
            dur = np.random.uniform(4.5, 6.0)
            
            y = generate_synthetic_heart_beat(duration=dur, fs=fs, heart_rate=hr, abnormal=False)
            sf.write(filepath, y, fs)
            f.write(f"{filename},normal\n")
            
        for idx in range(num_abnormal):
            filename = f"abnormal_{idx+1:04d}.wav"
            filepath = os.path.join(dest_dir, filename)
            
            hr = np.random.randint(70, 95)
            dur = np.random.uniform(4.5, 6.0)
            mur_type = 'systolic' if idx % 2 == 0 else 'diastolic'
            
            y = generate_synthetic_heart_beat(duration=dur, fs=fs, heart_rate=hr, abnormal=True, murmur_type=mur_type)
            sf.write(filepath, y, fs)
            f.write(f"{filename},abnormal\n")
            
    print(f"Synthetic dataset generation complete. Saved reference to {csv_path}")

def plot_waveform_and_mfcc(audio_path, save_plot_path=None):
    """
    Loads an audio file and plots its waveform and MFCC heatmap side-by-side.
    """
    y, fs = sf.read(audio_path)
    
    # Compute MFCC
    n_fft = 256
    hop_length = 80
    n_mfcc = 13
    
    mfcc = librosa.feature.mfcc(y=y, sr=fs, n_fft=n_fft, hop_length=hop_length, n_mfcc=n_mfcc)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Waveform
    t = np.arange(len(y)) / fs
    axes[0].plot(t, y, color='#1f77b4', alpha=0.8)
    axes[0].set_title('Waveform (Phonocardiogram)', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Time (seconds)', fontsize=12)
    axes[0].set_ylabel('Amplitude', fontsize=12)
    axes[0].grid(True, linestyle='--', alpha=0.6)
    
    # MFCC Heatmap
    img = librosa.display.specshow(mfcc, sr=fs, hop_length=hop_length, x_axis='time', ax=axes[1], cmap='coolwarm')
    axes[1].set_title('MFCC Heatmap (13 Coefficients)', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Time (seconds)', fontsize=12)
    axes[1].set_ylabel('MFCC Coefficient Index', fontsize=12)
    fig.colorbar(img, ax=axes[1])
    
    plt.tight_layout()
    if save_plot_path:
        os.makedirs(os.path.dirname(save_plot_path), exist_ok=True)
        plt.savefig(save_plot_path, dpi=150)
        plt.close()
    else:
        plt.show()
