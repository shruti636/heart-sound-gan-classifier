import os
import argparse
import numpy as np
import tensorflow as tf
import soundfile as sf
import librosa
from tensorflow.keras.optimizers import Adam
from wavegan_model import build_wavegan_generator, build_wavegan_discriminator
from preprocessing import preprocess_signal

# Set random seed for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

def gradient_penalty(critic, real_images, fake_images):
    """
    Computes the gradient penalty for WGAN-GP.
    Enforces the Lipschitz-1 constraint on the critic.
    """
    batch_size = tf.shape(real_images)[0]
    alpha = tf.random.uniform([batch_size, 1, 1], 0.0, 1.0)
    diff = fake_images - real_images
    interpolates = real_images + alpha * diff
    
    with tf.GradientTape() as gp_tape:
        gp_tape.watch(interpolates)
        critic_pred = critic(interpolates, training=True)
        
    grads = gp_tape.gradient(critic_pred, [interpolates])[0]
    # Reduce sum over sequence length (axis 1) and channel (axis 2)
    norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=[1, 2]) + 1e-12)
    gp = tf.reduce_mean((norm - 1.0) ** 2)
    return gp

def critic_loss(real_output, fake_output):
    return tf.reduce_mean(fake_output) - tf.reduce_mean(real_output)

def generator_loss(fake_output):
    return -tf.reduce_mean(fake_output)

class WaveGAN_Pipeline:
    def __init__(self, latent_dim=100, wave_shape=(5120, 1), gp_weight=10.0):
        self.latent_dim = latent_dim
        self.wave_shape = wave_shape
        self.gp_weight = gp_weight
        
        self.generator = build_wavegan_generator(self.latent_dim)
        self.discriminator = build_wavegan_discriminator(self.wave_shape) # acts as the Critic
        
        # WGAN-GP learning rate 1e-4, beta_1=0.0, beta_2=0.9
        self.generator_optimizer = Adam(learning_rate=0.0001, beta_1=0.0, beta_2=0.9)
        self.discriminator_optimizer = Adam(learning_rate=0.0001, beta_1=0.0, beta_2=0.9)
        
    @tf.function
    def train_step_critic(self, real_waves):
        batch_size = tf.shape(real_waves)[0]
        noise = tf.random.normal([batch_size, self.latent_dim])
        
        with tf.GradientTape() as critic_tape:
            fake_waves = self.generator(noise, training=True)
            
            real_output = self.discriminator(real_waves, training=True)
            fake_output = self.discriminator(fake_waves, training=True)
            
            c_loss = critic_loss(real_output, fake_output)
            gp = gradient_penalty(self.discriminator, real_waves, fake_waves)
            
            total_c_loss = c_loss + self.gp_weight * gp
            
        grads = critic_tape.gradient(total_c_loss, self.discriminator.trainable_variables)
        self.discriminator_optimizer.apply_gradients(zip(grads, self.discriminator.trainable_variables))
        return total_c_loss
        
    @tf.function
    def train_step_generator(self, batch_size):
        noise = tf.random.normal([batch_size, self.latent_dim])
        
        with tf.GradientTape() as gen_tape:
            fake_waves = self.generator(noise, training=True)
            fake_output = self.discriminator(fake_waves, training=True)
            g_loss = generator_loss(fake_output)
            
        grads = gen_tape.gradient(g_loss, self.generator.trainable_variables)
        self.generator_optimizer.apply_gradients(zip(grads, self.generator.trainable_variables))
        return g_loss

def load_and_segment_abnormal_waveforms(raw_dir='data/raw', fs=2000, segment_len=5040, pad_len=5120):
    """
    Loads raw WAV recordings, resamples, filters, segments, and pads to 5120 samples.
    Selects only abnormal recordings (based on manifest list or filename).
    """
    manifest_path = os.path.join(raw_dir, 'reference.csv')
    if not os.path.exists(manifest_path):
        raise FileNotFoundError(f"reference.csv manifest not found at {manifest_path}!")
        
    import pandas as pd
    df = pd.read_csv(manifest_path)
    
    abnormal_files = df[df['label'].str.lower() == 'abnormal']['filename'].values
    print(f"Manifest indicates {len(abnormal_files)} abnormal recording files.")
    
    waveforms = []
    
    for fname in abnormal_files:
        if not fname.endswith('.wav'):
            w_path = os.path.join(raw_dir, f"{fname}.wav")
        else:
            w_path = os.path.join(raw_dir, fname)
            
        if not os.path.exists(w_path):
            continue
            
        # Load wave
        y, orig_fs = sf.read(w_path)
        if len(y.shape) > 1:
            y = np.mean(y, axis=1)
            
        # Resample to 2000Hz
        if orig_fs != fs:
            y = librosa.resample(y, orig_sr=orig_fs, target_sr=fs)
            
        # Filter
        y_filt = preprocess_signal(y, fs=fs)
        
        # Segment into non-overlapping windows of segment_len
        step = segment_len
        for start in range(0, len(y_filt) - segment_len + 1, step):
            seg = y_filt[start:start + segment_len]
            # Pad from 5040 to 5120 to fit WaveGAN architecture
            seg_padded = np.pad(seg, (0, pad_len - segment_len), mode='constant')
            waveforms.append(seg_padded)
            
    waveforms = np.array(waveforms, dtype=np.float32)
    # Reshape to (N, 5120, 1) for 1D convolutions
    waveforms = np.expand_dims(waveforms, axis=-1)
    return waveforms

def train_wavegan(epochs=30, batch_size=16, latent_dim=100, raw_dir='data/raw', model_dir='models', outputs_dir='outputs/wavegan'):
    os.makedirs(outputs_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    
    # 1. Load raw audio waves
    print("Loading and segmenting abnormal heart audio waveforms...")
    real_waves = load_and_segment_abnormal_waveforms(raw_dir)
    real_waves = real_waves[:256] # Sub-sample to speed up CPU training while allowing more epochs
    print(f"Loaded {len(real_waves)} raw abnormal waves of shape {real_waves.shape}.")
    
    if len(real_waves) == 0:
        print("No abnormal waves found. Please verify raw files.")
        return
        
    dataset = tf.data.Dataset.from_tensor_slices(real_waves).shuffle(len(real_waves)).batch(batch_size, drop_remainder=True)
    
    # 2. Init WaveGAN
    gan = WaveGAN_Pipeline(latent_dim=latent_dim, wave_shape=(5120, 1))
    
    print("Starting WaveGAN raw audio training loop...")
    
    step = 0
    n_critic = 5 # Standard WGAN-GP Critic updates per Generator update
    
    for epoch in range(1, epochs + 1):
        epoch_g_loss = []
        epoch_c_loss = []
        
        for wave_batch in dataset:
            # Train Critic
            c_loss = gan.train_step_critic(wave_batch)
            epoch_c_loss.append(c_loss.numpy())
            
            # Train Generator every n_critic steps
            if step % n_critic == 0:
                g_loss = gan.train_step_generator(batch_size)
                epoch_g_loss.append(g_loss.numpy())
                
            step += 1
            
        mean_g_loss = np.mean(epoch_g_loss) if len(epoch_g_loss) > 0 else 0.0
        mean_c_loss = np.mean(epoch_c_loss)
        
        print(f"Epoch {epoch}/{epochs} | Gen Loss: {mean_g_loss:.4f} | Critic Loss: {mean_c_loss:.4f}")
        
        # Save generated audio samples every 10 epochs
        if epoch % 10 == 0 or epoch == 1 or epoch == epochs:
            noise = tf.random.normal([4, latent_dim])
            synth_waves = gan.generator(noise, training=False).numpy() # shape (4, 5120, 1)
            
            for i in range(len(synth_waves)):
                # Crop padding to get raw 5040 signal
                wave_out = synth_waves[i, :5040, 0]
                # Normalize audio
                max_val = np.max(np.abs(wave_out))
                if max_val > 0:
                    wave_out = wave_out / max_val
                
                # Write to disk
                fname = os.path.join(outputs_dir, f"epoch_{epoch:03d}_sample_{i+1}.wav")
                sf.write(fname, wave_out, 2000)
                
    # Save WaveGAN models
    gan.generator.save(os.path.join(model_dir, 'wavegan_generator.keras'))
    gan.discriminator.save(os.path.join(model_dir, 'wavegan_critic.keras'))
    print("WaveGAN models saved successfully!")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train 1D WaveGAN on raw heart audio signals")
    parser.add_argument('--epochs', type=int, default=30, help='Number of epochs to train')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for training')
    
    args = parser.parse_args()
    
    # Notice to user
    print("="*60)
    print("  WAVEGAN AUDIO WAVEFORM GENERATOR TRAINING TEMPLATE")
    print("  Note: Direct time-domain synthesis requires GPU acceleration.")
    print("  This pipeline is prepared for execution on GPU-enabled setups.")
    print("="*60)
    
    train_wavegan(epochs=args.epochs, batch_size=args.batch_size)
