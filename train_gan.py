import os
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from gan_model import build_generator, build_discriminator

# Set random seed for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

def discriminator_loss(real_output, fake_output):
    """
    Computes binary cross-entropy loss for the discriminator.
    We want real images to be classified as 1, and fake images as 0.
    """
    cross_entropy = tf.keras.losses.BinaryCrossentropy()
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss

def generator_loss(fake_output):
    """
    Computes binary cross-entropy loss for the generator.
    The generator wants the discriminator to classify fake images as 1.
    """
    cross_entropy = tf.keras.losses.BinaryCrossentropy()
    return cross_entropy(tf.ones_like(fake_output), fake_output)

class GAN:
    def __init__(self, latent_dim=100, img_shape=(64, 13)):
        self.latent_dim = latent_dim
        self.img_shape = img_shape
        
        self.generator = build_generator(self.latent_dim)
        self.discriminator = build_discriminator(self.img_shape)
        
        self.generator_optimizer = Adam(learning_rate=0.0002, beta_1=0.5)
        self.discriminator_optimizer = Adam(learning_rate=0.0002, beta_1=0.5)
        
    @tf.function
    def train_step(self, mfccs):
        noise = tf.random.normal([mfccs.shape[0], self.latent_dim])
        
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            generated_mfccs = self.generator(noise, training=True)
            
            real_output = self.discriminator(mfccs, training=True)
            fake_output = self.discriminator(generated_mfccs, training=True)
            
            gen_loss = generator_loss(fake_output)
            disc_loss = discriminator_loss(real_output, fake_output)
            
        gradients_of_generator = gen_tape.gradient(gen_loss, self.generator.trainable_variables)
        gradients_of_discriminator = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)
        
        self.generator_optimizer.apply_gradients(zip(gradients_of_generator, self.generator.trainable_variables))
        self.discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, self.discriminator.trainable_variables))
        
        return gen_loss, disc_loss

def save_gan_plots(generator, epoch, latent_dim, bounds=None, out_dir='outputs/mfcc_plots'):
    """
    Generates MFCC matrices using the generator and plots them as heatmaps.
    If bounds are provided, scales back the MFCC values.
    """
    os.makedirs(out_dir, exist_ok=True)
    num_examples = 4
    noise = tf.random.normal([num_examples, latent_dim])
    predictions = generator(noise, training=False).numpy()
    
    fig, axes = plt.subplots(1, num_examples, figsize=(16, 4))
    
    for i in range(num_examples):
        # Shape: (64, 13)
        mfcc = predictions[i]
        
        if bounds is not None:
            # Rescale back to raw MFCC range for visualization
            m_min, m_max = bounds[i]
            mfcc = ((mfcc + 1.0) / 2.0) * (m_max - m_min) + m_min
            
        # Transpose back to (13, 64) for display (standard representation: coefficients as Y, time as X)
        mfcc_display = mfcc.T
        
        im = axes[i].imshow(mfcc_display, cmap='coolwarm', aspect='auto', origin='lower')
        axes[i].set_title(f"Gen MFCC {i+1}")
        axes[i].set_xlabel('Time frames')
        axes[i].set_ylabel('MFCC Coeffs')
        
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f'epoch_{epoch:03d}.png'), dpi=150)
    plt.close()

def train_gan_pipeline(epochs=50, batch_size=32, latent_dim=100, processed_dir='data/processed', model_dir='models', outputs_dir='outputs'):
    # Load processed MFCC training data (Real train split only)
    X_mfcc = np.load(os.path.join(processed_dir, 'X_train.npy'))
    y = np.load(os.path.join(processed_dir, 'y_train.npy'))
    bounds = np.load(os.path.join(processed_dir, 'bounds_train.npy'))
    
    # Select only abnormal samples (labeled 1)
    abnormal_idx = np.where(y == 1)[0]
    X_abnormal = X_mfcc[abnormal_idx]
    bounds_abnormal = bounds[abnormal_idx]
    
    print(f"Loaded {len(X_abnormal)} abnormal MFCC samples for training the GAN.")
    
    if len(X_abnormal) == 0:
        raise ValueError("No abnormal samples found in dataset! Cannot train GAN.")
        
    dataset = tf.data.Dataset.from_tensor_slices(X_abnormal).shuffle(len(X_abnormal)).batch(batch_size, drop_remainder=False)
    
    # Initialize 1D GAN
    gan = GAN(latent_dim=latent_dim, img_shape=X_abnormal.shape[1:])
    
    gen_losses = []
    disc_losses = []
    
    print("Starting GAN training...")
    
    for epoch in range(1, epochs + 1):
        epoch_gen_loss = []
        epoch_disc_loss = []
        
        for mfcc_batch in dataset:
            g_loss, d_loss = gan.train_step(mfcc_batch)
            epoch_gen_loss.append(g_loss.numpy())
            epoch_disc_loss.append(d_loss.numpy())
            
        mean_g_loss = np.mean(epoch_gen_loss)
        mean_d_loss = np.mean(epoch_disc_loss)
        gen_losses.append(mean_g_loss)
        disc_losses.append(mean_d_loss)
        
        if epoch % 10 == 0 or epoch == 1 or epoch == epochs:
            print(f"Epoch {epoch}/{epochs} | Gen Loss: {mean_g_loss:.4f} | Disc Loss: {mean_d_loss:.4f}")
            # Save visual plots of MFCC heatmaps
            save_gan_plots(gan.generator, epoch, latent_dim, bounds_abnormal, os.path.join(outputs_dir, 'mfcc_plots'))
            
    # Save the models
    os.makedirs(model_dir, exist_ok=True)
    gan.generator.save(os.path.join(model_dir, 'gan_generator.keras'))
    gan.discriminator.save(os.path.join(model_dir, 'gan_discriminator.keras'))
    print("Saved generator and discriminator models.")
    
    # Plot training losses
    plot_dir = os.path.join(outputs_dir, 'plots')
    os.makedirs(plot_dir, exist_ok=True)
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, epochs + 1), gen_losses, label='Generator Loss', color='#1f77b4', linewidth=2)
    plt.plot(range(1, epochs + 1), disc_losses, label='Discriminator Loss', color='#ff7f0e', linewidth=2)
    plt.title('GAN Training Losses (MFCC-based)', fontsize=14, fontweight='bold')
    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel('Binary Cross-Entropy Loss', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, 'gan_loss.png'), dpi=150)
    plt.close()
    
    # Save a set of generated synthetic samples as .npy file
    print("Generating synthetic abnormal MFCC samples...")
    noise = tf.random.normal([50, latent_dim])
    synth_mfcc = gan.generator(noise, training=False).numpy()
    
    synth_dir = os.path.join(outputs_dir, 'synthetic_features')
    os.makedirs(synth_dir, exist_ok=True)
    np.save(os.path.join(synth_dir, 'synth_abnormal_mfcc.npy'), synth_mfcc)
    
    print(f"Saved 50 synthetic abnormal MFCC samples to {os.path.join(synth_dir, 'synth_abnormal_mfcc.npy')}")
    print("GAN training pipeline finished!")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train 1D GAN model on Abnormal MFCCs")
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs to train GAN')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--latent_dim', type=int, default=100, help='Dimension of latent space')
    
    args = parser.parse_args()
    train_gan_pipeline(epochs=args.epochs, batch_size=args.batch_size, latent_dim=args.latent_dim)
