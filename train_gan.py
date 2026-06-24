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

def gradient_penalty(critic, real_images, fake_images):
    """
    Computes the gradient penalty for WGAN-GP.
    Enforces the Lipschitz-1 constraint on the critic.
    """
    batch_size = tf.shape(real_images)[0]
    # alpha mixes real and fake samples sample-by-sample
    alpha = tf.random.uniform([batch_size, 1, 1], 0.0, 1.0)
    diff = fake_images - real_images
    interpolates = real_images + alpha * diff
    
    with tf.GradientTape() as gp_tape:
        gp_tape.watch(interpolates)
        critic_pred = critic(interpolates, training=True)
        
    grads = gp_tape.gradient(critic_pred, [interpolates])[0]
    # Compute L2 norm over time (axis 1) and features (axis 2)
    norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=[1, 2]) + 1e-12)
    gp = tf.reduce_mean((norm - 1.0) ** 2)
    return gp

def critic_loss(real_output, fake_output):
    """
    Wasserstein loss for the Critic.
    Critic wants to maximize (real_output - fake_output), i.e., minimize (fake - real).
    """
    return tf.reduce_mean(fake_output) - tf.reduce_mean(real_output)

def generator_loss(fake_output):
    """
    Wasserstein loss for the Generator.
    Generator wants to maximize fake_output, i.e., minimize -fake_output.
    """
    return -tf.reduce_mean(fake_output)

class WGAN_GP:
    def __init__(self, latent_dim=100, img_shape=(64, 39), gp_weight=10.0):
        self.latent_dim = latent_dim
        self.img_shape = img_shape
        self.gp_weight = gp_weight
        
        self.generator = build_generator(self.latent_dim)
        self.discriminator = build_discriminator(self.img_shape) # acts as the Critic
        
        # TTUR (Two Time-Scale Update Rule): learning rate 1e-4, beta_1=0.0, beta_2=0.9
        self.generator_optimizer = Adam(learning_rate=0.0001, beta_1=0.0, beta_2=0.9)
        self.discriminator_optimizer = Adam(learning_rate=0.0001, beta_1=0.0, beta_2=0.9)
        
    @tf.function
    def train_step_critic(self, real_images):
        batch_size = tf.shape(real_images)[0]
        noise = tf.random.normal([batch_size, self.latent_dim])
        
        with tf.GradientTape() as critic_tape:
            fake_images = self.generator(noise, training=True)
            
            real_output = self.discriminator(real_images, training=True)
            fake_output = self.discriminator(fake_images, training=True)
            
            c_loss = critic_loss(real_output, fake_output)
            gp = gradient_penalty(self.discriminator, real_images, fake_images)
            
            total_c_loss = c_loss + self.gp_weight * gp
            
        grads = critic_tape.gradient(total_c_loss, self.discriminator.trainable_variables)
        self.discriminator_optimizer.apply_gradients(zip(grads, self.discriminator.trainable_variables))
        return total_c_loss
        
    @tf.function
    def train_step_generator(self, batch_size):
        noise = tf.random.normal([batch_size, self.latent_dim])
        
        with tf.GradientTape() as gen_tape:
            fake_images = self.generator(noise, training=True)
            fake_output = self.discriminator(fake_images, training=True)
            g_loss = generator_loss(fake_output)
            
        grads = gen_tape.gradient(g_loss, self.generator.trainable_variables)
        self.generator_optimizer.apply_gradients(zip(grads, self.generator.trainable_variables))
        return g_loss

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
        # Shape: (64, 39)
        mfcc = predictions[i]
        
        if bounds is not None:
            # Rescale back to raw MFCC range for visualization
            m_min, m_max = bounds[i]
            mfcc = ((mfcc + 1.0) / 2.0) * (m_max - m_min) + m_min
            
        # Transpose back to (39, 64) for display (standard representation: coefficients as Y, time as X)
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
    
    print(f"Loaded {len(X_abnormal)} abnormal MFCC samples for training the WGAN-GP.")
    
    if len(X_abnormal) == 0:
        raise ValueError("No abnormal samples found in dataset! Cannot train GAN.")
        
    dataset = tf.data.Dataset.from_tensor_slices(X_abnormal).shuffle(len(X_abnormal)).batch(batch_size, drop_remainder=False)
    
    # Initialize WGAN-GP
    gan = WGAN_GP(latent_dim=latent_dim, img_shape=X_abnormal.shape[1:])
    
    gen_losses = []
    disc_losses = []
    
    print("Starting WGAN-GP training...")
    
    step = 0
    n_critic = 3 # Number of Critic updates per Generator update
    
    for epoch in range(1, epochs + 1):
        epoch_gen_loss = []
        epoch_disc_loss = []
        
        for mfcc_batch in dataset:
            batch_size_curr = tf.shape(mfcc_batch)[0]
            # Update Critic
            d_loss = gan.train_step_critic(mfcc_batch)
            epoch_disc_loss.append(d_loss.numpy())
            
            # Update Generator every n_critic steps
            if step % n_critic == 0:
                g_loss = gan.train_step_generator(batch_size_curr)
                epoch_gen_loss.append(g_loss.numpy())
                
            step += 1
            
        mean_g_loss = np.mean(epoch_gen_loss) if len(epoch_gen_loss) > 0 else (gen_losses[-1] if len(gen_losses) > 0 else 0.0)
        mean_d_loss = np.mean(epoch_disc_loss)
        gen_losses.append(mean_g_loss)
        disc_losses.append(mean_d_loss)
        
        if epoch % 10 == 0 or epoch == 1 or epoch == epochs:
            print(f"Epoch {epoch}/{epochs} | Gen Loss: {mean_g_loss:.4f} | Critic Loss: {mean_d_loss:.4f}")
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
    plt.plot(range(1, epochs + 1), disc_losses, label='Critic Loss', color='#ff7f0e', linewidth=2)
    plt.title('WGAN-GP Training Losses (Wasserstein Distance)', fontsize=14, fontweight='bold')
    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel('Wasserstein Loss', fontsize=12)
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
    print("WGAN-GP training pipeline finished!")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train WGAN-GP model on Abnormal MFCCs")
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs to train GAN')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--latent_dim', type=int, default=100, help='Dimension of latent space')
    
    args = parser.parse_args()
    train_gan_pipeline(epochs=args.epochs, batch_size=args.batch_size, latent_dim=args.latent_dim)

