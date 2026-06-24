import tensorflow as tf
from tensorflow.keras import layers, Model

class PhaseShuffle(layers.Layer):
    """
    Phase Shuffling layer from WaveGAN.
    Shifts the activations along the time dimension by a random integer offset.
    Prevents the discriminator from learning periodic phase artifacts.
    """
    def __init__(self, shift_factor=2, **kwargs):
        super(PhaseShuffle, self).__init__(**kwargs)
        self.shift_factor = shift_factor
        
    def call(self, x, training=None):
        if not training or self.shift_factor == 0:
            return x
            
        # Generate random shift in range [-shift_factor, shift_factor]
        shift = tf.random.uniform([], -self.shift_factor, self.shift_factor + 1, dtype=tf.int32)
        
        # Roll tensor along the time axis (axis 1)
        # x shape: (batch_size, sequence_length, channels)
        return tf.roll(x, shift, axis=1)

    def get_config(self):
        config = super(PhaseShuffle, self).get_config()
        config.update({"shift_factor": self.shift_factor})
        return config

def build_wavegan_generator(latent_dim=100):
    """
    Builds the 1D WaveGAN Generator.
    Maps a latent noise vector of shape (latent_dim,) to a 1D raw waveform of shape (5120, 1).
    Uses 1D transpose convolutions with large filter sizes (kernel size 25).
    """
    noise = layers.Input(shape=(latent_dim,))
    
    # Project to starting length: 5
    x = layers.Dense(5 * 512, use_bias=False)(noise)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(alpha=0.2)(x)
    
    x = layers.Reshape((5, 512))(x) # Shape: (5, 512)
    
    # Layer 1: Upsample by 4 -> (20, 256)
    x = layers.Conv1DTranspose(256, kernel_size=25, strides=4, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(alpha=0.2)(x)
    
    # Layer 2: Upsample by 4 -> (80, 128)
    x = layers.Conv1DTranspose(128, kernel_size=25, strides=4, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(alpha=0.2)(x)
    
    # Layer 3: Upsample by 4 -> (320, 64)
    x = layers.Conv1DTranspose(64, kernel_size=25, strides=4, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(alpha=0.2)(x)
    
    # Layer 4: Upsample by 4 -> (1280, 32)
    x = layers.Conv1DTranspose(32, kernel_size=25, strides=4, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(alpha=0.2)(x)
    
    # Layer 5: Upsample by 4 -> (5120, 1)
    out = layers.Conv1DTranspose(1, kernel_size=25, strides=4, padding='same', activation='tanh', use_bias=False)(x)
    
    model = Model(noise, out, name="WaveGAN_Generator")
    return model

def build_wavegan_discriminator(input_shape=(5120, 1)):
    """
    Builds the 1D WaveGAN Discriminator (Critic).
    Maps a 1D raw waveform of shape (5120, 1) to a raw scalar score.
    Uses 1D convolutions with large kernels and Phase Shuffling.
    """
    wave_input = layers.Input(shape=input_shape)
    
    # Layer 1: Downsample by 4 -> (1280, 32)
    x = layers.Conv1D(32, kernel_size=25, strides=4, padding='same')(wave_input)
    x = layers.LeakyReLU(alpha=0.2)(x)
    x = PhaseShuffle(shift_factor=2)(x)
    x = layers.Dropout(0.3)(x)
    
    # Layer 2: Downsample by 4 -> (320, 64)
    x = layers.Conv1D(64, kernel_size=25, strides=4, padding='same', use_bias=False)(x)
    x = layers.LayerNormalization()(x) # LN compatible with WGAN-GP
    x = layers.LeakyReLU(alpha=0.2)(x)
    x = PhaseShuffle(shift_factor=2)(x)
    x = layers.Dropout(0.3)(x)
    
    # Layer 3: Downsample by 4 -> (80, 128)
    x = layers.Conv1D(128, kernel_size=25, strides=4, padding='same', use_bias=False)(x)
    x = layers.LayerNormalization()(x)
    x = layers.LeakyReLU(alpha=0.2)(x)
    x = PhaseShuffle(shift_factor=2)(x)
    x = layers.Dropout(0.3)(x)
    
    # Layer 4: Downsample by 4 -> (20, 256)
    x = layers.Conv1D(256, kernel_size=25, strides=4, padding='same', use_bias=False)(x)
    x = layers.LayerNormalization()(x)
    x = layers.LeakyReLU(alpha=0.2)(x)
    x = PhaseShuffle(shift_factor=2)(x)
    x = layers.Dropout(0.3)(x)
    
    # Layer 5: Downsample by 4 -> (5, 512)
    x = layers.Conv1D(512, kernel_size=25, strides=4, padding='same', use_bias=False)(x)
    x = layers.LayerNormalization()(x)
    x = layers.LeakyReLU(alpha=0.2)(x)
    
    # Flatten & Output
    x = layers.Flatten()(x)
    out = layers.Dense(1)(x) # Raw critic score (no sigmoid activation)
    
    model = Model(wave_input, out, name="WaveGAN_Critic")
    return model

if __name__ == '__main__':
    # Verify model compile and shape dynamics
    gen = build_wavegan_generator(100)
    gen.summary()
    print("\n" + "="*50 + "\n")
    disc = build_wavegan_discriminator((5120, 1))
    disc.summary()
