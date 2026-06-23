import tensorflow as tf
from tensorflow.keras import layers, Model

def build_generator(latent_dim=100):
    """
    Builds the 1D Generator network.
    Maps a latent vector (noise) to a 1D MFCC feature matrix of shape (64, 13).
    """
    noise = layers.Input(shape=(latent_dim,))
    
    # Start with length 8 sequence
    x = layers.Dense(8 * 256, use_bias=False)(noise)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(alpha=0.2)(x)
    
    x = layers.Reshape((8, 256))(x)
    
    # Upsample sequence length to 16: (16, 128)
    x = layers.Conv1DTranspose(128, kernel_size=4, strides=2, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(alpha=0.2)(x)
    
    # Upsample sequence length to 32: (32, 64)
    x = layers.Conv1DTranspose(64, kernel_size=4, strides=2, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(alpha=0.2)(x)
    
    # Upsample sequence length to 64: (64, 13)
    # The output channels matches the number of MFCC coefficients (13)
    out = layers.Conv1DTranspose(13, kernel_size=4, strides=2, padding='same', activation='tanh', use_bias=False)(x)
    
    model = Model(noise, out, name="Generator")
    return model

def build_discriminator(input_shape=(64, 13)):
    """
    Builds the 1D Discriminator network.
    Classifies an MFCC matrix of shape (64, 13) as Real (1) or Fake (0).
    """
    mfcc_input = layers.Input(shape=input_shape)
    
    # Sequence length: 64 -> 32
    x = layers.Conv1D(64, kernel_size=4, strides=2, padding='same')(mfcc_input)
    x = layers.LeakyReLU(alpha=0.2)(x)
    x = layers.Dropout(0.3)(x)
    
    # Sequence length: 32 -> 16
    x = layers.Conv1D(128, kernel_size=4, strides=2, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(alpha=0.2)(x)
    x = layers.Dropout(0.3)(x)
    
    # Sequence length: 16 -> 8
    x = layers.Conv1D(256, kernel_size=4, strides=2, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(alpha=0.2)(x)
    x = layers.Dropout(0.3)(x)
    
    # Flatten & Output
    x = layers.Flatten()(x)
    out = layers.Dense(1, activation='sigmoid')(x)
    
    model = Model(mfcc_input, out, name="Discriminator")
    return model

if __name__ == '__main__':
    # Print network summaries to verify shapes
    gen = build_generator(100)
    gen.summary()
    print("\n" + "="*50 + "\n")
    disc = build_discriminator((64, 13))
    disc.summary()
