"""Progressive Wasserstein GAN (PWGAN) models for heart-sound features.

The generator and critic models progressively grow resolution along the time axis:
Stage 1: 16 time frames, 71 features
Stage 2: 32 time frames, 71 features
Stage 3: 64 time frames, 71 features
"""

import tensorflow as tf
from tensorflow.keras import Model, layers


class ProgressiveGenerator(Model):
    def __init__(self, latent_dim=100, feature_dim=71):
        super(ProgressiveGenerator, self).__init__()
        self.latent_dim = latent_dim
        self.feature_dim = feature_dim
        
        # Dense input mapping to resolution 16
        self.dense = layers.Dense(16 * 128, use_bias=False, name="dense_dense")
        self.bn_dense = layers.BatchNormalization(name="dense_bn")
        self.lrelu_dense = layers.LeakyReLU(negative_slope=0.2, name="dense_lrelu")
        
        # Stage 1: Resolution 16
        self.stage1_conv = layers.Conv1D(128, 3, padding="same", use_bias=False, name="stage1_conv")
        self.stage1_bn = layers.BatchNormalization(name="stage1_bn")
        self.stage1_lrelu = layers.LeakyReLU(negative_slope=0.2, name="stage1_lrelu")
        self.stage1_to_rgb = layers.Conv1D(feature_dim, 1, activation="tanh", name="stage1_to_rgb")
        
        # Stage 2: Resolution 32
        self.stage2_upsample = layers.UpSampling1D(size=2, name="stage2_upsample")
        self.stage2_conv1 = layers.Conv1D(64, 3, padding="same", use_bias=False, name="stage2_conv1")
        self.stage2_bn1 = layers.BatchNormalization(name="stage2_bn1")
        self.stage2_lrelu1 = layers.LeakyReLU(negative_slope=0.2, name="stage2_lrelu1")
        self.stage2_conv2 = layers.Conv1D(64, 3, padding="same", use_bias=False, name="stage2_conv2")
        self.stage2_bn2 = layers.BatchNormalization(name="stage2_bn2")
        self.stage2_lrelu2 = layers.LeakyReLU(negative_slope=0.2, name="stage2_lrelu2")
        self.stage2_to_rgb = layers.Conv1D(feature_dim, 1, activation="tanh", name="stage2_to_rgb")
        
        # Stage 3: Resolution 64
        self.stage3_upsample = layers.UpSampling1D(size=2, name="stage3_upsample")
        self.stage3_conv1 = layers.Conv1D(32, 3, padding="same", use_bias=False, name="stage3_conv1")
        self.stage3_bn1 = layers.BatchNormalization(name="stage3_bn1")
        self.stage3_lrelu1 = layers.LeakyReLU(negative_slope=0.2, name="stage3_lrelu1")
        self.stage3_conv2 = layers.Conv1D(32, 3, padding="same", use_bias=False, name="stage3_conv2")
        self.stage3_bn2 = layers.BatchNormalization(name="stage3_bn2")
        self.stage3_lrelu2 = layers.LeakyReLU(negative_slope=0.2, name="stage3_lrelu2")
        self.stage3_to_rgb = layers.Conv1D(feature_dim, 1, activation="tanh", name="stage3_to_rgb")

    def call(self, inputs, resolution=64, alpha=1.0, training=None):
        x = self.dense(inputs)
        x = self.bn_dense(x, training=training)
        x = self.lrelu_dense(x)
        x = layers.Reshape((16, 128), name="dense_reshape")(x)
        
        # Stage 1 block
        feat1 = self.stage1_conv(x)
        feat1 = self.stage1_bn(feat1, training=training)
        feat1 = self.stage1_lrelu(feat1)
        
        if resolution == 16:
            return self.stage1_to_rgb(feat1)
            
        # Stage 2 block
        x2 = self.stage2_upsample(feat1)
        feat2 = self.stage2_conv1(x2)
        feat2 = self.stage2_bn1(feat2, training=training)
        feat2 = self.stage2_lrelu1(feat2)
        feat2 = self.stage2_conv2(feat2)
        feat2 = self.stage2_bn2(feat2, training=training)
        feat2 = self.stage2_lrelu2(feat2)
        
        if resolution == 32:
            if alpha < 1.0:
                rgb1_up = self.stage2_upsample(self.stage1_to_rgb(feat1))
                rgb2 = self.stage2_to_rgb(feat2)
                return (1.0 - alpha) * rgb1_up + alpha * rgb2
            else:
                return self.stage2_to_rgb(feat2)
                
        # Stage 3 block
        x3 = self.stage3_upsample(feat2)
        feat3 = self.stage3_conv1(x3)
        feat3 = self.stage3_bn1(feat3, training=training)
        feat3 = self.stage3_lrelu1(feat3)
        feat3 = self.stage3_conv2(feat3)
        feat3 = self.stage3_bn2(feat3, training=training)
        feat3 = self.stage3_lrelu2(feat3)
        
        if resolution == 64:
            if alpha < 1.0:
                rgb2_up = self.stage3_upsample(self.stage2_to_rgb(feat2))
                rgb3 = self.stage3_to_rgb(feat3)
                return (1.0 - alpha) * rgb2_up + alpha * rgb3
            else:
                return self.stage3_to_rgb(feat3)
                
        raise ValueError(f"Unsupported resolution: {resolution}")


class ProgressiveCritic(Model):
    def __init__(self, feature_dim=71):
        super(ProgressiveCritic, self).__init__()
        self.feature_dim = feature_dim
        
        # Stage 1 rgb from_rgb & base classification
        self.stage1_from_rgb = layers.Conv1D(128, 1, name="stage1_from_rgb")
        self.stage1_lrelu_rgb = layers.LeakyReLU(negative_slope=0.2, name="stage1_lrelu_rgb")
        self.stage1_conv = layers.Conv1D(128, 3, padding="same", name="stage1_conv")
        self.stage1_ln = layers.LayerNormalization(name="stage1_ln")
        self.stage1_lrelu = layers.LeakyReLU(negative_slope=0.2, name="stage1_lrelu")
        self.stage1_flat = layers.Flatten(name="stage1_flat")
        self.stage1_out = layers.Dense(1, name="stage1_out")
        
        # Stage 2 (from 32 to 16 time frames)
        self.stage2_from_rgb = layers.Conv1D(64, 1, name="stage2_from_rgb")
        self.stage2_lrelu_rgb = layers.LeakyReLU(negative_slope=0.2, name="stage2_lrelu_rgb")
        self.stage2_conv1 = layers.Conv1D(64, 3, padding="same", name="stage2_conv1")
        self.stage2_ln1 = layers.LayerNormalization(name="stage2_ln1")
        self.stage2_lrelu1 = layers.LeakyReLU(negative_slope=0.2, name="stage2_lrelu1")
        self.stage2_pool = layers.AveragePooling1D(pool_size=2, name="stage2_pool")
        self.stage2_conv2 = layers.Conv1D(128, 3, padding="same", name="stage2_conv2")
        self.stage2_ln2 = layers.LayerNormalization(name="stage2_ln2")
        self.stage2_lrelu2 = layers.LeakyReLU(negative_slope=0.2, name="stage2_lrelu2")
        
        # Stage 3 (from 64 to 32 time frames)
        self.stage3_from_rgb = layers.Conv1D(32, 1, name="stage3_from_rgb")
        self.stage3_lrelu_rgb = layers.LeakyReLU(negative_slope=0.2, name="stage3_lrelu_rgb")
        self.stage3_conv1 = layers.Conv1D(32, 3, padding="same", name="stage3_conv1")
        self.stage3_ln1 = layers.LayerNormalization(name="stage3_ln1")
        self.stage3_lrelu1 = layers.LeakyReLU(negative_slope=0.2, name="stage3_lrelu1")
        self.stage3_pool = layers.AveragePooling1D(pool_size=2, name="stage3_pool")
        self.stage3_conv2 = layers.Conv1D(64, 3, padding="same", name="stage3_conv2")
        self.stage3_ln2 = layers.LayerNormalization(name="stage3_ln2")
        self.stage3_lrelu2 = layers.LeakyReLU(negative_slope=0.2, name="stage3_lrelu2")

        self.downsample = layers.AveragePooling1D(pool_size=2, name="downsample")

    def call(self, inputs, resolution=64, alpha=1.0, training=None):
        if resolution == 16:
            x = self.stage1_from_rgb(inputs)
            x = self.stage1_lrelu_rgb(x)
            
            x = self.stage1_conv(x)
            x = self.stage1_ln(x, training=training)
            x = self.stage1_lrelu(x)
            x = self.stage1_flat(x)
            return self.stage1_out(x)
            
        elif resolution == 32:
            x_new = self.stage2_from_rgb(inputs)
            x_new = self.stage2_lrelu_rgb(x_new)
            x_new = self.stage2_conv1(x_new)
            x_new = self.stage2_ln1(x_new, training=training)
            x_new = self.stage2_lrelu1(x_new)
            x_new = self.stage2_pool(x_new)
            x_new = self.stage2_conv2(x_new)
            x_new = self.stage2_ln2(x_new, training=training)
            x_new = self.stage2_lrelu2(x_new)
            
            if alpha < 1.0:
                inputs_down = self.downsample(inputs)
                x_old = self.stage1_from_rgb(inputs_down)
                x_old = self.stage1_lrelu_rgb(x_old)
                x = (1.0 - alpha) * x_old + alpha * x_new
            else:
                x = x_new
                
            x = self.stage1_conv(x)
            x = self.stage1_ln(x, training=training)
            x = self.stage1_lrelu(x)
            x = self.stage1_flat(x)
            return self.stage1_out(x)
            
        elif resolution == 64:
            x_new = self.stage3_from_rgb(inputs)
            x_new = self.stage3_lrelu_rgb(x_new)
            x_new = self.stage3_conv1(x_new)
            x_new = self.stage3_ln1(x_new, training=training)
            x_new = self.stage3_lrelu1(x_new)
            x_new = self.stage3_pool(x_new)
            x_new = self.stage3_conv2(x_new)
            x_new = self.stage3_ln2(x_new, training=training)
            x_new = self.stage3_lrelu2(x_new)
            
            x_new = self.stage2_conv1(x_new)
            x_new = self.stage2_ln1(x_new, training=training)
            x_new = self.stage2_lrelu1(x_new)
            x_new = self.stage2_pool(x_new)
            x_new = self.stage2_conv2(x_new)
            x_new = self.stage2_ln2(x_new, training=training)
            x_new = self.stage2_lrelu2(x_new)
            
            if alpha < 1.0:
                inputs_down = self.downsample(inputs)
                x_old = self.stage2_from_rgb(inputs_down)
                x_old = self.stage2_lrelu_rgb(x_old)
                x_old = self.stage2_conv1(x_old)
                x_old = self.stage2_ln1(x_old, training=training)
                x_old = self.stage2_lrelu1(x_old)
                x_old = self.stage2_pool(x_old)
                x_old = self.stage2_conv2(x_old)
                x_old = self.stage2_ln2(x_old, training=training)
                x_old = self.stage2_lrelu2(x_old)
                x = (1.0 - alpha) * x_old + alpha * x_new
            else:
                x = x_new
                
            x = self.stage1_conv(x)
            x = self.stage1_ln(x, training=training)
            x = self.stage1_lrelu(x)
            x = self.stage1_flat(x)
            return self.stage1_out(x)
            
        raise ValueError(f"Unsupported resolution: {resolution}")


# Final, non-progressive Functional builders for loading/inference
def build_final_generator(latent_dim=100, feature_dim=71):
    noise = layers.Input(shape=(latent_dim,), name="noise")
    x = layers.Dense(16 * 128, use_bias=False, name="dense_dense")(noise)
    x = layers.BatchNormalization(name="dense_bn")(x)
    x = layers.LeakyReLU(negative_slope=0.2, name="dense_lrelu")(x)
    x = layers.Reshape((16, 128), name="dense_reshape")(x)
    
    feat1 = layers.Conv1D(128, 3, padding="same", use_bias=False, name="stage1_conv")(x)
    feat1 = layers.BatchNormalization(name="stage1_bn")(feat1)
    feat1 = layers.LeakyReLU(negative_slope=0.2, name="stage1_lrelu")(feat1)
    
    x2 = layers.UpSampling1D(size=2, name="stage2_upsample")(feat1)
    feat2 = layers.Conv1D(64, 3, padding="same", use_bias=False, name="stage2_conv1")(x2)
    feat2 = layers.BatchNormalization(name="stage2_bn1")(feat2)
    feat2 = layers.LeakyReLU(negative_slope=0.2, name="stage2_lrelu1")(feat2)
    feat2 = layers.Conv1D(64, 3, padding="same", use_bias=False, name="stage2_conv2")(feat2)
    feat2 = layers.BatchNormalization(name="stage2_bn2")(feat2)
    feat2 = layers.LeakyReLU(negative_slope=0.2, name="stage2_lrelu2")(feat2)
    
    x3 = layers.UpSampling1D(size=2, name="stage3_upsample")(feat2)
    feat3 = layers.Conv1D(32, 3, padding="same", use_bias=False, name="stage3_conv1")(x3)
    feat3 = layers.BatchNormalization(name="stage3_bn1")(feat3)
    feat3 = layers.LeakyReLU(negative_slope=0.2, name="stage3_lrelu1")(feat3)
    feat3 = layers.Conv1D(32, 3, padding="same", use_bias=False, name="stage3_conv2")(feat3)
    feat3 = layers.BatchNormalization(name="stage3_bn2")(feat3)
    feat3 = layers.LeakyReLU(negative_slope=0.2, name="stage3_lrelu2")(feat3)
    
    out = layers.Conv1D(feature_dim, 1, activation="tanh", name="stage3_to_rgb")(feat3)
    return Model(noise, out, name="Generator")


def build_final_discriminator(input_shape=(64, 71)):
    inputs = layers.Input(shape=input_shape, name="features")
    
    # Stage 3 down to Stage 2
    x = layers.Conv1D(32, 1, name="stage3_from_rgb")(inputs)
    x = layers.LeakyReLU(negative_slope=0.2, name="stage3_lrelu_rgb")(x)
    x = layers.Conv1D(32, 3, padding="same", name="stage3_conv1")(x)
    x = layers.LayerNormalization(name="stage3_ln1")(x)
    x = layers.LeakyReLU(negative_slope=0.2, name="stage3_lrelu1")(x)
    x = layers.AveragePooling1D(pool_size=2, name="stage3_pool")(x)
    x = layers.Conv1D(64, 3, padding="same", name="stage3_conv2")(x)
    x = layers.LayerNormalization(name="stage3_ln2")(x)
    x = layers.LeakyReLU(negative_slope=0.2, name="stage3_lrelu2")(x)
    
    # Stage 2 down to Stage 1
    x = layers.Conv1D(64, 3, padding="same", name="stage2_conv1")(x)
    x = layers.LayerNormalization(name="stage2_ln1")(x)
    x = layers.LeakyReLU(negative_slope=0.2, name="stage2_lrelu1")(x)
    x = layers.AveragePooling1D(pool_size=2, name="stage2_pool")(x)
    x = layers.Conv1D(128, 3, padding="same", name="stage2_conv2")(x)
    x = layers.LayerNormalization(name="stage2_ln2")(x)
    x = layers.LeakyReLU(negative_slope=0.2, name="stage2_lrelu2")(x)
    
    # Stage 1
    x = layers.Conv1D(128, 3, padding="same", name="stage1_conv")(x)
    x = layers.LayerNormalization(name="stage1_ln")(x)
    x = layers.LeakyReLU(negative_slope=0.2, name="stage1_lrelu")(x)
    x = layers.Flatten(name="stage1_flat")(x)
    out = layers.Dense(1, name="stage1_out")(x)
    
    return Model(inputs, out, name="Discriminator")


def copy_matching_weights(source_model, target_model):
    """Copies weights between source and target models based on layer names."""
    for target_layer in target_model.layers:
        name = target_layer.name
        # If it's a layer containing weights
        if len(target_layer.weights) > 0:
            try:
                # Find matching layer in source
                source_layer = source_model.get_layer(name)
                target_layer.set_weights(source_layer.get_weights())
            except ValueError:
                print(f"Warning: Could not copy weights for layer {name}")


# Backward compatibility builders
def build_generator(latent_dim=100, feature_dim=71):
    return ProgressiveGenerator(latent_dim, feature_dim)


def build_discriminator(input_shape=(64, 71)):
    return ProgressiveCritic(input_shape[-1])


if __name__ == "__main__":
    # Test model shape dynamics
    gen = build_generator()
    critic = build_discriminator()
    
    noise = tf.random.normal((2, 100))
    for res in [16, 32, 64]:
        fake = gen(noise, resolution=res, alpha=0.5)
        score = critic(fake, resolution=res, alpha=0.5)
        print(f"Resolution: {res} | Gen Shape: {fake.shape} | Critic Score Shape: {score.shape}")
        
    print("\nVerifying final model building and weight copying...")
    final_gen = build_final_generator()
    final_critic = build_final_discriminator()
    
    copy_matching_weights(gen, final_gen)
    copy_matching_weights(critic, final_critic)
    
    fake_final = final_gen(noise)
    score_final = final_critic(fake_final)
    print(f"Final Gen Shape: {fake_final.shape} | Final Critic Score Shape: {score_final.shape}")
    print("All checks completed successfully!")
