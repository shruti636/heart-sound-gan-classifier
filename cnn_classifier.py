import tensorflow as tf
from tensorflow.keras import layers, Model

def build_cnn_classifier(input_shape=(64, 13)):
    """
    Builds the hybrid Conv1D + LSTM classifier model.
    Classifies MFCC feature matrices of shape (64, 13) into Normal (0) or Abnormal (1).
    """
    mfcc_input = layers.Input(shape=input_shape)
    
    # Block 1
    x = layers.Conv1D(32, kernel_size=3, padding='same')(mfcc_input)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(alpha=0.1)(x)
    x = layers.MaxPooling1D(pool_size=2)(x)
    x = layers.Dropout(0.25)(x)
    
    # Block 2
    x = layers.Conv1D(64, kernel_size=3, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(alpha=0.1)(x)
    x = layers.MaxPooling1D(pool_size=2)(x)
    x = layers.Dropout(0.25)(x)
    
    # Block 3
    x = layers.Conv1D(128, kernel_size=3, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(alpha=0.1)(x)
    x = layers.MaxPooling1D(pool_size=2)(x)
    x = layers.Dropout(0.25)(x)
    
    # Recurrent Temporal Modeling (Bidirectional LSTM)
    x = layers.Bidirectional(layers.LSTM(64, return_sequences=False))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)
    
    # Output layer (sigmoid for binary classification: Normal vs Abnormal)
    out = layers.Dense(1, activation='sigmoid')(x)
    
    model = Model(mfcc_input, out, name="CNN_LSTM_Classifier")
    return model

if __name__ == '__main__':
    # Print model summary
    model = build_cnn_classifier((64, 13))
    model.summary()
    
    # Compile model with default parameters to verify it compiles
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    print("CNN/LSTM Classifier compiled successfully!")
