import tensorflow as tf
from tensorflow.keras import layers, Model

def build_cnn_classifier(input_shape=(64, 39)):
    """
    Builds the simple hybrid Conv1D + Bidirectional LSTM classifier model.
    Classifies MFCC feature matrices of shape (64, 39) into Normal (0) or Abnormal (1).
    """
    inputs = layers.Input(shape=input_shape)
    
    # 1. First Conv1D layer
    x = layers.Conv1D(filters=32, kernel_size=3, padding='same', activation='relu')(inputs)
    
    # 2. MaxPooling layer (reduces sequence length by half: 64 -> 32)
    x = layers.MaxPooling1D(pool_size=2)(x)
    x = layers.Dropout(0.1)(x)
    
    # 3. Second Conv1D layer
    x = layers.Conv1D(filters=64, kernel_size=3, padding='same', activation='relu')(x)
    x = layers.Dropout(0.1)(x)
    
    # 4. Bidirectional LSTM layer for sequence modeling
    x = layers.Bidirectional(layers.LSTM(32, return_sequences=False))(x)
    x = layers.Dropout(0.2)(x)
    
    # 5. Output layer (Sigmoid for binary classification: Normal vs Abnormal)
    outputs = layers.Dense(1, activation='sigmoid')(x)
    
    model = Model(inputs, outputs, name="Conv1D_BiLSTM_Classifier")
    return model

if __name__ == '__main__':
    # Print model summary
    model = build_cnn_classifier((64, 39))
    model.summary()
    
    # Compile model with default parameters to verify it compiles
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    print("Conv1D-BiLSTM Classifier compiled successfully!")
