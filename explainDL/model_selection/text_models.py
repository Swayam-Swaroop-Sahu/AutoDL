"""
text_models.py
---------------
BiLSTM with attention for text classification. Supports binary and multi-class outputs.
"""

from tensorflow.keras import models, layers, optimizers
import tensorflow.keras.backend as K

def build_text_lstm(input_shape, output_units=1, activation='sigmoid', loss='binary_crossentropy', embedding_dim=128):
    """
    input_shape: (vocab_size, max_len)
    """
    vocab_size, max_len = input_shape

    inputs = layers.Input(shape=(max_len,))
    x = layers.Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_len)(inputs)
    x = layers.Bidirectional(layers.LSTM(128, return_sequences=True, dropout=0.3))(x)

    # Attention mechanism
    attention_weights = layers.Dense(1, activation='tanh')(x)            # (batch, timesteps, 1)
    attention_weights = layers.Flatten()(attention_weights)              # (batch, timesteps)
    attention_weights = layers.Activation('softmax')(attention_weights)  # (batch, timesteps)
    attention_weights = layers.RepeatVector(256)(attention_weights)      # (batch, 256, timesteps)
    attention_weights = layers.Permute([2, 1])(attention_weights)        # (batch, timesteps, 256)
    x = layers.multiply([x, attention_weights])
    x = layers.GlobalAveragePooling1D()(x)

    x = layers.Dense(128, activation='relu')(x)
    outputs = layers.Dense(output_units, activation=activation)(x)

    model = models.Model(inputs, outputs)
    model.compile(optimizer=optimizers.Adam(learning_rate=1e-4), loss=loss, metrics=['accuracy'])
    return model
