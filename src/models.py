import tensorflow as tf
from tensorflow.keras.layers import (
    Input, Masking, LSTM, GRU, Dense, Layer, Bidirectional,
    Conv1D, GlobalMaxPooling1D, MultiHeadAttention, Dropout,
    LayerNormalization, GlobalAveragePooling1D
)
from tensorflow.keras.models import Model
import tensorflow.keras.backend as K


# ======================================================
# Custom Complex Attention Layer
# ======================================================
class ComplexAttention(Layer):
    def __init__(self, return_attention=False, **kwargs):
        super().__init__(**kwargs)
        self.return_attention = return_attention

    def build(self, input_shape):
        self.W = self.add_weight(
            name='W',
            shape=(input_shape[-1], input_shape[-1]),
            initializer='glorot_uniform'
        )
        self.u = self.add_weight(
            name='u',
            shape=(input_shape[-1],),
            initializer='glorot_uniform'
        )
        super().build(input_shape)

    def call(self, x):
        uit = K.tanh(K.dot(x, self.W))
        ait = K.dot(uit, K.expand_dims(self.u, -1))
        ait = K.squeeze(ait, -1)
        a = K.softmax(ait)
        a_exp = K.expand_dims(a, axis=-1)
        weighted_input = x * a_exp
        output = K.sum(weighted_input, axis=1)
        return (output, a) if self.return_attention else output


# ======================================================
# 1. LSTM Model (Giữ nguyên)
# ======================================================
def build_lstm_model(timesteps, input_dim):
    inputs = Input(shape=(timesteps, input_dim))
    x = Masking(mask_value=0.)(inputs)
    x = LSTM(64, return_sequences=True)(x)
    x = ComplexAttention()(x)
    x = Dense(32, activation='relu')(x)
    outputs = Dense(1, activation='sigmoid')(x)
    model = Model(inputs, outputs)
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model


# ======================================================
# 2. GRU Model
# ======================================================
def build_gru_model(timesteps, input_dim):
    inputs = Input(shape=(timesteps, input_dim))
    x = Masking(mask_value=0.)(inputs)
    x = GRU(64, return_sequences=True)(x)
    x = GRU(32)(x)
    x = Dense(32, activation='relu')(x)
    outputs = Dense(1, activation='sigmoid')(x)
    model = Model(inputs, outputs)
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model


# ======================================================
# 3. Bidirectional LSTM Model
# ======================================================
def build_bilstm_model(timesteps, input_dim):
    inputs = Input(shape=(timesteps, input_dim))
    x = Masking(mask_value=0.)(inputs)
    x = Bidirectional(LSTM(64, return_sequences=True))(x)
    x = Bidirectional(LSTM(32))(x)
    x = Dense(32, activation='relu')(x)
    outputs = Dense(1, activation='sigmoid')(x)
    model = Model(inputs, outputs)
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model


# ======================================================
# 4. CNN-1D Model — nhanh nhất
# ======================================================
def build_cnn1d_model(timesteps, input_dim):
    inputs = Input(shape=(timesteps, input_dim))
    x = Conv1D(64, kernel_size=3, activation='relu', padding='same')(inputs)
    x = Conv1D(32, kernel_size=3, activation='relu', padding='same')(x)
    x = GlobalMaxPooling1D()(x)
    x = Dense(32, activation='relu')(x)
    x = Dropout(0.2)(x)
    outputs = Dense(1, activation='sigmoid')(x)
    model = Model(inputs, outputs)
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model


# ======================================================
# 5. Transformer Encoder Model — mạnh nhất
# ======================================================
def build_transformer_model(timesteps, input_dim, num_heads=4, ff_dim=64):
    inputs = Input(shape=(timesteps, input_dim))

    # Self-Attention block
    attn_output = MultiHeadAttention(num_heads=num_heads, key_dim=input_dim)(inputs, inputs)
    attn_output = Dropout(0.1)(attn_output)
    out1 = LayerNormalization(epsilon=1e-6)(inputs + attn_output)

    # Feed Forward network
    ffn = Dense(ff_dim, activation="relu")(out1)
    ffn = Dense(input_dim)(ffn)
    out2 = LayerNormalization(epsilon=1e-6)(out1 + ffn)

    # Pooling & output
    x = GlobalAveragePooling1D()(out2)
    x = Dense(32, activation='relu')(x)
    outputs = Dense(1, activation='sigmoid')(x)

    model = Model(inputs, outputs)
    model.compile(optimizer="adam",
                  loss="binary_crossentropy",
                  metrics=["accuracy"])
    return model


# ======================================================
# Tập hợp các mô hình chuỗi thời gian
# ======================================================
def get_time_series_models(timesteps, input_dim):
    return {
        "LSTM": build_lstm_model(timesteps, input_dim),
        "GRU": build_gru_model(timesteps, input_dim),
        "BiLSTM": build_bilstm_model(timesteps, input_dim),
        "CNN1D": build_cnn1d_model(timesteps, input_dim),
        "Transformer": build_transformer_model(timesteps, input_dim),
    }
