import tensorflow as tf
from tensorflow.keras.layers import Input, Masking, LSTM, GRU, Dense, Layer, Conv1D, Dropout, LayerNormalization, MultiHeadAttention, GlobalAveragePooling1D, Bidirectional
from tensorflow.keras.models import Model
import tensorflow.keras.backend as K

# -----------------------------
# Custom Attention Layer
# -----------------------------
class ComplexAttention(Layer):
    def __init__(self, return_attention=False, **kwargs):
        super().__init__(**kwargs)
        self.return_attention = return_attention

    def build(self, input_shape):
        self.W = self.add_weight(name='W', shape=(input_shape[-1], input_shape[-1]),
                                 initializer='glorot_uniform')
        self.u = self.add_weight(name='u', shape=(input_shape[-1],),
                                 initializer='glorot_uniform')
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


# -----------------------------
# LSTM Model
# -----------------------------
def build_lstm_model(timesteps, input_dim):
    inputs = Input(shape=(timesteps, input_dim))
    x = Masking(mask_value=0.)(inputs)
    x = LSTM(64, return_sequences=True)(x)
    x = ComplexAttention()(x)
    x = Dense(32, activation='relu')(x)
    outputs = Dense(1, activation='sigmoid')(x)
    model = Model(inputs, outputs)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model


# -----------------------------
# GRU Model
# -----------------------------
def build_gru_model(timesteps, input_dim):
    inputs = Input(shape=(timesteps, input_dim))
    x = Masking(mask_value=0.)(inputs)
    x = GRU(64, return_sequences=True)(x)
    x = GlobalAveragePooling1D()(x)
    x = Dense(32, activation='relu')(x)
    outputs = Dense(1, activation='sigmoid')(x)
    model = Model(inputs, outputs)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model


# -----------------------------
# Bi-LSTM Model
# -----------------------------
def build_bilstm_model(timesteps, input_dim):
    inputs = Input(shape=(timesteps, input_dim))
    x = Masking(mask_value=0.)(inputs)
    x = Bidirectional(LSTM(64, return_sequences=True))(x)
    x = GlobalAveragePooling1D()(x)
    x = Dense(32, activation='relu')(x)
    outputs = Dense(1, activation='sigmoid')(x)
    model = Model(inputs, outputs)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model


# -----------------------------
# Transformer Encoder Model
# -----------------------------
def build_transformer_model(timesteps, input_dim, num_heads=4):
    inputs = Input(shape=(timesteps, input_dim))
    x = Masking(mask_value=0.)(inputs)

    attn = MultiHeadAttention(num_heads=num_heads, key_dim=input_dim)(x, x)
    x = LayerNormalization()(x + attn)

    ff = Dense(128, activation='relu')(x)
    ff = Dense(input_dim)(ff)
    x = LayerNormalization()(x + ff)

    x = GlobalAveragePooling1D()(x)
    outputs = Dense(1, activation='sigmoid')(x)
    model = Model(inputs, outputs)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model


# -----------------------------
# TCN Model (Temporal Convolutional Network)
# -----------------------------
def build_tcn_model(timesteps, input_dim):
    inputs = Input(shape=(timesteps, input_dim))
    x = Masking(mask_value=0.)(inputs)

    x = Conv1D(64, kernel_size=3, padding="causal", activation="relu")(x)
    x = Conv1D(64, kernel_size=3, padding="causal", activation="relu")(x)
    x = GlobalAveragePooling1D()(x)
    x = Dense(32, activation="relu")(x)
    outputs = Dense(1, activation="sigmoid")(x)

    model = Model(inputs, outputs)
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model


# -----------------------------
# Return all time-series models
# -----------------------------
def get_timeseries_models():
    return {
        "LSTM_Attention": build_lstm_model,
        "GRU": build_gru_model,
        "BiLSTM": build_bilstm_model,
        "Transformer": build_transformer_model,
        "TCN": build_tcn_model
    }
