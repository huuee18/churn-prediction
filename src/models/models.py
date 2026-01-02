import tensorflow as tf
from tensorflow.keras.layers import (
    Input, Dense, Dropout, LayerNormalization,
    GlobalAveragePooling1D, Permute, Reshape,
    LSTM, GRU, Conv1D
)
from tensorflow.keras.models import Model


# =====================================================
# TSMixer
# =====================================================
def build_tsmixer_model(timesteps, input_dim, hidden_dim=64):
    inputs = Input(shape=(timesteps, input_dim))

    # ---- Time Mixing ----
    x = Permute((2, 1))(inputs)
    x = Dense(timesteps, activation="relu")(x)
    x = Permute((2, 1))(x)

    # ---- Feature Mixing ----
    y = Dense(hidden_dim, activation="relu")(x)
    y = Dense(input_dim)(y)

    x = LayerNormalization()(x + y)
    x = GlobalAveragePooling1D()(x)

    x = Dense(32, activation="relu")(x)
    outputs = Dense(1, activation="sigmoid")(x)

    model = Model(inputs, outputs)
    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.AUC(name="auc_roc"),
            tf.keras.metrics.AUC(name="auc_pr", curve="PR")
        ]
    )
    return model


# =====================================================
# N-BEATS (Classifier version)
# =====================================================
def build_nbeats_classifier(timesteps, input_dim, hidden_dim=128):
    inputs = Input(shape=(timesteps, input_dim))
    x = Reshape((timesteps * input_dim,))(inputs)

    for _ in range(4):
        x = Dense(hidden_dim, activation="relu")(x)
        x = Dense(hidden_dim, activation="relu")(x)

    x = Dense(32, activation="relu")(x)
    outputs = Dense(1, activation="sigmoid")(x)

    model = Model(inputs, outputs)
    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.AUC(name="auc_roc"),
            tf.keras.metrics.AUC(name="auc_pr", curve="PR")
        ]
    )
    return model


# =====================================================
# DLinear
# =====================================================
def build_dlinear_model(timesteps, input_dim):
    inputs = Input(shape=(timesteps, input_dim))

    x = GlobalAveragePooling1D()(inputs)
    x = Dense(input_dim)(x)

    x = Dense(32, activation="relu")(x)
    outputs = Dense(1, activation="sigmoid")(x)

    model = Model(inputs, outputs)
    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.AUC(name="auc_roc"),
            tf.keras.metrics.AUC(name="auc_pr", curve="PR")
        ]
    )
    return model


# =====================================================
# BASELINE 1: LSTM
# =====================================================
def build_lstm_model(timesteps, input_dim):
    inputs = Input(shape=(timesteps, input_dim))
    x = LSTM(64, return_sequences=False)(inputs)
    x = Dense(32, activation="relu")(x)
    outputs = Dense(1, activation="sigmoid")(x)

    model = Model(inputs, outputs)
    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.AUC(name="auc_roc"),
            tf.keras.metrics.AUC(name="auc_pr", curve="PR")
        ]
    )
    return model


# =====================================================
# BASELINE 2: GRU
# =====================================================
def build_gru_model(timesteps, input_dim):
    inputs = Input(shape=(timesteps, input_dim))
    x = GRU(64, return_sequences=False)(inputs)
    x = Dense(32, activation="relu")(x)
    outputs = Dense(1, activation="sigmoid")(x)

    model = Model(inputs, outputs)
    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.AUC(name="auc_roc"),
            tf.keras.metrics.AUC(name="auc_pr", curve="PR")
        ]
    )
    return model


# =====================================================
# BASELINE 3: Temporal CNN
# =====================================================
def build_cnn1d_model(timesteps, input_dim):
    inputs = Input(shape=(timesteps, input_dim))
    x = Conv1D(64, kernel_size=3, padding="same", activation="relu")(inputs)
    x = GlobalAveragePooling1D()(x)
    x = Dense(32, activation="relu")(x)
    outputs = Dense(1, activation="sigmoid")(x)

    model = Model(inputs, outputs)
    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.AUC(name="auc_roc"),
            tf.keras.metrics.AUC(name="auc_pr", curve="PR")
        ]
    )
    return model


# =====================================================
# Model registry
# =====================================================
def get_timeseries_models(timesteps, input_dim):
    return {
        "TSMixer": build_tsmixer_model(timesteps, input_dim),
        "LSTM": build_lstm_model(timesteps, input_dim),
        "GRU": build_gru_model(timesteps, input_dim),
        "CNN1D": build_cnn1d_model(timesteps, input_dim),
        "NBEATS_Classifier": build_nbeats_classifier(timesteps, input_dim),
        "DLinear": build_dlinear_model(timesteps, input_dim),
    }
