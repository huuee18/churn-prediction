
import tensorflow as tf
from tensorflow.keras.layers import (
    Input, Dense, Dropout, LayerNormalization,
    GlobalAveragePooling1D, Permute, Reshape
)
from tensorflow.keras.models import Model
def build_tsmixer_model(timesteps, input_dim, hidden_dim=64):
    inputs = Input(shape=(timesteps, input_dim))

    # -------- Time Mixing --------
    x = Permute((2, 1))(inputs)                 # (B, F, T)
    x = Dense(timesteps, activation="relu")(x)
    x = Permute((2, 1))(x)                      # (B, T, F)

    # -------- Feature Mixing --------
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
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc")]
    )
    return model
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
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc")]
    )
    return model
def build_dlinear_model(timesteps, input_dim):
    inputs = Input(shape=(timesteps, input_dim))

    x = GlobalAveragePooling1D()(inputs)
    x = Dense(input_dim, activation=None)(x)

    x = Dense(32, activation="relu")(x)
    outputs = Dense(1, activation="sigmoid")(x)

    model = Model(inputs, outputs)
    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc")]
    )
    return model
def get_timeseries_models(timesteps, input_dim):
    return {
        "TSMixer": build_tsmixer_model(timesteps, input_dim),
        "NBEATS_Classifier": build_nbeats_classifier(timesteps, input_dim),
        "DLinear": build_dlinear_model(timesteps, input_dim)
    }

