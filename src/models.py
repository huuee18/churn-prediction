import tensorflow as tf
from tensorflow.keras.layers import Input, Masking, LSTM, Dense, Layer
from tensorflow.keras.models import Model
import tensorflow.keras.backend as K
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression

# -------------------
# Custom Attention Layer
# -------------------
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

# -------------------
# LSTM Model
# -------------------
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

# -------------------
# ML Models
# -------------------
def get_ml_models():
    models_tree = {
        'Random Forest': RandomForestClassifier(
            random_state=42, max_depth=6, min_samples_leaf=20, class_weight='balanced'
        ),
        'XGBoost': XGBClassifier(
            random_state=42, max_depth=5, subsample=0.8, colsample_bytree=0.8, eval_metric="logloss"
        ),
        'Decision Tree': DecisionTreeClassifier(
            random_state=42, max_depth=5, min_samples_leaf=10, max_features=0.5
        )
    }
    model_linear = LogisticRegression(C=0.1, max_iter=3000, random_state=42)
    return models_tree, model_linear
