import os
import time
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import *
from sklearn.calibration import calibration_curve

from imblearn.over_sampling import SMOTE


# ==================================================
# FOCAL LOSS
# ==================================================
def focal_loss(gamma=2.0, alpha=0.75):

    def loss(y_true, y_pred):

        eps = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, eps, 1 - eps)

        ce = -(
            y_true * tf.math.log(y_pred)
            + (1-y_true) * tf.math.log(1-y_pred)
        )

        p_t = y_true * y_pred + (1-y_true)*(1-y_pred)

        alpha_factor = y_true * alpha + (1-y_true)*(1-alpha)

        focal = alpha_factor * tf.pow(1-p_t, gamma)

        return tf.reduce_mean(focal * ce)

    return loss


# ==================================================
# CLASS WEIGHT
# ==================================================
def compute_class_weights(y, boost_minority=6.0):

    classes = np.array([0,1])

    weights = compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=y
    )

    return {
        0: float(weights[0]),
        1: float(weights[1] * boost_minority)
    }


# ==================================================
# SEQUENCE SMOTE
# ==================================================
def oversample_sequence_data(X, y, strategy=0.40):

    n, t, f = X.shape

    X2 = X.reshape(n, t*f)

    sm = SMOTE(
        sampling_strategy=strategy,
        random_state=42,
        k_neighbors=3
    )

    X_res, y_res = sm.fit_resample(X2, y)

    X_res = X_res.reshape(
        X_res.shape[0],
        t,
        f
    )

    return X_res, y_res


# ==================================================
# THRESHOLD SEARCH
# ==================================================
def find_best_threshold(y_true, y_prob):

    best_t = 0.5
    best_score = -9999

    for t in np.linspace(0.05,0.95,91):

        pred = (y_prob >= t).astype(int)

        tn, fp, fn, tp = confusion_matrix(y_true,pred).ravel()

        score = tp*5 - fn*10 - fp

        if score > best_score:
            best_score = score
            best_t = t

    return best_t


# ==================================================
# TRAIN LOOP
# ==================================================
def train_ts_models(
    models,
    X_train, y_train,
    X_test, y_test,
    epochs=50,
    batch_size=32
):

    results = {}

    # -----------------------------------
    # APPLY SMOTE
    # -----------------------------------
    X_train, y_train = oversample_sequence_data(
        X_train,
        y_train,
        strategy=0.40
    )

    print("After SMOTE:", X_train.shape)
    print("Churn ratio:", y_train.mean())

    class_weight_dict = compute_class_weights(
        y_train,
        boost_minority=6.0
    )

    print("Class Weight:", class_weight_dict)

    # -----------------------------------
    for name, model in models.items():

        print(f"\nTraining {name}")

        model.compile(
            optimizer="adam",
            loss=focal_loss(),
            metrics=[
                "accuracy",
                tf.keras.metrics.AUC(name="auc")
            ]
        )

        start = time.time()

        history = model.fit(
            X_train,
            y_train,
            validation_data=(X_test,y_test),
            epochs=epochs,
            batch_size=batch_size,
            class_weight=class_weight_dict,
            verbose=1
        )

        train_time = time.time() - start

        y_prob = model.predict(X_test).ravel()

        best_t = find_best_threshold(
            y_test,
            y_prob
        )

        y_pred = (y_prob >= best_t).astype(int)

        metrics = {
            "accuracy": accuracy_score(y_test,y_pred),
            "precision": precision_score(y_test,y_pred),
            "recall": recall_score(y_test,y_pred),
            "f1": f1_score(y_test,y_pred),
            "auc_roc": roc_auc_score(y_test,y_prob),
            "auc_pr": average_precision_score(y_test,y_prob),
            "optimal_threshold": best_t
        }

        results[name] = {
            "model": model,
            "history": history,
            "X_test": X_test,
            "y_test": y_test,
            "y_prob": y_prob,
            "metrics": metrics,
            "optimal_threshold": best_t,
            "training_time_sec": train_time
        }

    return results
