import joblib
import uvicorn
import numpy as np
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
from tensorflow.keras.models import load_model

# ====== Load models ======
models_dir = "/content/outputs/models"
lstm_path = f"{models_dir}/lstm/lstm_model.h5"

# Tree-based models
model_rf = joblib.load(f"{models_dir}/Random Forest.joblib")
model_xgb = joblib.load(f"{models_dir}/XGBoost.joblib")
model_dt = joblib.load(f"{models_dir}/Decision Tree.joblib")
model_lr = joblib.load(f"{models_dir}/logistic_regression.joblib")

# LSTM model
model_lstm = load_model(lstm_path)

# ====== API app ======
app = FastAPI(title="Churn Prediction API")

# Input schema
class CustomerData(BaseModel):
    features: list   # danh sách giá trị feature (đúng thứ tự preprocess)
    sequence: list = None  # cho LSTM (chuỗi theo thời gian, dạng 2D)


# ====== Predict function ======
@app.post("/predict")
def predict(data: CustomerData):
    # Convert input to numpy
    X = np.array(data.features).reshape(1, -1)

    # ML model predictions
    preds = {
        "RandomForest": float(model_rf.predict_proba(X)[:, 1][0]),
        "XGBoost": float(model_xgb.predict_proba(X)[:, 1][0]),
        "DecisionTree": float(model_dt.predict_proba(X)[:, 1][0]),
        "LogisticRegression": float(model_lr.predict_proba(X)[:, 1][0]),
    }

    # LSTM nếu có sequence
    if data.sequence is not None:
        seq = np.array(data.sequence).reshape(1, len(data.sequence), -1)
        preds["LSTM"] = float(model_lstm.predict(seq)[0][0])

    return {
        "churn_probability": preds,
        "churn_prediction": {k: int(v > 0.5) for k, v in preds.items()}
    }


# ====== Run ======
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
