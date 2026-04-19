from fastapi import FastAPI, HTTPException
from app.feature_engineering import extract_features
from app.ml_model import predict
from app.models import Customer

import time
import psutil
import os

app = FastAPI(title="Churn Prediction ML Service")

# -----------------------------
# Health Check
# -----------------------------
@app.get("/")
def home():
    return {"message": "ML Pipeline Churn Prediction Service Running"}

@app.get("/health")
def health():
    return {"status": "ok"}

# -----------------------------
# 🔥 MAIN ENDPOINT
# -----------------------------
@app.post("/predict-risk")
def predict_risk(customer: Customer):
    try:
        # ⏱️ Start latency timer
        start_time = time.time()

        # Step 1: Feature extraction
        features = extract_features(customer.dict())

        # Step 2: Prediction
        risk = predict(features)

        # ⏱️ End latency timer
        latency = time.time() - start_time

        # 🧠 Memory usage
        process = psutil.Process(os.getpid())
        memory = process.memory_info().rss / (1024 * 1024)  # MB

        return {
            "risk": risk,
            "monitoring": {
                "latency_ms": round(latency * 1000, 2),
                "memory_mb": round(memory, 2)
            }
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))