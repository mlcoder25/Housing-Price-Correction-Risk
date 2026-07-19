"""
Serves the trained housing price-correction risk classifier.

Loads model/model.pkl -- the artifact `train.py` saves after registering the
champion run in MLflow. Run with:
    uvicorn api.main:app --reload --port 8000
"""
import os

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.data import FEATURE_NAMES

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "model", "model.pkl")

app = FastAPI(
    title="Housing Price Correction Risk Classifier",
    description=(
        "Predicts whether a country will report a year-over-year house "
        "price DECLINE this quarter, using only prior-quarter momentum "
        "and static country attributes (no same-quarter figures)."
    ),
    version="1.0.0",
)

_model = None


def get_model():
    global _model
    if _model is None:
        if not os.path.exists(MODEL_PATH):
            raise HTTPException(status_code=503, detail="Model artifact not found — run training first.")
        _model = joblib.load(MODEL_PATH)
    return _model


class QuarterFeatures(BaseModel):
    lag1_quarterly_change_pct: float = Field(..., description="Last quarter's QoQ price change (%)")
    lag1_yearly_change_pct: float = Field(..., description="Last quarter's YoY price change (%)")
    lag2_quarterly_change_pct: float = Field(..., description="QoQ price change two quarters ago (%)")
    momentum_change: float = Field(..., description="lag1_quarterly_change_pct - lag2_quarterly_change_pct")
    eu_member: int = Field(..., ge=0, le=1, description="1 if EU member, else 0")
    eurozone_member: int = Field(..., ge=0, le=1, description="1 if Eurozone member, else 0")
    quarter_num: int = Field(..., ge=1, le=4, description="Calendar quarter (1-4)")

    class Config:
        json_schema_extra = {
            "example": {
                "lag1_quarterly_change_pct": -1.2,
                "lag1_yearly_change_pct": -0.2,
                "lag2_quarterly_change_pct": 0.4,
                "momentum_change": -1.6,
                "eu_member": 1,
                "eurozone_member": 1,
                "quarter_num": 2,
            }
        }


class PredictionResponse(BaseModel):
    decline_risk: bool = Field(..., description="True if the model predicts a YoY price decline this quarter")
    probability: float = Field(..., description="Predicted probability of a YoY price decline")


@app.get("/health")
def health():
    model_ready = os.path.exists(MODEL_PATH)
    return {"status": "ok" if model_ready else "model_missing"}


@app.post("/predict", response_model=PredictionResponse)
def predict(features: QuarterFeatures):
    model = get_model()
    row = pd.DataFrame([[getattr(features, name) for name in FEATURE_NAMES]], columns=FEATURE_NAMES)
    proba = float(model.predict_proba(row)[0][1])
    return PredictionResponse(decline_risk=bool(proba >= 0.5), probability=proba)
