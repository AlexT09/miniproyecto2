"""
Etapa 3 — API REST para predicción de enfermedad cardíaca
Uso local: uvicorn app.api:app --reload --port 8000
Docs:      http://localhost:8000/docs
"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np
import os

# ── Cargar modelo al iniciar ──────────────────────────────────────────────────
MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.joblib")
model = joblib.load(MODEL_PATH)

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Heart Disease Prediction API",
    description=(
        "Predice la probabilidad de enfermedad cardíaca usando un modelo "
        "GradientBoosting entrenado con el dataset de Kaggle (918 pacientes)."
    ),
    version="1.0.0",
)

# ── Esquemas ──────────────────────────────────────────────────────────────────
# El modelo espera las features en el orden que produjo get_dummies sobre:
# Age, RestingBP, Cholesterol, FastingBS, MaxHR, Oldpeak,
# Sex_M, ChestPainType_ATA, ChestPainType_NAP, ChestPainType_TA,
# RestingECG_Normal, RestingECG_ST,
# ExerciseAngina_Y,
# ST_Slope_Flat, ST_Slope_Up
FEATURE_NAMES = [
    "Age", "RestingBP", "Cholesterol", "FastingBS", "MaxHR", "Oldpeak",
    "Sex_M",
    "ChestPainType_ATA", "ChestPainType_NAP", "ChestPainType_TA",
    "RestingECG_Normal", "RestingECG_ST",
    "ExerciseAngina_Y",
    "ST_Slope_Flat", "ST_Slope_Up",
]


class PatientInput(BaseModel):
    features: list

    model_config = {
        "json_schema_extra": {
            "example": {
                "features": [40, 140, 289, 0, 172, 0.0, 1, 1, 0, 0, 1, 0, 0, 0, 1]
            }
        }
    }


class PredictionResponse(BaseModel):
    heart_disease_probability: float
    prediction: int
    risk_level: str
    feature_count: int


# ── Endpoints ─────────────────────────────────────────────────────────────────
@app.get("/")
def root():
    return {
        "service": "Heart Disease Prediction API",
        "status": "running",
        "model": "GradientBoostingClassifier",
        "docs": "/docs",
    }


@app.get("/health")
def health():
    return {"status": "healthy", "model_loaded": model is not None}


@app.get("/features")
def get_features():
    """Devuelve los nombres y el orden esperado de las features."""
    return {
        "feature_count": len(FEATURE_NAMES),
        "features": {i: name for i, name in enumerate(FEATURE_NAMES)},
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(data: PatientInput):
    if len(data.features) != len(FEATURE_NAMES):
        raise HTTPException(
            status_code=422,
            detail=(
                f"Se esperan {len(FEATURE_NAMES)} features, "
                f"se recibieron {len(data.features)}. "
                f"Consulta GET /features para ver el orden correcto."
            ),
        )
    try:
        X = np.array(data.features, dtype=float).reshape(1, -1)
        proba = float(model.predict_proba(X)[0][1])
        pred  = int(proba > 0.5)

        if proba < 0.3:
            risk = "Bajo"
        elif proba < 0.6:
            risk = "Moderado"
        else:
            risk = "Alto"

        return {
            "heart_disease_probability": round(proba, 4),
            "prediction": pred,
            "risk_level": risk,
            "feature_count": len(data.features),
        }
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Error en predicción: {exc}")
