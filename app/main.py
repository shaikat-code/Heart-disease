from contextlib import asynccontextmanager
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException

from app.schemas import HeartInput, PredictionOutput

MODEL_PATH = Path(__file__).resolve().parent.parent / "model" / "heart_model.joblib"
FEATURES = list(HeartInput.model_fields.keys())

state = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    if not MODEL_PATH.exists():
        raise RuntimeError(f"Model file not found at {MODEL_PATH}. Run train.py first.")
    state["model"] = joblib.load(MODEL_PATH)
    yield
    state.clear()


app = FastAPI(title="Heart Disease Prediction API", version="1.0.0", lifespan=lifespan)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/info")
def info():
    model = state["model"]
    final_step = model.steps[-1][1] if hasattr(model, "steps") else model
    return {
        "model_type": type(final_step).__name__,
        "n_features": len(FEATURES),
        "features": FEATURES,
        "target": "heart_disease (true = disease present)",
    }


@app.post("/predict", response_model=PredictionOutput)
def predict(payload: HeartInput):
    try:
        row = pd.DataFrame([payload.model_dump()], columns=FEATURES)
        pred = state["model"].predict(row)[0]
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}")
    return PredictionOutput(heart_disease=bool(pred))
