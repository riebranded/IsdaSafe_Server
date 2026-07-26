import joblib
import pandas as pd
from fastapi import HTTPException

from app.config import FEATURE_ORDER, MODEL_PATH
from app.schemas import PredictionResponse, SensorInput

try:
    _model = joblib.load(MODEL_PATH)
    _load_error = None
except Exception as exc:  # noqa: BLE001 - report any load failure to the API caller
    _model = None
    _load_error = str(exc)


def predict_species(data: SensorInput) -> PredictionResponse:
    if _model is None:
        raise HTTPException(
            status_code=503,
            detail=f"Model not loaded from '{MODEL_PATH}': {_load_error}",
        )

    row = [[getattr(data, name) for name in FEATURE_ORDER]]
    df = pd.DataFrame(row, columns=FEATURE_ORDER)

    species = _model.predict(df)[0]

    confidence = None
    if hasattr(_model, "predict_proba"):
        confidence = float(max(_model.predict_proba(df)[0]))

    return PredictionResponse(suitable_species=str(species), confidence=confidence)
