import joblib
import pandas as pd
from fastapi import HTTPException

from app.cache import TTLCache
from app.config import MODEL_PATH
from app.schemas import PredictionResponse, SensorInput

# Sensor readings only change every 15 minutes, so predictions for the same
# readings are cached for that long instead of re-running the model each call.
_cache = TTLCache(ttl_seconds=15 * 60)

# The .joblib file stores a bundle: {"model": ..., "features": [...], "classes": [...]}.
# "features" gives the exact column names/order the model was trained on, which use
# different names than the API's SensorInput fields.
FIELD_TO_MODEL_FEATURE = {
    "temperature": "temp_c",
    "ph": "ph",
    "dissolved_oxygen": "do_mgL",
    "ammonia": "ammonia_mgL",
}

try:
    _bundle = joblib.load(MODEL_PATH)
    _model = _bundle["model"]
    _model_features = _bundle["features"]
    _load_error = None
except Exception as exc:  # noqa: BLE001 - report any load failure to the API caller
    _model = None
    _model_features = None
    _load_error = str(exc)


def _run_prediction(data: SensorInput) -> PredictionResponse:
    field_by_feature = {model_name: field for field, model_name in FIELD_TO_MODEL_FEATURE.items()}
    row = [[getattr(data, field_by_feature[feature]) for feature in _model_features]]
    df = pd.DataFrame(row, columns=_model_features)

    species = _model.predict(df)[0]

    confidence = None
    if hasattr(_model, "predict_proba"):
        confidence = float(max(_model.predict_proba(df)[0]))

    return PredictionResponse(suitable_species=str(species), confidence=confidence)


def predict_species(data: SensorInput) -> PredictionResponse:
    if _model is None:
        raise HTTPException(
            status_code=503,
            detail=f"Model not loaded from '{MODEL_PATH}': {_load_error}",
        )

    return _cache.get_or_set(data.model_dump_json(), lambda: _run_prediction(data))
