from fastapi import APIRouter, HTTPException

from app.predictor import predict_species
from app.schemas import PredictionResponse, SensorInput

router = APIRouter()


@router.post("/predict", response_model=PredictionResponse)
def predict(data: SensorInput) -> PredictionResponse:
    try:
        return predict_species(data)
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001 - never leak a raw stack trace to the client
        raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}") from exc
