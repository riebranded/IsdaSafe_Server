from fastapi import APIRouter, HTTPException

from app.feeding_advisor import get_feeding_recommendation
from app.predictor import predict_species
from app.schemas import (
    FeedingRecommendation,
    FeedingRecommendationRequest,
    PredictionResponse,
    SensorInput,
    SensorReading,
)

router = APIRouter()


@router.post("/sensor-readings", response_model=SensorReading)
def sensor_readings(data: SensorReading) -> SensorReading:
    return data


@router.post("/predict", response_model=PredictionResponse)
def predict(data: SensorInput) -> PredictionResponse:
    try:
        return predict_species(data)
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001 - never leak a raw stack trace to the client
        raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}") from exc


@router.post("/feeding-recommendation", response_model=FeedingRecommendation)
def feeding_recommendation(data: FeedingRecommendationRequest) -> FeedingRecommendation:
    try:
        return get_feeding_recommendation(data)
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001 - never leak a raw stack trace to the client
        raise HTTPException(status_code=500, detail=f"Feeding recommendation failed: {exc}") from exc
