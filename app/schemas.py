from typing import Optional

from pydantic import BaseModel, Field


class SensorInput(BaseModel):
    ammonia: float = Field(..., description="Dissolved ammonia level")
    dissolved_oxygen: float = Field(..., description="Dissolved oxygen level")
    ph: float = Field(..., description="Water pH level")
    temperature: float = Field(..., description="Water temperature")


class PredictionResponse(BaseModel):
    suitable_species: str
    confidence: Optional[float] = None


class FeedingRecommendationRequest(SensorInput):
    species: str = Field(..., description="Fish species being farmed, e.g. 'Nile Tilapia'")


class FeedingRecommendation(BaseModel):
    feeding_time: str = Field(..., description="Recommended time(s) of day to feed")
    feeding_frequency: str = Field(..., description="How often to feed (times per day/week)")
    feeding_amount: str = Field(..., description="Recommended feed amount/portion guidance")
    water_quality_recommendations: list[str] = Field(
        ..., description="Actionable water quality recommendations"
    )
    possible_risks: list[str] = Field(
        ..., description="Possible risks given the current readings/species"
    )
