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
