from pydantic import BaseModel, Field
from typing import Dict, List, Tuple, Optional, Union, Any

class FuelInferRequest(BaseModel):
    manufacturer: str = Field(..., json_schema_extra={"example": "Toyota"})
    model: str = Field(..., json_schema_extra={"example": "Prius"})
    variant: Optional[str] = Field(default="", json_schema_extra={"example": "Base"})
    vehicle_type: Optional[str] = Field(default="car", json_schema_extra={"example": "car"})
    track_id: Optional[Union[int, str]] = Field(default=0, json_schema_extra={"example": 101})

class FuelInferResponse(BaseModel):
    fuel_type: str = Field(..., json_schema_extra={"example": "HYBRID"})
    confidence: float = Field(..., json_schema_extra={"example": 0.95})
    mapping_source: str = Field(..., json_schema_extra={"example": "Exact Match"})
    mapping_notes: str = Field(..., json_schema_extra={"example": "Matched in reference database."})

class EmissionsEstimateRequest(BaseModel):
    vehicle_counts: Dict[str, int] = Field(
        ...,
        json_schema_extra={"example": {"car": 10, "truck": 3, "bus": 1, "motorcycle": 2}}
    )
    duration_seconds: Optional[float] = Field(default=3600.0, json_schema_extra={"example": 3600.0})

class EmissionsEstimateResponse(BaseModel):
    vehicle_pollution_index: float = Field(..., json_schema_extra={"example": 45.2})
    relative_noise_index: float = Field(..., json_schema_extra={"example": 56.4})
    hourly_emissions_g_hr: Dict[str, float] = Field(
        ...,
        json_schema_extra={"example": {"PM2.5": 0.42, "NOx": 4.12, "CO2": 2150.0}}
    )
    category_pollution: str = Field(..., json_schema_extra={"example": "MODERATE ESTIMATED VEHICLE CONTRIBUTION"})
    category_noise: str = Field(..., json_schema_extra={"example": "ELEVATED RELATIVE TRAFFIC NOISE"})

class DetectionBox(BaseModel):
    track_id: Optional[Union[int, str]] = None
    class_name: str
    confidence: float
    bbox: Tuple[float, float, float, float]
    fuel_type: str
    fuel_confidence: float

class ImageInferenceResponse(BaseModel):
    vehicle_count: int
    fuel_counts: Dict[str, int]
    pollution_index: float
    relative_noise_index: float
    pollutants_g_hr: Dict[str, float]
    detections: List[DetectionBox]
    processing_time_ms: float
