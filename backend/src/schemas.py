"""
Pydantic schemas for API request/response validation.
"""

from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field


# ============== Request Schemas ==============

class PredictRequest(BaseModel):
    """Request schema for RUL prediction."""
    unit_id: int = Field(..., ge=1, description="Engine unit ID")
    
    class Config:
        json_schema_extra = {
            "example": {
                "unit_id": 1
            }
        }


class ChatRequest(BaseModel):
    """Request schema for chat endpoint."""
    message: str = Field(..., min_length=1, description="User message")
    conversation_id: Optional[str] = Field(None, description="Optional conversation ID for context")
    
    class Config:
        json_schema_extra = {
            "example": {
                "message": "What is the RUL for engine unit 5?",
                "conversation_id": None
            }
        }


class AnalyzeRequest(BaseModel):
    """Request schema for sensor analysis."""
    unit_id: int = Field(..., ge=1, description="Engine unit ID")
    cycle: Optional[int] = Field(None, ge=1, description="Specific cycle to analyze")
    
    class Config:
        json_schema_extra = {
            "example": {
                "unit_id": 1,
                "cycle": 100
            }
        }


class SensorDataInput(BaseModel):
    """Schema for raw sensor data input."""
    unit_id: int = Field(..., ge=1)
    cycle: int = Field(..., ge=1)
    op_setting_1: float
    op_setting_2: float
    op_setting_3: float
    sensor_2: float = Field(..., description="Total temperature at LPC outlet")
    sensor_3: float = Field(..., description="Total temperature at HPC outlet")
    sensor_4: float = Field(..., description="Total temperature at LPT outlet")
    sensor_7: float = Field(..., description="Total pressure at HPC outlet")
    sensor_8: float = Field(..., description="Physical fan speed")
    sensor_9: float = Field(..., description="Physical core speed")
    sensor_11: float = Field(..., description="Static pressure at HPC outlet")
    sensor_12: float = Field(..., description="Fuel flow ratio")
    sensor_13: float = Field(..., description="Corrected fan speed")
    sensor_14: float = Field(..., description="Corrected core speed")
    sensor_15: float = Field(..., description="Bypass Ratio")
    sensor_17: float = Field(..., description="Bleed Enthalpy")
    sensor_20: float = Field(..., description="HPT coolant bleed")
    sensor_21: float = Field(..., description="LPT coolant bleed")


# ============== Response Schemas ==============

class PredictResponse(BaseModel):
    """Response schema for RUL prediction."""
    unit_id: int
    predicted_rul: float = Field(..., description="Predicted remaining useful life in cycles")
    state: str = Field(..., description="Engine state: healthy, degrading, or critical")
    state_probabilities: Dict[str, float] = Field(..., description="Probability for each state")
    confidence: str = Field(..., description="Prediction confidence level")
    
    class Config:
        json_schema_extra = {
            "example": {
                "unit_id": 1,
                "predicted_rul": 85.5,
                "state": "degrading",
                "state_probabilities": {
                    "healthy": 0.15,
                    "degrading": 0.70,
                    "critical": 0.15
                },
                "confidence": "medium"
            }
        }


class ChatResponse(BaseModel):
    """Response schema for chat endpoint."""
    response: str = Field(..., description="Assistant's response")
    conversation_id: str = Field(..., description="Conversation ID for follow-up")
    tools_used: List[str] = Field(default=[], description="Tools invoked during response")
    success: bool = Field(default=True, description="Whether the response was successfully generated")
    
    class Config:
        json_schema_extra = {
            "example": {
                "response": "Engine unit 5 has a predicted RUL of 72 cycles...",
                "conversation_id": "conv_abc123",
                "tools_used": ["predict_rul"]
            }
        }


class AnalyzeResponse(BaseModel):
    """Response schema for analysis endpoint."""
    unit_id: int
    cycle: int
    analysis: str = Field(..., description="Detailed analysis text")
    anomalies: List[Dict[str, Any]] = Field(default=[], description="Detected anomalies")
    
    class Config:
        json_schema_extra = {
            "example": {
                "unit_id": 1,
                "cycle": 100,
                "analysis": "Sensor readings within normal range...",
                "anomalies": []
            }
        }


class EngineListResponse(BaseModel):
    """Response schema for listing engines."""
    dataset: str
    total_engines: int
    engines: List[Dict[str, Any]]


class HealthResponse(BaseModel):
    """Response schema for health check."""
    status: str = Field(..., description="System status")
    models_loaded: bool
    dataset: str
    version: str


class MaintenanceResponse(BaseModel):
    """Response schema for maintenance recommendations."""
    unit_id: int
    predicted_rul: float
    urgency: str = Field(..., description="LOW, MEDIUM, or CRITICAL")
    recommendations: str
    next_inspection_cycles: Optional[int] = None


class ErrorResponse(BaseModel):
    """Standard error response."""
    detail: str
    error_code: Optional[str] = None
