"""
Pydantic schemas responsible for API request and response validation.
Defines structured data models with default examples, field descriptions, and validation rules.
"""


from pydantic import BaseModel, Field
from datetime import datetime
from typing import Optional
from enum import Enum

class RiskStatus(str, Enum):
    normal = "Normal"
    warning = "Warning"
    critical = "Critical"

class BaseSchema(BaseModel):
    model_config = {"from_attributes": True}

class UnitResponse(BaseSchema):
    id: str
    name: str
    capacity: Optional[float] = None
    max_temperature: Optional[float] = None
    max_pressure: Optional[float] = None
    location: Optional[str] = None
    created_at: datetime

class RiskAssessmentResponse(BaseSchema):
    id: str
    sensor_reading_id: str
    unit_id: str
    timestamp: datetime
    risk_score: float
    status: RiskStatus
    root_cause: Optional[str] = None
    prediction_confidence: Optional[float] = None

class Token(BaseModel):
    access_token: str
    token_type: str

class TokenData(BaseModel):
    username: Optional[str] = None

class UserCreate(BaseModel):
    username: str = Field(..., min_length=3, max_length=50)
    password: str = Field(..., min_length=6)
    full_name: Optional[str] = None
    role: str = Field(default="operator", pattern=r"^(operator|engineer|admin)$")

class UserResponse(BaseSchema):
    id: str
    username: str
    full_name: Optional[str] = None
    role: str
    is_active: bool

class OperatorReportCreate(BaseModel):
    unit_name: str
    description: str = Field(..., min_length=1)
    sensor_name: Optional[str] = None
    reported_issue: Optional[str] = None
    risk_impact: Optional[str] = Field(default="medium", pattern=r"^(low|medium|high)$")

class OperatorReportResponse(BaseSchema):
    id: str
    user_id: str
    unit_id: str
    unit_name: Optional[str] = None  
    timestamp: datetime
    description: str
    sensor_name: Optional[str] = None
    reported_issue: Optional[str] = None
    risk_impact: Optional[str] = None
    status: str