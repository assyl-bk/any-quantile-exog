from pydantic import BaseModel, EmailStr, Field, field_validator
from datetime import datetime
from typing import Optional, Literal
import re

UserRole = Literal[
    "energy_grid_operator",
    "energy_trader",
    "energy_planner",
    "system_administrator",
]

# User Schemas
class UserBase(BaseModel):
    email: EmailStr
    name: str

    @field_validator("email")
    @classmethod
    def normalize_email(cls, value: EmailStr) -> str:
        return str(value).strip().lower()

    @field_validator("name")
    @classmethod
    def validate_name(cls, value: str) -> str:
        normalized_name = value.strip()
        if len(normalized_name) < 2:
            raise ValueError("Name must be at least 2 characters long")
        if len(normalized_name) > 100:
            raise ValueError("Name must be at most 100 characters long")
        if not re.fullmatch(r"[A-Za-zÀ-ÖØ-öø-ÿ' -]+", normalized_name):
            raise ValueError("Name can only contain letters, spaces, apostrophes, and hyphens")
        return normalized_name

class UserCreate(UserBase):
    password: str = Field(..., min_length=8)
    role: UserRole

    @field_validator("password")
    @classmethod
    def validate_password(cls, value: str) -> str:
        if len(value) < 8:
            raise ValueError("Password must be at least 8 characters long")
        if len(value) > 128:
            raise ValueError("Password must be at most 128 characters long")
        if not re.search(r"[A-Z]", value):
            raise ValueError("Password must include at least one uppercase letter")
        if not re.search(r"[a-z]", value):
            raise ValueError("Password must include at least one lowercase letter")
        if not re.search(r"\d", value):
            raise ValueError("Password must include at least one number")
        if not re.search(r"[^A-Za-z0-9]", value):
            raise ValueError("Password must include at least one special character")
        return value

class UserLogin(BaseModel):
    email: EmailStr
    password: str

    @field_validator("email")
    @classmethod
    def normalize_email(cls, value: EmailStr) -> str:
        return str(value).strip().lower()

class UserResponse(UserBase):
    id: int
    role: UserRole
    is_active: bool
    created_at: datetime

    class Config:
        from_attributes = True

# Token Schemas
class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: UserResponse
    api_key: Optional[str] = None  # Full key returned once at login/signup

class TokenData(BaseModel):
    email: Optional[str] = None

# API Key Schemas
class APIKeyCreate(BaseModel):
    name: str
    permissions: str = "read"
    expires_in_days: Optional[int] = None

class APIKeyResponse(BaseModel):
    id: int
    name: str
    prefix: str
    key: Optional[str] = None  # Only returned on creation
    is_active: bool
    permissions: str
    created_at: datetime
    last_used: Optional[datetime]
    expires_at: Optional[datetime]


# Profile Schemas
class ProfileUpdate(BaseModel):
    name: Optional[str] = None
    email: Optional[str] = None

    class Config:
        from_attributes = True


# Notification Preference Schemas
class NotificationPreferenceUpdate(BaseModel):
    email: bool
    peak_demand: bool
    peak_demand_threshold: int = Field(gt=0, description="Threshold in MW")
    model_report: bool
    sys_updates: bool

    class Config:
        from_attributes = True


class NotificationPreferenceResponse(NotificationPreferenceUpdate):
    id: int
    user_id: int
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


# Alert History Schemas
class AlertHistoryResponse(BaseModel):
    id: int
    user_id: int
    alert_type: str
    message: str
    data: Optional[dict] = None
    acknowledged: bool
    created_at: datetime

    class Config:
        from_attributes = True

    class Config:
        from_attributes = True


class RoleCapabilitiesResponse(BaseModel):
    role: UserRole
    capabilities: list[str]


class RolesMatrixResponse(BaseModel):
    roles: dict[UserRole, list[str]]


# Forecast Schemas
class ForecastRequest(BaseModel):
    """Request for quantile forecasting.
    
    Attributes:
        historical_data: Historical univariate time series values (1D array)
        exogenous_data: Optional exogenous features (2D array: time_steps x features)
        quantiles: List of quantile levels (e.g., [0.1, 0.5, 0.9])
        apply_cqr: Apply CQR calibration for improved coverage (default: True)
        horizon: Optional forecast horizon override (default: use model config)
    """
    historical_data: list[float] = Field(
        ...,
        description="Historical univariate time series (e.g., 168 hourly values)"
    )
    exogenous_data: Optional[list[list[float]]] = Field(
        None,
        description="Exogenous features matrix (time_steps x num_features)"
    )
    quantiles: list[float] = Field(
        default=[0.1, 0.5, 0.9],
        description="Quantile levels (0.0 to 1.0)"
    )
    apply_cqr: bool = Field(
        default=True,
        description="Apply Conformalized Quantile Regression calibration"
    )
    horizon: Optional[int] = Field(
        default=None,
        description="Forecast horizon in time steps (optional, overrides config)"
    )
    
    @field_validator("historical_data")
    @classmethod
    def validate_historical_data(cls, value: list[float]) -> list[float]:
        if len(value) < 12:
            raise ValueError("Historical data must have at least 12 time steps")
        if len(value) > 10000:
            raise ValueError("Historical data cannot exceed 10000 time steps")
        if not all(isinstance(x, (int, float)) for x in value):
            raise ValueError("All historical data values must be numeric")
        return value
    
    @field_validator("quantiles")
    @classmethod
    def validate_quantiles(cls, value: list[float]) -> list[float]:
        if len(value) < 1:
            raise ValueError("Must specify at least one quantile")
        if len(value) > 50:
            raise ValueError("Cannot specify more than 50 quantiles")
        for q in value:
            if not (0 < q < 1):
                raise ValueError("Quantile values must be between 0 and 1 (exclusive)")
        if len(set(value)) != len(value):
            raise ValueError("Quantile values must be unique")
        return sorted(value)


class QuantileForecast(BaseModel):
    """Single quantile forecast for a time step."""
    quantile: float = Field(description="Quantile level")
    values: list[float] = Field(description="Forecast values for horizon")


class ForecastResponse(BaseModel):
    """Response containing quantile forecasts.
    
    Attributes:
        forecasts: List of quantile forecasts
        quantiles: List of quantile levels used
        forecast_horizon: Number of time steps forecasted
        cqr_applied: Whether CQR calibration was applied
        model_version: Version/variant of the model used
    """
    forecasts: list[QuantileForecast] = Field(
        description="Quantile forecasts for each level"
    )
    quantiles: list[float] = Field(description="Quantile levels")
    forecast_horizon: int = Field(description="Number of forecast steps")
    cqr_applied: bool = Field(description="CQR calibration applied")
    model_version: str = Field(default="AQ_NBEATS++_S2", description="Model version")
    generated_at: datetime = Field(default_factory=datetime.utcnow)
    
    class Config:
        from_attributes = True


class ForecastErrorResponse(BaseModel):
    """Error response for forecast requests."""
    error: str = Field(description="Error message")
    detail: Optional[str] = Field(None, description="Additional details")
    request_id: Optional[str] = Field(None, description="Request tracking ID")