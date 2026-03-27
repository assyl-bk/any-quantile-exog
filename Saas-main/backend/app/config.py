from pydantic_settings import BaseSettings
from typing import Optional
import os

class Settings(BaseSettings):
    # Database (default to SQLite for easy development)
    DATABASE_URL: str = "sqlite:///./energy_forecast.db"
    
    # Redis (optional - for caching and message queue)
    REDIS_URL: Optional[str] = "redis://localhost:6379"
    
    # JWT Settings
    SECRET_KEY: str = "your-secret-key-change-this-in-production"
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    
    # Environment
    ENVIRONMENT: str = "development"
    
    # Model Configuration - AQ_NBEATS++_S2 with CQR
    MODEL_BASE_PATH: str = "C:\\Users\\Pc\\OneDrive\\Documents\\PCD\\any-quantile-exog"
    MODEL_CHECKPOINT: str = "C:\\Users\\Pc\\OneDrive\\Documents\\PCD\\any-quantile-exog\\lightning_logs\\nbeatsaq-stage2-seed0\\checkpoints\\model-epoch=4.ckpt"
    MODEL_CONFIG: str = "C:\\Users\\Pc\\OneDrive\\Documents\\PCD\\any-quantile-exog\\config\\AQNBEATS++_S2.yaml"
    MODEL_CALIBRATOR: str = "C:\\Users\\Pc\\OneDrive\\Documents\\PCD\\any-quantile-exog\\results\\CQR\\calibrator_stage2_v2.pkl"
    LOAD_MODEL_ON_STARTUP: bool = True
    
    # Email Configuration
    EMAIL_PROVIDER: str = "console"  # console, smtp, sendgrid, ses
    SMTP_HOST: Optional[str] = None
    SMTP_PORT: int = 587
    SMTP_USER: Optional[str] = None
    SMTP_PASSWORD: Optional[str] = None
    SENDGRID_API_KEY: Optional[str] = None
    AWS_REGION: str = "us-east-1"
    
    # Notifications
    ENABLE_DEMAND_MONITORING: bool = True
    DEMAND_CHECK_INTERVAL: int = 60  # seconds
    DEMAND_ALERT_COOLDOWN: int = 900  # 15 minutes
    
    class Config:
        env_file = ".env"
        case_sensitive = True

settings = Settings()
