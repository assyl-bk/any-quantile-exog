from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import asyncio
import logging
from pathlib import Path
from app.routers import auth, api_keys, forecast, users
from app.database import engine, Base
from app.config import settings
from app.model_loader import get_model_loader
from app.services import get_email_service, get_demand_alert_service

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Note: Database tables will be created by lifespan context manager on startup

# Global task for demand monitoring
monitoring_task = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Handle startup and shutdown events.
    Loads the ML model on startup, creates database tables, and starts monitoring services.
    """
    global monitoring_task
    
    # Startup
    logger.info("🚀 Starting Energy Forecast API...")
    
    # Create database tables
    try:
        Base.metadata.create_all(bind=engine)
        logger.info("✅ Database tables created")
    except Exception as e:
        logger.warning(f"⚠️  Database table creation failed: {e}")
    
    if settings.LOAD_MODEL_ON_STARTUP:
        try:
            logger.info(f"Loading model from checkpoint: {settings.MODEL_CHECKPOINT}")
            
            loader = get_model_loader()
            loader.load_model(settings.MODEL_CHECKPOINT, settings.MODEL_CONFIG)
            loader.load_calibrator(settings.MODEL_CALIBRATOR)
            logger.info("✅ Model loaded successfully")
        except Exception as e:
            logger.error(f"❌ Failed to load model: {e}")
            logger.warning("⚠️  Model loading disabled - forecast endpoints will return errors")
    else:
        logger.info("Model loading disabled (LOAD_MODEL_ON_STARTUP=false)")
    
    # Start demand monitoring service
    try:
        logger.info("Starting demand alert monitoring service...")
        email_service = get_email_service(provider="console")  # Use "smtp", "sendgrid", or "ses" in production
        demand_alert_service = get_demand_alert_service(email_service)
        
        # Run monitoring in background
        monitoring_task = asyncio.create_task(demand_alert_service.monitor_demand())
        logger.info("✅ Demand alert service started")
    except Exception as e:
        logger.error(f"❌ Failed to start demand alert service: {e}")
        logger.warning("⚠️  Demand monitoring disabled")
    
    yield
    
    # Shutdown
    logger.info("🛑 Shutting down Energy Forecast API...")
    
    # Stop demand monitoring
    if monitoring_task:
        try:
            demand_alert_service = get_demand_alert_service()
            demand_alert_service.stop()
            monitoring_task.cancel()
            await asyncio.sleep(0.5)  # Give task time to cancel
            logger.info("✅ Demand alert service stopped")
        except Exception as e:
            logger.error(f"Error stopping demand alert service: {e}")


app = FastAPI(
    title="Energy Forecast Pro API",
    description="Probabilistic energy demand forecasting API with quantile predictions",
    version="1.0.0",
    lifespan=lifespan,
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:5173",
        "http://localhost:5174",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:5173",
        "http://127.0.0.1:5174",
    ],
    allow_origin_regex=r"https?://(localhost|127\.0\.0\.1)(:\d+)?$",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(auth.router, prefix="/api/auth", tags=["Authentication"])
app.include_router(api_keys.router, prefix="/api/keys", tags=["API Keys"])
app.include_router(users.router, prefix="/api/user", tags=["User Management"])
app.include_router(forecast.router, prefix="/api/forecast", tags=["Forecasting"])

@app.get("/")
async def root():
    return {
        "message": "Energy Forecast API",
        "version": "1.0.0",
        "status": "online",
        "endpoints": {
            "forecast": "/api/forecast",
            "forecast_info": "/api/forecast/info",
            "auth": "/api/auth",
            "api_keys": "/api/keys",
            "docs": "/docs"
        }
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "service": "Energy Forecast API"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
