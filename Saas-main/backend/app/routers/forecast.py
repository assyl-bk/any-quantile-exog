"""
Forecast router for quantile predictions using AQ_NBEATS++_S2 with CQR.
"""

import logging
from typing import Optional
import numpy as np
from fastapi import APIRouter, Depends, HTTPException, Header, status
from app.schemas import ForecastRequest, ForecastResponse, QuantileForecast, ForecastErrorResponse
from app.model_loader import get_model_loader
from app.security import verify_api_key

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post(
    "/forecast",
    response_model=ForecastResponse,
    responses={
        400: {"model": ForecastErrorResponse},
        401: {"model": ForecastErrorResponse},
        503: {"model": ForecastErrorResponse},
    },
)
async def create_forecast(
    request: ForecastRequest,
    x_api_key: str = Header(None),
    api_key_id: Optional[int] = Depends(verify_api_key),
):
    """
    Generate quantile forecasts using the trained AQ_NBEATS++_S2 model with CQR.
    
    **Authentication:**
    - Requires valid API key in `X-API-Key` header
    
    **Input:**
    - `historical_data`: Historical univariate time series (e.g., 168 hourly values)
    - `exogenous_data` (optional): Exogenous features for each time step
    - `quantiles`: Quantile levels to forecast (e.g., [0.1, 0.5, 0.9])
    - `apply_cqr`: Apply Conformalized Quantile Regression calibration
    
    **Output:**
    - `forecasts`: List of quantile forecasts (values for each quantile level)
    - `forecast_horizon`: Number of time steps forecasted (typically 24 hours)
    - `cqr_applied`: Whether CQR calibration was applied
    
    **Example Request:**
    ```json
    {
        "historical_data": [100.5, 102.3, ..., 105.2],
        "quantiles": [0.1, 0.5, 0.9],
        "apply_cqr": true
    }
    ```
    
    **Example Response:**
    ```json
    {
        "forecasts": [
            {
                "quantile": 0.1,
                "values": [98.5, 97.2, 96.8, ...]
            },
            {
                "quantile": 0.5,
                "values": [105.2, 104.8, 104.5, ...]
            },
            {
                "quantile": 0.9,
                "values": [112.5, 113.2, 113.8, ...]
            }
        ],
        "quantiles": [0.1, 0.5, 0.9],
        "forecast_horizon": 24,
        "cqr_applied": true,
        "model_version": "AQ_NBEATS++_S2"
    }
    ```
    """
    try:
        # Get model loader
        loader = get_model_loader()
        model = loader.get_model()
        
        if model is None:
            logger.error("Model not loaded")
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Model not initialized. Please try again later.",
            )
        
        # Convert to numpy arrays
        historical_data = np.array(request.historical_data, dtype=np.float32)
        exogenous_data = None
        if request.exogenous_data:
            exogenous_data = np.array(request.exogenous_data, dtype=np.float32)
        
        # Make prediction
        logger.info(
            f"Generating forecast: history_len={len(request.historical_data)}, "
            f"quantiles={request.quantiles}, apply_cqr={request.apply_cqr}"
        )
        
        result = loader.predict(
            historical_data=historical_data,
            exogenous_data=exogenous_data,
            quantiles=request.quantiles,
            apply_cqr=request.apply_cqr,
            horizon=request.horizon,
        )
        
        # Parse predictions
        forecasts_array = result['forecasts']  # (num_quantiles, forecast_horizon)
        quantiles = result['quantiles']
        cqr_applied = result['cqr_applied']
        
        # Build response
        forecast_items = []
        for i, q in enumerate(quantiles):
            forecast_items.append(
                QuantileForecast(
                    quantile=q,
                    values=forecasts_array[i].tolist()
                )
            )
        
        response = ForecastResponse(
            forecasts=forecast_items,
            quantiles=quantiles,
            forecast_horizon=forecasts_array.shape[1],
            cqr_applied=cqr_applied,
            model_version="AQ_NBEATS++_S2",
        )
        
        logger.info(f"Forecast generated successfully for API key {api_key_id}")
        return response
    
    except ValueError as e:
        logger.warning(f"Invalid input: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid forecast request: {str(e)}"
        )
    
    except RuntimeError as e:
        logger.error(f"Model error: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Forecasting service error: {str(e)}"
        )
    
    except Exception as e:
        logger.error(f"Unexpected error during forecast: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred during forecasting"
        )


@router.get(
    "/forecast/info",
    tags=["Forecast Info"],
)
async def forecast_info(
    x_api_key: str = Header(None),
    api_key_id: Optional[int] = Depends(verify_api_key),
):
    """
    Get information about the forecasting model.
    
    Returns model configuration, input/output specifications, and supported quantiles.
    """
    try:
        loader = get_model_loader()
        cfg = loader.get_config()
        calibrator = loader.get_calibrator()
        
        if cfg is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Model not initialized"
            )
        
        # Extract key info
        history_len = cfg.model.input_horizon_len if hasattr(cfg.model, 'input_horizon_len') else 168
        forecast_horizon = cfg.model.nn.backbone.size_out if hasattr(cfg.model.nn.backbone, 'size_out') else 24
        
        # ✅ Use .q_per_level dict, not subscript on the object
        supported_quantiles = sorted(list(calibrator.q_per_level.keys())) if calibrator else []
        
        return {
            "model_version": "AQ_NBEATS++_S2",
            "description": "Adaptive Quantile Neural Basis Expansion Transformation with CQR",
            "input_horizon": history_len,
            "forecast_horizon": forecast_horizon,
            "supported_quantiles": supported_quantiles,
            "cqr_enabled": calibrator is not None,
            "input_requirements": {
                "historical_data": {
                    "type": "array",
                    "length": history_len,
                    "description": f"Historical univariate time series ({history_len} values)"
                },
                "exogenous_data": {
                    "type": "optional array",
                    "dimensions": [history_len, -1],
                    "description": "Exogenous features (time_steps x num_features)"
                },
                "quantiles": {
                    "type": "array",
                    "items": "float",
                    "range": [0.001, 0.999],
                    "default": [0.1, 0.5, 0.9]
                }
            },
            "output": {
                "type": "quantile forecasts",
                "shape": [len(supported_quantiles) if supported_quantiles else 3, forecast_horizon],
                "description": "Probabilistic forecasts for each quantile level"
            }
        }
    
    except Exception as e:
        logger.error(f"Error getting forecast info: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Unable to retrieve model information"
        )


@router.post(
    "/forecast/batch",
    tags=["Batch Forecasting"],
)
async def batch_forecast(
    requests: list[ForecastRequest],
    x_api_key: str = Header(None),
    api_key_id: Optional[int] = Depends(verify_api_key),
):
    """
    Generate multiple forecasts in a single request (batch processing).
    
    **Input:** Array of forecast requests
    
    **Output:** Array of forecast responses
    
    **Note:** Batch requests are processed sequentially. For large batches,
    consider splitting into smaller requests.
    """
    if len(requests) > 100:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Batch size limited to 100 requests"
        )
    
    results = []
    for i, req in enumerate(requests):
        try:
            result = await create_forecast(req, x_api_key, api_key_id)
            results.append({"index": i, "status": "success", "data": result})
        except HTTPException as e:
            results.append({
                "index": i,
                "status": "error",
                "error": e.detail
            })
    
    return {"results": results, "total": len(requests), "successful": sum(1 for r in results if r["status"] == "success")}
