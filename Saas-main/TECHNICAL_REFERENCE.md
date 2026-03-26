# Technical Reference & API Documentation

## Module Reference

### 1. `app/model_loader.py` - Model Management

**Class: `ModelLoader` (Singleton)**

| Method              | Purpose            | Parameters                                              | Returns                      |
| ------------------- | ------------------ | ------------------------------------------------------- | ---------------------------- |
| `load_config()`     | Load YAML config   | `config_path: str`                                      | `OmegaConf`                  |
| `load_model()`      | Load checkpoint    | `checkpoint_path, config_path`                          | `AQNBEATSPlusPlus`           |
| `load_calibrator()` | Load CQR offsets   | `calibrator_path: str`                                  | `Dict[float, float]`         |
| `get_model()`       | Get loaded model   | -                                                       | `AQNBEATSPlusPlus`           |
| `get_calibrator()`  | Get calibrator     | -                                                       | `Dict[float, float] \| None` |
| `get_config()`      | Get configuration  | -                                                       | `OmegaConf \| None`          |
| `predict()`         | Generate forecasts | `historical_data, exogenous_data, quantiles, apply_cqr` | `Dict`                       |
| `_apply_cqr()`      | Apply calibration  | `forecasts: ndarray, quantiles: List[float]`            | `ndarray`                    |
| `reset()`           | Unload model       | -                                                       | `None`                       |

**Function: `get_model_loader()`**

- Returns singleton instance of ModelLoader
- Thread-safe
- Usage: `loader = get_model_loader()`

**Function: `initialize_model()`**

- Initialize model loader with paths
- Called once at application startup
- Parameters:
  - `base_path: str` - Project root directory
  - `checkpoint: str` - Relative path to checkpoint
  - `config: str` - Relative path to config
  - `calibrator: str` - Relative path to calibrator

### 2. `app/schemas.py` - Data Validation

**Classes:**

| Schema                  | Purpose           | Fields                                                     |
| ----------------------- | ----------------- | ---------------------------------------------------------- |
| `ForecastRequest`       | Input validation  | `historical_data, exogenous_data, quantiles, apply_cqr`    |
| `ForecastResponse`      | Output formatting | `forecasts, quantiles, forecast_horizon, cqr_applied, ...` |
| `QuantileForecast`      | Single quantile   | `quantile: float, values: List[float]`                     |
| `ForecastErrorResponse` | Error response    | `error, detail, request_id`                                |

**Validators:**

- `validate_historical_data()` - Check length (12-1000)
- `validate_quantiles()` - Check range (0,1), uniqueness

### 3. `app/routers/forecast.py` - API Endpoints

| Endpoint          | Method | Auth    | Purpose                     |
| ----------------- | ------ | ------- | --------------------------- |
| `/forecast`       | POST   | API Key | Generate quantile forecasts |
| `/forecast/info`  | GET    | API Key | Get model information       |
| `/forecast/batch` | POST   | API Key | Batch forecasting (max 100) |

**Endpoint Details:**

#### POST `/api/forecast`

**Headers:**

```
X-API-Key: efp_YOUR_API_KEY
Content-Type: application/json
```

**Request Body:**

```json
{
  "historical_data": [100.5, 102.3, ...],  // Required: 12-1000 values
  "exogenous_data": [[...], [...], ...],   // Optional: 2D array
  "quantiles": [0.1, 0.5, 0.9],            // Optional: default [0.1, 0.5, 0.9]
  "apply_cqr": true                        // Optional: default true
}
```

**Success Response (200):**

```json
{
  "forecasts": [
    {"quantile": 0.1, "values": [...]},
    {"quantile": 0.5, "values": [...]},
    {"quantile": 0.9, "values": [...]}
  ],
  "quantiles": [0.1, 0.5, 0.9],
  "forecast_horizon": 24,
  "cqr_applied": true,
  "model_version": "AQ_NBEATS++_S2",
  "generated_at": "2024-03-23T..."
}
```

**Error Responses:**

- `400 Bad Request` - Invalid input (wrong length, quantile out of range)
- `401 Unauthorized` - Missing or invalid API key
- `503 Service Unavailable` - Model not initialized

#### GET `/api/forecast/info`

**Headers:**

```
X-API-Key: efp_YOUR_API_KEY
```

**Response:**

```json
{
  "model_version": "AQ_NBEATS++_S2",
  "description": "Adaptive Quantile Neural Basis Expansion Transformation with CQR",
  "input_horizon": 168,
  "forecast_horizon": 24,
  "supported_quantiles": [0.01, 0.05, 0.1, ...],
  "cqr_enabled": true,
  "input_requirements": {...},
  "output": {...}
}
```

#### POST `/api/forecast/batch`

**Request Body:** Array of ForecastRequest objects (max 100)
**Response:**

```json
{
  "results": [
    {"index": 0, "status": "success", "data": {...}},
    {"index": 1, "status": "error", "error": "..."}
  ],
  "total": 2,
  "successful": 1
}
```

### 4. `app/security.py` - Authentication

**Function: `verify_api_key()`**

```python
async def verify_api_key(
    x_api_key: Optional[str] = Header(None),
    db: Session = Depends(get_db)
) -> Optional[int]
```

**Behavior:**

1. Checks if `X-API-Key` header is present
2. Queries database for matching key
3. Validates key is active and not expired
4. Updates `last_used` timestamp
5. Returns `api_key_id` on success
6. Raises `HTTPException` on failure

**Error Codes:**

- `401 Unauthorized` - Missing or invalid key
- `403 Forbidden` - Key inactive or expired
- `500 Internal Server Error` - Database error

### 5. `app/config.py` - Configuration

**Settings Class Variables:**

| Variable                | Type | Default                                 | Purpose                 |
| ----------------------- | ---- | --------------------------------------- | ----------------------- |
| `MODEL_BASE_PATH`       | str  | `"../"`                                 | Base path to models     |
| `MODEL_CHECKPOINT`      | str  | `lightning_logs/.../model-epoch=4.ckpt` | Checkpoint file         |
| `MODEL_CONFIG`          | str  | `config/AQNBEATS++_S2.yaml`             | Model config            |
| `MODEL_CALIBRATOR`      | str  | `results/CQR/calibrator_stage2_v2.pkl`  | CQR calibrator          |
| `LOAD_MODEL_ON_STARTUP` | bool | `True`                                  | Load model on app start |

**Usage:**

```python
from app.config import settings

print(settings.MODEL_BASE_PATH)
print(settings.LOAD_MODEL_ON_STARTUP)
```

### 6. `main.py` - Application Entry Point

**Lifespan Context Manager:**

```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    if settings.LOAD_MODEL_ON_STARTUP:
        initialize_model(...)

    yield

    # Shutdown
    logger.info("Shutting down...")
```

**App Configuration:**

- CORS enabled for localhost:3000, 5173, 5174
- Routers included: auth, api_keys, forecast
- Endpoints: root, health, docs, redoc

## Request Handling Flow

```
1. Client sends HTTP request with X-API-Key header
   ↓
2. FastAPI routes to endpoint handler
   ↓
3. Pydantic validates request body (ForecastRequest)
   ↓
4. Dependency resolve: verify_api_key()
   - Queries database
   - Checks key status/expiration
   - Returns user_id or 401 error
   ↓
5. Handler executes:
   a) Get model from singleton loader
   b) Convert inputs to numpy arrays
   c) Call loader.predict()
   d) Format response using ForecastResponse
   ↓
6. FastAPI returns JSON response with metadata
```

## Database Schema

### Users Table

```sql
CREATE TABLE users (
    id INTEGER PRIMARY KEY,
    email VARCHAR UNIQUE NOT NULL,
    name VARCHAR NOT NULL,
    hashed_password VARCHAR NOT NULL,
    role VARCHAR DEFAULT 'user',
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT now(),
    updated_at TIMESTAMP DEFAULT now()
);
```

### API Keys Table

```sql
CREATE TABLE api_keys (
    id INTEGER PRIMARY KEY,
    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    name VARCHAR NOT NULL,
    key VARCHAR UNIQUE NOT NULL,
    prefix VARCHAR,
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT now(),
    last_used TIMESTAMP,
    expires_at TIMESTAMP,
    permissions TEXT DEFAULT 'read'
);
```

## Configuration Files

### `.env` Format

```bash
# Database
DATABASE_URL=postgresql://user:pass@host:5432/db
REDIS_URL=redis://localhost:6379

# JWT
SECRET_KEY=your-secret-key
ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=30

# Environment
ENVIRONMENT=development

# Model
MODEL_BASE_PATH=../
MODEL_CHECKPOINT=lightning_logs/nbeatsaq-stage2-seed0/checkpoints/model-epoch=4.ckpt
MODEL_CONFIG=config/AQNBEATS++_S2.yaml
MODEL_CALIBRATOR=results/CQR/calibrator_stage2_v2.pkl
LOAD_MODEL_ON_STARTUP=true
```

### `config/AQNBEATS++_S2.yaml` Format

```yaml
model:
  _target_: model.AQNBEATSPlusPlus
  nn:
    backbone:
      _target_: modules.NBEATSAQCAT
      num_blocks: 30
      num_layers: 3
      layer_width: 1024
      size_in: 168 # Input: 168 hourly values
      size_out: 24 # Output: 24-hour forecast
      num_continuous: 4 # Exogenous continuous features
      num_calendar: 4 # Calendar features
  input_horizon_len: 168
  loss:
    _target_: losses.MQNLoss
```

## Dependencies

### Core FastAPI Stack

- `fastapi==0.115.0` - Web framework
- `uvicorn==0.32.0` - ASGI server
- `pydantic==2.10.0` - Data validation

### Database

- `sqlalchemy==2.0.36` - ORM
- `psycopg2-binary==2.9.10` - PostgreSQL driver
- `alembic==1.14.0` - Migrations

### Security

- `python-jose==3.3.0` - JWT tokens
- `passlib==1.7.4` - Password hashing
- `bcrypt` - Hashing algorithm

### Deep Learning

- `torch>=2.0.0` - PyTorch
- `pytorch-lightning>=2.0.0` - Training framework
- `numpy>=1.24.0` - Numerical computing
- `torchmetrics>=0.11.0` - Metrics
- `omegaconf>=2.3.0` - Configuration
- `PyYAML>=6.0` - YAML parsing

### Utilities

- `redis==5.2.0` - Caching (optional)
- `python-dotenv==1.0.1` - Environment loading
- `email-validator==2.1.0` - Email validation

## Common Tasks

### Adding a New Quantile Level

No code change needed! Just include in request:

```json
{
  "quantiles": [0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]
}
```

### Disabling CQR Calibration

```json
{
  "apply_cqr": false
}
```

Raw model predictions without calibration offset correction.

### Using GPU (CUDA)

```bash
# Auto-detected if NVIDIA GPU and CUDA available
# To force GPU: set CUDA_VISIBLE_DEVICES
export CUDA_VISIBLE_DEVICES=0

# To force CPU:
export CUDA_VISIBLE_DEVICES=""
```

### Checking Model Status

```python
from app.model_loader import get_model_loader

loader = get_model_loader()
print(f"Model loaded: {loader.get_model() is not None}")
print(f"CQR available: {loader.get_calibrator() is not None}")
print(f"Config: {loader.get_config()}")
```

### Logging Additional Debug Info

```python
import logging

# In any module
logger = logging.getLogger(__name__)
logger.debug("Debug message")
logger.info("Info message")
logger.warning("Warning message")
logger.error("Error message", exc_info=True)
```

### Monitoring API Key Usage

```sql
-- Find most used API keys
SELECT
    k.prefix,
    k.name,
    k.last_used,
    COUNT(*) as request_count
FROM api_keys k
GROUP BY k.id
ORDER BY k.last_used DESC
LIMIT 10;
```

## Performance Tuning

### For CPU-only Deployment

```python
# Disable GPU
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''

# Reduce model complexity (if needed)
# Set torch.set_num_threads() appropriately
```

### For GPU Deployment

```python
# Enable GPU memory growth
import torch
torch.cuda.empty_cache()

# Use half precision if available
# Note: Model must support mixed precision
```

### For Batch Processing

```python
# Process multiple requests efficiently
requests = [req1, req2, req3, ...]  # Up to 100

response = requests.post(
    "/api/forecast/batch",
    headers={"X-API-Key": api_key},
    json=requests
)
```

## Testing

### Unit Test Example

```python
import pytest
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

def test_forecast_valid_request():
    response = client.post(
        "/api/forecast",
        headers={"X-API-Key": "test_key"},
        json={
            "historical_data": [100.5] * 168,
            "quantiles": [0.1, 0.5, 0.9]
        }
    )
    assert response.status_code == 200
    data = response.json()
    assert len(data["forecasts"]) == 3
    assert data["forecast_horizon"] == 24
```

### Integration Test Example

```python
def test_full_workflow():
    # 1. Register user
    # 2. Login
    # 3. Create API key
    # 4. Make forecast request
    # 5. Verify response structure
    pass
```

## Troubleshooting Checklist

- [ ] Model files path correct in `.env`
- [ ] PostgreSQL running and accessible
- [ ] Python dependencies installed: `pip list`
- [ ] API key created and active
- [ ] Historical data has correct length
- [ ] Quantiles in range (0, 1)
- [ ] Server logs show "✅ Model loaded successfully"
- [ ] API responds to health check: `/health`
- [ ] Interactive docs accessible: `/docs`

## Next Steps for Extension

1. **Add REST endpoint for model retraining**
2. **Implement forecast caching with Redis**
3. **Add WebSocket for streaming predictions**
4. **Create dashboard for monitoring**
5. **Add data persistence for forecast results**
6. **Implement A/B testing framework**
7. **Add explainability features (SHAP)**
