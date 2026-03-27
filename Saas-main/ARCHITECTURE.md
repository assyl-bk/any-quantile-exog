# Architecture & Data Flow

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                      CLIENT APPLICATION                             │
│                  (Frontend, Mobile, or Scripts)                      │
└────────┬────────────────────────────────────────────────────────────┘
         │
         │ HTTPS / REST API
         │ Header: X-API-Key
         │
┌────────▼────────────────────────────────────────────────────────────┐
│                    FastAPI Application                              │
│                  (main.py - Port 8000)                              │
└────────┬────────────────────────────────────────────────────────────┘
         │
         ├─────────────────────────┬───────────────────┬─────────────┐
         │                         │                   │             │
    ┌────▼────┐         ┌──────────▼──┐      ┌────────▼──┐  ┌──────▼────┐
    │ Auth    │         │  API Key    │      │ Database  │  │  Forecast │
    │ Router  │         │  Router     │      │ (SQLAlch.)│  │  Router   │
    │         │         │             │      │           │  │           │
    │ signup, │         │ create,     │      │ Users,    │  │ /forecast │
    │ login   │         │ list,       │      │ API Keys  │  │ /info     │
    └────┬────┘         │ delete,     │      └───────────┘  │ /batch    │
         │              │ activate    │                     └──────┬────┘
         │              └──────────────┘                           │
         │                                                         │
         │                                    ┌────────────────────┘
         │                                    │
         │                              ┌─────▼──────────────────┐
         │                              │  Model Loader Module   │
         │                              │  (model_loader.py)     │
         │                              │                        │
         │                              │ ModelLoader class:     │
         │                              │ - load_model()         │
         │                              │ - load_calibrator()    │
         │                              │ - predict()            │
         │                              │ - _apply_cqr()         │
         │                              └─────┬──────────────────┘
         │                                    │
         │                    ┌───────────────┴──────────────┐
         │                    │                              │
         │              ┌─────▼────────┐          ┌─────────▼────┐
         │              │  Checkpoint  │          │  CQR         │
         │              │  (PyTorch)   │          │  Calibrator  │
         │              │              │          │              │
         │              │ model-epoch  │          │ calibrator_  │
         │              │ =4.ckpt      │          │ stage2_v2.pkl│
         │              └────────────┬─┘          └──────────────┘
         │                           │
         │                    ┌──────▼──────────┐
         │                    │  Config File    │
         │                    │  OmegaConf      │
         │                    │                 │
         │                    │ AQNBEATS++_S2   │
         │                    │ .yaml           │
         │                    └─────────────────┘
         │
    ┌────▼────────────────────────────────────────────────────────────┐
    │                      PostgreSQL Database                         │
    │  (Tables: users, api_keys, and app-specific models)            │
    └─────────────────────────────────────────────────────────────────┘
```

## Request/Response Flow for Forecast Endpoint

```
1. CLIENT REQUEST
   ┌─────────────────────────────────────┐
   │ POST /api/forecast                  │
   │ Headers:                            │
   │   X-API-Key: efp_...                │
   │   Content-Type: application/json    │
   │                                     │
   │ Body:                               │
   │ {                                   │
   │   "historical_data": [...],         │
   │   "quantiles": [0.1, 0.5, 0.9],    │
   │   "apply_cqr": true                 │
   │ }                                   │
   └──────────┬──────────────────────────┘
              │
              ▼
2. SECURITY VALIDATION
   ┌──────────────────────────────────────┐
   │ verify_api_key() dependency          │
   │                                      │
   │ Query: APIKey.key == x_api_key       │
   │ Checks:                              │
   │ - Key exists?                        │
   │ - Is active?                         │
   │ - Not expired?                       │
   │ - Update last_used                   │
   │                                      │
   │ Return: user_id (or 401 error)       │
   └──────────┬───────────────────────────┘
              │
              ▼
3. INPUT VALIDATION
   ┌──────────────────────────────────────┐
   │ ForecastRequest Pydantic model       │
   │                                      │
   │ Validates:                           │
   │ - historical_data length             │
   │ - quantiles in range (0, 1)          │
   │ - quantiles unique                   │
   │ - exogenous_data shape (if provided) │
   │                                      │
   │ Return: parsed request object        │
   │ (or 400 error)                       │
   └──────────┬───────────────────────────┘
              │
              ▼
4. MODEL INFERENCE
   ┌────────────────────────────────────────┐
   │ model_loader.predict()                 │
   │                                        │
   │ a) Convert arrays to tensors           │
   │    y_tensor: (batch, seq_len)          │
   │    x_tensor: (batch, seq_len, features)│
   │    q_tensor: (num_quantiles,)          │
   │                                        │
   │ b) Forward pass through N-BEATS        │
   │    Output: (batch, quantiles, horizon)│
   │                                        │
   │ c) If apply_cqr=true:                  │
   │    For each quantile:                  │
   │      offset = calibrator[q]            │
   │      predictions += offset             │
   │                                        │
   │ Return: dict with forecasts, metadata  │
   └────────────┬──────────────────────────┘
                │
                ▼
5. RESPONSE FORMATTING
   ┌─────────────────────────────────────┐
   │ Build ForecastResponse schema:       │
   │                                     │
   │ {                                   │
   │   "forecasts": [                    │
   │     {                               │
   │       "quantile": 0.1,              │
   │       "values": [...]               │
   │     },                              │
   │     ...                             │
   │   ],                                │
   │   "quantiles": [0.1, 0.5, 0.9],     │
   │   "forecast_horizon": 24,           │
   │   "cqr_applied": true,              │
   │   "model_version": "AQ_NBEATS++_S2",│
   │   "generated_at": "...",            │
   │ }                                   │
   │                                     │
   │ Status: 200 OK                      │
   └──────────┬──────────────────────────┘
              │
              ▼
6. CLIENT RESPONSE
   ┌──────────────────────────────────────┐
   │ JSON with forecast results            │
   │ (or error with status 400/401/503)    │
   └──────────────────────────────────────┘
```

## Data Models & Database Schema

### Users Table

```
users (id INTEGER PRIMARY KEY)
├── id: INTEGER (PK)
├── email: VARCHAR (UNIQUE)
├── name: VARCHAR
├── hashed_password: VARCHAR
├── role: VARCHAR (enum: energy_grid_operator, energy_trader, ...)
├── is_active: BOOLEAN
├── created_at: TIMESTAMP
└── updated_at: TIMESTAMP
```

### API Keys Table

```
api_keys (id INTEGER PRIMARY KEY)
├── id: INTEGER (PK)
├── user_id: INTEGER (FK → users.id)
├── name: VARCHAR
├── key: VARCHAR (UNIQUE) - "efp_..."
├── prefix: VARCHAR - first 12 chars for display
├── is_active: BOOLEAN
├── created_at: TIMESTAMP
├── last_used: TIMESTAMP (nullable)
├── expires_at: TIMESTAMP (nullable)
└── permissions: TEXT (enum: read, write, admin)
```

## Configuration Loading Priority

```
1. DEFAULT VALUES (in Settings class)
   ↓
2. ENVIRONMENT VARIABLES (.env file)
   ↓
3. ENVIRONMENT VARIABLES (OS level)

   Final values used by application
```

### Model Configuration Loading

```
.env file:
  MODEL_BASE_PATH = "../"
     ↓
Settings class:
  model_base_path = Path("../").resolve()
     ↓
Application startup:
  initialize_model(base_path, checkpoint, config, calibrator)
     ↓
ModelLoader:
  loads config YAML → OmegaConf object
  loads checkpoint → PyTorch model
  loads calibrator → pickle (CQR offsets)
```

## Tensor Flow Through Model

### Input Shapes

```
historical_data: (batch_size=1, seq_len=168)
  ↓
exogenous_data: (batch_size=1, seq_len=168, num_features=4)
  ↓
quantiles: (num_quantiles,) - e.g., (3,) for [0.1, 0.5, 0.9]
```

### Through N-BEATS Architecture

```
Input: y=(1, 168)
  ↓
Embedding layer: quantile embeddings (3, 64)
  ↓
Block 1: x = stack(residual, seasonal) (1, 168)
  ↓
Block 2-30: similar transformations
  ↓
Output: forecast (1, 3, 24)
        dimensions: (batch, quantiles, horizon)
```

### CQR Calibration

```
Raw forecasts: (3, 24)
  ├── For quantile 0.1: values += offset_0.1
  ├── For quantile 0.5: values += offset_0.5
  └── For quantile 0.9: values += offset_0.9
       ↓
Calibrated forecasts: (3, 24)
```

## File Organization

```
Saas-main/
├── CHANGES_SUMMARY.md          (What changed)
├── QUICKSTART_WINDOWS.md       (Windows setup guide)
├── INTEGRATION_GUIDE.md        (Complete documentation)
│
├── backend/
│   ├── main.py                 (FastAPI app - UPDATED)
│   ├── requirements.txt         (Dependencies - UPDATED)
│   ├── .env.example            (Config template - UPDATED)
│   │
│   ├── app/
│   │   ├── config.py           (Settings - UPDATED)
│   │   ├── schemas.py          (Pydantic models - UPDATED)
│   │   ├── security.py         (Auth helpers - UPDATED)
│   │   ├── model_loader.py     (NEW - model management)
│   │   │
│   │   ├── models.py           (SQLAlchemy ORM models)
│   │   ├── database.py         (DB connection)
│   │   │
│   │   └── routers/
│   │       ├── auth.py         (Sign up, login)
│   │       ├── api_keys.py     (Key management)
│   │       └── forecast.py     (NEW - quantile forecasts)
│   │
│   └── init_db.py              (Initialize database)
│
└── (parent folder: training scripts, data, configs, models)
    ├── model/                   (Training code)
    ├── config/AQNBEATS++_S2.yaml
    ├── lightning_logs/nbeatsaq-stage2-seed0/
    │   └── checkpoints/model-epoch=4.ckpt
    └── results/CQR/calibrator_stage2_v2.pkl
```

## Deployment Architecture Options

### Option 1: Single Container (Development)

```
┌─── Docker Container ───────────┐
│  FastAPI App + Model           │
│  (All in one)                  │
└────────────┬────────────────────┘
             │
      ┌──────▼──────┐      ┌──────────┐
      │ PostgreSQL  │      │  Redis   │
      │ (Container) │      │ (optional)
      └─────────────┘      └──────────┘
```

### Option 2: Separate Model Service (Production)

```
┌─ FastAPI Container ────┐    ┌─ Model Service ──┐
│  API + Auth + DB       │    │  PyTorch Model   │
│  (lightweight)         │◄──►│  (GPU optimized) │
└────────────────────────┘    └──────────────────┘

    Cache Layer (Redis)
         │
         ├─ Forecast cache
         └─ Model metadata
```

### Option 3: Serverless / Cloud (AWS Lambda)

```
┌─────────────────┐
│  API Gateway    │
│  (HTTP trigger) │
└────────┬────────┘
         │
    ┌────▼────────────────┐
    │  Lambda Function    │
    │  - Validate input   │
    │  - Call model       │
    │  - Return result    │
    └─────────────────────┘
         │
    ┌────▼──────────────┐
    │ S3 (model weights)│
    │ DynamoDB (cache)  │
    │ RDS (PostgreSQL)  │
    └───────────────────┘
```

## Performance Characteristics

### Inference Latency

```
CPU (AMD Ryzen):
  Load time: ~2-5 seconds (first call)
  Inference: 50-100ms per forecast

GPU (NVIDIA T4):
  Load time: ~1-2 seconds
  Inference: 10-20ms per forecast
```

### Memory Usage

```
Model weights: ~200MB
PyTorch overhead: ~1GB
Batch processing (1): ~2GB total
Batch processing (10): ~2.5GB total
Batch processing (100): ~4GB total
```

### Throughput

```
Sequential requests:
  CPU: ~50-100 req/sec
  GPU: ~100-200 req/sec

Batch mode (100 at once):
  CPU: Much slower (queues)
  GPU: ~500-1000 req/sec (with batching)
```

## Error Handling Flow

```
Request arrives
  ↓
[Validation Error?] ──→ 400 Bad Request
  │                   └─ Details in response
  ↓ No
[Auth Error?] ─────────→ 401 Unauthorized / 403 Forbidden
  │                   └─ Missing/invalid API key
  ↓ No
[Model not loaded?] ───→ 503 Service Unavailable
  │                   └─ Model initialization failed
  ↓ No
[Inference Error?] ────→ 500 Internal Server Error
  │                   └─ RuntimeError in prediction
  ↓ No
[Success] ──────────────→ 200 OK
                       └─ Forecast response
```

## Security Layers

```
1. API Key Validation (X-API-Key header)
   ├─ Key must exist in database
   ├─ Key must be active
   ├─ Key must not be expired
   └─ Update last_used timestamp

2. Input Validation (Pydantic)
   ├─ Array lengths checked
   ├─ Quantile ranges validated
   ├─ Data types validated
   └─ Reject malformed inputs

3. Model Safety
   ├─ Inference no_grad() context (no gradients)
   ├─ Tensor validation
   ├─ Error catching and logging
   └─ Fail-safe error responses

4. Database
   ├─ SQL injection prevention (SQLAlchemy ORM)
   ├─ Password hashing (bcrypt)
   ├─ JWT token signing
   └─ Connection encryption (configurable)
```

## Monitoring & Logging

```
Application Logs:
  ├─ INFO: Model loading status
  ├─ DEBUG: Inference details
  ├─ WARNING: Degraded performance
  └─ ERROR: Failures with stack traces

Metrics to Track:
  ├─ Inference latency (P50, P95, P99)
  ├─ API key usage by user
  ├─ Forecast cache hit rate
  ├─ Model loading time
  └─ Error rates by endpoint

Debugging:
  ├─ API docs: /docs
  ├─ ReDoc: /redoc
  ├─ Health: /health
  ├─ Model info: /api/forecast/info
  └─ Server logs
```
