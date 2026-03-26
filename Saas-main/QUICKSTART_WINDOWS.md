# Quick Start Guide for Windows

## Prerequisites

- Python 3.10+ installed
- PostgreSQL 15+ running
- Redis running (optional, for caching)

## Quick Setup (PowerShell)

### Step 1: Install Python Dependencies

```powershell
cd backend
pip install -r requirements.txt
```

_This will take a few minutes due to torch installation_

### Step 2: Configure Environment

```powershell
# Copy example to .env
Copy-Item .env.example .env

# Edit .env with your database credentials
notepad .env
```

**Essential variables to update:**

```
DATABASE_URL=postgresql://postgres:YOUR_PASSWORD@localhost:5432/energy_forecast
MODEL_BASE_PATH=../
MODEL_CHECKPOINT=lightning_logs/nbeatsaq-stage2-seed0/checkpoints/model-epoch=4.ckpt
MODEL_CONFIG=config/AQNBEATS++_S2.yaml
MODEL_CALIBRATOR=results/CQR/calibrator_stage2_v2.pkl
LOAD_MODEL_ON_STARTUP=true
```

### Step 3: Verify Model Files Exist

```powershell
# Check checkpoint
Test-Path "..\lightning_logs\nbeatsaq-stage2-seed0\checkpoints\model-epoch=4.ckpt"

# Check config
Test-Path "..\config\AQNBEATS++_S2.yaml"

# Check calibrator
Test-Path "..\results\CQR\calibrator_stage2_v2.pkl"
```

### Step 4: Initialize Database

```powershell
# One-time setup (creates tables)
$env:DATABASE_URL="postgresql://postgres:YOUR_PASSWORD@localhost:5432/energy_forecast"
python init_db.py
```

### Step 5: Start the API Server

```powershell
# Using uvicorn directly
python -m uvicorn main:app --reload --host 0.0.0.0 --port 8000

# Or using Python module
python main.py
```

You should see:

```
✅ Model loaded successfully
INFO:     Started server process
INFO:     Uvicorn running on http://0.0.0.0:8000
```

### Step 6: Create User & API Key

**Open PowerShell in another window:**

```powershell
# 1. Register user
$signupData = @{
    email = "forecast@example.com"
    name = "Forecast Admin"
    password = "SecurePassword123!"
    role = "energy_grid_operator"
} | ConvertTo-Json

Invoke-WebRequest -Uri "http://localhost:8000/api/auth/signup" `
    -Method POST `
    -Headers @{"Content-Type"="application/json"} `
    -Body $signupData

# 2. Login to get JWT token
$loginData = @{
    email = "forecast@example.com"
    password = "SecurePassword123!"
} | ConvertTo-Json

$loginResponse = Invoke-WebRequest -Uri "http://localhost:8000/api/auth/login" `
    -Method POST `
    -Headers @{"Content-Type"="application/json"} `
    -Body $loginData

$jwt = ($loginResponse.Content | ConvertFrom-Json).access_token

# 3. Create API key using JWT
$keyData = @{
    name = "Energy Forecast Key"
    permissions = "read"
    expires_in_days = 365
} | ConvertTo-Json

$keyResponse = Invoke-WebRequest -Uri "http://localhost:8000/api/keys/" `
    -Method POST `
    -Headers @{"Authorization"="Bearer $jwt"; "Content-Type"="application/json"} `
    -Body $keyData

$apiKey = ($keyResponse.Content | ConvertFrom-Json).key
Write-Host "API Key: $apiKey"
```

**Save the API key!** You'll need it for forecast requests.

### Step 7: Test the Forecast Endpoint

```powershell
$apiKey = "efp_YOUR_KEY_FROM_STEP_6"

$forecastData = @{
    historical_data = @(100.5, 102.3, 101.8, 103.2, 102.5, 104.1, 103.5, 102.8, 105.2, 104.5, 103.8, 106.2, 105.5, 104.8, 107.2, 106.5, 105.8, 108.2, 107.5, 106.8, 109.2, 108.5, 107.8, 110.2)
    quantiles = @(0.1, 0.5, 0.9)
    apply_cqr = $true
} | ConvertTo-Json

$response = Invoke-WebRequest -Uri "http://localhost:8000/api/forecast" `
    -Method POST `
    -Headers @{"X-API-Key"=$apiKey; "Content-Type"="application/json"} `
    -Body $forecastData

$response.Content | ConvertFrom-Json | ConvertTo-Json -Depth 10 | Write-Host
```

## Common Issues on Windows

### Issue: "inotify" platform error when installing

```powershell
# This is a known Windows npm issue, not related to our Python backend
# Just ignore it - it only affects file watching
```

### Issue: Port 8000 already in use

```powershell
# Use a different port
python -m uvicorn main:app --reload --host 0.0.0.0 --port 8001
```

### Issue: PostgreSQL not running

```powershell
# Start PostgreSQL service
Get-Service postgresql-x64-15 | Start-Service

# Or if using chocolatey installed version:
net start postgresql-x64-15
```

### Issue: Model loading fails

```powershell
# Check if paths are correct
$env:MODEL_BASE_PATH; `
"../lightning_logs/nbeatsaq-stage2-seed0/checkpoints/model-epoch=4.ckpt" | `
ForEach-Object {Test-Path $_}

# If paths wrong, update .env and restart
notepad .env
```

### Issue: CUDA out of memory

```powershell
# Use CPU only
$env:CUDA_VISIBLE_DEVICES=""
python -m uvicorn main:app --reload
```

## Alternative: Docker Setup (Recommended for Production)

If you have Docker installed:

```powershell
# Build and run with Docker
docker-compose up --build

# The API will be available at http://localhost:8000
```

## View API Documentation

Once server is running, open your browser:

- **Interactive Docs**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc
- **Health Check**: http://localhost:8000/health

## Python Client Quick Example

Create a file `test_forecast.py`:

```python
import requests
import json

API_KEY = "efp_YOUR_API_KEY"
BASE_URL = "http://localhost:8000/api"

def test_forecast():
    headers = {
        "X-API-Key": API_KEY,
        "Content-Type": "application/json"
    }

    data = {
        "historical_data": [100.5, 102.3, 101.8, 103.2, 102.5, 104.1, 103.5, 102.8,
                           105.2, 104.5, 103.8, 106.2, 105.5, 104.8, 107.2, 106.5,
                           105.8, 108.2, 107.5, 106.8, 109.2, 108.5, 107.8, 110.2],
        "quantiles": [0.1, 0.5, 0.9],
        "apply_cqr": True
    }

    response = requests.post(
        f"{BASE_URL}/forecast",
        headers=headers,
        json=data
    )

    print(json.dumps(response.json(), indent=2))

if __name__ == "__main__":
    test_forecast()
```

Run it:

```powershell
python test_forecast.py
```

## Next Steps

1. ✅ Setup complete - API is running
2. Generate forecasts using the `/api/forecast` endpoint
3. Integrate with your frontend/application
4. For production deployment, see `INTEGRATION_GUIDE.md`

## Additional Resources

- Full setup guide: `INTEGRATION_GUIDE.md`
- Summary of changes: `CHANGES_SUMMARY.md`
- API documentation: http://localhost:8000/docs
- Troubleshooting guide: See `INTEGRATION_GUIDE.md`

## Support

For issues:

1. Check error messages in terminal
2. Review logs in API output
3. Consult troubleshooting section in `INTEGRATION_GUIDE.md`
4. Verify `.env` configuration
