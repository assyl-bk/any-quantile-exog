# Energy Forecast Pro - Backend Setup Script for Windows
# Run this script to set up the development environment

Write-Host "`n╔═══════════════════════════════════════════════════════════╗" -ForegroundColor Cyan
Write-Host "║                                                           ║" -ForegroundColor Cyan
Write-Host "║        Energy Forecast - Backend Setup Wizard        ║" -ForegroundColor Cyan
Write-Host "║                                                           ║" -ForegroundColor Cyan
Write-Host "╚═══════════════════════════════════════════════════════════╝`n" -ForegroundColor Cyan

# Check Python
Write-Host "Checking Python installation..." -ForegroundColor Yellow
$pythonVersion = python --version 2>&1
if ($LASTEXITCODE -eq 0) {
    Write-Host "✅ $pythonVersion" -ForegroundColor Green
} else {
    Write-Host "❌ Python not found. Please install Python 3.10+" -ForegroundColor Red
    exit 1
}

# Create virtual environment
Write-Host "`nCreating virtual environment..." -ForegroundColor Yellow
if (Test-Path "venv") {
    Write-Host "Virtual environment already exists" -ForegroundColor Gray
} else {
    python -m venv venv
    if ($LASTEXITCODE -eq 0) {
        Write-Host "✅ Virtual environment created" -ForegroundColor Green
    } else {
        Write-Host "❌ Failed to create virtual environment" -ForegroundColor Red
        exit 1
    }
}

# Activate virtual environment
Write-Host "`nActivating virtual environment..." -ForegroundColor Yellow
& ".\venv\Scripts\Activate.ps1"

# Install dependencies
Write-Host "`nInstalling dependencies..." -ForegroundColor Yellow
pip install -r requirements.txt
if ($LASTEXITCODE -eq 0) {
    Write-Host "✅ Dependencies installed" -ForegroundColor Green
} else {
    Write-Host "❌ Failed to install dependencies" -ForegroundColor Red
    exit 1
}

# Create .env file
Write-Host "`nConfiguring environment..." -ForegroundColor Yellow
if (-not (Test-Path ".env")) {
    Copy-Item ".env.example" ".env"
    Write-Host "✅ .env file created from template" -ForegroundColor Green
    Write-Host "⚠️  Please update .env with your configuration" -ForegroundColor Yellow
} else {
    Write-Host ".env file already exists" -ForegroundColor Gray
}

Write-Host "`n╔═══════════════════════════════════════════════════════════╗" -ForegroundColor Cyan
Write-Host "║                   Setup Complete! 🎉                      ║" -ForegroundColor Cyan
Write-Host "╚═══════════════════════════════════════════════════════════╝`n" -ForegroundColor Cyan

Write-Host "Next steps:`n" -ForegroundColor White

Write-Host "1. Start PostgreSQL:" -ForegroundColor Yellow
Write-Host "   net start postgresql-x64-15`n" -ForegroundColor Gray

Write-Host "2. Start Redis:" -ForegroundColor Yellow
Write-Host "   redis-server`n" -ForegroundColor Gray

Write-Host "3. Create database:" -ForegroundColor Yellow
Write-Host "   psql -U postgres -c `"CREATE DATABASE energy_forecast;`"`n" -ForegroundColor Gray

Write-Host "4. Update .env file with your settings`n" -ForegroundColor Yellow

Write-Host "5. Run the API server:" -ForegroundColor Yellow
Write-Host "   python main.py" -ForegroundColor Gray
Write-Host "   Or: uvicorn main:app --reload`n" -ForegroundColor Gray

Write-Host "6. Access the API:" -ForegroundColor Yellow
Write-Host "   • API: http://localhost:8000" -ForegroundColor Gray
Write-Host "   • Docs: http://localhost:8000/docs" -ForegroundColor Gray
Write-Host "   • ReDoc: http://localhost:8000/redoc`n" -ForegroundColor Gray

Write-Host "For detailed instructions, see README.md`n" -ForegroundColor White
