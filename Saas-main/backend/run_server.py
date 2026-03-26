#!/usr/bin/env python
"""
Startup script for FastAPI server with proper environment configuration.
"""
import os
import sys
from pathlib import Path

# Set environment variables before any imports
os.environ['DATABASE_URL'] = 'sqlite:///./energy_forecast.db'
os.environ['LOAD_MODEL_ON_STARTUP'] = 'true'

# Set model base path to parent directory (where any-quantile-exog config is)
backend_dir = Path(__file__).parent
project_root = backend_dir.parent
os.environ['MODEL_BASE_PATH'] = str(project_root.parent)

# Add both backend and project root to path
sys.path.insert(0, str(backend_dir))
sys.path.insert(0, str(project_root.parent))  # Add any-quantile-exog to path for model imports

if __name__ == '__main__':
    import uvicorn
    print(f"Working directory: {os.getcwd()}")
    print(f"Backend directory: {backend_dir}")
    print(f"PROJECT_ROOT (MODEL_BASE_PATH): {os.environ.get('MODEL_BASE_PATH')}")
    print(f"DATABASE_URL: {os.environ.get('DATABASE_URL')}")
    print(f"LOAD_MODEL_ON_STARTUP: {os.environ.get('LOAD_MODEL_ON_STARTUP')}")
    
    uvicorn.run(
        'main:app',
        host='0.0.0.0',
        port=8000,
        reload=False
    )
