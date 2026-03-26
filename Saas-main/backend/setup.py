#!/usr/bin/env python3
"""
Setup script for Energy Forecast Pro Backend
This script helps set up the development environment
"""

import subprocess
import sys
import os
from pathlib import Path

def run_command(cmd, description):
    """Run a command and print status"""
    print(f"\n{'='*60}")
    print(f"📦 {description}")
    print(f"{'='*60}")
    try:
        result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
        if result.stdout:
            print(result.stdout)
        print(f"✅ {description} - SUCCESS")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} - FAILED")
        if e.stderr:
            print(f"Error: {e.stderr}")
        return False

def main():
    print("""
╔═══════════════════════════════════════════════════════════╗
║                                                           ║
║        Energy Forecast Pro - Backend Setup Wizard        ║
║                                                           ║
╚═══════════════════════════════════════════════════════════╝
    """)

    # Check Python version
    print(f"Python version: {sys.version}")
    if sys.version_info < (3, 10):
        print("❌ Python 3.10+ is required")
        sys.exit(1)

    # Create .env if not exists
    env_file = Path(".env")
    env_example = Path(".env.example")
    
    if not env_file.exists() and env_example.exists():
        print("\n📝 Creating .env file from template...")
        env_example.copy(env_file)
        print("✅ .env file created. Please update it with your configuration.")
    
    # Install dependencies
    if not run_command(
        f"{sys.executable} -m pip install -r requirements.txt",
        "Installing Python dependencies"
    ):
        print("\n⚠️  Failed to install dependencies")
        sys.exit(1)

    print("""
╔═══════════════════════════════════════════════════════════╗
║                   Setup Complete! 🎉                      ║
╚═══════════════════════════════════════════════════════════╝

Next steps:

1. Ensure PostgreSQL is running:
   • Windows: net start postgresql-x64-15
   • Or check Task Manager for postgres.exe

2. Ensure Redis is running:
   • Windows: Start Redis from Services
   • Or run: redis-server

3. Create the database:
   • psql -U postgres
   • CREATE DATABASE energy_forecast;
   • \\q

4. Update .env file with your configuration:
   • DATABASE_URL
   • REDIS_URL  
   • SECRET_KEY (generate with: openssl rand -hex 32)

5. Run the API server:
   • python main.py
   • Or: uvicorn main:app --reload

6. Access the API:
   • API: http://localhost:8000
   • Docs: http://localhost:8000/docs

For detailed instructions, see README.md
    """)

if __name__ == "__main__":
    main()
