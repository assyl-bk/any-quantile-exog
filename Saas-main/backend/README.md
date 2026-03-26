# Energy Forecast Pro - Backend API

Python FastAPI backend for the Energy Forecast Pro application with PostgreSQL and Redis.

## Prerequisites

- Python 3.10+
- PostgreSQL 15+
- Redis

## Setup Instructions

### 1. Install PostgreSQL

**Windows:**
```powershell
# Download and install from https://www.postgresql.org/download/windows/
# Or use chocolatey:
choco install postgresql15

# Start PostgreSQL service
net start postgresql-x64-15
```

### 2. Install Redis

**Windows:**
```powershell
# Download and install from https://github.com/microsoftarchive/redis/releases
# Or use chocolatey:
choco install redis-64

# Start Redis service
redis-server
```

### 3. Create Database

```powershell
# Connect to PostgreSQL
psql -U postgres

# Create database
CREATE DATABASE energy_forecast;

# Exit psql
\q
```

### 4. Install Python Dependencies

```powershell
# Navigate to backend directory
cd backend

# Create virtual environment
python -m venv venv

# Activate virtual environment
.\venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 5. Configure Environment

```powershell
# Copy example env file
copy .env.example .env

# Edit .env and update configurations
notepad .env
```

Update the following in `.env`:
- `DATABASE_URL`: PostgreSQL connection string
- `SECRET_KEY`: Generate with `openssl rand -hex 32`
- `REDIS_URL`: Redis connection string

### 6. Run the API Server

```powershell
# Activate virtual environment if not already activated
.\venv\Scripts\activate

# Run with uvicorn
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at:
- API: http://localhost:8000
- API Docs: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## API Endpoints

### Authentication

- `POST /api/auth/signup` - Register new user
- `POST /api/auth/login` - Login user
- `GET /api/auth/validate` - Validate token
- `GET /api/auth/me` - Get current user info

### API Keys

- `POST /api/keys/` - Create new API key
- `GET /api/keys/` - List user's API keys
- `DELETE /api/keys/{key_id}` - Delete API key
- `PATCH /api/keys/{key_id}/deactivate` - Deactivate API key
- `PATCH /api/keys/{key_id}/activate` - Activate API key

## Database Schema

### Users Table
- `id`: Primary key
- `name`: User full name
- `email`: Unique email address
- `hashed_password`: Bcrypt hashed password
- `role`: User role (user, admin)
- `is_active`: Account status
- `created_at`: Account creation timestamp
- `updated_at`: Last update timestamp

### API Keys Table
- `id`: Primary key
- `user_id`: Foreign key to users
- `name`: Key name/description
- `key`: Full API key (efp_...)
- `prefix`: Key prefix for display
- `is_active`: Key status
- `created_at`: Creation timestamp
- `last_used`: Last usage timestamp
- `expires_at`: Expiration timestamp
- `permissions`: Key permissions (read, write, admin)

## Security Features

- Bcrypt password hashing
- JWT token authentication
- Secure API key generation
- CORS protection
- Token expiration
- API key expiration support

## Development

### Run Tests
```powershell
pytest
```

### Database Migrations
```powershell
# Create migration
alembic revision --autogenerate -m "description"

# Run migrations
alembic upgrade head
```

### Format Code
```powershell
black .
```

## Production Deployment

1. Set `ENVIRONMENT=production` in `.env`
2. Generate strong `SECRET_KEY`
3. Use HTTPS
4. Configure proper CORS origins
5. Set up database backups
6. Configure Redis persistence
7. Use proper logging
8. Set up monitoring

## Troubleshooting

### PostgreSQL Connection Issues
- Ensure PostgreSQL service is running
- Check connection string in `.env`
- Verify database exists: `psql -U postgres -l`

### Redis Connection Issues
- Ensure Redis service is running: `redis-cli ping`
- Check Redis URL in `.env`

### Import Errors
- Activate virtual environment
- Verify all dependencies installed: `pip list`

## Support

For issues and questions, please refer to the project documentation.
