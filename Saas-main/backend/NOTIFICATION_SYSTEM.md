# Notification System Implementation Guide

## Overview

The notification system is now fully integrated with your Energy Forecast API. It includes:

✅ **User Profile Management** - Update name, email, profile image  
✅ **Notification Preferences** - Configure when and how users get alerts  
✅ **Peak Demand Alerts** - User-configurable threshold-based notifications  
✅ **Alert History** - Track all alerts sent to users  
✅ **Background Monitoring** - Continuous demand monitoring service  
✅ **Email Integration** - Multiple email provider support

---

## Features Implemented

### 1. User Management Endpoints

```
GET    /api/user/{user_id}                    - Get user profile
PUT    /api/user/{user_id}/profile            - Update name/email
POST   /api/user/{user_id}/profile-image      - Upload profile image
DELETE /api/user/{user_id}/profile-image      - Delete profile image
```

### 2. Notification Preferences

```
GET    /api/user/{user_id}/notifications      - Get preferences
PUT    /api/user/{user_id}/notifications      - Update preferences
```

Preferences include:

- **email**: Enable/disable email notifications
- **peak_demand**: Enable/disable peak demand alerts
- **peak_demand_threshold**: Custom MW threshold (default 8000)
- **model_report**: Weekly model performance reports
- **sys_updates**: System update notifications

### 3. Alert History

```
GET    /api/user/{user_id}/alerts             - Get alert history
POST   /api/user/{user_id}/alerts             - Create alert (internal)
```

### 4. Background Services

- **DemandAlertService**: Monitors current demand every 60 seconds
- **EmailService**: Sends emails via console, SMTP, SendGrid, or AWS SES

---

## Database Schema

### User Table (Extended)

```sql
ALTER TABLE users ADD profile_image VARCHAR(500);
```

### New Tables

**notification_preferences**

```sql
CREATE TABLE notification_preferences (
  id INTEGER PRIMARY KEY,
  user_id INTEGER UNIQUE NOT NULL,
  email BOOLEAN DEFAULT TRUE,
  peak_demand BOOLEAN DEFAULT TRUE,
  peak_demand_threshold INTEGER DEFAULT 8000,
  model_report BOOLEAN DEFAULT FALSE,
  sys_updates BOOLEAN DEFAULT FALSE,
  created_at TIMESTAMP,
  updated_at TIMESTAMP
);
```

**alert_history**

```sql
CREATE TABLE alert_history (
  id INTEGER PRIMARY KEY,
  user_id INTEGER NOT NULL,
  alert_type VARCHAR(50),
  message VARCHAR(255),
  data JSON,
  acknowledged BOOLEAN DEFAULT FALSE,
  created_at TIMESTAMP
);
```

---

## Testing the System

### Prerequisites

```bash
cd backend
pip install -r requirements.txt
```

### Run Tests

```bash
# Run all notification tests
pytest test_notifications.py -v -s

# Run specific test class
pytest test_notifications.py::TestProfileEndpoints -v

# Run with coverage
pytest test_notifications.py --cov=app --cov-report=html
```

### Test Coverage

- ✅ Profile management (get, update)
- ✅ Image upload/delete
- ✅ Notification preferences (get, update)
- ✅ Threshold validation
- ✅ Alert history (get, create)
- ✅ Email service (console, mock providers)
- ✅ Demand alert service (cooldown logic)
- ✅ Complete integration flow

---

## Manual Testing

### 1. Start the Backend Server

```bash
cd backend
python -m uvicorn main:app --reload
```

You should see:

```
🚀 Starting Energy Forecast API...
✅ Database tables created
✅ Demand alert service started
```

### 2. Test Profile Updates

```bash
# Get user profile
curl -X GET http://localhost:8000/api/user/1 \
  -H "Authorization: Bearer YOUR_TOKEN"

# Update profile
curl -X PUT http://localhost:8000/api/user/1/profile \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"name": "New Name", "email": "new@example.com"}'
```

### 3. Test Notification Preferences

```bash
# Get preferences (creates defaults if not exist)
curl -X GET http://localhost:8000/api/user/1/notifications \
  -H "Authorization: Bearer YOUR_TOKEN"

# Update preferences
curl -X PUT http://localhost:8000/api/user/1/notifications \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "email": true,
    "peak_demand": true,
    "peak_demand_threshold": 8500,
    "model_report": false,
    "sys_updates": false
  }'
```

### 4. Test Alert Creation

```bash
# Create a test alert
curl -X POST http://localhost:8000/api/user/1/alerts \
  -H "Content-Type: application/json" \
  -d '{
    "alert_type": "peak_demand",
    "message": "Demand exceeded 8000 MW",
    "data": {"current_demand": 8500, "threshold": 8000}
  }'

# Get alert history
curl -X GET http://localhost:8000/api/user/1/alerts \
  -H "Authorization: Bearer YOUR_TOKEN"
```

### 5. Test Profile Image Upload

```bash
# Upload image
curl -X POST http://localhost:8000/api/user/1/profile-image \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -F "image=@/path/to/image.jpg"

# Delete image
curl -X DELETE http://localhost:8000/api/user/1/profile-image \
  -H "Authorization: Bearer YOUR_TOKEN"
```

---

## Email Configuration

### Console Provider (Development/Testing)

Prints emails to console. Used by default.

```python
# In main.py
email_service = get_email_service(provider="console")
```

### SMTP Provider (Gmail, Office 365, etc.)

Set environment variables:

```bash
EMAIL_PROVIDER=smtp
SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
SMTP_USER=your_email@gmail.com
SMTP_PASSWORD=your_app_password
```

Then use:

```python
email_service = get_email_service(provider="smtp")
```

### SendGrid Provider

Set environment variables:

```bash
EMAIL_PROVIDER=sendgrid
SENDGRID_API_KEY=your_sendgrid_key
```

### AWS SES Provider

Set environment variables:

```bash
EMAIL_PROVIDER=ses
AWS_REGION=us-east-1
```

---

## Demand Monitoring Integration

### How It Works

1. **Background Task**: `DemandAlertService` runs continuously
2. **Check Interval**: Every 60 seconds (configurable)
3. **Threshold Check**: Compares current demand against user thresholds
4. **Cooldown**: Won't spam - max 1 alert per 15 minutes per user
5. **Notifications**: Sends email if user has email enabled

### Example Flow

```
1. User sets threshold to 8500 MW
2. System monitors demand every 60 seconds
3. Current demand reaches 8600 MW
4. Alert triggered for user
5. Email sent (if enabled)
6. Alert saved to history
7. 15-min cooldown starts (no more alerts for this user until then)
8. Demand drops to 7000 MW
9. No alert (threshold not exceeded)
```

### Setting Current Demand

In production, you need to update the current demand from your data source. Example:

```python
from app.services import get_demand_alert_service

alert_service = get_demand_alert_service()
alert_service.set_current_demand(8600)  # Update from your API/database
```

Or modify `app/services/demand_alert.py`:

```python
async def get_current_demand_from_source(self) -> float:
    """Replace with your actual data source"""
    # Query database
    # result = db.query(DemandData).order_by(DemandData.timestamp.desc()).first()
    # return result.demand_mw

    # Or call external API
    # response = await httpx.get("https://grid-api.com/current-demand")
    # return response.json()["demand"]

    # For now, return the set value
    return self.current_demand
```

---

## Production Deployment Checklist

- [ ] Disable console email provider
- [ ] Configure SMTP/SendGrid/SES
- [ ] Set strong SECRET_KEY in environment
- [ ] Use PostgreSQL instead of SQLite
- [ ] Configure CORS for production domains
- [ ] Set LOAD_MODEL_ON_STARTUP = True
- [ ] Enable HTTPS
- [ ] Set up monitoring/logging
- [ ] Test email delivery
- [ ] Test demand monitoring with real data
- [ ] Set up database backups
- [ ] Monitor background task health

---

## Troubleshooting

### Demand Alert Service Not Starting

```
❌ Failed to start demand alert service: ...
```

Check:

1. Database connection is working
2. No import errors in `app/services/`
3. logs for detailed error

### Email Not Sending

```python
# Check if email service is working
email_service = get_email_service(provider="console")
result = await email_service.send_email(
    to_email="test@example.com",
    subject="Test",
    html_content="<h1>Test</h1>"
)
print(result)  # Should be True
```

### Database Errors

```sql
-- Check tables were created
SELECT name FROM sqlite_master WHERE type='table';

-- Check notification preferences
SELECT * FROM notification_preferences;

-- Check alert history
SELECT * FROM alert_history;
```

---

## API Documentation

Full API documentation available at: `http://localhost:8000/docs`

Interactive API testing with Swagger UI showing:

- All endpoints
- Request/response schemas
- Try-it-out functionality
- Example data

---

## Next Steps

1. **Test Desktop Frontend**: Verify settings page works end-to-end
2. **Configure Real Email**: Set up SMTP/SendGrid for actual emails
3. **Data Source Integration**: Wire up real demand data feed
4. **Monitoring**: Add health checks and alerting for background service
5. **Analytics**: Track alert delivery rates and user engagement
