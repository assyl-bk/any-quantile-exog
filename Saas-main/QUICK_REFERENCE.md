# Quick Reference Guide

## Quick Start

### 1. Start Backend

```bash
cd Saas-main/backend
python -m uvicorn main:app --reload
```

### 2. Access API Documentation

```
http://localhost:8000/docs
```

Interactive Swagger UI for testing all endpoints

---

## Common Endpoints (with examples)

### Get User Profile

```bash
curl http://localhost:8000/api/user/1 \
  -H "Authorization: Bearer TOKEN"
```

Response:

```json
{
  "id": 1,
  "email": "user@example.com",
  "name": "John Doe",
  "role": "energy_grid_operator",
  "is_active": true,
  "created_at": "2024-01-15T10:30:00"
}
```

### Update Profile

```bash
curl -X PUT http://localhost:8000/api/user/1/profile \
  -H "Authorization: Bearer TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Jane Doe",
    "email": "jane@example.com"
  }'
```

### Upload Profile Image

```bash
curl -X POST http://localhost:8000/api/user/1/profile-image \
  -H "Authorization: Bearer TOKEN" \
  -F "image=@profile.jpg"
```

Response:

```json
{
  "profile_image": "uploads/profiles/user_1_1234567890.jpg",
  "message": "Profile image uploaded successfully"
}
```

### Get Notification Preferences

```bash
curl http://localhost:8000/api/user/1/notifications \
  -H "Authorization: Bearer TOKEN"
```

Response:

```json
{
  "id": 1,
  "user_id": 1,
  "email": true,
  "peak_demand": true,
  "peak_demand_threshold": 8000,
  "model_report": false,
  "sys_updates": false,
  "created_at": "2024-01-15T10:30:00",
  "updated_at": "2024-01-15T10:30:00"
}
```

### Update Notification Preferences

```bash
curl -X PUT http://localhost:8000/api/user/1/notifications \
  -H "Authorization: Bearer TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "email": true,
    "peak_demand": true,
    "peak_demand_threshold": 9000,
    "model_report": true,
    "sys_updates": false
  }'
```

### Get Alert History

```bash
curl "http://localhost:8000/api/user/1/alerts?limit=10&offset=0" \
  -H "Authorization: Bearer TOKEN"
```

Response:

```json
[
  {
    "id": 1,
    "user_id": 1,
    "alert_type": "peak_demand",
    "message": "Demand exceeded 8000 MW",
    "data": {
      "current_demand": 8500,
      "threshold": 8000,
      "exceeded_by": 500
    },
    "acknowledged": false,
    "created_at": "2024-01-15T14:30:00"
  }
]
```

### Create Test Alert (Internal)

```bash
curl -X POST http://localhost:8000/api/user/1/alerts \
  -H "Content-Type: application/json" \
  -d '{
    "alert_type": "peak_demand",
    "message": "Test demand alert",
    "data": {
      "current_demand": 8500,
      "threshold": 8000
    }
  }'
```

---

## Testing

### Run Full Test Suite

```bash
cd backend
pytest test_notifications.py -v
```

### Run Specific Test Class

```bash
pytest test_notifications.py::TestProfileEndpoints -v
```

### Run with Output

```bash
pytest test_notifications.py -v -s
```

### Run with Coverage Report

```bash
pytest test_notifications.py --cov=app --cov-report=html
# View: htmlcov/index.html
```

---

## Environment Configuration

### Email Provider Setup

**Console (Default - Development)**

```bash
EMAIL_PROVIDER=console
```

**SMTP (Gmail/Office 365)**

```bash
EMAIL_PROVIDER=smtp
SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
SMTP_USER=your_email@gmail.com
SMTP_PASSWORD=your_app_password
```

**SendGrid**

```bash
EMAIL_PROVIDER=sendgrid
SENDGRID_API_KEY=your_sendgrid_key
```

**AWS SES**

```bash
EMAIL_PROVIDER=ses
AWS_REGION=us-east-1
```

---

## Monitoring & Debugging

### Check Demand Alert Service Status

Look for in startup logs:

```
✅ Demand alert service started
```

### Enable Debug Logging

Logs are shown in console. For demand monitoring debug:

1. Check console output for "Alert created for user"
2. Check database: `SELECT * FROM alert_history`
3. For email testing: Use console provider to see "📧 EMAIL" log

### Database Inspection

```bash
# Using SQLite CLI
sqlite3 energy_forecast.db

# Check tables
.tables

# View notification preferences
SELECT * FROM notification_preferences;

# View alert history
SELECT * FROM alert_history;

# View users
SELECT id, name, email, role FROM users;
```

---

## Common Issues & Solutions

### "User not found" (404)

- Ensure user exists in database first
- Verify user_id in URL is correct
- Check token is valid for that user

### "Not authorized" (403)

- Users can only access their own data
- Verify token belongs to the user_id in URL
- Admin token might be needed for other users

### "Invalid file type" (400)

- Profile image must be jpg, jpeg, png, or webp
- Max file size: 5MB
- Check file doesn't have wrong extension

### "Token expired"

- Get new token by logging in
- Access token expires every 30 min (configurable)
- Check ALGORITHM = "HS256" in settings

### Demand monitoring not running

- Check app startup logs for "Demand alert service started"
- If error, see error message in logs
- Verify database tables were created
- Use console provider for email (no SMTP needed)

---

## File Locations

| Component         | Location                                        |
| ----------------- | ----------------------------------------------- |
| Models            | `backend/app/models.py`                         |
| Schemas           | `backend/app/schemas.py`                        |
| User Router       | `backend/app/routers/users.py`                  |
| Email Service     | `backend/app/services/email_service.py`         |
| Demand Monitoring | `backend/app/services/demand_alert.py`          |
| Tests             | `backend/test_notifications.py`                 |
| Main App          | `backend/main.py`                               |
| Settings          | `backend/app/config.py`                         |
| Frontend Settings | `Saas-main/src/app/components/SettingsPage.tsx` |

---

## Useful Parameters

### Alert List Query Parameters

```
?limit=10      # Default: 20
?offset=0      # Start at item 0
```

### Notification Preferences Fields

```
email: bool
peak_demand: bool
peak_demand_threshold: int (>0)
model_report: bool
sys_updates: bool
```

### Alert History Fields

```
id: int
user_id: int
alert_type: string (peak_demand, model_report, sys_updates, etc.)
message: string
data: dict (JSON)
acknowledged: bool
created_at: datetime
```

---

## Performance Notes

- Demand checking: Every 60 seconds (configurable)
- Alert cooldown: 15 minutes per user (prevents spam)
- Database indexes: user_id, alert_type, created_at
- Background task: Async/non-blocking

---

## Support Resources

- **API Docs**: http://localhost:8000/docs
- **Full Guide**: `backend/NOTIFICATION_SYSTEM.md`
- **Implementation Details**: `IMPLEMENTATION_COMPLETE.md`
- **Requirements**: `BACKEND_API_REQUIREMENTS.md`
- **Tests**: `backend/test_notifications.py`
