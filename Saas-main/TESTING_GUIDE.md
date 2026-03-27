# 🧪 Testing & Database Visualization Guide

## 📊 Database Visualization Tools

### Option 1: DB Browser for SQLite (Recommended - GUI)
**Download**: https://sqlitebrowser.org/dl/

1. Download and install DB Browser for SQLite (free)
2. Open the application
3. Click **"Open Database"**
4. Navigate to: `C:\Users\Assil\Desktop\Saas-main\Saas-main\backend\energy_forecast.db`
5. Browse tables: **users**, **api_keys**

**Features**:
- Browse data in tables
- Execute SQL queries
- See table structures and relationships
- Export data to CSV/SQL
- Real-time updates (refresh to see new data)

---

### Option 2: VS Code SQLite Extension
**Install**: Search for "SQLite Viewer" in VS Code Extensions

1. Install the extension
2. Navigate to `backend/energy_forecast.db` in the file explorer
3. Right-click → **"Open Database"**
4. Query and view data directly in VS Code

---

### Option 3: Command Line (Built-in)
```powershell
cd backend
sqlite3 energy_forecast.db

# View all tables
.tables

# View users
SELECT * FROM users;

# View API keys
SELECT * FROM api_keys;

# Exit
.quit
```

---

## 🔧 API Testing Tools

### Option 1: FastAPI Interactive Docs (Easiest)
**URL**: http://localhost:8000/docs

**Features**:
- ✅ Beautiful interactive UI (Swagger UI)
- ✅ Test all endpoints directly in browser
- ✅ See request/response schemas
- ✅ Automatic JWT token handling
- ✅ Try it out with real data

**How to use**:
1. Open http://localhost:8000/docs
2. Click on any endpoint (e.g., `/api/auth/signup`)
3. Click **"Try it out"**
4. Fill in the request body
5. Click **"Execute"**
6. See the response with status code and data

**Testing Authentication**:
1. **Signup**: POST `/api/auth/signup`
   ```json
   {
     "name": "Test User",
     "email": "test@example.com",
     "password": "testpassword123"
   }
   ```
   - Copy the `access_token` from the response

2. **Authorize**: Click the **🔒 Authorize** button at the top
   - Paste your token
   - Click "Authorize"
   - Now all protected endpoints will use this token

3. **Get Current User**: GET `/api/auth/me`
   - Click "Try it out" → "Execute"
   - You'll see your user info

---

### Option 2: ReDoc (Alternative Documentation)
**URL**: http://localhost:8000/redoc

- Clean, scrollable documentation
- Better for reading API specs
- Cannot test endpoints (read-only)

---

### Option 3: Postman/Thunder Client

#### Using Thunder Client (VS Code Extension)
1. Install "Thunder Client" extension in VS Code
2. Create a new request:

**Create Account**:
```
Method: POST
URL: http://localhost:8000/api/auth/signup
Headers: 
  Content-Type: application/json
Body (JSON):
{
  "name": "Test User",
  "email": "test@example.com",  
  "password": "password123456"
}
```

**Login**:
```
Method: POST
URL: http://localhost:8000/api/auth/login
Headers:
  Content-Type: application/json
Body (JSON):
{
  "email": "test@example.com",
  "password": "password123456"
}
```

**Get Current User (Protected)**:
```
Method: GET
URL: http://localhost:8000/api/auth/me
Headers:
  Authorization: Bearer YOUR_TOKEN_HERE
```

---

### Option 4: cURL (Command Line)

**Create Account**:
```powershell
curl -X POST "http://localhost:8000/api/auth/signup" `
  -H "Content-Type: application/json" `
  -d '{\"name\":\"Test User\",\"email\":\"test@example.com\",\"password\":\"password123\"}'
```

**Login**:
```powershell
curl -X POST "http://localhost:8000/api/auth/login" `
  -H "Content-Type: application/json" `
  -d '{\"email\":\"test@example.com\",\"password\":\"password123\"}'
```

**Get Current User**:
```powershell
curl -X GET "http://localhost:8000/api/auth/me" `
  -H "Authorization: Bearer YOUR_TOKEN_HERE"
```

---

## 🎯 Complete Testing Flow

### 1. Test Signup

**Frontend**: http://localhost:5174
- Click "Sign Up"
- Fill in: Name, Email, Password (min 8 chars)
- Click "Create Account"
- You should be logged in automatically and see the dashboard

**Backend API**: http://localhost:8000/docs
- Go to POST `/api/auth/signup`
- Click "Try it out"
- Enter test data
- Click "Execute"
- Check response: status 201, token, user object

**Database**:
- Open `energy_forecast.db` in DB Browser
- Check `users` table
- You should see your new user with hashed password

---

### 2. Test Login

**Frontend**: http://localhost:5174
- Logout (click avatar → Sign Out)
- Enter email and password
- Click "Sign In"
- Should redirect to dashboard

**Backend API**: http://localhost:8000/docs
- Go to POST `/api/auth/login`
- Enter your credentials
- Check response: token, user object

---

### 3. Test Protected Endpoints

**Backend API**:
1. Copy your access token from login/signup
2. Click 🔒 **Authorize** button at top of docs
3. Paste token in format: `YOUR_TOKEN` (no "Bearer" prefix needed)
4. Click "Authorize"

Now test:
- GET `/api/auth/me` - Get your user info
- GET `/api/auth/validate` - Validate token
- POST `/api/keys/` - Create API key
- GET `/api/keys/` - List your API keys

---

### 4. Test API Keys

**Create API Key**:
```
POST /api/keys/
{
  "name": "Production API Key",
  "permissions": "read",
  "expires_in_days": 90
}
```

Response will include the full key (only shown once!):
```json
{
  "id": 1,
  "name": "Production API Key",
  "key": "efp_xxxxxxxxxxxxxxxxxxxxx",
  "prefix": "efp_",
  "is_active": true,
  "permissions": "read",
  "created_at": "2026-02-21T..."
}
```

**List Keys**:
```
GET /api/keys/
```

---

## 🐛 Debugging Tips

### Check Frontend Logs
1. Open browser DevTools (F12)
2. Go to **Console** tab
3. You should see logs like:
   - "Signup successful: { user: {...}, access_token: '...' }"
   - "Login successful: { user: {...}, access_token: '...' }"
   - "User validated: { id: 1, email: '...', ... }"

### Check Backend Logs
Look at the terminal running `python main.py`:
```
INFO: 127.0.0.1:xxxxx - "POST /api/auth/signup HTTP/1.1" 201 Created
INFO: 127.0.0.1:xxxxx - "POST /api/auth/login HTTP/1.1" 200 OK
```

### Check Network Requests
1. Open DevTools → **Network** tab
2. Perform an action (signup/login)
3. Click on the request (e.g., `signup`)
4. Check:
   - **Request Headers**: Content-Type, Authorization
   - **Request Payload**: Your data
   - **Response**: Status code, data

### Common Issues & Fixes

**Issue**: "Cannot connect to backend"
- ✅ Check backend is running: http://localhost:8000
- ✅ Check CORS settings in `main.py`

**Issue**: "Token validation failed"
- ✅ Check token in localStorage (DevTools → Application → Local Storage)
- ✅ Try logging out and logging in again

**Issue**: "Database is locked"
- ✅ Close DB Browser while running the app
- ✅ SQLite only allows one writer at a time

---

## 📈 Production Checklist

Before deploying to production:

- [ ] Change SECRET_KEY in `.env` to a secure random string
- [ ] Switch from SQLite to PostgreSQL
- [ ] Enable HTTPS
- [ ] Set up Redis for caching
- [ ] Configure proper CORS origins
- [ ] Add rate limiting
- [ ] Set up logging and monitoring
- [ ] Back up database regularly

---

## 🎓 Quick Reference

| Tool | Purpose | URL |
|------|---------|-----|
| Frontend App | User interface | http://localhost:5174 |
| Backend API | REST API | http://localhost:8000 |
| Swagger Docs | Interactive API testing | http://localhost:8000/docs |
| ReDoc | API documentation | http://localhost:8000/redoc |
| Database | SQLite file | `backend/energy_forecast.db` |

**Authentication Flow**:
1. User signs up/logs in
2. Backend returns JWT token + user object
3. Frontend stores token in localStorage
4. All subsequent requests include token in Authorization header
5. Backend validates token and returns protected data
