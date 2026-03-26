# 🚀 Energy Forecast Pro - Quick Start Guide

## ✅ System Status

### Backend Server
- **Status**: Running ✓
- **URL**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs
- **Database**: SQLite (energy_forecast.db)

### Frontend Server
- **Status**: Running ✓
- **URL**: http://localhost:5174
- **Framework**: React 18 + TypeScript + Vite

---

## 📝 How to Use

### 1. Access the Application
Open your browser and go to: **http://localhost:5174**

You'll see the login/signup page with a beautiful animated interface.

### 2. Create an Account
1. Click on "Sign Up" (toggle button at the top)
2. Enter your details:
   - Name: Your full name
   - Email: Your email address
   - Password: Your secure password
3. Click "Create Account"

The system will:
- Create your user account
- Hash your password securely using Bcrypt
- Generate a JWT token
- Log you in automatically

### 3. Login
If you already have an account:
1. Enter your email and password
2. Click "Sign In"

### 4. Use the Dashboard
Once logged in, you'll see:
- **Header**: Your user avatar with initials
- **Dashboard**: Energy forecasting interface with customizable quantiles
- **Dark/Light Mode**: Toggle theme using the button in the header
- **User Menu**: Click your avatar to see profile and sign out

---

## 🔧 Development

### Start Servers Manually

**Backend:**
```powershell
cd backend
python main.py
```

**Frontend:**
```powershell
npm run dev
```

### Stop Servers
Press `Ctrl+C` in the terminal running each server.

---

## 📚 API Endpoints

All API endpoints are available at: http://localhost:8000/docs

### Authentication Endpoints
- `POST /api/auth/signup` - Create new user account
- `POST /api/auth/login` - Login and get JWT token
- `GET /api/auth/validate` - Validate JWT token
- `GET /api/auth/me` - Get current user info

### API Key Endpoints (Protected - requires JWT)
- `POST /api/keys/` - Create new API key
- `GET /api/keys/` - List all your API keys
- `DELETE /api/keys/{id}` - Delete an API key
- `PATCH /api/keys/{id}/activate` - Activate an API key
- `PATCH /api/keys/{id}/deactivate` - Deactivate an API key

---

## 🗄️ Database

The application uses **SQLite** for easy development (no installation required).

**Database file**: `backend/energy_forecast.db`

**Tables**:
- `users` - User accounts with hashed passwords
- `api_keys` - Generated API keys with permissions

To view the database:
```powershell
cd backend
sqlite3 energy_forecast.db
# Then run SQL commands like:
# SELECT * FROM users;
# .quit to exit
```

---

## 🔐 Security Features

- **Bcrypt Password Hashing**: Passwords are never stored in plain text
- **JWT Tokens**: Secure token-based authentication with 30-minute expiry
- **Protected Routes**: Frontend routes require valid authentication
- **API Key System**: Generate and manage API keys for external integrations
- **CORS Protection**: Backend configured to accept requests only from frontend

---

## 🎨 Features

### Dashboard
- Real-time energy forecasting
- Customizable quantile selection (1-99%)
- Confidence interval visualization (50-99%)
- Dark/Light mode toggle
- Responsive design

### Authentication
- Beautiful animated login/signup page
- Email validation
- Password visibility toggle
- Error handling with user-friendly messages
- Auto-login after signup
- Persistent sessions (JWT in localStorage)

### User Management
- User profile display
- Avatar with initials
- Dropdown menu with quick actions
- Secure logout

---

## 🔄 Switching to PostgreSQL (Production)

For production, you may want to use PostgreSQL instead of SQLite:

1. Install PostgreSQL 15+
2. Create a database: `CREATE DATABASE energy_forecast;`
3. Update `backend/.env`:
   ```
   DATABASE_URL=postgresql://postgres:yourpassword@localhost:5432/energy_forecast
   ```
4. Restart the backend server

The system will automatically use PostgreSQL when the DATABASE_URL starts with `postgresql://`.

---

## 📞 Support

If you encounter any issues:
1. Check both servers are running
2. Check the browser console for frontend errors (F12)
3. Check the backend terminal for API errors
4. Verify the database file exists: `backend/energy_forecast.db`

---

## 🎉 You're All Set!

Visit **http://localhost:5174** and start using Energy Forecast Pro!
