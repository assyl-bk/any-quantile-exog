# Backend API Requirements for Settings & Notifications

This document outlines the API endpoints required to support the updated SettingsPage UI.

## Database Schema Updates

### User Table Extension

```sql
ALTER TABLE users ADD COLUMN profile_image VARCHAR(500) NULL;
ALTER TABLE users ADD COLUMN settings JSON DEFAULT '{}';
```

### Notifications Preferences Table

```sql
CREATE TABLE notification_preferences (
  id INT PRIMARY KEY AUTO_INCREMENT,
  user_id INT NOT NULL UNIQUE,
  email BOOLEAN DEFAULT TRUE,
  peak_demand BOOLEAN DEFAULT TRUE,
  peak_demand_threshold INT DEFAULT 8000,
  model_report BOOLEAN DEFAULT FALSE,
  sys_updates BOOLEAN DEFAULT FALSE,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
  FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
);
```

## API Endpoints

### 1. Get User Profile

**Endpoint:** `GET /api/user/{user_id}`
**Headers:** `Authorization: Bearer {token}`
**Response:**

```json
{
  "id": 1,
  "email": "user@example.com",
  "name": "John Doe",
  "role": "energy_grid_operator",
  "profile_image": "https://example.com/images/profile-1.jpg"
}
```

### 2. Update User Profile

**Endpoint:** `PUT /api/user/{user_id}/profile`
**Headers:**

```
Authorization: Bearer {token}
Content-Type: application/json
```

**Request Body:**

```json
{
  "name": "John Doe",
  "email": "john@example.com"
}
```

**Response:** `200 OK` with updated user object

### 3. Upload Profile Image

**Endpoint:** `POST /api/user/{user_id}/profile-image`
**Headers:** `Authorization: Bearer {token}`
**Body:** `multipart/form-data` with field named `image`
**Response:**

```json
{
  "profile_image": "https://example.com/images/profile-1.jpg"
}
```

**Implementation Notes:**

- Store image in cloud storage (S3, Azure Blob, etc.)
- Compress/optimize image before storage
- Return HTTPS URL
- Max file size: 5MB
- Allowed formats: jpg, jpeg, png, webp

### 4. Get Notification Preferences

**Endpoint:** `GET /api/user/{user_id}/notifications`
**Headers:** `Authorization: Bearer {token}`
**Response:**

```json
{
  "email": true,
  "peakDemand": true,
  "peakDemandThreshold": 8000,
  "modelReport": false,
  "sysUpdates": false
}
```

### 5. Update Notification Preferences

**Endpoint:** `PUT /api/user/{user_id}/notifications`
**Headers:**

```
Authorization: Bearer {token}
Content-Type: application/json
```

**Request Body:**

```json
{
  "email": true,
  "peakDemand": true,
  "peakDemandThreshold": 8500,
  "modelReport": false,
  "sysUpdates": false
}
```

**Response:** `200 OK` with updated preferences
**Validation:**

- `peakDemandThreshold` must be > 0
- All boolean fields must be valid booleans

## Peak Demand Alert System

### Real-time Monitoring

Implement a background job/service that:

1. **Monitors Demand Data**
   - Polls or subscribes to real-time energy demand data
   - Updates every minute or as data becomes available

2. **Check Thresholds**

   ```pseudocode
   FOR EACH user WITH peakDemand = true:
     IF current_demand > user.peakDemandThreshold:
       IF user.email = true:
         SEND_EMAIL(user.email, "Peak Demand Alert", ...)
       SEND_IN_APP_NOTIFICATION(user.id, "Demand exceeded...")
   ```

3. **Alert Management**
   - Lock alerts to prevent spam (e.g., one alert max per 15 minutes)
   - Store alert history for user review
   - Track which alerts have been received

### Alert History Table

```sql
CREATE TABLE alert_history (
  id INT PRIMARY KEY AUTO_INCREMENT,
  user_id INT NOT NULL,
  alert_type VARCHAR(50),
  message TEXT,
  data JSON,
  acknowledged BOOLEAN DEFAULT FALSE,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
  INDEX (user_id, created_at)
);
```

### Email Template

```
Subject: ⚠️ Peak Demand Alert

Hello {user_name},

Current energy demand has exceeded your configured threshold:

Configured Threshold: {threshold} MW
Current Demand: {current_demand} MW
Exceeded by: {overage} MW

Time: {timestamp}
Grid Region: {region}

View more details: [Link to dashboard]

---
You can adjust your threshold or disable these alerts in Settings.
```

## Implementation Checklist

- [ ] Add columns to users table
- [ ] Create notification_preferences table
- [ ] Create alert_history table
- [ ] Implement GET /api/user/{user_id}
- [ ] Implement PUT /api/user/{user_id}/profile
- [ ] Implement POST /api/user/{user_id}/profile-image (with image storage)
- [ ] Implement GET /api/user/{user_id}/notifications
- [ ] Implement PUT /api/user/{user_id}/notifications
- [ ] Create background job for peak demand monitoring
- [ ] Add email service integration
- [ ] Add in-app notification system
- [ ] Create alert history endpoint: GET /api/user/{user_id}/alerts
- [ ] Add error handling and validation
- [ ] Add rate limiting
- [ ] Add audit logging

## Testing Checklist

- [ ] Test profile updates persist
- [ ] Test image upload with various formats/sizes
- [ ] Test notification preferences save/load
- [ ] Test threshold validation
- [ ] Test peak demand alert trigger
- [ ] Test alert deduplication (no spam)
- [ ] Test email notifications
- [ ] Test authorization (users can only access their own data)
