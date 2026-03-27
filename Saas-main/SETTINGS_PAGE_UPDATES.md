# SettingsPage Updates - Changes Summary

## ✅ Issues Fixed

### 1. Username Display Issue

**Problem:** Username wasn't appearing in profile settings  
**Root Cause:** Component used `user?.username` but the User interface defines `user?.name`  
**Fix:** Changed all references to use `user?.name` throughout the component

### 2. Profile Image Selection

**Problem:** No way to upload/select a profile image  
**Fix:**

- Added upload button (icon overlay on avatar)
- Displays uploaded image or initials gradient
- Calls `POST /api/user/{user_id}/profile-image` to upload
- Handles loading state during upload

### 3. Notification System (API-Ready)

**Problem:** Notifications were local state only, not persisted  
**Fix:**

- Integrated with backend API for loading/saving preferences
- Calls `GET /api/user/{user_id}/notifications` on mount
- Calls `PUT /api/user/{user_id}/notifications` when saving
- Added error handling for API failures

### 4. Peak Demand Alert Threshold

**Problem:** No way for users to configure demand threshold  
**Fix:**

- Added configurable threshold input (MW)
- Shows conditionally when "Peak Demand Alerts" is enabled
- Threshold is saved to backend with notification preferences
- Backend can use this value to trigger alerts when demand exceeds it

## 📝 Code Changes Made

### SettingsPage.tsx

- ✅ Added `useEffect` hook to load notification preferences on mount
- ✅ Added profile image upload handler with API integration
- ✅ Converted notification toggles to controlled inputs backed by API
- ✅ Added threshold input for peak demand with validation
- ✅ Fixed username display to use `name` field
- ✅ Added save buttons with success feedback
- ✅ Improved initials calculation (handles full names correctly)
- ✅ Added token retrieval for API authentication

### AuthContext.tsx

- ✅ Added `profile_image?: string` field to User interface

## 🔧 Backend Implementation Needed

To make this fully functional, implement these API endpoints:

### Profile Endpoints

```
PUT /api/user/{user_id}/profile
Updates user name and email

POST /api/user/{user_id}/profile-image
Uploads profile image (multipart/form-data)
```

### Notification Endpoints

```
GET /api/user/{user_id}/notifications
Returns notification preferences with threshold

PUT /api/user/{user_id}/notifications
Saves notification preferences including peakDemandThreshold
```

### Peak Demand Alert System

Background job that:

- Monitors real-time energy demand
- Checks against each user's `peakDemandThreshold`
- Sends notifications when exceeded
- Respects user's email notification preference

See `BACKEND_API_REQUIREMENTS.md` for complete technical specifications.

## 🚀 Features Now Available in UI

✅ Edit user name (displays correctly)
✅ Edit user email
✅ Upload and display custom profile image
✅ Toggle email notifications
✅ Toggle peak demand alerts
✅ **SET CUSTOM DEMAND THRESHOLD** (configurable MW value)
✅ Toggle model performance reports
✅ Toggle system update notifications
✅ Save/load preferences from API
✅ Visual feedback on successful saves

## 💾 What Gets Saved

When user clicks "Save Preferences":

```json
{
  "email": true,
  "peakDemand": true,
  "peakDemandThreshold": 8500,
  "modelReport": false,
  "sysUpdates": false
}
```

This data is sent to: `PUT /api/user/{user_id}/notifications`
