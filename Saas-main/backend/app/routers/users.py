"""
User profile and notification management endpoints
"""

import os
import shutil
from pathlib import Path
from datetime import datetime
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, status
from sqlalchemy.orm import Session

from app.database import get_db
from app.models import User, NotificationPreference, AlertHistory
from app.schemas import (
    UserResponse,
    ProfileUpdate,
    NotificationPreferenceUpdate,
    NotificationPreferenceResponse,
    AlertHistoryResponse,
)
from app.security import get_current_user

router = APIRouter()

# Configuration for file uploads
UPLOAD_DIR = Path("uploads/profiles")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
MAX_FILE_SIZE = 5 * 1024 * 1024  # 5MB


def validate_image_file(filename: str) -> bool:
    """Validate image file extension"""
    ext = Path(filename).suffix.lower()
    return ext in ALLOWED_EXTENSIONS


# ──────────────────────────────────────────────────────────────────────────────
# GET / FETCH ENDPOINTS
# ──────────────────────────────────────────────────────────────────────────────


@router.get("/{user_id}", response_model=UserResponse)
async def get_user(
    user_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Get user profile information"""
    # Users can only access their own profile
    if user_id != current_user.id:
        raise HTTPException(status_code=403, detail="Not authorized to access this user's profile")

    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    return user


@router.get("/{user_id}/notifications", response_model=NotificationPreferenceResponse)
async def get_notification_preferences(
    user_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    Get user's notification preferences.
    Creates default preferences if they don't exist.
    """
    # Users can only access their own preferences
    if user_id != current_user.id:
        raise HTTPException(
            status_code=403, detail="Not authorized to access this user's preferences"
        )

    # Get or create notification preferences
    prefs = db.query(NotificationPreference).filter(
        NotificationPreference.user_id == user_id
    ).first()

    if not prefs:
        # Create default preferences
        prefs = NotificationPreference(
            user_id=user_id,
            email=True,
            peak_demand=True,
            peak_demand_threshold=8000,
            model_report=False,
            sys_updates=False,
        )
        db.add(prefs)
        db.commit()
        db.refresh(prefs)

    return prefs


@router.get("/{user_id}/alerts", response_model=list[AlertHistoryResponse])
async def get_alerts(
    user_id: int,
    limit: int = 20,
    offset: int = 0,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Get user's alert history"""
    # Users can only access their own alerts
    if user_id != current_user.id:
        raise HTTPException(status_code=403, detail="Not authorized to access this user's alerts")

    alerts = (
        db.query(AlertHistory)
        .filter(AlertHistory.user_id == user_id)
        .order_by(AlertHistory.created_at.desc())
        .offset(offset)
        .limit(limit)
        .all()
    )

    return alerts


# ──────────────────────────────────────────────────────────────────────────────
# UPDATE ENDPOINTS
# ──────────────────────────────────────────────────────────────────────────────


@router.put("/{user_id}/profile", response_model=UserResponse)
async def update_profile(
    user_id: int,
    profile_update: ProfileUpdate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Update user profile (name and/or email)"""
    # Users can only update their own profile
    if user_id != current_user.id:
        raise HTTPException(status_code=403, detail="Not authorized to update this user's profile")

    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    # Update fields if provided
    if profile_update.name is not None:
        user.name = profile_update.name

    if profile_update.email is not None:
        # Check if email already exists for another user
        existing_user = db.query(User).filter(
            User.email == profile_update.email,
            User.id != user_id
        ).first()
        if existing_user:
            raise HTTPException(status_code=400, detail="Email already in use")
        user.email = profile_update.email

    user.updated_at = datetime.utcnow()
    db.commit()
    db.refresh(user)

    return user


@router.put("/{user_id}/notifications", response_model=NotificationPreferenceResponse)
async def update_notification_preferences(
    user_id: int,
    prefs_update: NotificationPreferenceUpdate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Update user's notification preferences"""
    # Users can only update their own preferences
    if user_id != current_user.id:
        raise HTTPException(
            status_code=403, detail="Not authorized to update this user's preferences"
        )

    # Get or create preferences
    prefs = db.query(NotificationPreference).filter(
        NotificationPreference.user_id == user_id
    ).first()

    if not prefs:
        prefs = NotificationPreference(user_id=user_id)
        db.add(prefs)

    # Update preferences
    prefs.email = prefs_update.email
    prefs.peak_demand = prefs_update.peak_demand
    prefs.peak_demand_threshold = prefs_update.peak_demand_threshold
    prefs.model_report = prefs_update.model_report
    prefs.sys_updates = prefs_update.sys_updates
    prefs.updated_at = datetime.utcnow()

    db.commit()
    db.refresh(prefs)

    return prefs


# ──────────────────────────────────────────────────────────────────────────────
# FILE UPLOAD ENDPOINTS
# ──────────────────────────────────────────────────────────────────────────────


@router.post("/{user_id}/profile-image")
async def upload_profile_image(
    user_id: int,
    image: UploadFile = File(...),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Upload user profile image"""
    # Users can only upload their own profile image
    if user_id != current_user.id:
        raise HTTPException(
            status_code=403, detail="Not authorized to upload this user's profile image"
        )

    # Validate file
    if not image.filename:
        raise HTTPException(status_code=400, detail="No filename provided")

    if not validate_image_file(image.filename):
        raise HTTPException(
            status_code=400,
            detail="Invalid file type. Allowed: jpg, jpeg, png, webp"
        )

    # Check file size
    contents = await image.read()
    if len(contents) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Maximum size: {MAX_FILE_SIZE / 1024 / 1024}MB"
        )

    # Get user
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    # Delete old profile image if it exists
    if user.profile_image:
        try:
            old_path = Path(user.profile_image)
            if old_path.exists():
                old_path.unlink()
        except Exception as e:
            print(f"Warning: Could not delete old profile image: {e}")

    # Save new image
    try:
        # Generate unique filename
        file_ext = Path(image.filename).suffix.lower()
        timestamp = datetime.utcnow().timestamp()
        new_filename = f"user_{user_id}_{timestamp}{file_ext}"
        file_path = UPLOAD_DIR / new_filename

        # Save file
        with open(file_path, "wb") as f:
            f.write(contents)

        # Update user profile_image path
        user.profile_image = str(file_path)
        user.updated_at = datetime.utcnow()
        db.commit()
        db.refresh(user)

        return {
            "profile_image": str(file_path),
            "message": "Profile image uploaded successfully"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error uploading file: {str(e)}")


@router.delete("/{user_id}/profile-image")
async def delete_profile_image(
    user_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Delete user profile image"""
    # Users can only delete their own profile image
    if user_id != current_user.id:
        raise HTTPException(
            status_code=403, detail="Not authorized to delete this user's profile image"
        )

    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    if not user.profile_image:
        raise HTTPException(status_code=404, detail="No profile image to delete")

    try:
        # Delete file
        file_path = Path(user.profile_image)
        if file_path.exists():
            file_path.unlink()

        # Update user
        user.profile_image = None
        user.updated_at = datetime.utcnow()
        db.commit()

        return {"message": "Profile image deleted successfully"}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error deleting file: {str(e)}")


# ──────────────────────────────────────────────────────────────────────────────
# INTERNAL ENDPOINTS (for services/scheduler)
# ──────────────────────────────────────────────────────────────────────────────


@router.post("/{user_id}/alerts", response_model=AlertHistoryResponse)
async def create_alert(
    user_id: int,
    alert_type: str,
    message: str,
    data: Optional[dict] = None,
    db: Session = Depends(get_db),
):
    """
    Create an alert for a user.
    This endpoint is for internal use by monitoring services.
    Should be protected with API key in production.
    """
    # Verify user exists
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    # Create alert
    alert = AlertHistory(
        user_id=user_id,
        alert_type=alert_type,
        message=message,
        data=data,
    )

    db.add(alert)
    db.commit()
    db.refresh(alert)

    return alert


@router.get("/{user_id}/alerts")
async def get_alerts(
    user_id: int,
    limit: int = 10,
    offset: int = 0,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Get user's alert history"""
    # Users can only access their own alerts
    if user_id != current_user.id:
        raise HTTPException(
            status_code=403, detail="Not authorized to access this user's alerts"
        )

    # Get alerts
    alerts = (
        db.query(AlertHistory)
        .filter(AlertHistory.user_id == user_id)
        .order_by(AlertHistory.created_at.desc())
        .offset(offset)
        .limit(limit)
        .all()
    )

    total = db.query(AlertHistory).filter(AlertHistory.user_id == user_id).count()

    return {
        "alerts": [AlertHistoryResponse.from_orm(a) for a in alerts],
        "total": total,
        "limit": limit,
        "offset": offset,
    }


@router.put("/{user_id}/alerts/{alert_id}/read")
async def mark_alert_as_read(
    user_id: int,
    alert_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Mark an alert as read/acknowledged"""
    # Users can only update their own alerts
    if user_id != current_user.id:
        raise HTTPException(
            status_code=403, detail="Not authorized to update this user's alerts"
        )

    # Get alert
    alert = (
        db.query(AlertHistory)
        .filter(AlertHistory.id == alert_id, AlertHistory.user_id == user_id)
        .first()
    )
    if not alert:
        raise HTTPException(status_code=404, detail="Alert not found")

    # Mark as read
    alert.acknowledged = True
    db.commit()
    db.refresh(alert)

    return {"message": "Alert marked as read"}
