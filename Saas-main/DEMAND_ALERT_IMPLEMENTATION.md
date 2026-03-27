# Peak Demand Alert System - Backend Implementation Guide

## Overview

The frontend now allows users to set a custom demand threshold. When energy demand in the grid exceeds this threshold, the user should receive a notification (if they have peak demand alerts enabled).

## Implementation Steps

### 1. Store Notification Preferences

When user saves notification preferences, store in database:

```python
# Example FastAPI endpoint
@app.put("/api/user/{user_id}/notifications")
async def update_notifications(
    user_id: int,
    prefs: NotificationPreferences,  # Has peakDemandThreshold
    current_user: User = Depends(get_current_user)
):
    # Verify user owns this data
    if user_id != current_user.id:
        raise HTTPException(status_code=403, detail="Forbidden")

    # Save to database
    db.update_notification_prefs(user_id, prefs)

    # IMPORTANT: If peakDemand was just enabled, trigger monitoring
    if prefs.peak_demand and not was_monitoring(user_id):
        start_monitoring_for_user(user_id)

    return prefs
```

### 2. Real-Time Demand Monitoring

Create a background service that runs continuously:

```python
import asyncio
from background_tasks import BackgroundTaskManager

class DemandAlertService:
    def __init__(self, db, email_service, notification_service):
        self.db = db
        self.email_service = email_service
        self.notification_service = notification_service
        self.alert_cooldown = {}  # user_id -> last_alert_time

    async def monitor_demand(self):
        """Run continuously to check demand against thresholds"""
        while True:
            try:
                # Get current demand (from your grid API/database)
                current_demand = await self.get_current_demand()

                # Get all users with peak demand alerts enabled
                users_to_check = self.db.get_users_with_peak_alerts()

                for user in users_to_check:
                    threshold = user.notification_prefs.peak_demand_threshold

                    # Check if demand exceeds threshold
                    if current_demand > threshold:
                        # Check cooldown (avoid spam)
                        if self.should_send_alert(user.id):
                            await self.send_alert(user, current_demand, threshold)
                            self.alert_cooldown[user.id] = datetime.now()

                # Check every minute
                await asyncio.sleep(60)

            except Exception as e:
                logger.error(f"Error in demand monitoring: {e}")
                await asyncio.sleep(60)

    def should_send_alert(self, user_id: int) -> bool:
        """Prevent alert spam - max one per 15 minutes"""
        last_alert = self.alert_cooldown.get(user_id)
        if not last_alert:
            return True

        time_since_alert = datetime.now() - last_alert
        return time_since_alert.total_seconds() > 900  # 15 minutes

    async def send_alert(self, user: User, current_demand: float, threshold: float):
        """Send notification to user"""
        prefs = user.notification_prefs

        alert_data = {
            "current_demand": current_demand,
            "threshold": threshold,
            "exceeded_by": current_demand - threshold,
            "timestamp": datetime.now().isoformat()
        }

        # Save alert history
        self.db.save_alert_history(
            user_id=user.id,
            alert_type="peak_demand",
            message=f"Demand {current_demand:.0f} MW exceeded threshold {threshold:.0f} MW",
            data=alert_data
        )

        # Send via email if enabled
        if prefs.email:
            await self.email_service.send_peak_demand_email(user, alert_data)

        # Send in-app notification
        await self.notification_service.send_notification(
            user_id=user.id,
            title="⚠️ Peak Demand Alert",
            message=f"Demand exceeded {threshold} MW ({current_demand:.0f} MW)",
            type="peak_demand",
            data=alert_data
        )
```

### 3. Start the Service on App Startup

```python
from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    demand_service = DemandAlertService(db, email_service, notification_service)

    # Run monitoring in background
    task = asyncio.create_task(demand_service.monitor_demand())

    yield

    # Shutdown
    task.cancel()

app = FastAPI(lifespan=lifespan)
```

### 4. Email Template

```python
async def send_peak_demand_email(self, user: User, alert_data: dict):
    """Send peak demand alert email"""

    subject = "⚠️ Peak Demand Alert - Action Required"

    html_content = f"""
    <html>
        <body style="font-family: Arial, sans-serif;">
            <div style="max-width: 600px; margin: 0 auto;">
                <h2 style="color: #ef4444;">Peak Demand Alert</h2>

                <p>Hello {user.name},</p>

                <p>Current energy demand has exceeded your configured threshold:</p>

                <div style="background-color: #f3f4f6; padding: 20px; border-radius: 8px; margin: 20px 0;">
                    <p style="margin: 10px 0;"><strong>Your Threshold:</strong> {alert_data['threshold']:.0f} MW</p>
                    <p style="margin: 10px 0;"><strong>Current Demand:</strong> <span style="color: #ef4444; font-size: 24px; font-weight: bold;">{alert_data['current_demand']:.0f} MW</span></p>
                    <p style="margin: 10px 0;"><strong>Exceeded By:</strong> {alert_data['exceeded_by']:.0f} MW</p>
                    <p style="margin: 10px 0;"><strong>Time:</strong> {alert_data['timestamp']}</p>
                </div>

                <p>
                    <a href="https://yourapp.com/dashboard"
                       style="background-color: #3b82f6; color: white; padding: 10px 20px;
                              text-decoration: none; border-radius: 5px; display: inline-block;">
                        View Dashboard
                    </a>
                </p>

                <hr style="margin: 30px 0;"/>

                <p style="color: #6b7280; font-size: 12px;">
                    You can adjust your threshold or disable these alerts in
                    <a href="https://yourapp.com/settings">Settings</a>.
                </p>
            </div>
        </body>
    </html>
    """

    await self.send_email(
        to=user.email,
        subject=subject,
        html_content=html_content
    )
```

### 5. Database Queries

```python
# Get users with peak demand alerts enabled
def get_users_with_peak_alerts():
    """
    SELECT u.* FROM users u
    JOIN notification_preferences np ON u.id = np.user_id
    WHERE np.peak_demand = true
    """
    pass

# Save alert history
def save_alert_history(user_id: int, alert_type: str, message: str, data: dict):
    """
    INSERT INTO alert_history
    (user_id, alert_type, message, data, created_at)
    VALUES (?, ?, ?, ?, NOW())
    """
    pass

# Get notification preferences
def get_notification_preferences(user_id: int):
    """
    SELECT * FROM notification_preferences
    WHERE user_id = ?
    """
    pass
```

### 6. Optional: Alert History Endpoint

Let users see past alerts:

```python
@app.get("/api/user/{user_id}/alerts")
async def get_alerts(
    user_id: int,
    limit: int = 20,
    current_user: User = Depends(get_current_user)
):
    if user_id != current_user.id:
        raise HTTPException(status_code=403)

    alerts = db.get_alerts(user_id, limit=limit)
    return alerts
```

## Key Considerations

1. **Timezone Handling**: Store timestamps in UTC
2. **Cooldown Period**: Prevent alert spam (15-30 min cooldown)
3. **Demand Data Source**:
   - Real-time API from grid operator
   - Database table with latest values
   - WebSocket connection for live data
4. **Performance**: Index user_id in notification_preferences and alert_history
5. **Scalability**: For many users, consider:
   - Batch processing
   - Caching user thresholds
   - Distributed task queue (Celery, RabbitMQ)
6. **Testing**: Mock demand data for testing without real grid data

## Testing the System

```python
# Test: Verify alert is sent when threshold exceeded
async def test_demand_alert():
    # Setup
    user = create_test_user()
    set_notification_preference(user.id, peak_demand=True, threshold=8000)

    # Simulate demand
    demand_service.set_mock_demand(8500)  # Higher than threshold

    # Wait for monitoring cycle
    await asyncio.sleep(61)

    # Verify
    alerts = db.get_alerts(user.id)
    assert len(alerts) > 0
    assert "Peak Demand" in alerts[0].message
```
