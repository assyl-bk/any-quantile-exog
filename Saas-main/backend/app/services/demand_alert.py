"""
Background service for monitoring energy demand and sending alerts to users
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, Optional

from sqlalchemy.orm import Session
from app.database import SessionLocal
from app.models import NotificationPreference, AlertHistory, User
from app.services.email_service import EmailService

logger = logging.getLogger(__name__)


class DemandAlertService:
    """
    Background service that monitors energy demand and sends alerts when
    demand exceeds user-configured thresholds.
    """

    def __init__(self, email_service: Optional[EmailService] = None):
        self.email_service = email_service
        self.alert_cooldown: Dict[int, datetime] = {}  # user_id -> last_alert_time
        self.cooldown_period = timedelta(minutes=15)  # Don't send more than once per 15 min
        self.check_interval = 60  # Check every 60 seconds
        self.current_demand: float = 0.0  # This would come from real-time API/DB
        self.is_running = False

    def set_current_demand(self, demand: float) -> None:
        """Set the current demand value (called by monitoring system)"""
        self.current_demand = demand

    def should_send_alert(self, user_id: int) -> bool:
        """Check if enough time has passed since the last alert for this user"""
        if user_id not in self.alert_cooldown:
            return True

        last_alert_time = self.alert_cooldown[user_id]
        time_since_alert = datetime.utcnow() - last_alert_time

        return time_since_alert >= self.cooldown_period

    async def send_alert(
        self,
        db: Session,
        user: User,
        current_demand: float,
        threshold: float,
    ) -> bool:
        """
        Send alert to user and save to history.
        Returns True if alert was sent, False otherwise.
        """
        try:
            prefs = user.notification_preferences
            if not prefs:
                return False

            alert_data = {
                "current_demand": current_demand,
                "threshold": threshold,
                "exceeded_by": round(current_demand - threshold, 2),
                "timestamp": datetime.utcnow().isoformat(),
            }

            message = (
                f"Peak demand alert: Demand {current_demand:.0f} MW exceeded "
                f"your threshold of {threshold:.0f} MW"
            )

            # Save to alert history
            alert = AlertHistory(
                user_id=user.id,
                alert_type="peak_demand",
                message=message,
                data=alert_data,
            )
            db.add(alert)
            db.commit()

            logger.info(f"Alert created for user {user.id}: {message}")

            # Send email if enabled
            if prefs.email and self.email_service:
                try:
                    await self.email_service.send_peak_demand_alert(
                        user, current_demand, threshold
                    )
                    logger.info(f"Email alert sent to {user.email}")
                except Exception as e:
                    logger.error(f"Failed to send email to {user.email}: {e}")

            # Update cooldown
            self.alert_cooldown[user.id] = datetime.utcnow()

            return True

        except Exception as e:
            logger.error(f"Error sending alert to user {user.id}: {e}")
            return False

    async def check_demand_threshold(self, current_demand: Optional[float] = None) -> None:
        """
        Check if current demand exceeds any user's threshold.
        Send alerts to users who need them.
        """
        if current_demand is None:
            current_demand = self.current_demand

        db = SessionLocal()
        try:
            # Get all users with peak demand alerts enabled
            users_to_check = (
                db.query(User)
                .join(NotificationPreference)
                .filter(NotificationPreference.peak_demand == True)
                .all()
            )

            for user in users_to_check:
                prefs = user.notification_preferences
                threshold = prefs.peak_demand_threshold

                # Check if demand exceeds threshold
                if current_demand > threshold:
                    # Check cooldown to avoid spam
                    if self.should_send_alert(user.id):
                        await self.send_alert(db, user, current_demand, threshold)

        except Exception as e:
            logger.error(f"Error checking demand thresholds: {e}")
        finally:
            db.close()

    async def get_current_demand_from_source(self) -> float:
        """
        Fetch current demand from the real-time data source.
        This is a placeholder - implement based on your actual data source.

        Could be:
        - SQL query to get latest demand from database
        - API call to grid operator
        - WebSocket connection to live data
        """
        # TODO: Implement actual demand data retrieval
        # For now, return the set value or a default
        return self.current_demand

    async def monitor_demand(self) -> None:
        """
        Main monitoring loop. Runs continuously to check demand against thresholds.
        """
        self.is_running = True
        logger.info("🚀 Demand Alert Service started")

        while self.is_running:
            try:
                # Get current demand from data source
                demand = await self.get_current_demand_from_source()

                # Check thresholds and send alerts if needed
                await self.check_demand_threshold(demand)

                # Wait before next check
                await asyncio.sleep(self.check_interval)

            except Exception as e:
                logger.error(f"Error in demand monitoring loop: {e}")
                # Wait before retrying on error
                await asyncio.sleep(self.check_interval)

    def stop(self) -> None:
        """Stop the monitoring service"""
        self.is_running = False
        logger.info("🛑 Demand Alert Service stopped")


# Global instance
_demand_alert_service: Optional[DemandAlertService] = None


def get_demand_alert_service(
    email_service: Optional[EmailService] = None,
) -> DemandAlertService:
    """Get or create the demand alert service (singleton)"""
    global _demand_alert_service

    if _demand_alert_service is None:
        _demand_alert_service = DemandAlertService(email_service)

    return _demand_alert_service
