"""
Tests for notification system and user management endpoints
"""

import pytest
import asyncio
from datetime import datetime
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session

from main import app
from app.database import Base, get_db
from app.models import User, NotificationPreference, AlertHistory
from app.schemas import NotificationPreferenceUpdate, ProfileUpdate
from app.security import create_access_token
from app.services import get_email_service, get_demand_alert_service


# Use in-memory SQLite for testing
SQLALCHEMY_DATABASE_URL = "sqlite:///:memory:"
engine = create_engine(SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False})
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def override_get_db():
    try:
        db = TestingSessionLocal()
        yield db
    finally:
        db.close()


@pytest.fixture()
def db():
    """Create a fresh database for each test"""
    Base.metadata.create_all(bind=engine)
    yield TestingSessionLocal()
    Base.metadata.drop_all(bind=engine)


@pytest.fixture()
def client(db):
    """Create test client with overridden database"""
    app.dependency_overrides[get_db] = override_get_db
    yield TestClient(app)
    app.dependency_overrides.clear()


@pytest.fixture()
def test_user(db):
    """Create a test user"""
    user = User(
        id=1,
        name="Test User",
        email="test@example.com",
        hashed_password="hashed_pwd",
        role="energy_grid_operator",
        is_active=True,
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


@pytest.fixture()
def test_token(test_user):
    """Create access token for test user"""
    return create_access_token(data={"sub": test_user.email})


class TestProfileEndpoints:
    """Test user profile management endpoints"""

    def test_get_user_profile(self, client, test_user, test_token):
        """Test fetching user profile"""
        response = client.get(
            f"/api/user/{test_user.id}",
            headers={"Authorization": f"Bearer {test_token}"}
        )
        assert response.status_code == 200
        data = response.json()
        assert data["email"] == test_user.email
        assert data["name"] == test_user.name
        assert data["role"] == test_user.role

    def test_update_profile(self, client, test_user, test_token):
        """Test updating user profile"""
        update_data = {
            "name": "Updated Name",
            "email": "newemail@example.com"
        }
        response = client.put(
            f"/api/user/{test_user.id}/profile",
            json=update_data,
            headers={"Authorization": f"Bearer {test_token}"}
        )
        assert response.status_code == 200
        data = response.json()
        assert data["name"] == "Updated Name"
        assert data["email"] == "newemail@example.com"

    def test_get_user_forbidden(self, client, test_user, test_token):
        """Test that users cannot access other users' profiles"""
        response = client.get(
            f"/api/user/999",
            headers={"Authorization": f"Bearer {test_token}"}
        )
        assert response.status_code == 403


class TestNotificationPreferences:
    """Test notification preferences management"""

    def test_get_default_preferences(self, client, test_user, test_token):
        """Test getting default notification preferences"""
        response = client.get(
            f"/api/user/{test_user.id}/notifications",
            headers={"Authorization": f"Bearer {test_token}"}
        )
        assert response.status_code == 200
        data = response.json()
        assert data["email"] is True
        assert data["peak_demand"] is True
        assert data["peak_demand_threshold"] == 8000
        assert data["model_report"] is False
        assert data["sys_updates"] is False

    def test_update_preferences(self, client, test_user, test_token, db):
        """Test updating notification preferences"""
        # Create initial preferences
        prefs = NotificationPreference(
            user_id=test_user.id,
            email=True,
            peak_demand=True,
            peak_demand_threshold=8000,
        )
        db.add(prefs)
        db.commit()

        # Update preferences
        update_data = {
            "email": False,
            "peak_demand": True,
            "peak_demand_threshold": 9000,
            "model_report": True,
            "sys_updates": False,
        }
        response = client.put(
            f"/api/user/{test_user.id}/notifications",
            json=update_data,
            headers={"Authorization": f"Bearer {test_token}"}
        )
        assert response.status_code == 200
        data = response.json()
        assert data["email"] is False
        assert data["peak_demand_threshold"] == 9000
        assert data["model_report"] is True

    def test_threshold_validation(self, client, test_user, test_token):
        """Test that threshold must be positive"""
        invalid_data = {
            "email": True,
            "peak_demand": True,
            "peak_demand_threshold": -100,  # Invalid negative threshold
            "model_report": False,
            "sys_updates": False,
        }
        response = client.put(
            f"/api/user/{test_user.id}/notifications",
            json=invalid_data,
            headers={"Authorization": f"Bearer {test_token}"}
        )
        # Pydantic should reject negative values
        assert response.status_code == 422


class TestAlertHistory:
    """Test alert history endpoints"""

    def test_get_alerts_empty(self, client, test_user, test_token):
        """Test getting alerts when none exist"""
        response = client.get(
            f"/api/user/{test_user.id}/alerts",
            headers={"Authorization": f"Bearer {test_token}"}
        )
        assert response.status_code == 200
        data = response.json()
        assert len(data) == 0

    def test_get_alerts_with_data(self, client, test_user, test_token, db):
        """Test getting alerts"""
        # Create some test alerts
        for i in range(3):
            alert = AlertHistory(
                user_id=test_user.id,
                alert_type="peak_demand",
                message=f"Test alert {i}",
                data={"demand": 8000 + i*100}
            )
            db.add(alert)
        db.commit()

        response = client.get(
            f"/api/user/{test_user.id}/alerts",
            headers={"Authorization": f"Bearer {test_token}"}
        )
        assert response.status_code == 200
        data = response.json()
        assert len(data) == 3

    def test_create_internal_alert(self, client, test_user, db):
        """Test creating an alert (internal endpoint)"""
        alert_data = {
            "alert_type": "peak_demand",
            "message": "Demand exceeded 8000 MW",
            "data": {"current_demand": 8500, "threshold": 8000}
        }
        response = client.post(
            f"/api/user/{test_user.id}/alerts",
            json=alert_data
        )
        assert response.status_code == 200
        data = response.json()
        assert data["alert_type"] == "peak_demand"
        assert data["message"] == "Demand exceeded 8000 MW"


class TestEmailService:
    """Test email service functionality"""

    @pytest.mark.asyncio
    async def test_email_service_console(self):
        """Test email service with console provider"""
        email_service = get_email_service(provider="console")
        
        result = await email_service.send_email(
            to_email="test@example.com",
            subject="Test Email",
            html_content="<h1>Test</h1>",
        )
        assert result is True

    @pytest.mark.asyncio
    async def test_peak_demand_email(self, test_user):
        """Test peak demand alert email"""
        email_service = get_email_service(provider="console")
        
        result = await email_service.send_peak_demand_alert(
            user=test_user,
            current_demand=9000,
            threshold=8000,
        )
        assert result is True


class TestDemandAlertService:
    """Test demand alert service"""

    @pytest.mark.asyncio
    async def test_demand_alert_service_init(self):
        """Test initializing demand alert service"""
        email_service = get_email_service(provider="console")
        alert_service = get_demand_alert_service(email_service)
        
        assert alert_service is not None
        assert alert_service.cooldown_period.total_seconds() == 900

    @pytest.mark.asyncio
    async def test_should_send_alert_first_time(self):
        """Test alert is sent on first occurrence"""
        alert_service = get_demand_alert_service()
        
        # First alert should always be sent
        assert alert_service.should_send_alert(user_id=1)

    @pytest.mark.asyncio
    async def test_should_send_alert_cooldown(self):
        """Test cooldown prevents spam"""
        alert_service = get_demand_alert_service()
        user_id = 1
        
        # First alert should be sent
        assert alert_service.should_send_alert(user_id)
        alert_service.alert_cooldown[user_id] = datetime.utcnow()
        
        # Subsequent alert should be blocked
        assert not alert_service.should_send_alert(user_id)

    @pytest.mark.asyncio
    async def test_set_current_demand(self):
        """Test setting current demand value"""
        alert_service = get_demand_alert_service()
        alert_service.set_current_demand(8500)
        assert alert_service.current_demand == 8500


class TestIntegration:
    """Integration tests for the complete notification flow"""

    def test_complete_notification_flow(self, client, test_user, test_token, db):
        """Test complete flow: set preferences -> create alert -> retrieve history"""
        
        # 1. Set notification preferences with threshold
        prefs_data = {
            "email": True,
            "peak_demand": True,
            "peak_demand_threshold": 8000,
            "model_report": False,
            "sys_updates": False,
        }
        response = client.put(
            f"/api/user/{test_user.id}/notifications",
            json=prefs_data,
            headers={"Authorization": f"Bearer {test_token}"}
        )
        assert response.status_code == 200
        
        # 2. Create an alert
        alert_data = {
            "alert_type": "peak_demand",
            "message": "Demand exceeded 8000 MW",
            "data": {"current_demand": 8500, "threshold": 8000, "exceeded_by": 500}
        }
        response = client.post(
            f"/api/user/{test_user.id}/alerts",
            json=alert_data
        )
        assert response.status_code == 200
        
        # 3. Retrieve alert history
        response = client.get(
            f"/api/user/{test_user.id}/alerts",
            headers={"Authorization": f"Bearer {test_token}"}
        )
        assert response.status_code == 200
        alerts = response.json()
        assert len(alerts) == 1
        assert alerts[0]["alert_type"] == "peak_demand"
        assert alerts[0]["acknowledged"] is False


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
