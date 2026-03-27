"""
Email notification service
"""

import logging
from typing import Optional
from datetime import datetime

from app.models import User

logger = logging.getLogger(__name__)


class EmailService:
    """
    Service for sending email notifications.
    Can be configured to use SMTP, SendGrid, AWS SES, or other providers.
    """

    def __init__(self, provider: str = "console"):
        """
        Initialize email service.

        Args:
            provider: Email provider type ('console', 'smtp', 'sendgrid', 'ses')
        """
        self.provider = provider

    async def send_email(
        self,
        to_email: str,
        subject: str,
        html_content: str,
        text_content: Optional[str] = None,
    ) -> bool:
        """
        Send email using configured provider.

        Args:
            to_email: Recipient email address
            subject: Email subject
            html_content: HTML email body
            text_content: Plain text email body (optional)

        Returns:
            True if email was sent successfully, False otherwise
        """
        try:
            if self.provider == "console":
                logger.info(f"📧 EMAIL TO: {to_email}")
                logger.info(f"📧 SUBJECT: {subject}")
                logger.info(f"📧 BODY: {html_content[:200]}...")
                return True

            elif self.provider == "smtp":
                return await self._send_smtp(to_email, subject, html_content, text_content)

            elif self.provider == "sendgrid":
                return await self._send_sendgrid(to_email, subject, html_content, text_content)

            elif self.provider == "ses":
                return await self._send_ses(to_email, subject, html_content, text_content)

            else:
                logger.error(f"Unknown email provider: {self.provider}")
                return False

        except Exception as e:
            logger.error(f"Error sending email to {to_email}: {e}")
            return False

    async def _send_smtp(
        self,
        to_email: str,
        subject: str,
        html_content: str,
        text_content: Optional[str],
    ) -> bool:
        """Send email via SMTP (requires aiosmtplib)"""
        try:
            import aiosmtplib
            from email.mime.text import MIMEText
            from email.mime.multipart import MIMEMultipart

            msg = MIMEMultipart("alternative")
            msg["Subject"] = subject
            msg["From"] = "noreply@energyforecast.com"
            msg["To"] = to_email

            if text_content:
                part1 = MIMEText(text_content, "plain")
                msg.attach(part1)

            part2 = MIMEText(html_content, "html")
            msg.attach(part2)

            # Configure these in environment variables
            smtp_host = "smtp.gmail.com"  # Change based on provider
            smtp_port = 587

            async with aiosmtplib.SMTP(hostname=smtp_host, port=smtp_port) as smtp:
                await smtp.login("your_email@gmail.com", "your_app_password")
                await smtp.send_message(msg)

            logger.info(f"✅ Email sent to {to_email} via SMTP")
            return True

        except Exception as e:
            logger.error(f"SMTP error: {e}")
            return False

    async def _send_sendgrid(
        self,
        to_email: str,
        subject: str,
        html_content: str,
        text_content: Optional[str],
    ) -> bool:
        """Send email via SendGrid API (requires sendgrid package)"""
        try:
            from sendgrid import SendGridAPIClient
            from sendgrid.helpers.mail import Mail

            # Get API key from environment
            import os
            sg_key = os.getenv("SENDGRID_API_KEY")
            if not sg_key:
                logger.error("SENDGRID_API_KEY not configured")
                return False

            message = Mail(
                from_email="noreply@energyforecast.com",
                to_emails=to_email,
                subject=subject,
                plain_text_content=text_content or "",
                html_content=html_content,
            )

            sg = SendGridAPIClient(sg_key)
            response = sg.send(message)

            logger.info(f"✅ Email sent to {to_email} via SendGrid (status: {response.status_code})")
            return response.status_code in [200, 201, 202]

        except Exception as e:
            logger.error(f"SendGrid error: {e}")
            return False

    async def _send_ses(
        self,
        to_email: str,
        subject: str,
        html_content: str,
        text_content: Optional[str],
    ) -> bool:
        """Send email via AWS SES (requires boto3)"""
        try:
            import aioboto3

            session = aioboto3.Session()

            async with session.client("ses", region_name="us-east-1") as client:
                await client.send_email(
                    Source="noreply@energyforecast.com",
                    Destination={"ToAddresses": [to_email]},
                    Message={
                        "Subject": {"Data": subject},
                        "Body": {
                            "Html": {"Data": html_content},
                            "Text": {"Data": text_content or ""},
                        },
                    },
                )

            logger.info(f"✅ Email sent to {to_email} via AWS SES")
            return True

        except Exception as e:
            logger.error(f"AWS SES error: {e}")
            return False

    async def send_peak_demand_alert(
        self,
        user: User,
        current_demand: float,
        threshold: float,
    ) -> bool:
        """Send peak demand alert email to user"""

        exceeded_by = current_demand - threshold

        html_content = f"""
        <html>
            <body style="font-family: Arial, sans-serif; background-color: #f5f5f5;">
                <div style="max-width: 600px; margin: 0 auto; background-color: white; padding: 30px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
                    <h2 style="color: #ef4444; margin-top: 0;">⚠️ Peak Demand Alert</h2>

                    <p style="color: #374151; line-height: 1.6;">
                        Hello <strong>{user.name}</strong>,
                    </p>

                    <p style="color: #374151; line-height: 1.6;">
                        Current energy demand has exceeded your configured threshold:
                    </p>

                    <div style="background-color: #f9fafb; padding: 20px; border-left: 4px solid #ef4444; margin: 20px 0; border-radius: 4px;">
                        <p style="margin: 10px 0; color: #374151;">
                            <strong>Your Threshold:</strong> {threshold:.0f} MW
                        </p>
                        <p style="margin: 10px 0; color: #374151;">
                            <strong>Current Demand:</strong>
                            <span style="font-size: 24px; color: #ef4444; font-weight: bold;">
                                {current_demand:.0f} MW
                            </span>
                        </p>
                        <p style="margin: 10px 0; color: #374151;">
                            <strong>Exceeded By:</strong> <span style="color: #ef4444;">{exceeded_by:.0f} MW</span>
                        </p>
                        <p style="margin: 10px 0; color: #6b7280; font-size: 12px;">
                            <strong>Time:</strong> {datetime.utcnow().isoformat()}
                        </p>
                    </div>

                    <p style="margin-top: 30px;">
                        <a href="http://localhost:3000/dashboard" 
                           style="display: inline-block; background-color: #3b82f6; color: white; padding: 12px 24px; text-decoration: none; border-radius: 6px; font-weight: bold;">
                            View Dashboard
                        </a>
                    </p>

                    <hr style="margin: 30px 0; border: none; border-top: 1px solid #e5e7eb;">

                    <p style="color: #6b7280; font-size: 12px; margin-bottom: 0;">
                        You can adjust your threshold or disable these alerts in
                        <a href="http://localhost:3000/settings" style="color: #3b82f6; text-decoration: none;">Settings</a>.
                    </p>
                </div>
            </body>
        </html>
        """

        text_content = f"""
        Peak Demand Alert

        Hello {user.name},

        Current energy demand has exceeded your configured threshold:

        Your Threshold: {threshold:.0f} MW
        Current Demand: {current_demand:.0f} MW
        Exceeded By: {exceeded_by:.0f} MW
        Time: {datetime.utcnow().isoformat()}

        View Dashboard: http://localhost:3000/dashboard

        You can adjust your threshold or disable these alerts in Settings:
        http://localhost:3000/settings
        """

        return await self.send_email(
            to_email=user.email,
            subject=f"⚠️ Peak Demand Alert - {current_demand:.0f} MW Demand",
            html_content=html_content,
            text_content=text_content,
        )

    async def send_model_report(self, user: User, report_content: str) -> bool:
        """Send weekly model performance report"""

        html_content = f"""
        <html>
            <body style="font-family: Arial, sans-serif;">
                <div style="max-width: 600px; margin: 0 auto; padding: 20px;">
                    <h2>📊 Weekly Model Performance Report</h2>
                    <p>Hello {user.name},</p>
                    <p>Here is your weekly model performance summary:</p>
                    <div style="background-color: #f0f0f0; padding: 20px; border-radius: 8px;">
                        {report_content}
                    </div>
                    <p>
                        <a href="http://localhost:3000/analytics">View Full Analytics</a>
                    </p>
                    <hr>
                    <p style="font-size: 12px; color: #666;">
                        You can manage these notifications in
                        <a href="http://localhost:3000/settings">Settings</a>.
                    </p>
                </div>
            </body>
        </html>
        """

        return await self.send_email(
            to_email=user.email,
            subject="Weekly Model Performance Report",
            html_content=html_content,
        )

    async def send_system_update(self, user: User, update_message: str) -> bool:
        """Send system update notification"""

        html_content = f"""
        <html>
            <body style="font-family: Arial, sans-serif;">
                <div style="max-width: 600px; margin: 0 auto; padding: 20px;">
                    <h2>ℹ️ System Update Notification</h2>
                    <p>Hello {user.name},</p>
                    <p>{update_message}</p>
                    <p>
                        <a href="http://localhost:3000/settings">View Settings</a>
                    </p>
                </div>
            </body>
        </html>
        """

        return await self.send_email(
            to_email=user.email,
            subject="System Update Notification",
            html_content=html_content,
        )


# Global instance
_email_service: Optional[EmailService] = None


def get_email_service(provider: str = "console") -> EmailService:
    """Get or create the email service (singleton)"""
    global _email_service

    if _email_service is None:
        _email_service = EmailService(provider)

    return _email_service
