# src/notification_system.py
"""
Notification and Alert System
Handles notifications for escalations, approvals, and system alerts
"""

import smtplib
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from dataclasses import dataclass
import json
import os
from enum import Enum

class NotificationType(Enum):
    ESCALATION = "escalation"
    APPROVAL_REQUEST = "approval_request"
    SYSTEM_ALERT = "system_alert"
    PERFORMANCE_REPORT = "performance_report"
    ERROR_NOTIFICATION = "error"

@dataclass
class Notification:
    """Structure for notifications"""
    notification_id: str
    type: NotificationType
    priority: str  # high, medium, normal
    title: str
    message: str
    details: Dict[str, Any]
    created_at: datetime
    recipient: str
    sent: bool = False
    sent_at: Optional[datetime] = None
    
class NotificationSystem:
    """Comprehensive notification system"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or self._load_default_config()
        self.notification_history = []
        self.pending_notifications = []
        self.email_config = self.config.get("email", {})
        
    def _load_default_config(self) -> Dict[str, Any]:
        """Load default notification configuration"""
        return {
            "email": {
                "smtp_server": "smtp.gmail.com",
                "smtp_port": 587,
                "sender_email": os.getenv("NOTIFICATION_EMAIL", ""),
                "sender_password": os.getenv("NOTIFICATION_PASSWORD", ""),
                "enabled": bool(os.getenv("NOTIFICATION_EMAIL"))
            },
            "recipients": {
                "escalations": [os.getenv("ESCALATION_EMAIL", "admin@institute.edu")],
                "approvals": [os.getenv("APPROVAL_EMAIL", "reviewer@institute.edu")],
                "system_alerts": [os.getenv("ADMIN_EMAIL", "admin@institute.edu")],
                "reports": [os.getenv("REPORTS_EMAIL", "manager@institute.edu")]
            },
            "notification_preferences": {
                "escalation": {"email": True, "priority_threshold": "normal"},
                "approval_request": {"email": True, "priority_threshold": "normal"},
                "system_alert": {"email": True, "priority_threshold": "high"},
                "error": {"email": True, "priority_threshold": "medium"}
            }
        }
    
    def send_escalation_notification(self, escalation_data: Dict) -> bool:
        """Send notification for query escalation"""
        
        notification = Notification(
            notification_id=f"escalation_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            type=NotificationType.ESCALATION,
            priority=escalation_data.get("priority", "normal"),
            title=f"Query Escalated - {escalation_data.get('reason', 'Manual Review Required')}",
            message=self._create_escalation_message(escalation_data),
            details=escalation_data,
            created_at=datetime.now(),
            recipient=self._get_recipients("escalations")[0]  # Primary escalation recipient
        )
        
        return self._send_notification(notification)
    
    def send_approval_notification(self, approval_request: Any) -> bool:
        """Send notification for approval request"""
        
        notification = Notification(
            notification_id=f"approval_{approval_request.request_id}",
            type=NotificationType.APPROVAL_REQUEST,
            priority=approval_request.priority,
            title=f"Response Approval Required - {approval_request.query_id}",
            message=self._create_approval_message(approval_request),
            details={
                "request_id": approval_request.request_id,
                "query_id": approval_request.query_id,
                "ai_confidence": approval_request.ai_confidence,
                "quality_scores": approval_request.quality_scores,
                "expires_at": approval_request.expires_at.isoformat()
            },
            created_at=datetime.now(),
            recipient=self._get_recipients("approvals")[0]
        )
        
        return self._send_notification(notification)
    
    def send_system_alert(self, alert_type: str, message: str, details: Dict = None) -> bool:
        """Send system alert notification"""
        
        priority_map = {
            "system_failure": "high",
            "performance_degradation": "medium",
            "configuration_issue": "medium",
            "api_error": "medium",
            "resource_warning": "normal"
        }
        
        notification = Notification(
            notification_id=f"alert_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            type=NotificationType.SYSTEM_ALERT,
            priority=priority_map.get(alert_type, "normal"),
            title=f"System Alert: {alert_type.replace('_', ' ').title()}",
            message=message,
            details=details or {},
            created_at=datetime.now(),
            recipient=self._get_recipients("system_alerts")[0]
        )
        
        return self._send_notification(notification)
    
    def send_performance_report(self, report_data: Dict) -> bool:
        """Send performance report notification"""
        
        notification = Notification(
            notification_id=f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            type=NotificationType.PERFORMANCE_REPORT,
            priority="normal",
            title="AI Query Handler - Performance Report",
            message=self._create_performance_report_message(report_data),
            details=report_data,
            created_at=datetime.now(),
            recipient=self._get_recipients("reports")[0]
        )
        
        return self._send_notification(notification)
    
    def send_error_notification(self, error_type: str, error_message: str, 
                              context: Dict = None) -> bool:
        """Send error notification"""
        
        notification = Notification(
            notification_id=f"error_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            type=NotificationType.ERROR_NOTIFICATION,
            priority="medium",
            title=f"System Error: {error_type}",
            message=f"An error occurred in the AI Query Handler system:\n\n{error_message}",
            details=context or {},
            created_at=datetime.now(),
            recipient=self._get_recipients("system_alerts")[0]
        )
        
        return self._send_notification(notification)
    
    def _send_notification(self, notification: Notification) -> bool:
        """Send notification through configured channels"""
        
        # Check if notification should be sent based on preferences
        if not self._should_send_notification(notification):
            logging.info(f"Notification {notification.notification_id} skipped based on preferences")
            return True
        
        # Add to pending
        self.pending_notifications.append(notification)
        
        success = True
        
        # Send email if enabled
        if self.email_config.get("enabled", False):
            email_success = self._send_email_notification(notification)
            success = success and email_success
        
        # Mark as sent if successful
        if success:
            notification.sent = True
            notification.sent_at = datetime.now()
            self.notification_history.append(notification)
            self.pending_notifications.remove(notification)
            
            logging.info(f"Notification {notification.notification_id} sent successfully")
        else:
            logging.error(f"Failed to send notification {notification.notification_id}")
        
        return success
    
    def _send_email_notification(self, notification: Notification) -> bool:
        """Send email notification"""
        
        try:
            # Create email message
            msg = MIMEMultipart()
            msg['From'] = self.email_config['sender_email']
            msg['To'] = notification.recipient
            msg['Subject'] = f"[AI Query Handler] {notification.title}"
            
            # Create email body
            email_body = self._create_email_body(notification)
            msg.attach(MIMEText(email_body, 'html'))
            
            # Send email
            server = smtplib.SMTP(self.email_config['smtp_server'], self.email_config['smtp_port'])
            server.starttls()
            server.login(self.email_config['sender_email'], self.email_config['sender_password'])
            
            server.sendmail(
                self.email_config['sender_email'],
                notification.recipient,
                msg.as_string()
            )
            server.quit()
            
            return True
            
        except Exception as e:
            logging.error(f"Failed to send email notification: {e}")
            return False
    
    def _create_escalation_message(self, escalation_data: Dict) -> str:
        """Create escalation notification message"""
        
        return f"""
A query has been escalated for manual review.

Query ID: {escalation_data.get('query_id', 'Unknown')}
Escalation Reason: {escalation_data.get('reason', 'Not specified')}
Priority: {escalation_data.get('priority', 'Normal')}
Escalated At: {escalation_data.get('escalated_at', 'Unknown')}

Original Email:
Subject: {escalation_data.get('context', {}).get('email_content', {}).get('subject', 'No subject')}
From: {escalation_data.get('context', {}).get('email_content', {}).get('sender', 'Unknown sender')}

AI Decision Context:
Classification: {escalation_data.get('context', {}).get('classification', {}).get('category', 'Unknown')}
Confidence: {escalation_data.get('context', {}).get('classification', {}).get('confidence', 0.0):.2f}

Please review this query and provide appropriate response.
        """.strip()
    
    def _create_approval_message(self, approval_request) -> str:
        """Create approval notification message"""
        
        return f"""
A response requires your approval before sending.

Request ID: {approval_request.request_id}
Query ID: {approval_request.query_id}
Priority: {approval_request.priority}
AI Confidence: {approval_request.ai_confidence:.2f}
Quality Score: {approval_request.quality_scores.get('overall', 0.0):.2f}
Expires: {approval_request.expires_at.strftime('%Y-%m-%d %H:%M:%S')}

Original Query:
Subject: {approval_request.original_email.get('subject', 'No subject')}
From: {approval_request.original_email.get('sender', 'Unknown')}

Generated Response:
{approval_request.generated_response[:500]}...

Please review and approve/reject this response.
        """.strip()
    
    def _create_performance_report_message(self, report_data: Dict) -> str:
        """Create performance report message"""
        
        metrics = report_data.get('detailed_metrics', {})
        
        return f"""
AI Query Handler Performance Report

Report Period: {report_data.get('timestamp', 'Unknown')}

Key Metrics:
- Classification Accuracy: {metrics.get('classification_accuracy', 0.0):.2%}
- Response Quality Score: {metrics.get('response_relevance_score', 0.0):.2%}
- Automation Rate: {metrics.get('automation_rate', 0.0):.2%}
- Escalation Rate: {metrics.get('escalation_rate', 0.0):.2%}
- Average Response Time: {metrics.get('avg_response_time_seconds', 0.0):.1f}s

System Status: {report_data.get('evaluation_summary', {}).get('overall_score', 0.0):.2%} Overall Performance

Recommendations:
{chr(10).join('- ' + rec for rec in report_data.get('recommendations', []))}
        """.strip()
    
    def _create_email_body(self, notification: Notification) -> str:
        """Create HTML email body"""
        
        priority_colors = {
            "high": "#dc3545",
            "medium": "#ffc107", 
            "normal": "#28a745"
        }
        
        priority_color = priority_colors.get(notification.priority, "#6c757d")
        
        html_body = f"""
        <html>
        <body style="font-family: Arial, sans-serif; line-height: 1.6; color: #333;">
            <div style="max-width: 600px; margin: 0 auto; padding: 20px;">
                <div style="background-color: {priority_color}; color: white; padding: 15px; border-radius: 5px; margin-bottom: 20px;">
                    <h2 style="margin: 0;">{notification.title}</h2>
                    <p style="margin: 5px 0 0 0;">Priority: {notification.priority.upper()}</p>
                </div>
                
                <div style="background-color: #f8f9fa; padding: 15px; border-radius: 5px; margin-bottom: 20px;">
                    <pre style="white-space: pre-wrap; font-family: inherit;">{notification.message}</pre>
                </div>
                
                <div style="border-top: 1px solid #dee2e6; padding-top: 15px; font-size: 12px; color: #6c757d;">
                    <p><strong>Notification ID:</strong> {notification.notification_id}</p>
                    <p><strong>Timestamp:</strong> {notification.created_at.strftime('%Y-%m-%d %H:%M:%S')}</p>
                    <p><strong>System:</strong> AI Query Handler</p>
                </div>
            </div>
        </body>
        </html>
        """
        
        return html_body
    
    def _should_send_notification(self, notification: Notification) -> bool:
        """Check if notification should be sent based on preferences"""
        
        preferences = self.config.get("notification_preferences", {})
        notification_prefs = preferences.get(notification.type.value, {})
        
        # Check if email notifications are enabled for this type
        if not notification_prefs.get("email", True):
            return False
        
        # Check priority threshold
        priority_threshold = notification_prefs.get("priority_threshold", "normal")
        priority_levels = {"normal": 0, "medium": 1, "high": 2}
        
        if priority_levels.get(notification.priority, 0) < priority_levels.get(priority_threshold, 0):
            return False
        
        return True
    
    def _get_recipients(self, category: str) -> List[str]:
        """Get recipient list for notification category"""
        
        recipients = self.config.get("recipients", {})
        return recipients.get(category, ["admin@institute.edu"])
    
    def get_notification_history(self, limit: int = 50) -> List[Dict]:
        """Get notification history"""
        
        history = self.notification_history[-limit:] if limit else self.notification_history
        
        return [
            {
                "notification_id": notif.notification_id,
                "type": notif.type.value,
                "priority": notif.priority,
                "title": notif.title,
                "recipient": notif.recipient,
                "created_at": notif.created_at.isoformat(),
                "sent": notif.sent,
                "sent_at": notif.sent_at.isoformat() if notif.sent_at else None
            }
            for notif in history
        ]
    
    def get_pending_notifications(self) -> List[Dict]:
        """Get pending notifications"""
        
        return [
            {
                "notification_id": notif.notification_id,
                "type": notif.type.value,
                "priority": notif.priority,
                "title": notif.title,
                "created_at": notif.created_at.isoformat(),
                "recipient": notif.recipient
            }
            for notif in self.pending_notifications
        ]
    
    def get_notification_stats(self) -> Dict[str, Any]:
        """Get notification system statistics"""
        
        total_sent = len(self.notification_history)
        pending_count = len(self.pending_notifications)
        
        # Count by type
        type_counts = {}
        for notif in self.notification_history:
            type_name = notif.type.value
            type_counts[type_name] = type_counts.get(type_name, 0) + 1
        
        # Count by priority
        priority_counts = {}
        for notif in self.notification_history:
            priority_counts[notif.priority] = priority_counts.get(notif.priority, 0) + 1
        
        return {
            "total_notifications_sent": total_sent,
            "pending_notifications": pending_count,
            "notifications_by_type": type_counts,
            "notifications_by_priority": priority_counts,
            "email_enabled": self.email_config.get("enabled", False),
            "configured_recipients": {
                category: len(recipients) 
                for category, recipients in self.config.get("recipients", {}).items()
            }
        }
    
    def test_notification_system(self) -> Dict[str, Any]:
        """Test notification system connectivity"""
        
        results = {
            "email_config_valid": False,
            "smtp_connection_successful": False,
            "test_notification_sent": False,
            "errors": []
        }
        
        # Check email configuration
        if self.email_config.get("enabled") and self.email_config.get("sender_email"):
            results["email_config_valid"] = True
            
            # Test SMTP connection
            try:
                server = smtplib.SMTP(self.email_config['smtp_server'], self.email_config['smtp_port'])
                server.starttls()
                server.login(self.email_config['sender_email'], self.email_config['sender_password'])
                server.quit()
                results["smtp_connection_successful"] = True
                
                # Send test notification
                test_notification = Notification(
                    notification_id="test_notification",
                    type=NotificationType.SYSTEM_ALERT,
                    priority="normal",
                    title="Test Notification",
                    message="This is a test notification from the AI Query Handler system.",
                    details={},
                    created_at=datetime.now(),
                    recipient=self._get_recipients("system_alerts")[0]
                )
                
                if self._send_email_notification(test_notification):
                    results["test_notification_sent"] = True
                
            except Exception as e:
                results["errors"].append(f"SMTP connection failed: {str(e)}")
        else:
            results["errors"].append("Email configuration incomplete")
        
        return results