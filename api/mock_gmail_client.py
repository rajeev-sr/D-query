"""
Mock Gmail Client for Demo Mode
Provides sample data when credentials.json is not available
"""
import json
from datetime import datetime, timedelta
from typing import Dict, List, Any
import random

class MockGmailClient:
    """Mock Gmail client that provides sample data for demonstration"""
    
    def __init__(self):
        self.demo_mode = True
        self._setup_mock_data()
    
    def _setup_mock_data(self):
        """Generate realistic sample email data"""
        self.sample_emails = [
            {
                "id": "mock_001",
                "subject": "Question about Assignment Submission",
                "body": "Hi, I need help with submitting my assignment. The deadline is approaching and I'm having trouble with the portal.",
                "sender": "student1@university.edu",
                "date": (datetime.now() - timedelta(hours=2)).isoformat(),
                "labels": ["INBOX", "UNREAD"],
                "processed": False,
                "confidence": 0.85,
                "category": "academic",
                "auto_response": "Your question about assignment submission has been received. Please check the academic calendar for deadlines and use the student portal for submissions."
            },
            {
                "id": "mock_002", 
                "subject": "Fee Payment Query",
                "body": "I want to know about the fee structure for this semester. Can you provide details?",
                "sender": "student2@university.edu",
                "date": (datetime.now() - timedelta(hours=5)).isoformat(),
                "labels": ["INBOX"],
                "processed": True,
                "confidence": 0.92,
                "category": "fee",
                "auto_response": "Thank you for your fee inquiry. Please refer to the fee structure document available on the student portal."
            },
            {
                "id": "mock_003",
                "subject": "Technical Support Needed",
                "body": "My account is locked and I cannot access the learning management system. Please help.",
                "sender": "student3@university.edu", 
                "date": (datetime.now() - timedelta(hours=8)).isoformat(),
                "labels": ["INBOX", "UNREAD"],
                "processed": False,
                "confidence": 0.78,
                "category": "technical",
                "auto_response": "Your technical support request has been logged. Please contact IT support at support@university.edu or visit the help desk."
            },
            {
                "id": "mock_004",
                "subject": "Course Registration Issue",
                "body": "I'm trying to register for courses but getting an error message. What should I do?",
                "sender": "student4@university.edu",
                "date": (datetime.now() - timedelta(days=1)).isoformat(), 
                "labels": ["INBOX"],
                "processed": True,
                "confidence": 0.88,
                "category": "registration",
                "auto_response": "Course registration issues can be resolved by contacting the registrar's office during office hours."
            },
            {
                "id": "mock_005",
                "subject": "Urgent: Exam Schedule Clarification",
                "body": "There seems to be a conflict in my exam schedule. This is urgent as exams start tomorrow.",
                "sender": "student5@university.edu",
                "date": (datetime.now() - timedelta(hours=1)).isoformat(),
                "labels": ["INBOX", "UNREAD", "IMPORTANT"],
                "processed": False,
                "confidence": 0.95,
                "category": "academic",
                "needs_human_review": True,
                "auto_response": None  # Requires human review due to urgency
            }
        ]
    
    def get_unread_emails(self, max_results=10):
        """Get unread emails (mock data)"""
        unread = [email for email in self.sample_emails if "UNREAD" in email.get("labels", [])]
        return unread[:max_results]
    
    def get_emails(self, max_results=50, unread_only=False):
        """Get emails for dashboard"""
        if unread_only:
            return self.get_unread_emails(max_results)
        return self.sample_emails[:max_results]
    
    def fetch_emails(self, max_results=50, unread_only=True, label_ids=None, query=None):
        """Fetch emails (alias for get_emails for compatibility)"""
        # Ignore query parameter in mock mode, just return sample emails
        return self.get_emails(max_results, unread_only)
    
    def get_email_stats(self):
        """Get email statistics"""
        total_emails = len(self.sample_emails)
        processed_emails = len([e for e in self.sample_emails if e.get("processed", False)])
        unread_emails = len([e for e in self.sample_emails if "UNREAD" in e.get("labels", [])])
        auto_responded = len([e for e in self.sample_emails if e.get("auto_response") and e.get("processed", False)])
        pending_review = len([e for e in self.sample_emails if e.get("needs_human_review", False)])
        
        avg_confidence = sum(e.get("confidence", 0) for e in self.sample_emails) / total_emails if total_emails > 0 else 0
        auto_response_rate = (auto_responded / processed_emails * 100) if processed_emails > 0 else 0
        
        return {
            "total_emails": total_emails,
            "total_emails_processed": processed_emails,
            "unread_emails": unread_emails,
            "auto_response_rate": round(auto_response_rate, 1),
            "average_confidence": round(avg_confidence, 3),
            "emails_pending_review": pending_review
        }
    
    def get_processing_metrics(self, days=7):
        """Get processing metrics for the last N days"""
        # Generate mock daily processing data
        daily_data = {}
        for i in range(days):
            date = (datetime.now() - timedelta(days=i)).strftime("%Y-%m-%d")
            daily_data[date] = random.randint(0, 5)  # Random processed emails per day
        
        return {
            "daily_processed": daily_data,
            "response_times": [1.2, 0.8, 1.5, 0.9, 2.1],  # Mock response times in seconds
            "confidence_distribution": {
                "high": 3,  # confidence > 0.8
                "medium": 1,  # confidence 0.5-0.8
                "low": 1   # confidence < 0.5
            }
        }
    
    def get_real_time_data(self):
        """Get real-time system data"""
        return {
            "unread_emails": len(self.get_unread_emails()),
            "last_update": datetime.now().isoformat(),
            "system_status": "running",
            "recent_activity": [
                {
                    "email_id": "mock_005",
                    "action": "Received",
                    "timestamp": (datetime.now() - timedelta(minutes=5)).isoformat()
                },
                {
                    "email_id": "mock_004", 
                    "action": "Auto-responded",
                    "timestamp": (datetime.now() - timedelta(minutes=15)).isoformat()
                },
                {
                    "email_id": "mock_003",
                    "action": "Classified",
                    "timestamp": (datetime.now() - timedelta(minutes=30)).isoformat()
                }
            ]
        }
    
    def is_authenticated(self):
        """Check if client is authenticated (always True for mock)"""
        return True
        
    def send_email(self, to_email: str, subject: str, body: str):
        """Mock sending email (just log the action)"""
        print(f"DEMO: Would send email to {to_email} with subject: {subject}")
        return {"success": True, "message": "Demo email sent"}