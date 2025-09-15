# src/email_sender.py
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
import os
from typing import Dict, List, Optional
from datetime import datetime
import logging

class EmailSender:
    def __init__(self, use_gmail_api=True):
        self.use_gmail_api = use_gmail_api
        self.gmail_service = None
        self.smtp_config = None
        
        if use_gmail_api:
            self._setup_gmail_api()
        else:
            self._setup_smtp()
    
    def _setup_gmail_api(self):
        """Setup Gmail API for sending emails"""
        try:
            from src.gmail_client import GmailClient
            gmail_client = GmailClient()
            self.gmail_service = gmail_client.service
            print("Gmail API sender initialized")
        except Exception as e:
            print(f"⚠️ Gmail API setup failed: {e}")
            print("Falling back to SMTP...")
            self._setup_smtp()
    
    def _setup_smtp(self):
        """Setup SMTP configuration"""
        self.smtp_config = {
            'smtp_server': os.getenv('SMTP_SERVER', 'smtp.gmail.com'),
            'smtp_port': int(os.getenv('SMTP_PORT', '587')),
            'email': os.getenv('EMAIL_ADDRESS'),
            'password': os.getenv('EMAIL_PASSWORD')
        }
        
        if not self.smtp_config['email'] or not self.smtp_config['password']:
            print("⚠️ SMTP credentials not found in environment variables")
            print("Please set EMAIL_ADDRESS and EMAIL_PASSWORD")
    
    def send_email(self, to_email: str, subject: str, body: str, 
                   reply_to_id: str = None, is_html: bool = False) -> bool:
        """Send email response"""
        
        if self.use_gmail_api and self.gmail_service:
            return self._send_via_gmail_api(to_email, subject, body, reply_to_id, is_html)
        else:
            return self._send_via_smtp(to_email, subject, body, is_html)
    
    def _send_via_gmail_api(self, to_email: str, subject: str, body: str, 
                           reply_to_id: str = None, is_html: bool = False) -> bool:
        """Send email via Gmail API"""
        try:
            from googleapiclient.errors import HttpError
            import base64
            from email.mime.text import MIMEText
            
            # Create message
            if is_html:
                message = MIMEMultipart('alternative')
                message.attach(MIMEText(body, 'html'))
            else:
                message = MIMEText(body, 'plain')
            
            message['to'] = to_email
            message['subject'] = subject
            
            # Add In-Reply-To header if this is a reply
            if reply_to_id:
                message['In-Reply-To'] = reply_to_id
                message['References'] = reply_to_id
            
            # Encode message
            raw_message = base64.urlsafe_b64encode(message.as_bytes()).decode('utf-8')
            
            # Send message
            send_message = self.gmail_service.users().messages().send(
                userId='me',
                body={'raw': raw_message}
            ).execute()
            
            print(f"Email sent via Gmail API: {send_message['id']}")
            return True
            
        except Exception as e:
            print(f"Gmail API send failed: {e}")
            return False
    
    def _send_via_smtp(self, to_email: str, subject: str, body: str, is_html: bool = False) -> bool:
        """Send email via SMTP"""
        try:
            if not self.smtp_config['email'] or not self.smtp_config['password']:
                print("SMTP credentials not configured")
                return False
            
            # Create message
            msg = MIMEMultipart()
            msg['From'] = self.smtp_config['email']
            msg['To'] = to_email
            msg['Subject'] = subject
            
            # Add body
            if is_html:
                msg.attach(MIMEText(body, 'html'))
            else:
                msg.attach(MIMEText(body, 'plain'))
            
            # Send email
            server = smtplib.SMTP(self.smtp_config['smtp_server'], self.smtp_config['smtp_port'])
            server.starttls()
            server.login(self.smtp_config['email'], self.smtp_config['password'])
            
            text = msg.as_string()
            server.sendmail(self.smtp_config['email'], to_email, text)
            server.quit()
            
            print(f"Email sent via SMTP to {to_email}")
            return True
            
        except Exception as e:
            print(f"SMTP send failed: {e}")
            return False
    
    def format_response_email(self, original_email: Dict, response: str, 
                            include_signature: bool = True) -> Dict:
        """Format response email with proper structure"""
        
        # Extract original details
        original_subject = original_email.get('subject', 'Re: Your Query')
        sender = original_email.get('sender', '')
        
        # Create reply subject
        if not original_subject.startswith('Re:'):
            reply_subject = f"Re: {original_subject}"
        else:
            reply_subject = original_subject
        
        # Format body
        formatted_body = f"\n\n{response}\n\n"
        
#         if include_signature:
#             formatted_body += """Best regards,
# AI Assistant
# Institute Support Team

# ---
# This is an automated response. If you need further assistance, please contact the support team directly.
# """
        
        return {
            'to': sender,
            'subject': reply_subject,
            'body': formatted_body,
            'reply_to_id': original_email.get('id'),
            'is_html': False
        }