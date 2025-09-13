import logging
import os
import pickle
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
import base64
import email
from datetime import datetime, timedelta

class GmailClient:
    def __init__(self):
        self.SCOPES = ['https://www.googleapis.com/auth/gmail.readonly',
                       'https://www.googleapis.com/auth/gmail.send']
        self.service = self._authenticate()
    
    def _authenticate(self):
        try:
            creds = None
            if os.path.exists('token.pickle'):
                with open('token.pickle', 'rb') as token:
                    creds = pickle.load(token)
            
            if not creds or not creds.valid:
                if creds and creds.expired and creds.refresh_token:
                    creds.refresh(Request())
                else:
                    flow = InstalledAppFlow.from_client_secrets_file(
                        'credentials.json', self.SCOPES)
                    creds = flow.run_local_server(port=0)
                
                with open('token.pickle', 'wb') as token:
                    pickle.dump(creds, token)
            
            return build('gmail', 'v1', credentials=creds)
        except Exception as e:
            logging.error(f"Gmail authentication failed: {e}")
            raise Exception("Please check credentials.json file and internet connection")
        
    def fetch_emails(self, query="is:unread", max_results=10):
        # Fetch emails based on query
        try:
            results = self.service.users().messages().list(
                userId='me', q=query, maxResults=max_results).execute()
            
            messages = results.get('messages', [])
            emails = []
            
            for msg in messages:
                msg_data = self.service.users().messages().get(
                    userId='me', id=msg['id']).execute()
                
                email_data = self._parse_email(msg_data)
                emails.append(email_data)
                
            return emails
        
        except Exception as e:
            logging.error(f"Error fetching emails: {e}")
            return []

    def _parse_email(self, msg_data):
        """Parse email data into structured format"""
        payload = msg_data['payload']
        headers = payload.get('headers', [])
        
        # Extract headers
        subject = next((h['value'] for h in headers if h['name'] == 'Subject'), '')
        sender = next((h['value'] for h in headers if h['name'] == 'From'), '')
        date = next((h['value'] for h in headers if h['name'] == 'Date'), '')
        
        # Extract body
        body = self._extract_body(payload)
        
        return {
            'id': msg_data['id'],
            'subject': subject,
            'sender': sender,
            'date': date,
            'body': body,
            'thread_id': msg_data['threadId']
        }

    def _extract_body(self, payload):
        """Extract email body text"""
        body = ""
        
        if 'parts' in payload:
            for part in payload['parts']:
                if part['mimeType'] == 'text/plain':
                    data = part['body']['data']
                    body = base64.urlsafe_b64decode(data).decode('utf-8')
                    break
        else:
            if payload['mimeType'] == 'text/plain':
                data = payload['body']['data']
                body = base64.urlsafe_b64decode(data).decode('utf-8')
        
        return body