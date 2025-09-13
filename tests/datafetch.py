import base64
from src.gmail_client import GmailClient

client = GmailClient()
emails = client.fetch_emails(query="is:unread", max_results=2)
print(f"Fetched {len(emails)} emails")
for email in emails[:1]:
    print(f"Subject: {email['subject']}")
    print(f"From: {email['sender']}")
    # print(f"Body: {email['body']}")
    print(f"Date: {email['date']}")

    body = email['body']
    print(f"Body: {body[:50]}")

    print("---")