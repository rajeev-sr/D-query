from src.gmail_client import GmailClient

client = GmailClient()
emails = client.fetch_emails(query="is:unread", max_results=2)
print(f"Fetched {len(emails)} emails")
for email in emails[:2]:
    print(f"Subject: {email['subject']}")
    print(f"From: {email['sender']}")
    print("---")