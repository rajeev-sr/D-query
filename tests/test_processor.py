# test_processor.py
from src.gmail_client import GmailClient
from src.email_processor import EmailProcessor

client = GmailClient()
processor = EmailProcessor()

# Fetch recent emails
all_emails = client.fetch_emails(query="is:unread", max_results=4)
print(f"Total emails: {len(all_emails)}")

# Filter student emails
student_emails = processor.filter_student_emails(all_emails)
print(f"Student emails: {len(student_emails)}")

# Categorize
categories = processor.categorize_emails(student_emails)
for category, emails in categories.items():
    print(f"{category}: {len(emails)} emails")