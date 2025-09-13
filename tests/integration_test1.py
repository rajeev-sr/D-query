from src.gmail_client import GmailClient
from src.email_processor import EmailProcessor
from src.data_manager import DataManager

def main():
    print("=== DAY 1 INTEGRATION TEST ===")
    
    # Step 1: Connect to Gmail
    print("1. Connecting to Gmail...")
    client = GmailClient()
    
    # Step 2: Fetch emails
    print("2. Fetching recent emails...")
    emails = client.fetch_emails(query="newer_than:30d", max_results=100)
    print(f"Fetched {len(emails)} emails")
    
    # Step 3: Process emails
    print("3. Processing emails...")
    processor = EmailProcessor()
    student_emails = processor.filter_student_emails(emails)
    print(f"Found {len(student_emails)} student emails")
    
    categories = processor.categorize_emails(student_emails)
    for cat, emails_list in categories.items():
        print(f"  {cat}: {len(emails_list)} emails")
    
    # Step 4: Save data
    print("4. Saving data...")
    data_manager = DataManager()
    data_manager.save_raw_emails(student_emails)
    training_file = data_manager.create_training_dataset(student_emails)
    
    print(f"Next: Manually label the training data in {training_file}")
    
if __name__ == "__main__":
    main()


# {
#     "query": "subject + body",
#     "sender": "sender",
#     "category": "category",
#     "response": "response",
#     "timestamp": "timestamp" 
# }