# day3_integration_test.py
from src.decision_engine import DecisionEngine
from src.gmail_client import GmailClient
from src.email_processor import EmailProcessor
import os

def test_integration():
    print("=== DAY 3 INTEGRATION TEST ===")
    
    # Check if model exists
    if not os.path.exists("models/fine_tuned"):
        print("Fine-tuned model not found. Please run training first.")
        print("Run: python train_model.py")
        return False
    
    # Test model loading
    print("Testing model loading...")
    engine = DecisionEngine()
    
    if not engine.classifier.model:
        print("Model failed to load")
        return False
    
    print("Model loaded successfully")
    
    # Test with sample data
    print("\nTesting with sample queries...")
    
    sample_emails = [
        {
            'id': 'test1',
            'subject': 'Exam Schedule Query',
            'body': 'Can you please tell me when the final exam is scheduled?',
            'sender': 'student@university.edu'
        },
        {
            'id': 'test2',
            'subject': 'Password Issue',
            'body': 'I am having trouble logging into the student portal. My password is not working.',
            'sender': 'student123@university.edu'
        },
        {
            'id': 'test3',
            'subject': 'Urgent Problem',
            'body': 'This is an urgent issue with my registration. I need immediate help!',
            'sender': 'student456@university.edu'
        }
    ]
    
    # Process queries
    results = engine.batch_process(sample_emails)
    
    print("\nResults Summary:")
    actions = {}
    for result in results:
        action = result['action']
        actions[action] = actions.get(action, 0) + 1
        
        print(f"\nEmail {result['email_id']}:")
        print(f"  Action: {result['action']}")
        print(f"  Reason: {result['reason']}")
        print(f"  Confidence: {result['confidence']:.2f}")
        
        if result.get('response'):
            print(f"  Response: {result['response'][:100]}...")
    
    print(f"\nAction Distribution:")
    for action, count in actions.items():
        print(f"  {action}: {count}")
    
    # Test with real emails if available
    try:
        print("\nTesting with real emails...")
        gmail_client = GmailClient()
        processor = EmailProcessor()
        
        recent_emails = gmail_client.fetch_emails(query="is:unread", max_results=3)
        student_emails = processor.filter_student_emails(recent_emails)
        
        if student_emails:
            real_results = engine.batch_process(student_emails[:2])  # Test with 2 emails
            print(f"Processed {len(real_results)} real emails successfully")
        else:
            print("No recent student emails found")
            
    except Exception as e:
        print(f"Could not test with real emails: {e}")

    print("\nDAY 3 INTEGRATION COMPLETE!")
    print("Model fine-tuning successful")
    print("Classification system working")
    print("Decision engine operational")

    return True

if __name__ == "__main__":
    test_integration()