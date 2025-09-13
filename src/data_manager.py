import json
import pandas as pd
from datetime import datetime
import os
from typing import List, Dict

class DataManager:
    def __init__(self, data_dir="data"):
        self.data_dir = data_dir
        os.makedirs(f"{data_dir}/raw_emails", exist_ok=True)
        os.makedirs(f"{data_dir}/processed", exist_ok=True)
    
    def save_raw_emails(self, emails: List[Dict], filename=None):
        """Save raw email data"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"emails_{timestamp}.json"
        
        filepath = os.path.join(self.data_dir, "raw_emails", filename)
        
        with open(filepath, 'w') as f:
            json.dump(emails, f, indent=2)
        
        print(f"Saved {len(emails)} emails to {filepath}")
        return filepath
    
    def create_training_dataset(self, emails: List[Dict]):
        """Create training dataset for fine-tuning"""
        training_data = []
        
        for email in emails:
            # Create training examples
            example = {
                'query': email['subject'] + '\n' + email['body'],
                'sender': email['sender'],
                'category': 'administrative',  # To be labeled manually
                'response': '',  # To be filled with your typical responses
                'timestamp': email['date']
            }
            training_data.append(example)
        
        # Save as CSV for easy editing
        df = pd.DataFrame(training_data)
        filepath = os.path.join(self.data_dir, "processed", "training_dataset.csv")
        df.to_csv(filepath, index=False)
        
        print(f"Created training dataset with {len(training_data)} examples")
        print(f"Please manually label the 'category' and 'response' columns in {filepath}")
        
        return filepath