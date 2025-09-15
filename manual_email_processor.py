#!/usr/bin/env python3
"""Manual email processing test"""

import sys
import os
sys.path.append('/home/rajeev-kumar/Desktop/D-query')

from src.automated_processor import AutomatedEmailProcessor
import json
from datetime import datetime

def manual_process_emails():
    """Manually process emails to see what's happening"""
    
    print("=== MANUAL EMAIL PROCESSING TEST ===\n")
    
    try:
        # Initialize processor
        print("1. Initializing Automated Processor...")
        processor = AutomatedEmailProcessor()
        print("   ✅ Processor initialized")
        
        # Show current configuration
        print(f"\n2. Configuration:")
        print(f"   Email query: {processor.config.get('email_query', 'is:unread')}")
        print(f"   Max emails per batch: {processor.config.get('max_emails_per_batch', 10)}")
        print(f"   Auto-respond enabled: {processor.config.get('auto_respond_enabled', True)}")
        
        # Process a batch
        print(f"\n3. Processing current batch...")
        start_time = datetime.now()
        result = processor.process_emails_batch()
        end_time = datetime.now()
        
        print(f"   ✅ Processing completed in {(end_time - start_time).total_seconds():.2f} seconds")
        
        # Show results
        print(f"\n4. Results:")
        print(f"   Status: {result.get('status', 'unknown')}")
        print(f"   Total processed: {result.get('processed', 0)}")
        print(f"   Auto-responded: {result.get('auto_responded', 0)}")
        print(f"   Review needed: {result.get('review_needed', 0)}")
        print(f"   Escalated: {result.get('escalated', 0)}")
        print(f"   Errors: {result.get('errors', 0)}")
        
        if result.get('message'):
            print(f"   Message: {result['message']}")
        
        # Show details if any
        details = result.get('details', [])
        if details:
            print(f"\n5. Email Details:")
            for i, detail in enumerate(details, 1):
                print(f"   📧 Email {i}:")
                print(f"      Subject: {detail.get('subject', 'No subject')}")
                print(f"      From: {detail.get('sender', 'Unknown')}")
                print(f"      Action: {detail.get('action', 'unknown')}")
                if 'ai_decision' in detail:
                    ai_decision = detail['ai_decision']
                    print(f"      AI Confidence: {ai_decision.get('confidence', 0):.2f}")
                    print(f"      AI Category: {ai_decision.get('category', 'unknown')}")
                    print(f"      AI Reason: {ai_decision.get('reason', 'No reason')}")
        
        # Check for review queue
        print(f"\n6. Checking review queue...")
        review_file = "data/admin_data/review_queue.json"
        if os.path.exists(review_file):
            with open(review_file, 'r') as f:
                review_queue = json.load(f)
            print(f"   📋 {len(review_queue)} items in review queue")
            
            if review_queue:
                print(f"   Recent items:")
                for item in review_queue[-3:]:  # Show last 3 items
                    print(f"      - Subject: {item.get('subject', 'No subject')}")
                    print(f"        Action: {item.get('action', 'unknown')}")
                    print(f"        Date: {item.get('processed_at', 'unknown')}")
        else:
            print(f"   📋 No review queue file found")
        
        return True
        
    except Exception as e:
        print(f"❌ Manual processing failed: {e}")
        import traceback
        print(f"Error details: {traceback.format_exc()}")
        return False

def show_system_status():
    """Show overall system status"""
    
    print(f"\n=== SYSTEM STATUS ===\n")
    
    # Check files
    files_to_check = [
        ("config/automation_config.json", "Configuration"),
        ("credentials.json", "Gmail Credentials"),
        ("token.pickle", "Gmail Token"),
        ("models/fine_tuned", "AI Model"),
        ("data/vector_db", "RAG Database"),
        ("logs/email_automation.log", "Processing Log")
    ]
    
    for file_path, description in files_to_check:
        if os.path.exists(file_path):
            print(f"   ✅ {description}: {file_path}")
        else:
            print(f"   ❌ {description}: {file_path} (missing)")
    
    # Check log size
    log_file = "logs/email_automation.log"
    if os.path.exists(log_file):
        size = os.path.getsize(log_file)
        print(f"   📊 Log file size: {size} bytes")

if __name__ == "__main__":
    show_system_status()
    success = manual_process_emails()
    
    if success:
        print(f"\n🎉 Manual processing completed successfully!")
        print(f"\n💡 TIPS:")
        print(f"1. Run this script periodically to process emails manually")
        print(f"2. Check 'data/admin_data/review_queue.json' for items needing review")
        print(f"3. Install missing dependencies to use the automated scheduler")
        print(f"4. Use the dashboard once dependencies are resolved")
    else:
        print(f"\n🔧 TROUBLESHOOTING:")
        print(f"1. Check if all required files exist")
        print(f"2. Verify Gmail credentials are working")
        print(f"3. Install missing Python packages")