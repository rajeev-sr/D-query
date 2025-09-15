# config/create_config.py
import json
import os

def create_automation_config():
    """Create automation configuration file"""
    
    config = {
        "processing_interval_minutes": 30,
        "max_emails_per_batch": 10,
        "auto_respond_enabled": True,
        
        "confidence_thresholds": {
            "auto_respond": 0.8,
            "review_needed": 0.5,
            "escalate": 0.0
        },
        
        "human_review_required_categories": [
            "complex", 
            "complaint",
            "technical"
        ],
        
        "escalation_keywords": [
            "urgent",
            "emergency", 
            "complaint",
            "angry",
            "frustrated",
            "disappointed",
            "wrong",
            "mistake",
            "problem",
            "issue"
        ],
        
        "email_query": "is:unread",
        
        "working_hours": {
            "enabled": True,
            "start": "09:00",
            "end": "17:00",
            "timezone": "UTC",
            "weekdays_only": True
        },
        
        "notifications": {
            "enabled": True,
            "email": None,  # Set your email for notifications
            "escalation_only": False,
            "daily_summary": True,
            "weekly_summary": True
        },
        
        "email_settings": {
            "use_gmail_api": True,
            "signature_enabled": True,
            "mark_as_read": True,
            "add_label": "AI_PROCESSED"
        },
        
        "safety_settings": {
            "max_auto_responses_per_hour": 50,
            "require_human_approval_for_new_senders": False,
            "blacklist_domains": [],
            "whitelist_domains": []
        }
    }
    
    # Create config directory
    os.makedirs("config", exist_ok=True)
    
    # Save config
    config_file = "config/automation_config.json"
    with open(config_file, 'w') as f:
        json.dump(config, f, indent=2)

    print(f"Configuration file created: {config_file}")
    print("Please review and customize the settings before running automation")
    
    return config_file

if __name__ == "__main__":
    create_automation_config()