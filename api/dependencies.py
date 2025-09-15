from fastapi import Depends, HTTPException, status
from functools import lru_cache
import sys
import os
import json
import logging

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logger = logging.getLogger(__name__)

# Global instances (singleton pattern)
_gmail_client = None
_decision_engine = None  
_email_processor = None
_automated_processor = None
_config = None

from api.mock_gmail_client import MockGmailClient

@lru_cache()
def get_config():
    """Load and cache configuration"""
    global _config
    if _config is None:
        try:
            config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                                     'config', 'automation_config.json')
            with open(config_path, 'r') as f:
                _config = json.load(f)
            logger.info("Configuration loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to load system configuration"
            )
    return _config

async def get_gmail_client():
    """Get or create Gmail client instance"""
    global _gmail_client
    if _gmail_client is None:
        try:
            # Check if credentials.json exists
            credentials_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'credentials.json')
            if os.path.exists(credentials_path):
                # Use real Gmail client when credentials are available
                from src.gmail_client import GmailClient
                _gmail_client = GmailClient()
                logger.info("Gmail client initialized")
            else:
                # Use mock client for demo mode
                from .mock_gmail_client import MockGmailClient
                _gmail_client = MockGmailClient()
                logger.info("Mock Gmail client initialized (Demo Mode)")
        except Exception as e:
            # Fallback to mock client if real client fails
            try:
                from .mock_gmail_client import MockGmailClient
                _gmail_client = MockGmailClient()
                logger.info("Fallback to Mock Gmail client (Demo Mode)")
            except Exception as fallback_error:
                logger.error(f"Failed to initialize any Gmail client: {e}, {fallback_error}")
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="Failed to initialize Gmail client"
                )
    return _gmail_client

async def get_decision_engine():
    """Get or create decision engine instance"""
    global _decision_engine
    if _decision_engine is None:
        try:
            from src.enhanced_decision_engine import EnhancedDecisionEngine
            config = get_config()
            # Initialize without parameters since EnhancedDecisionEngine has defaults
            _decision_engine = EnhancedDecisionEngine()
            logger.info("Decision engine initialized")
        except Exception as e:
            logger.error(f"Failed to initialize decision engine: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to initialize decision engine"
            )
    return _decision_engine

async def get_email_processor():
    """Get or create email processor instance"""
    global _email_processor
    if _email_processor is None:
        try:
            from src.email_processor import EmailProcessor
            _email_processor = EmailProcessor()
            logger.info("Email processor initialized")
        except Exception as e:
            logger.error(f"Failed to initialize email processor: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to initialize email processor"
            )
    return _email_processor

async def get_automated_processor():
    """Get or create automated processor instance"""
    global _automated_processor
    if _automated_processor is None:
        try:
            from src.automated_processor import AutomatedEmailProcessor
            config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                                     'config', 'automation_config.json')
            _automated_processor = AutomatedEmailProcessor(config_path=config_path)
            logger.info("Automated processor initialized")
        except Exception as e:
            logger.error(f"Failed to initialize automated processor: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to initialize automated processor"
            )
    return _automated_processor

def get_email_sender():
    """Get or create email sender instance"""
    try:
        from src.email_sender import EmailSender
        config = get_config()
        use_gmail_api = config.get('email_settings', {}).get('use_gmail_api', True)
        return EmailSender(use_gmail_api=use_gmail_api)
    except Exception as e:
        logger.error(f"Failed to initialize email sender: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to initialize email sender"
        )

def update_config(new_config: dict):
    """Update configuration and reload dependencies"""
    global _config, _decision_engine, _automated_processor
    
    try:
        config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                                 'config', 'automation_config.json')
        
        # Save the new configuration
        with open(config_path, 'w') as f:
            json.dump(new_config, f, indent=2)
        
        # Clear cached config
        _config = new_config
        get_config.cache_clear()
        
        # Reinitialize components that depend on config
        _decision_engine = None
        _automated_processor = None
        
        logger.info("Configuration updated successfully")
        return True
        
    except Exception as e:
        logger.error(f"Failed to update configuration: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to update configuration: {str(e)}"
        )

def get_automated_processor():
    """Get automated email processor instance"""
    global _automated_processor
    
    if _automated_processor is None:
        try:
            # Try to import and initialize the automated processor
            from src.automated_processor import AutomatedEmailProcessor
            
            # Use mock Gmail client if real one is not available
            gmail_client = get_gmail_client()
            
            # Create a custom processor that uses our Gmail client
            _automated_processor = AutomatedEmailProcessor()
            
            # Replace the Gmail client with our dependency-injected one
            _automated_processor.gmail_client = gmail_client
            
            logger.info("✅ Automated processor initialized")
            
        except ImportError as e:
            logger.warning(f"⚠️ Could not import automated processor: {e}")
            logger.info("Using mock automated processor for demo mode")
            # Create a mock processor for demo purposes
            _automated_processor = MockAutomatedProcessor()
        except Exception as e:
            logger.error(f"❌ Failed to initialize automated processor: {e}")
            _automated_processor = MockAutomatedProcessor()
    
    return _automated_processor

class MockAutomatedProcessor:
    """Mock automated processor for demo mode"""
    
    def __init__(self):
        self.config = get_config()
        self.last_processed_time = None
        self.processing_log = []
        self._is_running = False
        
        # Use the mock Gmail client
        self.gmail_client = get_gmail_client()
    
    def process_emails_batch(self):
        """Mock batch processing"""
        import random
        from datetime import datetime
        
        # Simulate processing some emails
        processed = random.randint(1, 5)
        auto_responded = random.randint(0, processed)
        review_needed = processed - auto_responded
        
        result = {
            "status": "success",
            "processed": processed,
            "auto_responded": auto_responded,
            "escalated": 0,
            "review_needed": review_needed,
            "errors": 0,
            "details": [],
            "processing_time": random.uniform(1.0, 5.0),
            "message": f"Mock processing completed: {processed} emails"
        }
        
        # Add mock details
        for i in range(processed):
            action = "auto_responded" if i < auto_responded else "review_needed"
            result["details"].append({
                "email_id": f"mock_{i}",
                "subject": f"Mock Email {i}",
                "sender": f"student{i}@university.edu",
                "action": action,
                "ai_decision": {
                    "confidence": random.uniform(0.3, 0.9),
                    "response": "Mock AI response"
                },
                "processed_at": datetime.now().isoformat()
            })
        
        self.last_processed_time = datetime.now()
        return result
    
    def _is_within_working_hours(self):
        """Mock working hours check"""
        return True