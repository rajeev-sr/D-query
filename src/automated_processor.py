# src/automated_processor.py
from src.enhanced_decision_engine import EnhancedDecisionEngine
from src.gmail_client import GmailClient
from src.email_processor import EmailProcessor
from src.email_sender import EmailSender
from src.gemini_filter import GeminiQueryFilter
from typing import Dict, List
import json
import os
from datetime import datetime, timedelta
import logging
import time

class AutomatedEmailProcessor:
    def __init__(self, config_file="config/automation_config.json"):
        self.config_file = config_file
        self.config = self._load_config()
        
        # Initialize components
        self.decision_engine = EnhancedDecisionEngine()
        self.gmail_client = GmailClient()
        self.email_processor = EmailProcessor()
        self.email_sender = EmailSender()
        
        # Initialize Gemini filter if API key is available
        try:
            self.gemini_filter = GeminiQueryFilter()
            self.gemini_enabled = True
            logging.info("Gemini query filter initialized successfully")
        except Exception as e:
            self.gemini_filter = None
            self.gemini_enabled = False
            logging.warning(f"Gemini filter not available: {e}")
        
        # Processing tracking
        self.processing_log = []
        self.last_processed_time = None
        self.filtered_stats = {
            'total_fetched': 0,
            'query_emails': 0,
            'non_query_emails': 0,
            'student_emails': 0
        }
        
        # Setup logging
        logging.basicConfig(
            filename='logs/email_automation.log',
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
    
    def _load_config(self):
        """Load automation configuration"""
        default_config = {
            "processing_interval_minutes": 30,
            "max_emails_per_batch": 10,
            "auto_respond_enabled": True,
            "gemini_filter_enabled": True,  # Enable Gemini query filtering
            "human_review_required_categories": ["complex", "complaint"],
            "escalation_keywords": ["urgent", "complaint", "angry", "frustrated"],
            "confidence_thresholds": {
                "auto_respond": 0.8,
                "review_needed": 0.5
            },
            "email_query": "is:unread",
            "working_hours": {
                "start": "09:00",
                "end": "17:00",
                "timezone": "UTC"
            },
            "notifications": {
                "enabled": True,
                "email": None,
                "escalation_only": False
            }
        }
        
        try:
            if os.path.exists(self.config_file):
                with open(self.config_file, 'r') as f:
                    user_config = json.load(f)
                    default_config.update(user_config)
            else:
                # Create config file with defaults
                os.makedirs(os.path.dirname(self.config_file), exist_ok=True)
                with open(self.config_file, 'w') as f:
                    json.dump(default_config, f, indent=2)
                print(f"📁 Created default config file: {self.config_file}")
        
        except Exception as e:
            print(f"⚠️ Config loading failed: {e}")
            print("Using default configuration")
        
        return default_config
    
    def process_emails_batch(self) -> Dict:
        """Process a batch of emails with Gemini query filtering"""
        start_time = datetime.now()
        
        try:
            # Fetch emails
            query = self.config.get("email_query", "is:unread")
            max_emails = self.config.get("max_emails_per_batch", 10)
            
            logging.info(f"Fetching emails with query: {query}, max: {max_emails}")
            emails = self.gmail_client.fetch_emails(query=query, max_results=max_emails)
            
            # Update stats
            self.filtered_stats['total_fetched'] = len(emails)
            
            if not emails:
                return {
                    "status": "success",
                    "processed": 0,
                    "auto_responded": 0,
                    "escalated": 0,
                    "review_needed": 0,
                    "errors": 0,
                    "filtered_out": 0,
                    "message": "No new emails found"
                }
            
            # Filter student emails
            student_emails = self.email_processor.filter_student_emails(emails)
            self.filtered_stats['student_emails'] = len(student_emails)
            
            if not student_emails:
                return {
                    "status": "success",
                    "processed": 0,
                    "auto_responded": 0,
                    "escalated": 0,
                    "review_needed": 0,
                    "errors": 0,
                    "filtered_out": len(emails),
                    "message": "No student emails found"
                }
            
            # Apply Gemini query filtering if enabled
            query_emails = student_emails
            filtered_out_count = 0
            
            if self.gemini_enabled and self.config.get("gemini_filter_enabled", True):
                logging.info(f"Applying Gemini query filter to {len(student_emails)} student emails...")
                try:
                    query_emails, non_query_emails = self.gemini_filter.filter_query_emails(student_emails)
                    filtered_out_count = len(non_query_emails)
                    
                    # Update stats
                    self.filtered_stats['query_emails'] = len(query_emails)
                    self.filtered_stats['non_query_emails'] = len(non_query_emails)
                    
                    logging.debug(f"Gemini filtering: {len(query_emails)} queries, {len(non_query_emails)} non-queries")
                    
                    # Log filtered out emails
                    for email in non_query_emails:
                        logging.info(f"Filtered out: {email.get('subject', 'No Subject')} - {email.get('gemini_filter_reason', 'Unknown reason')}")
                        
                except Exception as e:
                    logging.error(f"Gemini filtering failed, processing all emails: {e}")
                    query_emails = student_emails
            
            if not query_emails:
                return {
                    "status": "success",
                    "processed": 0,
                    "auto_responded": 0,
                    "escalated": 0,
                    "review_needed": 0,
                    "errors": 0,
                    "filtered_out": filtered_out_count,
                    "message": f"No query-related emails found. Filtered out {filtered_out_count} non-query emails"
                }
            
            # Process each query email
            results = {
                "processed": 0,
                "auto_responded": 0,
                "escalated": 0,
                "review_needed": 0,
                "errors": 0,
                "filtered_out": filtered_out_count,
                "details": []
            }
            
            for email in query_emails:
                try:
                    result = self._process_single_email(email)
                    
                    # Update counters
                    results["processed"] += 1
                    results[result["action"]] = results.get(result["action"], 0) + 1
                    results["details"].append(result)
                    
                    # Log processing
                    self._log_processing(email, result)
                    
                except Exception as e:
                    results["errors"] += 1
                    logging.error(f"Error processing email {email.get('id')}: {e}")
            
            # Update last processed time
            self.last_processed_time = start_time
            
            # Send notifications if configured
            if results["escalated"] > 0 or results["review_needed"] > 0:
                self._send_notification(results)
            
            results["status"] = "success"
            results["processing_time"] = (datetime.now() - start_time).total_seconds()
            
            return results
            
        except Exception as e:
            logging.error(f"Batch processing failed: {e}")
            return {
                "status": "error",
                "error": str(e),
                "processed": 0
            }
    
    def _process_single_email(self, email: Dict) -> Dict:
        """Process a single email"""
        
        # Get AI decision
        decision = self.decision_engine.process_query(email)
        
        # Apply automation rules
        action = self._apply_automation_rules(decision, email)
        
        result = {
            "email_id": email.get('id'),
            "subject": email.get('subject'),
            "sender": email.get('sender'),
            "action": action,
            "ai_decision": decision,
            "processed_at": datetime.now().isoformat(),
            "automated": True
        }
        
        # Execute action
        if action == "auto_respond" and self.config.get("auto_respond_enabled", True):
            sent = self._send_auto_response(email, decision)
            result["response_sent"] = sent
            
            if sent:
                # Mark email as read/handled
                self._mark_email_handled(email['id'])
        
        elif action == "review_needed":
            # Add to review queue (would integrate with dashboard)
            self._add_to_review_queue(email, decision)
        
        elif action == "escalate":
            # Handle escalation
            self._handle_escalation(email, decision)
        
        return result
    
    def _apply_automation_rules(self, decision: Dict, email: Dict) -> str:
        """Apply automation business rules"""
        
        # Rule 1: Check escalation keywords
        email_content = f"{email.get('subject', '')} {email.get('body', '')}".lower()
        escalation_keywords = self.config.get("escalation_keywords", [])
        
        if any(keyword in email_content for keyword in escalation_keywords):
            return "escalate"
        
        # Rule 2: Check confidence thresholds
        confidence = decision.get('confidence', 0)
        thresholds = self.config.get("confidence_thresholds", {})
        
        if confidence >= thresholds.get("auto_respond", 0.8):
            # Rule 3: Check category restrictions
            category = decision.get('category', '')
            restricted_categories = self.config.get("human_review_required_categories", [])
            
            if category in restricted_categories:
                return "review_needed"
            
            return "auto_respond"
        
        elif confidence >= thresholds.get("review_needed", 0.5):
            return "review_needed"
        
        else:
            return "escalate"
    
    def _send_auto_response(self, email: Dict, decision: Dict) -> bool:
        """Send automated response"""
        try:
            response_text = decision.get('response', 'Thank you for your query. We will get back to you soon.')
            
            # Format email
            formatted_email = self.email_sender.format_response_email(email, response_text)
            
            # Send email
            success = self.email_sender.send_email(
                to_email=formatted_email['to'],
                subject=formatted_email['subject'],
                body=formatted_email['body'],
                reply_to_id=formatted_email.get('reply_to_id')
            )
            
            if success:
                logging.debug(f"Auto-response sent for email {email.get('id')}")
            else:
                logging.error(f"Failed to send auto-response for email {email.get('id')}")
            
            return success
            
        except Exception as e:
            logging.error(f"Auto-response error for email {email.get('id')}: {e}")
            return False
    
    def _mark_email_handled(self, email_id: str):
        """Mark email as handled in Gmail"""
        try:
            # Mark as read and add label
            self.gmail_client.service.users().messages().modify(
                userId='me',
                id=email_id,
                body={
                    'removeLabelIds': ['UNREAD'],
                    'addLabelIds': []  # Could add custom label like 'AI_HANDLED'
                }
            ).execute()
            
        except Exception as e:
            # This is expected with limited Gmail API permissions - not a critical error
            logging.debug(f"Could not mark email {email_id} as handled (expected with limited permissions): {e}")
    
    def _add_to_review_queue(self, email: Dict, decision: Dict):
        """Add email to review queue"""
        review_item = {
            "email": email,
            "decision": decision,
            "added_at": datetime.now().isoformat(),
            "status": "pending_review"
        }
        
        # Save to review queue file
        queue_file = "data/review_queue.json"
        os.makedirs(os.path.dirname(queue_file), exist_ok=True)
        
        try:
            if os.path.exists(queue_file):
                with open(queue_file, 'r') as f:
                    queue = json.load(f)
            else:
                queue = []
            
            queue.append(review_item)
            
            with open(queue_file, 'w') as f:
                json.dump(queue, f, indent=2, default=str)
                
        except Exception as e:
            logging.error(f"Could not add to review queue: {e}")
    
    def _handle_escalation(self, email: Dict, decision: Dict):
        """Handle escalated emails"""
        escalation_item = {
            "email": email,
            "decision": decision,
            "escalated_at": datetime.now().isoformat(),
            "reason": decision.get('reason', 'Unknown'),
            "status": "escalated"
        }
        
        # Save to escalation file
        escalation_file = "data/escalations.json"
        os.makedirs(os.path.dirname(escalation_file), exist_ok=True)
        
        try:
            if os.path.exists(escalation_file):
                with open(escalation_file, 'r') as f:
                    escalations = json.load(f)
            else:
                escalations = []
            
            escalations.append(escalation_item)
            
            with open(escalation_file, 'w') as f:
                json.dump(escalations, f, indent=2, default=str)
                
            logging.warning(f"Email escalated: {email.get('subject')} from {email.get('sender')}")
                
        except Exception as e:
            logging.error(f"Could not save escalation: {e}")
    
    def _send_notification(self, results: Dict):
        """Send notification about processing results"""
        if not self.config.get("notifications", {}).get("enabled", False):
            return
        
        notification_email = self.config.get("notifications", {}).get("email")
        if not notification_email:
            return
        
        # Create notification message
        subject = f"AI Email Processor - {results['escalated']} Escalations, {results['review_needed']} Reviews"
        
        body = f"""
Email Processing Summary:

Processed: {results['processed']} emails
Auto Responded: {results['auto_responded']}
Need Review: {results['review_needed']}
Escalated: {results['escalated']}
Errors: {results['errors']}

Processing Time: {results.get('processing_time', 0):.1f} seconds

Please check the dashboard for details.
"""
        
        try:
            self.email_sender.send_email(notification_email, subject, body)
            logging.info("Notification sent successfully")
        except Exception as e:
            logging.error(f"Failed to send notification: {e}")
    
    def _log_processing(self, email: Dict, result: Dict):
        """Log processing details"""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "email_id": email.get('id'),
            "subject": email.get('subject'),
            "sender": email.get('sender'),
            "action": result.get('action'),
            "confidence": result.get('ai_decision', {}).get('confidence', 0),
            "response_sent": result.get('response_sent', False)
        }
        
        self.processing_log.append(log_entry)
        
        # Keep only last 1000 entries
        if len(self.processing_log) > 1000:
            self.processing_log = self.processing_log[-1000:]
    
    def run_continuous_processing(self, run_once=False):
        """Run continuous email processing"""
        interval = self.config.get("processing_interval_minutes", 30) * 60  # Convert to seconds
        
        print(f" Starting automated email processing...")
        print(f" Processing interval: {interval/60:.1f} minutes")
        print(f" Max emails per batch: {self.config.get('max_emails_per_batch', 10)}")
        print(f" Auto-respond enabled: {self.config.get('auto_respond_enabled', True)}")
        
        while True:
            try:
                print(f"\n⏰ Starting processing batch at {datetime.now()}")
                results = self.process_emails_batch()
                
                print(f"  Batch completed:")
                print(f"  Processed: {results.get('processed', 0)}")
                print(f"  Auto-responded: {results.get('auto_responded', 0)}")
                print(f"  Review needed: {results.get('review_needed', 0)}")
                print(f"  Escalated: {results.get('escalated', 0)}")
                print(f"  Errors: {results.get('errors', 0)}")
                
                if run_once:
                    break
                
                print(f"😴 Sleeping for {interval/60:.1f} minutes...")
                time.sleep(interval)
                
            except KeyboardInterrupt:
                print("\n🛑 Processing stopped by user")
                break
            except Exception as e:
                print(f"❌ Processing error: {e}")
                logging.error(f"Continuous processing error: {e}")
                
                if not run_once:
                    print(f"⏳ Retrying in 5 minutes...")
                    time.sleep(300)  # Wait 5 minutes before retrying
                else:
                    break
    
    def get_processing_stats(self) -> Dict:
        """Get processing statistics"""
        if not self.processing_log:
            return {"message": "No processing data available"}
        
        # Calculate stats
        total_processed = len(self.processing_log)
        actions = {}
        
        for entry in self.processing_log:
            action = entry.get('action', 'unknown')
            actions[action] = actions.get(action, 0) + 1
        
        # Calculate rates
        auto_response_rate = actions.get('auto_respond', 0) / total_processed * 100
        escalation_rate = actions.get('escalate', 0) / total_processed * 100
        
        return {
            "total_processed": total_processed,
            "action_breakdown": actions,
            "auto_response_rate": f"{auto_response_rate:.1f}%",
            "escalation_rate": f"{escalation_rate:.1f}%",
            "last_processed": self.last_processed_time.isoformat() if self.last_processed_time else None,
            "average_confidence": sum(entry.get('confidence', 0) for entry in self.processing_log) / total_processed
        }
    
    def get_filtering_stats(self) -> Dict:
        """Get email filtering statistics"""
        stats = self.filtered_stats.copy()
        
        # Add Gemini filter status
        stats['gemini_filter_enabled'] = self.gemini_enabled and self.config.get('gemini_filter_enabled', True)
        stats['gemini_available'] = self.gemini_enabled
        
        # Calculate rates
        if stats['total_fetched'] > 0:
            stats['student_email_rate'] = (stats['student_emails'] / stats['total_fetched']) * 100
            if stats['student_emails'] > 0:
                stats['query_rate'] = (stats['query_emails'] / stats['student_emails']) * 100
                stats['filter_efficiency'] = (stats['non_query_emails'] / stats['student_emails']) * 100
            else:
                stats['query_rate'] = 0
                stats['filter_efficiency'] = 0
        else:
            stats['student_email_rate'] = 0
            stats['query_rate'] = 0
            stats['filter_efficiency'] = 0
        
        return stats
    
    def toggle_gemini_filter(self, enabled: bool) -> bool:
        """Enable or disable Gemini filtering"""
        if not self.gemini_enabled:
            return False
        
        self.config['gemini_filter_enabled'] = enabled
        self._save_config()
        
        logging.info(f"Gemini filtering {'enabled' if enabled else 'disabled'}")
        return True