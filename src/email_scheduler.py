import schedule
import time
import threading
from datetime import datetime, timedelta
from src.automated_processor import AutomatedEmailProcessor
import logging
import os
import signal
import sys

class EmailScheduler:
    def __init__(self, config_file="config/automation_config.json"):
        self.processor = AutomatedEmailProcessor(config_file)
        self.running = False
        self.thread = None
        
        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        # Setup logging
        os.makedirs('logs', exist_ok=True)
        logging.basicConfig(
            filename='logs/scheduler.log',
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully"""
        print(f"\n🛑 Received signal {signum}. Shutting down gracefully...")
        self.stop()
        sys.exit(0)
    
    def setup_schedule(self):
        """Setup processing schedule"""
        config = self.processor.config
        
        # Get schedule configuration
        interval = config.get("processing_interval_minutes", 30)
        working_hours = config.get("working_hours", {})
        
        if working_hours.get("start") and working_hours.get("end"):
            # Schedule within working hours
            start_time = working_hours["start"]
            end_time = working_hours["end"]
            
            # Schedule every X minutes during working hours
            current_time = datetime.strptime(start_time, "%H:%M")
            end_datetime = datetime.strptime(end_time, "%H:%M")
            
            while current_time <= end_datetime:
                time_str = current_time.strftime("%H:%M")
                schedule.every().day.at(time_str).do(self._scheduled_processing)
                current_time += timedelta(minutes=interval)
                
            print(f"📅 Scheduled processing every {interval} minutes from {start_time} to {end_time}")
        
        else:
            # Schedule every X minutes all day
            schedule.every(interval).minutes.do(self._scheduled_processing)
            print(f"📅 Scheduled processing every {interval} minutes (24/7)")
        
        # Additional schedules
        schedule.every().day.at("09:00").do(self._daily_summary)
        schedule.every().monday.at("08:00").do(self._weekly_summary)
    
    def _scheduled_processing(self):
        """Execute scheduled processing"""
        try:
            print(f"\n⏰ Scheduled processing started at {datetime.now()}")
            results = self.processor.process_emails_batch()
            
            # Log results
            logging.info(f"Scheduled processing completed: {results}")
            
            print(f"  Scheduled processing completed:")
            print(f"  Processed: {results.get('processed', 0)}")
            print(f"  Auto-responded: {results.get('auto_responded', 0)}")
            print(f"  Escalated: {results.get('escalated', 0)}")
            
        except Exception as e:
            print(f"❌ Scheduled processing failed: {e}")
            logging.error(f"Scheduled processing error: {e}")
    
    def _daily_summary(self):
        """Generate daily processing summary"""
        try:
            stats = self.processor.get_processing_stats()
            
            # Send daily summary email if configured
            notification_config = self.processor.config.get("notifications", {})
            if notification_config.get("enabled") and notification_config.get("email"):
                
                subject = f"Daily AI Email Processing Summary - {datetime.now().strftime('%Y-%m-%d')}"
                body = f"""
Daily Email Processing Summary:

Total Processed: {stats.get('total_processed', 0)}
Auto Response Rate: {stats.get('auto_response_rate', '0%')}
Escalation Rate: {stats.get('escalation_rate', '0%')}
Average Confidence: {stats.get('average_confidence', 0):.2f}

Action Breakdown:
{chr(10).join([f"- {action}: {count}" for action, count in stats.get('action_breakdown', {}).items()])}

System Status: Running
Last Processed: {stats.get('last_processed', 'N/A')}

Dashboard: http://localhost:8501
"""
                
                self.processor.email_sender.send_email(
                    notification_config["email"], 
                    subject, 
                    body
                )
            
            logging.info(f"Daily summary generated: {stats}")
            
        except Exception as e:
            logging.error(f"Daily summary error: {e}")
    
    def _weekly_summary(self):
        """Generate weekly processing summary"""
        # Similar to daily summary but with weekly stats
        print("Weekly summary generated")
        logging.info("Weekly summary generated")
    
    def start(self):
        """Start the scheduler"""
        if self.running:
            print("cheduler is already running")
            return
        
        print("Starting Email Scheduler...")
        
        # Setup schedule
        self.setup_schedule()
        
        # Start scheduler in separate thread
        self.running = True
        self.thread = threading.Thread(target=self._run_scheduler, daemon=True)
        self.thread.start()
        
        print("Email Scheduler started successfully")
        print("Use Ctrl+C to stop gracefully")
    
    def _run_scheduler(self):
        """Run the scheduler loop"""
        while self.running:
            try:
                schedule.run_pending()
                time.sleep(60)  # Check every minute
            except Exception as e:
                logging.error(f"Scheduler loop error: {e}")
                if not self.running:
                    break
                time.sleep(60)
    
    def stop(self):
        """Stop the scheduler"""
        if not self.running:
            print("⚠️ Scheduler is not running")
            return
        
        print("🛑 Stopping Email Scheduler...")
        self.running = False
        
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=5)
        
        print("Email Scheduler stopped")
    
    def run_once(self):
        """Run processing once for testing"""
        print("Running one-time processing...")
        results = self.processor.process_emails_batch()
        
        print("Results:")
        for key, value in results.items():
            if key != "details":
                print(f"  {key}: {value}")
        
        return results
    
    def status(self):
        """Get scheduler status"""
        return {
            "running": self.running,
            "scheduled_jobs": len(schedule.jobs),
            "next_run": str(schedule.next_run()) if schedule.jobs else None,
            "processor_stats": self.processor.get_processing_stats()
        }

# Command-line interface
def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="AI Email Scheduler")
    parser.add_argument("--config", default="config/automation_config.json", 
                       help="Configuration file path")
    parser.add_argument("--once", action="store_true", 
                       help="Run processing once and exit")
    parser.add_argument("--status", action="store_true", 
                       help="Show scheduler status")
    
    args = parser.parse_args()
    
    scheduler = EmailScheduler(args.config)
    
    if args.once:
        scheduler.run_once()
    elif args.status:
        status = scheduler.status()
        print("Scheduler Status:")
        for key, value in status.items():
            print(f"  {key}: {value}")
    else:
        # Run continuously
        scheduler.start()
        
        try:
            # Keep main thread alive
            while scheduler.running:
                time.sleep(1)
        except KeyboardInterrupt:
            scheduler.stop()

if __name__ == "__main__":
    main()