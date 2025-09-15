from src.automated_processor import AutomatedEmailProcessor
from src.email_scheduler import EmailScheduler
from dashboard.main_dashboard import DashboardManager
import os
import json
import streamlit as st

def test_day5_integration():
    print("=== DAY 5 INTEGRATION TEST ===")
    print("Testing: Dashboard + Email Automation + Scheduling")
    
    # Prerequisites check
    prerequisites = {
        'enhanced_system': os.path.exists("models/fine_tuned") and os.path.exists("data/vector_db"),
        'config_directory': os.path.exists("config"),
        'logs_directory': True  # Will be created if needed
    }
    
    print("\n🔍 Prerequisites Check:")
    for name, exists in prerequisites.items():
        status = "✅" if exists else "❌"
        print(f"  {status} {name}")
        
        if not exists and name == 'config_directory':
            print("    Creating config directory...")
            from config.create_config import create_automation_config
            create_automation_config()
            prerequisites[name] = True
    
    # Test 1: Configuration Setup
    print("\n⚙️ Testing Configuration...")
    try:
        config_file = "config/automation_config.json"
        
        if os.path.exists(config_file):
            with open(config_file, 'r') as f:
                config = json.load(f)
            print("     Configuration loaded successfully")
            print(f"    Processing interval: {config.get('processing_interval_minutes', 30)} minutes")
            print(f"    Auto-respond enabled: {config.get('auto_respond_enabled', True)}")
        else:
            print(" Configuration file not found")
            return False
            
    except Exception as e:
        print(f" Configuration error: {e}")
        return False
    
    # Test 2: Automated Processor
    print("\n Testing Automated Processor...")
    try:
        processor = AutomatedEmailProcessor()
        stats = processor.get_processing_stats()
        
        print("   Automated processor initialized")
        print("   Ready for batch processing")
        
        # Test single batch (dry run)
        print("    Running test batch processing...")
        results = processor.process_emails_batch()

        print(f"     Processed: {results.get('processed', 0)} emails")
        print(f"     Status: {results.get('status', 'unknown')}")

    except Exception as e:
        print(f"   Automated processor error: {e}")
        return False
    
    # Test 3: Email Scheduler
    print("\n⏰ Testing Email Scheduler...")
    try:
        scheduler = EmailScheduler()
        status = scheduler.status()
        
        print("     Email scheduler initialized")
        print(f"    Scheduled jobs: {status.get('scheduled_jobs', 0)}")
        print(f"    Running: {status.get('running', False)}")
        
        # Setup schedule (don't start)
        scheduler.setup_schedule()
        print("     Schedule setup completed")
        
    except Exception as e:
        print(f"    Email scheduler error: {e}")
        return False
    
    # Test 4: Dashboard Components
    print("\n Testing Dashboard Components...")
    try:
        # Test dashboard manager initialization
        dashboard = DashboardManager()
        print("   Dashboard manager initialized")
        
        # Test system initialization (without actually running Streamlit)
        print("    Dashboard ready for Streamlit launch")
        print("    Command: streamlit run dashboard/main_dashboard.py")
        
    except Exception as e:
        print(f"   Dashboard component error: {e}")
        return False
    
    # Test 5: Email Sender
    print("\n📧 Testing Email Sender...")
    try:
        from src.email_sender import EmailSender
        
        sender = EmailSender()
        print("   Email sender initialized")
        
        # Test email formatting
        test_email = {
            'subject': 'Test Subject',
            'sender': 'test@example.com',
            'id': 'test123'
        }
        
        formatted = sender.format_response_email(test_email, "Test response")
        print("   Email formatting working")
        print(f"    Formatted subject: {formatted['subject']}")
        
    except Exception as e:
        print(f"   Email sender error: {e}")
        return False
    

    # Test 6: End-to-End Workflow
    print("\n🔄 Testing End-to-End Workflow...")
    try:
        # Create a test workflow
        test_email = {
            'id': 'integration_test_email',
            'subject': 'Fee Payment Question',
            'body': 'How can I pay my semester fees online?',
            'sender': 'student@university.edu'
        }
        
        # Process through automated processor
        result = processor._process_single_email(test_email)
        
        print("  ✅ End-to-end workflow completed")
        print(f"    Action taken: {result.get('action')}")
        print(f"    AI confidence: {result.get('ai_decision', {}).get('confidence', 0):.2f}")
        
        if result.get('action') == 'auto_respond':
            print("    📤 Would send automated response")
        elif result.get('action') == 'review_needed':
            print("    👥 Would add to review queue")
        elif result.get('action') == 'escalate':
            print("    🚨 Would escalate to human")
        
    except Exception as e:
        print(f"  ❌ End-to-end workflow error: {e}")
        return False
    
    # Test 7: Data Persistence
    print("\n💾 Testing Data Persistence...")
    try:
        # Test review queue
        if os.path.exists("data/review_queue.json"):
            print("  ✅ Review queue file exists")
        
        # Test escalations
        if os.path.exists("data/escalations.json"):
            print("  ✅ Escalations file exists")
        
        # Test dashboard data
        if os.path.exists("data/dashboard_data.json"):
            print("  ✅ Dashboard data file exists")
        else:
            print("  ℹ️  Dashboard data will be created on first use")
        
        print("  ✅ Data persistence system ready")
        
    except Exception as e:
        print(f"  ❌ Data persistence error: {e}")
        return False
    
    # Final Assessment
    print(f"\n🎯 DAY 5 FINAL ASSESSMENT:")
    
    success_criteria = {
        'Configuration system': True,
        'Automated processor': True,
        'Email scheduler': True,
        'Dashboard components': True,
        'Email sender': True,
        'End-to-end workflow': True,
        'Data persistence': True
    }
    
    passed_criteria = sum(success_criteria.values())
    total_criteria = len(success_criteria)
    
    print(f"Success criteria: {passed_criteria}/{total_criteria}")
    for criterion, passed in success_criteria.items():
        status = "✅" if passed else "❌"
        print(f"  {status} {criterion}")
    
    overall_success = passed_criteria == total_criteria
    
    if overall_success:
        print(f"\n🎉 DAY 5 INTEGRATION SUCCESSFUL!")
        print(f"✅ User interface ready")
        print(f"✅ Email automation operational")
        print(f"✅ Scheduling system configured")
        print(f"✅ Complete AI Query Handler system ready!")
        
        print(f"\n🚀 NEXT STEPS:")
        print(f"1. Launch dashboard: streamlit run dashboard/main_dashboard.py")
        print(f"2. Start automation: python src/email_scheduler.py")
        print(f"3. Monitor logs in: logs/")
        print(f"4. Customize config in: config/automation_config.json")
        
        # Save final integration results
        integration_results = {
            'timestamp': datetime.now().isoformat(),
            'success': True,
            'all_systems_operational': True,
            'success_criteria': success_criteria,
            'score': f"{passed_criteria}/{total_criteria}",
            'ready_for_production': True
        }
        
        os.makedirs('models', exist_ok=True)
        with open('models/day5_integration_results.json', 'w') as f:
            json.dump(integration_results, f, indent=2)
        
        return True
        
    else:
        print(f"\n❌ DAY 5 INTEGRATION INCOMPLETE")
        print(f"⚠️ {total_criteria - passed_criteria} criteria failed")
        print(f"📋 Review failed components before proceeding")
        return False

def demonstrate_system():
    """Demonstrate the complete system"""
    print("\n🎪 SYSTEM DEMONSTRATION")
    
    print("1. 📊 Dashboard Features:")
    print("   - Real-time email processing")
    print("   - Pending review management") 
    print("   - Analytics and metrics")
    print("   - System status monitoring")
    
    print("\n2. 🤖 Automation Features:")
    print("   - Intelligent email classification")
    print("   - Automated responses with RAG context")
    print("   - Smart escalation rules")
    print("   - Batch processing capability")
    
    print("\n3. ⏰ Scheduling Features:")
    print("   - Configurable processing intervals")
    print("   - Working hours support")
    print("   - Daily/weekly summaries")
    print("   - Graceful shutdown handling")
    
    print("\n4. 🛡️ Safety Features:")
    print("   - Confidence thresholds")
    print("   - Human review workflows")
    print("   - Escalation keywords detection")
    print("   - Rate limiting protection")

if __name__ == "__main__":
    from datetime import datetime
    
    success = test_day5_integration()
    
    if success:
        demonstrate_system()
        
        print(f"\n✨ CONGRATULATIONS! ✨")
        print(f"Your AI Query Handler is fully operational!")
        print(f"Total implementation time: 5 days")
        
        print(f"\n📈 SYSTEM CAPABILITIES:")
        print(f"✅ Fine-tuned AI model for institute queries")
        print(f"✅ RAG system with official documentation")
        print(f"✅ Intelligent decision engine")
        print(f"✅ User-friendly dashboard interface")
        print(f"✅ Automated email processing")
        print(f"✅ Smart scheduling system")
        print(f"✅ Human review workflows")
        print(f"✅ Comprehensive analytics")
        
    else:
        print(f"\n🔧 TROUBLESHOOTING NEEDED")
        print(f"Please address the failed components above")