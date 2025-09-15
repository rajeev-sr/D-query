# dashboard/main_dashboard.py
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import json
import os
import sys
import time
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables first
load_dotenv()

# Add src to path for direct imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# Direct imports of your components
from src.enhanced_decision_engine import EnhancedDecisionEngine
from src.gmail_client import GmailClient
from src.email_processor import EmailProcessor
from src.automated_processor import AutomatedEmailProcessor
from src.gemini_filter import GeminiQueryFilter

class DashboardManager:
    def __init__(self):
        self.data_file = "data/dashboard_data.json"
        
        # Initialize session state
        if 'system_status' not in st.session_state:
            st.session_state.system_status = {}
        if 'processing_status' not in st.session_state:
            st.session_state.processing_status = {}
        if 'processed_emails' not in st.session_state:
            st.session_state.processed_emails = []
        if 'automation_running' not in st.session_state:
            st.session_state.automation_running = False
        if 'last_refresh' not in st.session_state:
            st.session_state.last_refresh = None
        
        # Initialize components directly
        self.automated_processor = None
        self.decision_engine = None
        self.gmail_client = None
        self.email_processor = None
        self.gemini_filter = None
        
        # Component initialization status
        self.components_initialized = False
    
    def initialize_components(self):
        """Initialize all system components directly"""
        if self.components_initialized:
            return True
        
        try:
            with st.spinner("Initializing system components..."):
                # Initialize decision engine
                self.decision_engine = EnhancedDecisionEngine()
                
                # Initialize Gmail client
                try:
                    self.gmail_client = GmailClient()
                    gmail_connected = True
                except Exception as e:
                    st.warning(f"Gmail connection failed: {e}")
                    gmail_connected = False
                
                # Initialize email processor
                self.email_processor = EmailProcessor()
                
                # Initialize Gemini filter
                try:
                    self.gemini_filter = GeminiQueryFilter()
                    gemini_available = True
                except Exception as e:
                    st.info(f"Gemini filter not available: {e}")
                    gemini_available = False
                
                # Initialize automated processor
                try:
                    self.automated_processor = AutomatedEmailProcessor()
                    # Override components with our initialized ones
                    self.automated_processor.decision_engine = self.decision_engine
                    self.automated_processor.gmail_client = self.gmail_client
                    self.automated_processor.email_processor = self.email_processor
                    if gemini_available:
                        self.automated_processor.gemini_filter = self.gemini_filter
                        self.automated_processor.gemini_enabled = True
                    else:
                        self.automated_processor.gemini_enabled = False
                    
                except Exception as e:
                    st.error(f"Failed to initialize automated processor: {e}")
                    return False
                
                # Get system stats
                stats = self.decision_engine.get_system_stats()
                
                # Update system status
                st.session_state.system_status = {
                    'status': 'healthy' if stats.get('model_loaded') and gmail_connected else 'degraded',
                    'gmail_connection': gmail_connected,
                    'ai_model_loaded': stats.get('model_loaded', False),
                    'decision_engine': True,
                    'database_connection': True,  # Assuming local file storage works
                    'gemini_available': gemini_available
                }
                
                self.components_initialized = True
                st.success("✅ All components initialized successfully!")
                return True
                
        except Exception as e:
            st.error(f"❌ Failed to initialize components: {e}")
            return False
    
    def get_processing_status(self):
        """Get current processing status directly from components"""
        if not self.components_initialized:
            return False
        
        try:
            # Get stats from automated processor
            processing_stats = self.automated_processor.get_processing_stats() if hasattr(self.automated_processor, 'get_processing_stats') else {}
            filtering_stats = self.automated_processor.get_filtering_stats() if hasattr(self.automated_processor, 'get_filtering_stats') else {}
            
            st.session_state.processing_status = {
                'last_run_time': self.automated_processor.last_processed_time.isoformat() if self.automated_processor.last_processed_time else 'Never',
                'config': self.automated_processor.config,
                'automation_running': getattr(self.automated_processor, '_is_running', False),
                'processing_stats': processing_stats,
                'filtering_stats': filtering_stats,
                'gemini_available': self.automated_processor.gemini_enabled,
                'gemini_filter_enabled': self.automated_processor.config.get('gemini_filter_enabled', True),
                'total_processed': len(self.automated_processor.processing_log),
                'today_stats': self._get_today_stats()
            }
            
            st.session_state.automation_running = getattr(self.automated_processor, '_is_running', False)
            return True
            
        except Exception as e:
            st.error(f"Failed to get processing status: {e}")
            return False
    
    def _get_today_stats(self):
        """Get today's processing statistics"""
        today = datetime.now().strftime("%Y-%m-%d")
        today_logs = [log for log in self.automated_processor.processing_log 
                     if log.get('timestamp', '').startswith(today)]
        
        stats = {
            'processed': len(today_logs),
            'auto_responses': len([log for log in today_logs if log.get('action') == 'auto_respond']),
            'escalations': len([log for log in today_logs if log.get('action') == 'escalate'])
        }
        return stats
    
    def start_automation(self):
        """Start email automation directly"""
        if not self.components_initialized:
            st.error("Components not initialized")
            return False
        
        try:
            # Set running flag
            self.automated_processor._is_running = True
            st.session_state.automation_running = True
            st.success("✅ Automation started successfully")
            return True
            
        except Exception as e:
            st.error(f"Failed to start automation: {e}")
            return False
    
    def stop_automation(self):
        """Stop email automation directly"""
        if not self.components_initialized:
            st.error("Components not initialized")
            return False
        
        try:
            # Clear running flag
            self.automated_processor._is_running = False
            st.session_state.automation_running = False
            st.success("✅ Automation stopped successfully")
            return True
            
        except Exception as e:
            st.error(f"Failed to stop automation: {e}")
            return False
    
    def process_batch_emails(self, max_emails=10):
        """Process batch of emails directly"""
        if not self.components_initialized:
            st.error("Components not initialized")
            return None
        
        try:
            # Update batch size in config
            self.automated_processor.config['max_emails_per_batch'] = max_emails
            
            # Process batch
            result = self.automated_processor.process_emails_batch()
            
            if result.get("status") == "success":
                st.success(f"✅ Processed {result.get('processed', 0)} emails")
                
                # Update processed emails in session
                self._update_processed_emails_from_result(result)
                return result
            else:
                st.error(f"❌ Batch processing failed: {result.get('error', 'Unknown error')}")
                return None
                
        except Exception as e:
            st.error(f"Cannot process emails: {e}")
            return None
    
    def _update_processed_emails_from_result(self, result):
        """Update session state with processed emails"""
        processed_emails = []
        for detail in result.get('details', []):
            email_data = {
                'id': detail.get('email_id', 'unknown'),
                'subject': detail.get('subject', 'No Subject'),
                'sender': detail.get('sender', 'Unknown'),
                'body': detail.get('body', 'No Content'),
                'action': detail.get('action', 'unknown'),
                'confidence': detail.get('ai_decision', {}).get('confidence', 0.0),
                'category': detail.get('ai_decision', {}).get('category', 'unknown'),
                'response': detail.get('ai_decision', {}).get('response'),
                'rag_sources': detail.get('ai_decision', {}).get('rag_sources', []),
                'processed_at': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'gemini_filtered': detail.get('gemini_filtered', False),
                'gemini_filter_result': detail.get('gemini_filter_result')
            }
            processed_emails.append(email_data)
        
        # Add to session state
        st.session_state.processed_emails.extend(processed_emails)
        
        # Keep only last 100 emails
        if len(st.session_state.processed_emails) > 100:
            st.session_state.processed_emails = st.session_state.processed_emails[-100:]
    
    def get_recent_emails(self, limit=20):
        """Get recent processed emails from session state"""
        return st.session_state.processed_emails[-limit:] if st.session_state.processed_emails else []
    
    def get_dashboard_analytics(self):
        """Get analytics data directly from components"""
        if not self.components_initialized or not st.session_state.processed_emails:
            return None
        
        try:
            emails = st.session_state.processed_emails
            df = pd.DataFrame(emails)
            
            analytics = {
                'total_processed': len(emails),
                'automation_rate': len([e for e in emails if e.get('action') == 'auto_respond']) / len(emails) * 100,
                'average_confidence': df['confidence'].mean() if 'confidence' in df.columns else 0,
                'avg_response_time_minutes': 2.5,  # Placeholder
                'growth_rate': 0  # Placeholder
            }
            
            return analytics
            
        except Exception as e:
            st.error(f"Failed to get analytics: {e}")
            return None
    
    def update_config(self, config_data):
        """Update automation configuration directly"""
        if not self.components_initialized:
            st.error("Components not initialized")
            return False
        
        try:
            # Update the config
            self.automated_processor.config.update(config_data)
            
            # Save config
            self.automated_processor._save_config()
            
            st.success("✅ Configuration updated successfully")
            return True
            
        except Exception as e:
            st.error(f"Failed to update config: {e}")
            return False
    
    def toggle_gemini_filter(self, enabled):
        """Toggle Gemini filter directly"""
        if not self.components_initialized:
            st.error("Components not initialized")
            return False
        
        if not self.automated_processor.gemini_enabled:
            st.warning("Gemini filter not available (API key missing)")
            return False
        
        try:
            self.automated_processor.config['gemini_filter_enabled'] = enabled
            self.automated_processor._save_config()
            
            message = f"Gemini filtering {'enabled' if enabled else 'disabled'}"
            st.success(f"✅ {message}")
            return True
            
        except Exception as e:
            st.error(f"Failed to toggle Gemini filter: {e}")
            return False
    def get_filter_stats(self):
        """Get Gemini filter statistics directly"""
        if not self.components_initialized:
            return {}
        
        try:
            if hasattr(self.automated_processor, 'get_filtering_stats'):
                return self.automated_processor.get_filtering_stats()
            return {}
        except Exception as e:
            st.error(f"Failed to get filter stats: {e}")
            return {}

def main():
    st.set_page_config(
        page_title="AI Email Processing Dashboard",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS
    st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .status-good { color: #28a745; font-weight: bold; }
    .status-warning { color: #ffc107; font-weight: bold; }
    .status-error { color: #dc3545; font-weight: bold; }
    .automation-running { 
        color: #28a745; 
        font-size: 1.2rem; 
        font-weight: bold;
        animation: blink 2s linear infinite;
    }
    @keyframes blink {
        0% { opacity: 1; }
        50% { opacity: 0.5; }
        100% { opacity: 1; }
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.markdown('<h1 class="main-header">🤖 AI Email Processing Dashboard</h1>', unsafe_allow_html=True)
    
    # Initialize dashboard manager
    dashboard = DashboardManager()
    
    # Sidebar
    with st.sidebar:
        st.header("🎛️ Control Panel")
        
        # System Initialization
        st.subheader("� System Initialize")
        if st.button("🔄 Initialize System", type="primary"):
            with st.spinner("Initializing system components..."):
                dashboard.initialize_components()
                dashboard.get_processing_status()
        
        # System Status Display
        if st.session_state.system_status:
            st.subheader("📊 System Status")
            status = st.session_state.system_status
            processing_status = st.session_state.processing_status
            
            # Overall status
            overall_status = status.get('status', 'unknown')
            status_color = 'status-good' if overall_status == 'healthy' else 'status-warning' if overall_status == 'degraded' else 'status-error'
            st.markdown(f'<p class="{status_color}">Status: {overall_status.title()}</p>', unsafe_allow_html=True)
            
            # Component status
            components = {
                "Gmail": status.get('gmail_connection', False),
                "AI Model": status.get('ai_model_loaded', False),
                "Decision Engine": status.get('decision_engine', False),
                "Database": status.get('database_connection', False),
                "Gemini Filter": processing_status.get('gemini_available', False)
            }
            
            for component, is_healthy in components.items():
                icon = "✅" if is_healthy else "❌"
                st.write(f"{icon} {component}")
            
            # Gemini Filter Status
            if processing_status.get('gemini_available', False):
                gemini_enabled = processing_status.get('gemini_filter_enabled', False)
                st.write(f"🤖 Gemini Filter: {'🟢 ON' if gemini_enabled else '🔴 OFF'}")
                
                # Toggle button
                if st.button("🔄 Toggle Gemini Filter"):
                    dashboard.toggle_gemini_filter(not gemini_enabled)
                    dashboard.get_processing_status()  # Refresh status
        
        # Automation Controls
        st.subheader("🔄 Email Automation")
        
        # Show automation status
        if st.session_state.automation_running:
            st.markdown('<p class="automation-running">🟢 AUTOMATION RUNNING</p>', unsafe_allow_html=True)
        else:
            st.markdown('<p class="status-error">🔴 Automation Stopped</p>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("▶️ Start", disabled=st.session_state.automation_running):
                dashboard.start_automation()
                dashboard.get_processing_status()
        
        with col2:
            if st.button("⏹️ Stop", disabled=not st.session_state.automation_running):
                dashboard.stop_automation()
                dashboard.get_processing_status()
        
        # Manual Processing
        st.subheader("⚡ Quick Actions")
        
        if st.button("📧 Process Batch Emails"):
            with st.spinner("Processing emails..."):
                result = dashboard.process_batch_emails()
                if result:
                    dashboard.get_recent_emails()
        
        if st.button("📈 Refresh Data"):
            with st.spinner("Refreshing data..."):
                dashboard.check_api_health()
                dashboard.get_processing_status()
                dashboard.get_recent_emails()
                st.session_state.last_refresh = datetime.now()
        
        # Auto-refresh toggle
        auto_refresh = st.checkbox("Auto-refresh (30s)", value=False)
        if auto_refresh:
            if st.session_state.last_refresh is None or \
               (datetime.now() - st.session_state.last_refresh).seconds > 30:
                dashboard.check_api_health()
                dashboard.get_processing_status()
                dashboard.get_recent_emails()
                st.session_state.last_refresh = datetime.now()
                st.rerun()
    
    # Main dashboard content
    if not st.session_state.system_status:
        st.info("👆 Please check API health using the sidebar to begin")
        st.markdown("""
        ### 🚀 Getting Started
        
        1. Make sure the FastAPI backend is running: `python -m uvicorn api.main:app --reload`
        2. Click "Check API Health" in the sidebar
        3. Use the automation controls to start processing emails
        
        ### 📊 Dashboard Features
        
        - **Real-time monitoring** of email processing automation
        - **System health** monitoring with component status
        - **Email processing** controls and batch operations
        - **Analytics and insights** with interactive charts
        - **Configuration management** for automation settings
        """)
        return
    
    # Create tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "📧 Email Processing", "⚙️ Configuration", "📈 Analytics"])
    
    with tab1:
        show_overview_tab(dashboard)
    
    with tab2:
        show_email_processing_tab(dashboard)
    
    with tab3:
        show_configuration_tab(dashboard)
    
    with tab4:
        show_analytics_tab(dashboard)

def show_overview_tab(dashboard):
    """Show system overview and metrics"""
    st.header("📊 System Overview")
    
    # Get latest data
    with st.spinner("Loading overview data..."):
        dashboard.get_processing_status()
        recent_emails = dashboard.get_recent_emails(limit=10)
    
    # System Status Cards
    col1, col2, col3, col4 = st.columns(4)
    
    status = st.session_state.system_status
    processing_status = st.session_state.processing_status
    
    with col1:
        system_health = status.get('status', 'unknown').title()
        health_color = "🟢" if system_health == "Healthy" else "🟡" if system_health == "Degraded" else "🔴"
        st.metric("System Health", f"{health_color} {system_health}")
    
    with col2:
        automation_status = "🟢 Running" if st.session_state.automation_running else "🔴 Stopped"
        st.metric("Automation", automation_status)
    
    with col3:
        total_processed = processing_status.get('total_processed', 0)
        st.metric("Total Processed", total_processed)
    
    with col4:
        auto_responses = processing_status.get('auto_responses', 0)
        st.metric("Auto Responses", auto_responses)
    
    # Processing Metrics Row
    st.subheader("📈 Processing Metrics")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        last_batch_count = processing_status.get('last_batch_count', 0)
        st.metric("Last Batch Size", last_batch_count)
    
    with col2:
        avg_confidence = processing_status.get('average_confidence', 0)
        st.metric("Avg Confidence", f"{avg_confidence:.2f}")
    
    with col3:
        escalations = processing_status.get('escalations', 0)
        st.metric("Escalations", escalations)
    
    with col4:
        success_rate = processing_status.get('success_rate', 0)
        st.metric("Success Rate", f"{success_rate:.1f}%")
    
    # Gemini Filtering Statistics
    filter_stats = processing_status.get('filtering_stats', {})
    if filter_stats.get('gemini_available', False):
        st.subheader("🤖 Gemini Filter Statistics")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            total_fetched = filter_stats.get('total_fetched', 0)
            st.metric("Total Fetched", total_fetched)
        
        with col2:
            query_emails = filter_stats.get('query_emails', 0)
            st.metric("Query Emails", query_emails)
        
        with col3:
            filtered_out = filter_stats.get('non_query_emails', 0)
            st.metric("Filtered Out", filtered_out)
        
        with col4:
            filter_efficiency = filter_stats.get('filter_efficiency', 0)
            st.metric("Filter Rate", f"{filter_efficiency:.1f}%")
    
    # Recent Activity Timeline
    st.subheader("📋 Recent Activity")
    
    if recent_emails:
        # Create a timeline view of recent emails
        for i, email in enumerate(recent_emails[:5]):  # Show last 5
            with st.expander(f"📧 {email.get('subject', 'No Subject')[:60]}... ({email.get('action', 'unknown').title()})"):
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.write(f"**From:** {email.get('sender', 'Unknown')}")
                    st.write(f"**Action:** {email.get('action', 'Unknown').replace('_', ' ').title()}")
                    st.write(f"**Confidence:** {email.get('confidence', 0):.2f}")
                    
                    if email.get('category'):
                        st.write(f"**Category:** {email.get('category').title()}")
                    
                    if email.get('rag_sources'):
                        sources = email.get('rag_sources', [])
                        st.write(f"**Knowledge Sources:** {len(sources)} sources used")
                
                with col2:
                    processed_at = email.get('processed_at', 'Unknown')
                    st.write(f"**Processed:** {processed_at}")
                    
                    if email.get('response_sent'):
                        st.success("✅ Response Sent")
                    elif email.get('action') == 'auto_respond':
                        st.info("📤 Queued for Response")
                    elif email.get('action') == 'escalate':
                        st.warning("⚠️ Escalated")
                
                # Show response if available
                if email.get('response'):
                    st.write("**Generated Response:**")
                    with st.expander("View Response"):
                        st.write(email['response'])
    else:
        st.info("No recent emails processed. Start automation or process a batch to see activity.")
    
    # System Performance Chart
    if processing_status.get('hourly_stats'):
        st.subheader("⏱️ Processing Performance")
        
        hourly_data = processing_status['hourly_stats']
        if hourly_data:
            df = pd.DataFrame(hourly_data)
            
            fig = px.line(df, x='hour', y='count', 
                         title='Emails Processed Per Hour',
                         labels={'hour': 'Hour', 'count': 'Emails Processed'})
            st.plotly_chart(fig, use_container_width=True)
    
    # Current Configuration Summary
    st.subheader("⚙️ Current Configuration")
    
    config_col1, config_col2 = st.columns(2)
    
    with config_col1:
        st.write("**Processing Settings:**")
        auto_respond = processing_status.get('config', {}).get('auto_respond_enabled', 'Unknown')
        st.write(f"• Auto-respond: {auto_respond}")
        
        confidence_threshold = processing_status.get('config', {}).get('confidence_thresholds', {}).get('auto_respond', 'Unknown')
        st.write(f"• Confidence threshold: {confidence_threshold}")
        
        max_batch = processing_status.get('config', {}).get('max_emails_per_batch', 'Unknown')
        st.write(f"• Max batch size: {max_batch}")
    
    with config_col2:
        st.write("**System Info:**")
        st.write(f"• Gmail connected: {'✅' if status.get('gmail_connection') else '❌'}")
        st.write(f"• AI model loaded: {'✅' if status.get('ai_model_loaded') else '❌'}")
        st.write(f"• Database connected: {'✅' if status.get('database_connection') else '❌'}")
    
    # Quick Health Check
    if st.button("🔍 Run Health Check"):
        with st.spinner("Running comprehensive health check..."):
            dashboard.check_api_health()
            dashboard.get_processing_status()
            
            if status.get('status') == 'healthy':
                st.success("✅ All systems operational!")
            else:
                st.warning("⚠️ Some systems need attention. Check the sidebar for details.")

def show_email_processing_tab(dashboard):
    """Show email processing interface"""
    st.header("📧 Email Processing")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("� Automation Control")
        
        # Current status
        if st.session_state.automation_running:
            st.success("🟢 **Email automation is RUNNING**")
            st.write("The system is automatically processing emails in the background.")
            
            if st.button("⏹️ Stop Automation", type="secondary"):
                dashboard.stop_automation()
                st.rerun()
        else:
            st.warning("🔴 **Email automation is STOPPED**")
            st.write("Click below to start automatic email processing.")
            
            if st.button("▶️ Start Automation", type="primary"):
                dashboard.start_automation()
                st.rerun()
        
        # Processing status
        processing_status = st.session_state.processing_status
        if processing_status:
            st.subheader("📊 Processing Status")
            
            last_run = processing_status.get('last_run_time', 'Never')
            st.write(f"**Last Run:** {last_run}")
            
            next_run = processing_status.get('next_run_time', 'Not scheduled')
            st.write(f"**Next Run:** {next_run}")
            
            interval = processing_status.get('config', {}).get('processing_interval_minutes', 30)
            st.write(f"**Interval:** {interval} minutes")
            
            # Show working hours if configured
            working_hours = processing_status.get('config', {}).get('working_hours')
            if working_hours:
                start_time = working_hours.get('start', 'Not set')
                end_time = working_hours.get('end', 'Not set')
                st.write(f"**Working Hours:** {start_time} - {end_time}")
    
    with col2:
        st.subheader("⚡ Manual Processing")
        
        # Batch processing
        st.write("**Process Batch of Emails**")
        batch_size = st.slider("Batch size:", min_value=1, max_value=50, value=10)
        
        if st.button("🔄 Process Batch Now", type="primary"):
            with st.spinner(f"Processing {batch_size} emails..."):
                result = dashboard.process_batch_emails(batch_size)
                if result:
                    st.success(f"✅ Processed {result.get('processed_count', 0)} emails")
                    
                    # Show summary
                    summary = result.get('summary', {})
                    if summary:
                        st.write("**Batch Summary:**")
                        for action, count in summary.items():
                            st.write(f"• {action.replace('_', ' ').title()}: {count}")
        
        # Quick stats
        st.write("**Today's Stats:**")
        today_stats = processing_status.get('today_stats', {})
        
        processed_today = today_stats.get('processed', 0)
        st.metric("Processed Today", processed_today)
        
        auto_responses_today = today_stats.get('auto_responses', 0)
        st.metric("Auto Responses", auto_responses_today)
        
        escalations_today = today_stats.get('escalations', 0)
        st.metric("Escalations", escalations_today)
    
    # Recent Processing Results
    st.subheader("📋 Recent Processing Results")
    
    # Get fresh data
    recent_emails = dashboard.get_recent_emails(limit=20)
    
    if recent_emails:
        # Create tabs for different views
        result_tab1, result_tab2, result_tab3 = st.tabs(["📊 Summary View", "📧 Detailed View", "📈 Processing Chart"])
        
        with result_tab1:
            # Summary table
            summary_data = []
            for email in recent_emails:
                summary_data.append({
                    'Subject': email.get('subject', 'No Subject')[:50] + '...' if len(email.get('subject', '')) > 50 else email.get('subject', ''),
                    'From': email.get('sender', 'Unknown')[:30],
                    'Action': email.get('action', 'Unknown').replace('_', ' ').title(),
                    'Confidence': f"{email.get('confidence', 0):.2f}",
                    'Category': email.get('category', 'Unknown').title(),
                    'Knowledge Used': '✅' if email.get('rag_sources') else '❌',
                    'Processed At': email.get('processed_at', 'Unknown')
                })
            
            df = pd.DataFrame(summary_data)
            st.dataframe(df, use_container_width=True)
        
        with result_tab2:
            # Detailed view with expandable emails
            for i, email in enumerate(recent_emails[:10]):  # Show top 10 in detail
                action_emoji = {
                    'auto_respond': '🤖',
                    'escalate': '⚠️',
                    'review_needed': '👀',
                    'ignore': '🚫'
                }.get(email.get('action'), '📧')
                
                with st.expander(f"{action_emoji} {email.get('subject', 'No Subject')[:60]}... ({email.get('confidence', 0):.2f})"):
                    col1, col2 = st.columns([2, 1])
                    
                    with col1:
                        st.write(f"**From:** {email.get('sender', 'Unknown')}")
                        st.write(f"**Subject:** {email.get('subject', 'No Subject')}")
                        
                        # Show partial body
                        body = email.get('body', 'No content')[:200] + '...' if len(email.get('body', '')) > 200 else email.get('body', 'No content')
                        st.write(f"**Body Preview:** {body}")
                        
                        # AI Decision details
                        st.write(f"**AI Decision:** {email.get('action', 'Unknown').replace('_', ' ').title()}")
                        st.write(f"**Category:** {email.get('category', 'Unknown').title()}")
                        st.write(f"**Confidence:** {email.get('confidence', 0):.2f}")
                        
                        if email.get('reason'):
                            st.write(f"**Reason:** {email.get('reason')}")
                    
                    with col2:
                        st.write(f"**Processed At:** {email.get('processed_at', 'Unknown')}")
                        
                        # Show RAG sources if used
                        rag_sources = email.get('rag_sources', [])
                        if rag_sources:
                            st.write(f"**Knowledge Sources Used:** {len(rag_sources)}")
                            for source in rag_sources[:3]:  # Show first 3
                                st.write(f"• {source}")
                        
                        # Show if response was generated
                        if email.get('response'):
                            if st.button(f"View Response {i}", key=f"view_resp_{i}"):
                                st.text_area("Generated Response:", email['response'], height=100, disabled=True, key=f"resp_{i}")
        
        with result_tab3:
            # Processing performance chart
            if len(recent_emails) > 1:
                # Group by hour
                email_df = pd.DataFrame(recent_emails)
                if 'processed_at' in email_df.columns:
                    try:
                        email_df['hour'] = pd.to_datetime(email_df['processed_at']).dt.hour
                        hourly_counts = email_df.groupby('hour').size().reset_index(name='count')
                        
                        fig = px.bar(hourly_counts, x='hour', y='count',
                                   title='Emails Processed by Hour',
                                   labels={'hour': 'Hour of Day', 'count': 'Number of Emails'})
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Action distribution
                        action_counts = email_df['action'].value_counts()
                        fig2 = px.pie(values=action_counts.values, names=action_counts.index,
                                     title='Action Distribution')
                        st.plotly_chart(fig2, use_container_width=True)
                    except:
                        st.info("Unable to generate time-based charts")
            else:
                st.info("Not enough data for charts. Process more emails to see visualizations.")
    else:
        st.info("No recent emails found. Start automation or process a batch to see results here.")

def show_configuration_tab(dashboard):
    """Show configuration management interface"""
    st.header("⚙️ System Configuration")
    
    # Get current configuration
    processing_status = st.session_state.processing_status
    current_config = processing_status.get('config', {})
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🤖 Automation Settings")
        
        with st.form("automation_config"):
            # Processing interval
            interval = st.number_input(
                "Processing Interval (minutes)",
                min_value=5,
                max_value=1440,  # 24 hours
                value=current_config.get('processing_interval_minutes', 30),
                help="How often to check for new emails"
            )
            
            # Max batch size
            max_batch = st.number_input(
                "Max Emails Per Batch",
                min_value=1,
                max_value=100,
                value=current_config.get('max_emails_per_batch', 10),
                help="Maximum emails to process at once"
            )
            
            # Auto-respond enabled
            auto_respond = st.checkbox(
                "Enable Auto-Response",
                value=current_config.get('auto_respond_enabled', True),
                help="Allow system to automatically send responses"
            )
            
            # Working hours
            st.write("**Working Hours:**")
            working_hours = current_config.get('working_hours', {'start': '09:00', 'end': '17:00'})
            
            col_start, col_end = st.columns(2)
            with col_start:
                start_time = st.time_input(
                    "Start Time",
                    value=datetime.strptime(working_hours.get('start', '09:00'), '%H:%M').time()
                )
            
            with col_end:
                end_time = st.time_input(
                    "End Time", 
                    value=datetime.strptime(working_hours.get('end', '17:00'), '%H:%M').time()
                )
            
            # Submit button
            if st.form_submit_button("Update Automation Settings", type="primary"):
                new_config = {
                    'processing_interval_minutes': interval,
                    'max_emails_per_batch': max_batch,
                    'auto_respond_enabled': auto_respond,
                    'working_hours': {
                        'start': start_time.strftime('%H:%M'),
                        'end': end_time.strftime('%H:%M')
                    }
                }
                
                if dashboard.update_config(new_config):
                    dashboard.get_processing_status()  # Refresh status
                    st.success("✅ Configuration updated successfully!")
    
    with col2:
        st.subheader("🎯 AI Decision Settings")
        
        with st.form("ai_config"):
            # Gemini Filter Toggle
            gemini_available = current_config.get('gemini_available', False)
            if gemini_available:
                gemini_enabled = st.checkbox(
                    "Enable Gemini Query Filtering",
                    value=current_config.get('gemini_filter_enabled', True),
                    help="Use Gemini AI to filter out non-query emails before processing"
                )
            else:
                st.info("🔸 Gemini filtering not available (API key not configured)")
                gemini_enabled = False
            
            # Confidence thresholds
            st.write("**Confidence Thresholds:**")
            
            current_thresholds = current_config.get('confidence_thresholds', {})
            
            auto_threshold = st.slider(
                "Auto-Response Threshold",
                min_value=0.0,
                max_value=1.0,
                value=current_thresholds.get('auto_respond', 0.8),
                step=0.05,
                help="Minimum confidence to automatically respond"
            )
            
            review_threshold = st.slider(
                "Review Needed Threshold", 
                min_value=0.0,
                max_value=1.0,
                value=current_thresholds.get('review_needed', 0.5),
                step=0.05,
                help="Minimum confidence to mark for human review"
            )
            
            # Escalation keywords
            st.write("**Escalation Keywords:**")
            escalation_keywords = current_config.get('escalation_keywords', [])
            keywords_text = st.text_area(
                "Keywords (one per line)",
                value='\n'.join(escalation_keywords),
                help="Emails containing these words will be escalated"
            )
            
            # Human review categories
            st.write("**Categories Requiring Human Review:**")
            review_categories = current_config.get('human_review_required_categories', [])
            categories_text = st.text_area(
                "Categories (one per line)",
                value='\n'.join(review_categories),
                help="These categories will always require human review"
            )
            
            # Submit button
            if st.form_submit_button("Update AI Settings", type="primary"):
                ai_config = {
                    'gemini_filter_enabled': gemini_enabled,
                    'confidence_thresholds': {
                        'auto_respond': auto_threshold,
                        'review_needed': review_threshold
                    },
                    'escalation_keywords': [kw.strip() for kw in keywords_text.split('\n') if kw.strip()],
                    'human_review_required_categories': [cat.strip() for cat in categories_text.split('\n') if cat.strip()]
                }
                
                if dashboard.update_config(ai_config):
                    dashboard.get_processing_status()  # Refresh status
                    st.success("✅ AI configuration updated successfully!")
    
    # Email Query Configuration
    st.subheader("📧 Email Query Settings")
    
    col1, col2 = st.columns(2)
    
    with col1:
        with st.form("email_config"):
            # Email query
            current_query = current_config.get('email_query', 'is:unread')
            email_query = st.text_input(
                "Email Query",
                value=current_query,
                help="Gmail search query for finding emails to process"
            )
            
            # Query examples
            st.write("**Common Queries:**")
            st.write("• `is:unread` - Unread emails")
            st.write("• `newer_than:1d` - Last 24 hours")
            st.write("• `from:students.edu` - From specific domain")
            st.write("• `has:attachment` - Emails with attachments")
            
            if st.form_submit_button("Update Email Query", type="primary"):
                query_config = {'email_query': email_query}
                if dashboard.update_config(query_config):
                    st.success("✅ Email query updated successfully!")
    
    with col2:
        st.subheader("📊 Current Status")
        
        if processing_status:
            st.write("**System Status:**")
            st.write(f"• Automation: {'🟢 Running' if st.session_state.automation_running else '🔴 Stopped'}")
            st.write(f"• Last run: {processing_status.get('last_run_time', 'Never')}")
            st.write(f"• Total processed: {processing_status.get('total_processed', 0)}")
            
            st.write("**Current Configuration:**")
            st.write(f"• Interval: {current_config.get('processing_interval_minutes', 30)} min")
            st.write(f"• Batch size: {current_config.get('max_emails_per_batch', 10)}")
            st.write(f"• Auto-respond: {'✅' if current_config.get('auto_respond_enabled') else '❌'}")
            
            st.write("**Thresholds:**")
            thresholds = current_config.get('confidence_thresholds', {})
            st.write(f"• Auto-respond: {thresholds.get('auto_respond', 0.8)}")
            st.write(f"• Review needed: {thresholds.get('review_needed', 0.5)}")
        else:
            st.info("Load processing status to see current configuration")
    
    # Configuration Export/Import
    st.subheader("💾 Configuration Backup")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("📥 Export Configuration"):
            if current_config:
                config_json = json.dumps(current_config, indent=2)
                st.download_button(
                    label="Download Config JSON",
                    data=config_json,
                    file_name=f"email_config_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )
            else:
                st.warning("No configuration available to export")
    
    with col2:
        uploaded_config = st.file_uploader("📤 Import Configuration", type=['json'])
        if uploaded_config:
            try:
                imported_config = json.load(uploaded_config)
                st.json(imported_config)
                
                if st.button("Apply Imported Configuration"):
                    if dashboard.update_config(imported_config):
                        st.success("✅ Configuration imported and applied successfully!")
                    else:
                        st.error("❌ Failed to apply imported configuration")
                        
            except json.JSONDecodeError:
                st.error("❌ Invalid JSON file")

def show_analytics_tab(dashboard):
    """Show comprehensive analytics and insights"""
    st.header("📈 Analytics & Insights")
    
    # Get analytics data
    with st.spinner("Loading analytics data..."):
        analytics_data = dashboard.get_dashboard_analytics()
        recent_emails = dashboard.get_recent_emails(limit=100)  # Get more data for analytics
    
    if not analytics_data and not recent_emails:
        st.info("No analytics data available yet. Process some emails to see insights.")
        return
    
    # Overview Metrics
    st.subheader("📊 Key Performance Indicators")
    
    col1, col2, col3, col4 = st.columns(4)
    
    if analytics_data:
        with col1:
            total_processed = analytics_data.get('total_processed', 0)
            growth = analytics_data.get('growth_rate', 0)
            st.metric("Total Processed", total_processed, delta=f"{growth:+.1f}%" if growth else None)
        
        with col2:
            automation_rate = analytics_data.get('automation_rate', 0)
            st.metric("Automation Rate", f"{automation_rate:.1f}%")
        
        with col3:
            avg_confidence = analytics_data.get('average_confidence', 0)
            st.metric("Avg Confidence", f"{avg_confidence:.2f}")
        
        with col4:
            response_time = analytics_data.get('avg_response_time_minutes', 0)
            st.metric("Avg Response Time", f"{response_time:.1f} min")
    
    # Charts Row 1
    col1, col2 = st.columns(2)
    
    if recent_emails:
        df = pd.DataFrame(recent_emails)
        
        with col1:
            st.subheader("🎯 Action Distribution")
            
            if 'action' in df.columns:
                action_counts = df['action'].value_counts()
                
                # Create pie chart
                fig = px.pie(
                    values=action_counts.values,
                    names=[action.replace('_', ' ').title() for action in action_counts.index],
                    title="Email Actions Distribution",
                    color_discrete_sequence=px.colors.qualitative.Set3
                )
                fig.update_traces(textposition='inside', textinfo='percent+label')
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("📊 Confidence Distribution")
            
            if 'confidence' in df.columns:
                # Create histogram
                fig = px.histogram(
                    df, 
                    x='confidence',
                    nbins=20,
                    title="AI Confidence Score Distribution",
                    labels={'confidence': 'Confidence Score', 'count': 'Number of Emails'}
                )
                fig.update_layout(bargap=0.1)
                st.plotly_chart(fig, use_container_width=True)
    
    # Time-based Analytics
    if recent_emails and 'processed_at' in pd.DataFrame(recent_emails).columns:
        st.subheader("⏰ Time-based Analysis")
        
        try:
            df['processed_datetime'] = pd.to_datetime(df['processed_at'])
            df['date'] = df['processed_datetime'].dt.date
            df['hour'] = df['processed_datetime'].dt.hour
            df['day_of_week'] = df['processed_datetime'].dt.day_name()
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Daily processing volume
                daily_counts = df.groupby('date').size().reset_index(name='count')
                
                fig = px.line(
                    daily_counts, 
                    x='date', 
                    y='count',
                    title='Daily Email Processing Volume',
                    markers=True
                )
                fig.update_xaxes(title='Date')
                fig.update_yaxes(title='Emails Processed')
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Hourly distribution
                hourly_counts = df.groupby('hour').size().reset_index(name='count')
                
                fig = px.bar(
                    hourly_counts, 
                    x='hour', 
                    y='count',
                    title='Email Processing by Hour of Day'
                )
                fig.update_xaxes(title='Hour of Day')
                fig.update_yaxes(title='Emails Processed')
                st.plotly_chart(fig, use_container_width=True)
            
            # Day of week analysis
            dow_counts = df.groupby('day_of_week').size().reindex(['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']).reset_index(name='count')
            
            fig = px.bar(
                dow_counts,
                x='day_of_week',
                y='count', 
                title='Email Volume by Day of Week'
            )
            fig.update_xaxes(title='Day of Week')
            fig.update_yaxes(title='Emails Processed')
            st.plotly_chart(fig, use_container_width=True)
            
        except Exception as e:
            st.warning(f"Could not generate time-based charts: {e}")
    
    # Performance Analytics
    st.subheader("🎯 Performance Metrics")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.write("**Efficiency Metrics**")
        
        if recent_emails:
            df = pd.DataFrame(recent_emails)
            
            if 'action' in df.columns:
                # Auto-response rate
                auto_responses = len(df[df['action'] == 'auto_respond'])
                auto_rate = (auto_responses / len(df)) * 100 if len(df) > 0 else 0
                st.metric("Auto-Response Rate", f"{auto_rate:.1f}%")
                
                # Escalation rate
                escalations = len(df[df['action'] == 'escalate'])
                escalation_rate = (escalations / len(df)) * 100 if len(df) > 0 else 0
                st.metric("Escalation Rate", f"{escalation_rate:.1f}%")
    
    with col2:
        st.write("**Quality Metrics**")
        
        if recent_emails and 'confidence' in pd.DataFrame(recent_emails).columns:
            df = pd.DataFrame(recent_emails)
            
            # High confidence rate (>0.8)
            high_conf = len(df[df['confidence'] > 0.8])
            high_conf_rate = (high_conf / len(df)) * 100 if len(df) > 0 else 0
            st.metric("High Confidence Rate", f"{high_conf_rate:.1f}%")
            
            # Average confidence by action
            if 'action' in df.columns:
                avg_conf_by_action = df.groupby('action')['confidence'].mean()
                st.write("**Avg Confidence by Action:**")
                for action, conf in avg_conf_by_action.items():
                    st.write(f"• {action.replace('_', ' ').title()}: {conf:.2f}")
    
    with col3:
        st.write("**Knowledge Usage**")
        
        if recent_emails:
            df = pd.DataFrame(recent_emails)
            
            # RAG usage rate
            if 'rag_sources' in df.columns:
                rag_used = df['rag_sources'].notna().sum()
                rag_rate = (rag_used / len(df)) * 100 if len(df) > 0 else 0
                st.metric("Knowledge Base Usage", f"{rag_rate:.1f}%")
                
                # Average sources per email
                avg_sources = df[df['rag_sources'].notna()]['rag_sources'].apply(len).mean()
                if not pd.isna(avg_sources):
                    st.metric("Avg Sources per Email", f"{avg_sources:.1f}")
    
    # Category Analysis
    if recent_emails and 'category' in pd.DataFrame(recent_emails).columns:
        st.subheader("📂 Category Analysis")
        
        df = pd.DataFrame(recent_emails)
        category_counts = df['category'].value_counts()
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Category distribution
            fig = px.bar(
                x=category_counts.index,
                y=category_counts.values,
                title="Email Categories",
                labels={'x': 'Category', 'y': 'Count'}
            )
            fig.update_xaxes(tickangle=45)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Category performance (confidence)
            category_performance = df.groupby('category')['confidence'].agg(['mean', 'count']).reset_index()
            category_performance.columns = ['Category', 'Avg_Confidence', 'Count']
            
            fig = px.scatter(
                category_performance,
                x='Count',
                y='Avg_Confidence',
                size='Count',
                hover_data=['Category'],
                title="Category Performance: Volume vs Confidence"
            )
            st.plotly_chart(fig, use_container_width=True)
    
    # Export Analytics
    st.subheader("📤 Export Analytics")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("📊 Generate Report"):
            if analytics_data or recent_emails:
                # Create comprehensive report
                report_data = {
                    'generated_at': datetime.now().isoformat(),
                    'summary_metrics': analytics_data if analytics_data else {},
                    'recent_emails_count': len(recent_emails) if recent_emails else 0
                }
                
                if recent_emails:
                    df = pd.DataFrame(recent_emails)
                    report_data['action_distribution'] = df['action'].value_counts().to_dict()
                    
                    if 'confidence' in df.columns:
                        report_data['confidence_stats'] = {
                            'mean': float(df['confidence'].mean()),
                            'median': float(df['confidence'].median()),
                            'std': float(df['confidence'].std())
                        }
                
                report_json = json.dumps(report_data, indent=2, default=str)
                
                st.download_button(
                    label="Download Analytics Report (JSON)",
                    data=report_json,
                    file_name=f"email_analytics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )
            else:
                st.warning("No data available for report generation")
    
    with col2:
        if recent_emails and st.button("📈 Export Raw Data"):
            df = pd.DataFrame(recent_emails)
            csv_data = df.to_csv(index=False)
            
            st.download_button(
                label="Download Raw Data (CSV)",
                data=csv_data,
                file_name=f"email_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )

if __name__ == "__main__":
    main()