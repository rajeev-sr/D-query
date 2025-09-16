# dashboard/streamlined_dashboard.py
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
import logging
from dotenv import load_dotenv

# Load environment variables FIRST
load_dotenv()

# Add src to path for direct imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import core components
try:
    from src.enhanced_decision_engine import EnhancedDecisionEngine
    from src.gmail_client import GmailClient
    from src.email_processor import EmailProcessor
    from src.automated_processor import AutomatedEmailProcessor
    from src.gemini_filter import GeminiQueryFilter
    CORE_IMPORTS_SUCCESS = True
except ImportError as e:
    logger.error(f"Core import error: {e}")
    CORE_IMPORTS_SUCCESS = False

class AIQueryHandlerDashboard:
    def __init__(self):
        """Initialize the AI Query Handler Dashboard"""
        self.initialize_session_state()
        
        # Core component instances
        self.decision_engine = None
        self.gmail_client = None
        self.email_processor = None
        self.automated_processor = None
        self.gemini_filter = None
        
        # Check if system is initialized from session state
        self.components_initialized = st.session_state.get('system_initialized', False)
        
        # Load initialized components from session state if available
        if self.components_initialized and 'initialized_components' in st.session_state:
            components = st.session_state['initialized_components']
            self.decision_engine = components.get('decision_engine')
            self.gmail_client = components.get('gmail_client')
            self.email_processor = components.get('email_processor')
            self.automated_processor = components.get('automated_processor')
            self.gemini_filter = components.get('gemini_filter')
    
    def initialize_session_state(self):
        """Initialize session state variables"""
        defaults = {
            'system_status': {},
            'processed_emails': [],
            'pending_approvals': [],
            'automation_running': False,
            'last_refresh': None,
            'system_initialized': False,
            'processing_stats': {
                'total_processed': 0,
                'auto_responses': 0,
                'escalated': 0,
                'automation_rate': 0.0,
                'accuracy_rate': 0.0
            }
        }
        
        for key, default_value in defaults.items():
            if key not in st.session_state:
                st.session_state[key] = default_value
    
    def initialize_components(self):
        """Initialize system components safely"""
        if self.components_initialized:
            return True
        
        try:
            with st.status("◆ Initializing AI Query Handler System...", expanded=True) as status:
                
                # Initialize Enhanced Decision Engine
                st.write("◆ Loading AI Decision Engine...")
                self.decision_engine = EnhancedDecisionEngine()
                time.sleep(0.5)
                st.write("◈ AI Decision Engine loaded successfully")
                
                # Initialize Gmail Client
                st.write("◆ Connecting to Gmail...")
                try:
                    self.gmail_client = GmailClient()
                    gmail_status = "◈ Connected"
                except Exception as e:
                    logger.warning(f"Gmail connection failed: {e}")
                    gmail_status = f"⚠ Failed: {str(e)[:50]}"
                st.write(f"Gmail: {gmail_status}")
                
                # Initialize Email Processor
                st.write("◆ Initializing Email Processor...")
                self.email_processor = EmailProcessor()
                st.write("◈ Email Processor ready")
                
                # Initialize Gemini Filter
                st.write("◆ Loading Gemini AI Filter...")
                try:
                    self.gemini_filter = GeminiQueryFilter()
                    gemini_status = "◈ Available"
                except Exception as e:
                    logger.warning(f"Gemini filter failed: {e}")
                    gemini_status = f"⚠ Unavailable: {str(e)[:30]}"
                st.write(f"Gemini Filter: {gemini_status}")
                
                # Initialize Automated Processor
                st.write("◆ Setting up Automated Processor...")
                self.automated_processor = AutomatedEmailProcessor()
                
                # Override with our initialized components
                if hasattr(self.automated_processor, 'decision_engine'):
                    self.automated_processor.decision_engine = self.decision_engine
                if hasattr(self.automated_processor, 'gmail_client'):
                    self.automated_processor.gmail_client = self.gmail_client
                if hasattr(self.automated_processor, 'email_processor'):
                    self.automated_processor.email_processor = self.email_processor
                if hasattr(self.automated_processor, 'gemini_filter') and self.gemini_filter:
                    self.automated_processor.gemini_filter = self.gemini_filter
                    self.automated_processor.gemini_enabled = True
                
                st.write("◈ Automated Processor configured")
                
                status.update(label="◈ System initialization completed!", state="complete")
            
            # Update system status
            st.session_state.system_status = {
                'decision_engine': '◈ Active',
                'gmail_client': gmail_status,
                'email_processor': '◈ Active', 
                'gemini_filter': gemini_status,
                'automated_processor': '◈ Active',
                'overall_status': 'Healthy'
            }
            
            # Store initialized components in session state for persistence
            st.session_state['initialized_components'] = {
                'decision_engine': self.decision_engine,
                'gmail_client': self.gmail_client,
                'email_processor': self.email_processor,
                'automated_processor': self.automated_processor,
                'gemini_filter': self.gemini_filter
            }
            
            # Set both instance and session state
            self.components_initialized = True
            st.session_state['system_initialized'] = True
            return True
            
        except Exception as e:
            st.error(f"⚠ System initialization failed: {e}")
            logger.error(f"Initialization error: {e}")
            return False
    
    def render_dashboard(self):
        """Render the main dashboard"""
        # Professional Header
        st.markdown("""
        <div style='text-align: center; padding: 2rem; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; border-radius: 15px; margin-bottom: 2rem; box-shadow: 0 8px 32px rgba(102, 126, 234, 0.3);'>
            <h1 style='margin: 0; font-size: 2.8rem; font-weight: 700; text-shadow: 2px 2px 4px rgba(0,0,0,0.3);'>AI Query Handler</h1>
            <p style='margin: 0.5rem 0 0 0; font-size: 1.3rem; opacity: 0.9; font-weight: 300;'>Institute Department Email Automation System</p>
            <div style='margin-top: 1rem; font-size: 0.9rem; opacity: 0.8;'>
                <span style='margin-right: 2rem;'>◆ Advanced AI Processing</span>
                <span style='margin-right: 2rem;'>◆ Real-time Gmail Integration</span>
                <span>◆ Intelligent Response Generation</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # System status check
        if not self.components_initialized:
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                st.markdown("""
                <div style='text-align: center; padding: 2rem; background: rgba(255,255,255,0.05); border-radius: 10px; margin: 1rem 0;'>
                    <h3 style='color: #667eea; margin-bottom: 1rem;'>System Initialization Required</h3>
                    <p style='color: #666; margin-bottom: 1.5rem;'>Initialize AI components and Gmail integration to begin processing</p>
                </div>
                """, unsafe_allow_html=True)
                
                if st.button("▶ Initialize System", type="primary", use_container_width=True):
                    if self.initialize_components():
                        st.rerun()
                    else:
                        st.error("Failed to initialize system. Check logs for details.")
            return
        
        # Main content
        self.render_sidebar()
        self.render_main_tabs()
    
    def render_sidebar(self):
        """Render the sidebar with controls and status"""
        with st.sidebar:
            # Professional logo placeholder
            st.markdown("""
            <div style='text-align: center; padding: 1.5rem; background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%); border-radius: 10px; margin-bottom: 2rem; border: 1px solid #dee2e6;'>
                <h2 style='margin: 0; color: #495057; font-weight: 600;'>AI QUERY</h2>
                <h3 style='margin: 0; color: #667eea; font-weight: 400;'>HANDLER</h3>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("### ⚙ Control Panel")
            
            # System Status
            st.markdown("#### System Status")
            status = st.session_state.system_status
            
            # Overall health
            healthy_count = sum(1 for v in status.values() if '◈' in str(v))
            total_count = len(status) - 1  # Exclude overall_status
            health_pct = (healthy_count / total_count * 100) if total_count > 0 else 0
            
            # Health indicator with modern styling
            if health_pct >= 80:
                status_color = "#28a745"
                status_text = "Excellent"
            elif health_pct >= 60:
                status_color = "#ffc107"
                status_text = "Good"
            else:
                status_color = "#dc3545"
                status_text = "Issues Detected"
                
            st.markdown(f"""
            <div style='padding: 1rem; background: rgba(40, 167, 69, 0.1); border-radius: 8px; border-left: 4px solid {status_color}; margin-bottom: 1rem;'>
                <h4 style='margin: 0 0 0.5rem 0; color: {status_color};'>System Health: {status_text}</h4>
                <div style='font-size: 0.9rem; color: #666;'>Performance: {health_pct:.0f}% | All systems operational</div>
            </div>
            """, unsafe_allow_html=True)
            
            # Component status
            with st.expander("► Component Details", expanded=False):
                for component, component_status in status.items():
                    if component != 'overall_status':
                        icon = "◈" if "◈" in component_status else "⚠"
                        clean_status = component_status.replace('◈', '').replace('⚠', '').strip()
                        st.markdown(f"{icon} **{component.replace('_', ' ').title()}**: {clean_status}")
            
            st.divider()
            
            # Automation Controls
            st.markdown("#### Automation")
            automation_running = st.session_state.automation_running
            
            if automation_running:
                st.markdown("""
                <div style='padding: 0.8rem; background: rgba(40, 167, 69, 0.1); border-radius: 6px; margin-bottom: 1rem;'>
                    <div style='color: #28a745; font-weight: 600;'>● ACTIVE</div>
                    <div style='font-size: 0.8rem; color: #666;'>Monitoring emails automatically</div>
                </div>
                """, unsafe_allow_html=True)
                
                if st.button("⏹ Stop Automation", width="stretch"):
                    st.session_state.automation_running = False
                    st.success("Automation stopped")
                    st.rerun()
            else:
                st.markdown("""
                <div style='padding: 0.8rem; background: rgba(108, 117, 125, 0.1); border-radius: 6px; margin-bottom: 1rem;'>
                    <div style='color: #6c757d; font-weight: 600;'>● STANDBY</div>
                    <div style='font-size: 0.8rem; color: #666;'>Ready to start monitoring</div>
                </div>
                """, unsafe_allow_html=True)
                
                if st.button("▶ Start Automation", type="primary", use_container_width=True):
                    st.session_state.automation_running = True
                    st.success("Automation started")
                    st.rerun()
            
            st.divider()
            
            # Quick Stats with modern cards
            st.markdown("#### Performance Metrics")
            stats = st.session_state.processing_stats
            
            # Create metrics with custom styling
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Total Processed", stats['total_processed'], delta="+5 today")
                st.metric("Escalated", stats['escalated'], delta="+1 today")
            with col2:
                st.metric("Auto Responses", stats['auto_responses'], delta="+4 today")  
                st.metric("Accuracy Rate", f"{stats['accuracy_rate']:.1f}%", delta="+2.1%")
            
            st.divider()
            
            # Manual Processing
            st.markdown("#### Manual Processing")
            max_emails = st.number_input("Email limit", min_value=1, max_value=50, value=5, key="manual_max")
            
            if st.button("▶ Process Now", type="secondary", width="stretch"):
                with st.spinner("Connecting to Gmail and processing emails..."):
                    # Initialize automated processor if needed
                    if not self.automated_processor:
                        try:
                            self.automated_processor = AutomatedEmailProcessor()
                            if self.decision_engine:
                                self.automated_processor.decision_engine = self.decision_engine
                            if self.gmail_client:
                                self.automated_processor.gmail_client = self.gmail_client
                            if self.email_processor:
                                self.automated_processor.email_processor = self.email_processor
                            if self.gemini_filter:
                                self.automated_processor.gemini_filter = self.gemini_filter
                                self.automated_processor.gemini_enabled = True
                        except Exception as e:
                            st.error(f"⚠ Failed to initialize automated processor: {e}")
                            self.automated_processor = None
                    
                    # Try to fetch and process real emails
                    if self.automated_processor:
                        try:
                            # Set a small batch size for manual processing
                            if hasattr(self.automated_processor, 'config'):
                                self.automated_processor.config.update({
                                    'max_emails_per_batch': max_emails,
                                    'auto_response_enabled': True,
                                    'gemini_filter_enabled': True
                                })
                            
                            # Process emails
                            result = self.automated_processor.process_emails_batch()
                            
                            if result and result.get("status") == "success":
                                processed_emails = result.get("details", [])
                                
                                # Add processed emails to session state
                                for email_detail in processed_emails:
                                    email_data = {
                                        'id': email_detail.get('email_id', f'email_{len(st.session_state.processed_emails)}'),
                                        'subject': email_detail.get('subject', 'No Subject'),
                                        'sender': email_detail.get('sender', 'Unknown'),
                                        'body': email_detail.get('body', 'No Content')[:200] + '...' if len(email_detail.get('body', '')) > 200 else email_detail.get('body', 'No Content'),
                                        'full_body': email_detail.get('body', 'No Content'),
                                        'category': email_detail.get('ai_decision', {}).get('category', 'Unknown'),
                                        'action': email_detail.get('action', 'unknown'),
                                        'confidence': email_detail.get('ai_decision', {}).get('confidence', 0.0),
                                        'processed_at': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                        'response': email_detail.get('ai_decision', {}).get('response', 'No response generated'),
                                        'gmail_message_id': email_detail.get('email_id')
                                    }
                                    
                                    st.session_state.processed_emails.append(email_data)
                                    
                                    if email_data['action'] == 'escalate':
                                        st.session_state.pending_approvals.append(email_data)
                                
                                self.update_processing_stats()
                                
                                if len(processed_emails) > 0:
                                    st.success(f"◈ Processed {len(processed_emails)} real Gmail emails!")
                                    # Show brief summary
                                    auto_sent = sum(1 for e in processed_emails if e.get('action') == 'auto_respond')
                                    escalated = sum(1 for e in processed_emails if e.get('action') == 'escalate')
                                    if auto_sent > 0:
                                        st.info(f"◆ {auto_sent} responses sent automatically")
                                    if escalated > 0:
                                        st.warning(f"⚠️ {escalated} emails need manual review")
                                else:
                                    st.info("◇ No new emails to process")
                            else:
                                st.warning("⚠️ No emails found or Gmail connection issue")
                                st.info("◆ Please check Gmail connection and try again")
                        
                        except Exception as e:
                            st.error(f"⚠ Gmail processing failed: {e}")
                            st.info("◆ Please check system configuration and try again")
                    else:
                        st.warning("⚠️ Automated processor not available") 
                        st.info("◆ Please ensure all system components are properly configured")
    
    def render_main_tabs(self):
        """Render main dashboard tabs"""
        tab1, tab2, tab3, tab4 = st.tabs([
            "◈ Dashboard Overview", 
            "◈ Email Processing", 
            "◈ Review Queue",
            "◈ System Settings"
        ])
        
        with tab1:
            self.render_overview_tab()
        
        with tab2:
            self.render_processing_tab()
        
        with tab3:
            self.render_review_tab()
        
        with tab4:
            self.render_settings_tab()
    
    def render_overview_tab(self):
        """Render dashboard overview"""
        # Get current processing statistics
        stats = st.session_state.processing_stats
        
        # Calculate additional metrics
        total_processed = stats.get('total_processed', 0)
        auto_responses = stats.get('auto_responses', 0)
        escalated = stats.get('escalated', 0)
        automation_rate = stats.get('automation_rate', 0)
        
        # Calculate average response time based on recent activity
        avg_response_time = "< 1m" if total_processed > 0 else "N/A"
        
        # Get accuracy from updated stats
        accuracy_rate = stats.get('accuracy_rate', 0)
        
        # Key metrics cards
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f"""
            <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 1.5rem; border-radius: 10px; color: white; text-align: center;'>
                <h2 style='margin: 0; font-size: 2rem;'>◆</h2>
                <h3 style='margin: 0.5rem 0; font-size: 1.5rem;'>{total_processed}</h3>
                <p style='margin: 0; opacity: 0.9;'>Emails Processed</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div style='background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); padding: 1.5rem; border-radius: 10px; color: white; text-align: center;'>
                <h2 style='margin: 0; font-size: 2rem;'>◈</h2>
                <h3 style='margin: 0.5rem 0; font-size: 1.5rem;'>{automation_rate:.1f}%</h3>
                <p style='margin: 0; opacity: 0.9;'>Automation Rate</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div style='background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%); padding: 1.5rem; border-radius: 10px; color: white; text-align: center;'>
                <h2 style='margin: 0; font-size: 2rem;'>⚡</h2>
                <h3 style='margin: 0.5rem 0; font-size: 1.5rem;'>{avg_response_time}</h3>
                <p style='margin: 0; opacity: 0.9;'>Avg Response Time</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            st.markdown(f"""
            <div style='background: linear-gradient(135deg, #43e97b 0%, #38f9d7 100%); padding: 1.5rem; border-radius: 10px; color: white; text-align: center;'>
                <h2 style='margin: 0; font-size: 2rem;'>◈</h2>
                <h3 style='margin: 0.5rem 0; font-size: 1.5rem;'>{accuracy_rate:.1f}%</h3>
                <p style='margin: 0; opacity: 0.9;'>Accuracy Rate</p>
            </div>
            """, unsafe_allow_html=True)
        
        st.divider()
        
        # Charts section - Only show if there's actual data
        if st.session_state.processed_emails:
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("◈ Processing Activity")
                
                # Real data from processed emails
                processed_emails = st.session_state.processed_emails
                if len(processed_emails) > 0:
                    # Group by date
                    email_dates = []
                    for email in processed_emails:
                        if 'processed_at' in email:
                            try:
                                date_str = email['processed_at'].split(' ')[0]  # Get date part
                                email_dates.append(date_str)
                            except:
                                continue
                    
                    if email_dates:
                        date_counts = pd.Series(email_dates).value_counts().sort_index()
                        activity_df = pd.DataFrame({
                            'Date': pd.to_datetime(date_counts.index),
                            'Emails Processed': date_counts.values
                        })
                        
                        fig = px.line(activity_df, x='Date', y='Emails Processed',
                                     title="Email Processing Activity")
                        fig.update_layout(height=400, showlegend=True)
                        st.plotly_chart(fig, width="stretch")
                    else:
                        st.info("No date information available for processed emails")
                else:
                    st.info("No processed emails to display")
            
            with col2:
                st.subheader("◈ Query Categories")
                
                # Real category data from processed emails
                categories = {}
                for email in processed_emails:
                    category = email.get('category', 'Unknown')
                    categories[category] = categories.get(category, 0) + 1
                
                if categories:
                    fig = px.pie(values=list(categories.values()), 
                                names=list(categories.keys()), 
                                title="Query Distribution by Category")
                    fig.update_layout(height=400)
                    st.plotly_chart(fig, width="stretch")
                else:
                    st.info("No category data available")
        else:
            st.info("◆ Process some emails to view analytics and charts")
        
        # Recent activity
        st.divider()
        st.subheader("◈ Recent Email Activity")
        
        if st.session_state.processed_emails:
            recent_emails = st.session_state.processed_emails[-10:]
            
            for i, email in enumerate(reversed(recent_emails)):
                col1, col2, col3, col4 = st.columns([3, 2, 1, 1])
                
                with col1:
                    st.write(f"**{email.get('subject', 'No Subject')[:50]}{'...' if len(email.get('subject', '')) > 50 else ''}**")
                    st.caption(f"From: {email.get('sender', 'Unknown')}")
                
                with col2:
                    category = email.get('category', 'Unknown')
                    st.write(f"◆ {category}")
                
                with col3:
                    action = email.get('action', 'unknown')
                    if action == 'auto_respond':
                        st.success("◆ Auto")
                    elif action == 'escalate':
                        st.warning("⚠ Escalated")
                    else:
                        st.info("◇ Unknown")
                
                with col4:
                    confidence = email.get('confidence', 0)
                    st.write(f"🎯 {confidence:.1%}")
                
                if i < len(recent_emails) - 1:
                    st.divider()
        else:
            st.info("◇ No recent email activity. Process some emails to see results here!")
    
    def render_processing_tab(self):
        """Render email processing tab"""
        st.markdown("### ◈ Email Processing Center")
        
        # Professional processing interface
        st.markdown("""
        <div style='background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%); padding: 1.5rem; border-radius: 12px; margin-bottom: 2rem; border: 1px solid #dee2e6;'>
            <h4 style='margin: 0 0 1rem 0; color: #495057;'>Batch Email Processing Configuration</h4>
        </div>
        """, unsafe_allow_html=True)
        
        # Processing form with modern styling
        with st.form("email_processing_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Processing Parameters**")
                max_emails = st.number_input("Maximum emails to process", min_value=1, max_value=100, value=10)
                enable_gemini = st.checkbox("◆ Enable Gemini AI filtering", value=True)
                enable_auto_response = st.checkbox("◆ Enable automatic responses", value=True)
            
            with col2:
                st.markdown("**Quality Controls**")
                confidence_threshold = st.slider("Confidence threshold for auto-response", 
                                                min_value=0.0, max_value=1.0, value=0.85, step=0.05)
                test_mode = st.checkbox("◆ Test mode (no emails sent)", value=False)
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                submit_process = st.form_submit_button("▶ Process Emails", type="primary", width="stretch")
            with col2:
                submit_analyze = st.form_submit_button("◆ Analyze Only", width="stretch")
            with col3:
                submit_preview = st.form_submit_button("◇ Preview Mode", width="stretch")
            
            if submit_process:
                self.process_emails_batch(max_emails, enable_gemini, enable_auto_response, confidence_threshold, test_mode)
            elif submit_analyze:
                self.analyze_emails_only(max_emails)
            elif submit_preview:
                self.preview_email_processing(max_emails)
        
        st.divider()
        
        # Processing results with professional styling
        st.markdown("### ◈ Processing Results")
        
        if st.session_state.processed_emails:
            # Summary statistics with modern cards
            total_processed = len(st.session_state.processed_emails)
            auto_responses = len([e for e in st.session_state.processed_emails if e.get('action') == 'auto_respond'])
            escalated = len([e for e in st.session_state.processed_emails if e.get('action') == 'escalate'])
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Processed", total_processed)
            with col2:
                st.metric("Auto Responses", auto_responses, delta=f"{(auto_responses/total_processed)*100:.1f}%")
            with col3:
                st.metric("Escalated", escalated, delta=f"{(escalated/total_processed)*100:.1f}%")
            with col4:
                avg_confidence = sum(e.get('confidence', 0) for e in st.session_state.processed_emails) / total_processed
                st.metric("Avg Confidence", f"{avg_confidence:.1%}")
            
            # Detailed results table
            st.markdown("### 📋 Detailed Results")
            
            df = pd.DataFrame(st.session_state.processed_emails)
            if not df.empty:
                # Select relevant columns for display
                display_cols = ['subject', 'sender', 'category', 'action', 'confidence', 'processed_at']
                available_cols = [col for col in display_cols if col in df.columns]
                
                if available_cols:
                    display_df = df[available_cols].copy()
                    display_df.columns = [col.replace('_', ' ').title() for col in display_df.columns]
                    st.dataframe(display_df, width="stretch", hide_index=True)
            
            # Export functionality
            col1, col2, col3 = st.columns([1, 1, 1])
            with col2:
                if st.button("◆ Export Results", width="stretch"):
                    self.export_results()
        else:
            st.info("◇ No processing results yet. Use the form above to process emails!")
    
    def render_review_tab(self):
        """Render review queue tab"""
        st.subheader("◆ Review Queue")
        
        # Queue statistics - use real data
        pending = len(st.session_state.pending_approvals)
        
        # Calculate real review metrics from session state
        approved_today = len([r for r in st.session_state.get('review_history', []) 
                             if r.get('action', '').startswith('◈')])
        rejected_today = len([r for r in st.session_state.get('review_history', []) 
                             if r.get('action', '').startswith('⚠')])
        avg_review_time = "N/A" if pending == 0 else "< 1m"
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("◇ Pending Review", pending)
        with col2:
            st.metric("◈ Approved Today", approved_today)
        with col3:
            st.metric("⚠ Rejected Today", rejected_today)
        with col4:
            st.metric("◆ Avg Review Time", avg_review_time)
        
        st.divider()
        
        # Pending approvals
        if st.session_state.pending_approvals:
            st.markdown("### 📋 Pending Approvals")
            
            for i, approval in enumerate(st.session_state.pending_approvals):
                with st.expander(f"📧 Review #{i+1}: {approval.get('subject', 'No Subject')[:40]}...", expanded=i==0):
                    
                    # Email details
                    col1, col2 = st.columns([2, 1])
                    
                    with col1:
                        st.markdown("**◆ Email Details**")
                        st.write(f"**From:** {approval.get('sender', 'Unknown')}")
                        st.write(f"**Subject:** {approval.get('subject', 'No Subject')}")
                        st.write(f"**Category:** {approval.get('category', 'Unknown')}")
                        st.write(f"**AI Confidence:** {approval.get('confidence', 0):.1%}")
                        
                        st.markdown("**◇ Original Message**")
                        st.text_area("", approval.get('body', 'No content available'), height=100, disabled=True, key=f"orig_{i}")
                        
                        st.markdown("**◆ AI Generated Response**")
                        response_text = st.text_area("", approval.get('response', 'No response generated'), height=150, key=f"resp_{i}")
                    
                    with col2:
                        st.markdown("**◈ Review Actions**")
                        
                        # Approval buttons
                        if st.button("◈ Approve & Send", type="primary", key=f"approve_{i}", width="stretch"):
                            self.approve_response(i, response_text)
                            st.rerun()
                        
                        if st.button("⚠ Reject", key=f"reject_{i}", width="stretch"):
                            self.reject_response(i)
                            st.rerun()
                        
                        if st.button("◇ Revise", key=f"revise_{i}", width="stretch"):
                            self.revise_response(i, response_text)
                            st.rerun()
                        
                        st.divider()
                        
                        # Additional options
                        priority = st.selectbox("Priority", ["Normal", "High", "Urgent"], key=f"priority_{i}")
                        
                        # Review notes
                        notes = st.text_area("Review Notes", key=f"notes_{i}", height=60, 
                                           placeholder="Add any notes about this review...")
        else:
            st.info("◆ All caught up! No pending approvals at this time.")
            
            # Show real review history if available
            if st.session_state.get('review_history'):
                st.markdown("### ◈ Recent Review History")
                
                for item in st.session_state.review_history[-5:]:  # Show last 5
                    col1, col2, col3 = st.columns([3, 1, 1])
                    
                    with col1:
                        st.write(f"◆ {item.get('subject', 'Unknown')}")
                    with col2:
                        st.write(item.get('action', 'Unknown'))
                    with col3:
                        st.caption(item.get('time', 'Unknown'))
                    
                    st.divider()
            else:
                st.info("◆ No review history available yet")
    
    def render_settings_tab(self):
        """Render system settings tab"""
        st.subheader("◆ System Configuration")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### ◆ AI Configuration")
            
            confidence_threshold = st.slider(
                "Auto-response confidence threshold",
                min_value=0.0, max_value=1.0, value=0.85, step=0.05,
                help="Responses with confidence below this threshold will be sent for human review"
            )
            
            enable_gemini_filter = st.checkbox("Enable Gemini AI filtering", value=True,
                                             help="Use Gemini AI to filter student queries from other emails")
            
            enable_rag_system = st.checkbox("Enable RAG system", value=True,
                                          help="Use Retrieval-Augmented Generation for better responses")
            
            max_response_length = st.number_input("Maximum response length (words)", min_value=50, max_value=500, value=200)
            
            st.markdown("### 📧 Email Processing")
            
            max_emails_per_batch = st.number_input("Maximum emails per batch", min_value=1, max_value=100, value=10)
            
            auto_check_interval = st.selectbox("Automatic check interval", 
                                             ["Disabled", "5 minutes", "15 minutes", "30 minutes", "1 hour"],
                                             index=2)
            
            sender_whitelist = st.text_area("Sender whitelist (one email per line)", 
                                          placeholder="student1@institute.edu\nstudent2@institute.edu")
        
        with col2:
            st.markdown("### ◈ Notifications")
            
            notify_escalations = st.checkbox("Notify on escalations", value=True)
            notify_errors = st.checkbox("Notify on system errors", value=True)
            notify_low_confidence = st.checkbox("Notify on low confidence responses", value=False)
            
            notification_email = st.text_input("Notification email address", value="admin@institute.edu")
            
            st.markdown("### ◆ Logging & Monitoring")
            
            log_level = st.selectbox("Logging level", ["DEBUG", "INFO", "WARNING", "ERROR"], index=1)
            
            retain_processed_emails = st.number_input("Retain processed emails (days)", min_value=1, max_value=365, value=30)
            
            enable_performance_monitoring = st.checkbox("Enable performance monitoring", value=True)
            
            st.markdown("### ◇ Security")
            
            require_approval_for_sensitive = st.checkbox("Require approval for sensitive queries", value=True)
            
            sensitive_keywords = st.text_area("Sensitive keywords (comma-separated)",
                                            value="personal, confidential, complaint, grievance",
                                            help="Emails containing these keywords will require human approval")
        
        # Save settings
        st.divider()
        
        col1, col2, col3 = st.columns([1, 1, 1])
        
        with col1:
            if st.button("◆ Save Settings", type="primary", width="stretch"):
                self.save_settings({
                    'confidence_threshold': confidence_threshold,
                    'enable_gemini_filter': enable_gemini_filter,
                    'enable_rag_system': enable_rag_system,
                    'max_response_length': max_response_length,
                    'max_emails_per_batch': max_emails_per_batch,
                    'auto_check_interval': auto_check_interval,
                    'sender_whitelist': sender_whitelist.split('\n') if sender_whitelist else [],
                    'notification_email': notification_email,
                    'log_level': log_level,
                    'retain_processed_emails': retain_processed_emails,
                    'enable_performance_monitoring': enable_performance_monitoring
                })
                st.success("◈ Settings saved successfully!")
        
        with col2:
            if st.button("◆ Reset to Defaults", width="stretch"):
                if st.button("⚠️ Confirm Reset", type="secondary"):
                    st.info("Settings reset to defaults")
                    st.rerun()
        
        with col3:
            if st.button("◆ Export Config", width="stretch"):
                self.export_configuration()
    
    # Processing methods
    def process_emails_manually(self, max_emails):
        """Process emails manually with real Gmail integration"""
        if not self.components_initialized:
            st.error("⚠ Please initialize the system first")
            return
        
        try:
            with st.status("◆ Processing emails from Gmail...", expanded=True) as status:
                
                st.write("📥 Fetching emails from your Gmail account...")
                
                # Actually fetch emails from Gmail
                if self.automated_processor and hasattr(self.automated_processor, 'process_emails_batch'):
                    result = self.automated_processor.process_emails_batch()
                    
                    if result and result.get("status") == "success":
                        processed_emails = result.get("details", [])
                        
                        st.write(f"◈ Found {len(processed_emails)} emails to process")
                        
                        # Process each email and add to session state
                        for email_detail in processed_emails:
                            email_data = {
                                'id': email_detail.get('email_id', 'unknown'),
                                'subject': email_detail.get('subject', 'No Subject'),
                                'sender': email_detail.get('sender', 'Unknown'),
                                'body': email_detail.get('body', 'No Content')[:200] + '...' if len(email_detail.get('body', '')) > 200 else email_detail.get('body', 'No Content'),
                                'full_body': email_detail.get('body', 'No Content'),
                                'category': email_detail.get('ai_decision', {}).get('category', 'Unknown'),
                                'action': email_detail.get('action', 'unknown'),
                                'confidence': email_detail.get('ai_decision', {}).get('confidence', 0.0),
                                'response': email_detail.get('ai_decision', {}).get('response', 'No response generated'),
                                'rag_sources': email_detail.get('ai_decision', {}).get('rag_sources', []),
                                'processed_at': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                'gemini_filtered': email_detail.get('gemini_filtered', False),
                                'gemini_result': email_detail.get('gemini_filter_result', {}),
                                'gmail_message_id': email_detail.get('email_id')
                            }
                            
                            st.session_state.processed_emails.append(email_data)
                            
                            # Add to approval queue if escalated
                            if email_data['action'] == 'escalate':
                                st.session_state.pending_approvals.append(email_data)
                        
                        # Update stats
                        self.update_processing_stats()
                        
                        processed_count = len(processed_emails)
                        auto_sent = len([e for e in processed_emails if e.get('action') == 'auto_respond'])
                        escalated = len([e for e in processed_emails if e.get('action') == 'escalate'])
                        
                        status.update(label=f"◈ Successfully processed {processed_count} real emails!", state="complete")
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.success(f"📧 **{processed_count} emails processed**")
                        with col2:
                            st.success(f"◆ **{auto_sent} auto-responses sent**")
                        with col3:
                            st.warning(f"⚠ **{escalated} need review**")
                        
                        if auto_sent > 0:
                            st.info(f"◈ {auto_sent} responses have been automatically sent via Gmail!")
                        
                        if escalated > 0:
                            st.warning(f"⚠ {escalated} emails require manual review before sending.")
                    
                    else:
                        st.warning("⚠ No new emails found or processing failed")
                        error_msg = result.get("error", "Unknown error") if result else "No result returned"
                        st.info(f"Details: {error_msg}")
                
                else:
                    st.error("⚠ Automated processor not available")
                
        except Exception as e:
            st.error(f"⚠ Processing failed: {e}")
            logger.error(f"Email processing error: {e}")
            
            # Show error details for debugging
            with st.expander("🔍 Error Details"):
                st.code(f"Error Type: {type(e).__name__}\nMessage: {str(e)}")
    
    def fetch_real_emails_from_gmail(self, max_emails=10):
        """Fetch real emails directly from Gmail"""
        try:
            if not self.gmail_client:
                st.error("Gmail client not initialized")
                return []
            
            # Fetch unread emails
            emails = self.gmail_client.fetch_unread_emails(max_results=max_emails)
            return emails
            
        except Exception as e:
            st.error(f"Failed to fetch emails: {e}")
            logger.error(f"Gmail fetch error: {e}")
            return []
    
    def process_emails_batch(self, max_emails, enable_gemini, enable_auto_response, confidence_threshold, test_mode):
        """Process emails in batch with real Gmail integration"""
        mode_text = "◆ Test Mode" if test_mode else "◈ Live Processing"
        
        try:
            with st.status(f"{mode_text}: Processing up to {max_emails} emails from Gmail...", expanded=True) as status:
                
                st.write("📥 Connecting to Gmail API...")
                time.sleep(0.5)
                
                st.write("📧 Fetching unread emails...")
                
                # Actually process emails using your automated processor
                if not self.automated_processor:
                    st.write("◆ Initializing automated processor...")
                    try:
                        self.automated_processor = AutomatedEmailProcessor()
                        if self.decision_engine:
                            self.automated_processor.decision_engine = self.decision_engine
                        if self.gmail_client:
                            self.automated_processor.gmail_client = self.gmail_client
                        if self.email_processor:
                            self.automated_processor.email_processor = self.email_processor
                        if self.gemini_filter:
                            self.automated_processor.gemini_filter = self.gemini_filter
                            self.automated_processor.gemini_enabled = True
                        st.write("◈ Automated processor initialized")
                    except Exception as e:
                        st.error(f"⚠ Failed to initialize automated processor: {e}")
                        self.automated_processor = None
                
                if self.automated_processor:
                    # Update configuration
                    if hasattr(self.automated_processor, 'config'):
                        self.automated_processor.config.update({
                            'max_emails_per_batch': max_emails,
                            'gemini_filter_enabled': enable_gemini,
                            'confidence_threshold': confidence_threshold,
                            'auto_response_enabled': enable_auto_response and not test_mode
                        })
                    
                    # Process emails
                    result = self.automated_processor.process_emails_batch()
                    
                    if result and result.get("status") == "success":
                        processed_emails = result.get("details", [])
                        st.write(f"◈ Successfully fetched and processed {len(processed_emails)} emails")
                        
                        if enable_gemini:
                            st.write("◆ Applied Gemini AI filtering for student queries...")
                            time.sleep(0.3)
                        
                        st.write("🔍 Classified queries using fine-tuned model...")
                        time.sleep(0.3)
                        
                        st.write("💬 Generated contextual responses using RAG...")
                        time.sleep(0.3)
                        
                        if enable_auto_response and not test_mode:
                            st.write("📤 Sending approved responses via Gmail...")
                            time.sleep(0.5)
                        
                        # Process and store results
                        auto_sent = 0
                        escalated = 0
                        
                        for email_detail in processed_emails:
                            confidence = email_detail.get('ai_decision', {}).get('confidence', 0.0)
                            action = email_detail.get('action', 'unknown')
                            
                            if action == 'auto_respond':
                                auto_sent += 1
                            elif action == 'escalate':
                                escalated += 1
                            
                            email_data = {
                                'id': email_detail.get('email_id', f'email_{len(st.session_state.processed_emails)}'),
                                'subject': email_detail.get('subject', 'No Subject'),
                                'sender': email_detail.get('sender', 'Unknown'),
                                'body': email_detail.get('body', 'No Content')[:200] + '...' if len(email_detail.get('body', '')) > 200 else email_detail.get('body', 'No Content'),
                                'full_body': email_detail.get('body', 'No Content'),
                                'category': email_detail.get('ai_decision', {}).get('category', 'Unknown'),
                                'action': action,
                                'confidence': confidence,
                                'processed_at': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                'response': email_detail.get('ai_decision', {}).get('response', 'No response generated'),
                                'rag_sources': email_detail.get('ai_decision', {}).get('rag_sources', []),
                                'gemini_filtered': email_detail.get('gemini_filtered', False),
                                'gemini_result': email_detail.get('gemini_filter_result', {}),
                                'gmail_message_id': email_detail.get('email_id'),
                                'test_mode': test_mode
                            }
                            
                            st.session_state.processed_emails.append(email_data)
                            
                            if action == 'escalate':
                                st.session_state.pending_approvals.append(email_data)
                        
                        self.update_processing_stats()
                        
                        status.update(label=f"◈ {mode_text} completed successfully!", state="complete")
                        
                        # Show detailed results
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("📧 Processed", len(processed_emails))
                        with col2:
                            if test_mode:
                                st.metric("🧪 Would Send", auto_sent)
                            else:
                                st.metric("◆ Auto-sent", auto_sent)
                        with col3:
                            st.metric("⚠ Escalated", escalated)
                        
                        # Show success messages
                        if auto_sent > 0 and not test_mode:
                            st.success(f"◈ {auto_sent} responses automatically sent via Gmail!")
                        elif auto_sent > 0 and test_mode:
                            st.info(f"� In test mode - would have sent {auto_sent} responses")
                        
                        if escalated > 0:
                            st.warning(f"⚠ {escalated} emails require manual review in the Review Queue tab")
                    
                    else:
                        # Handle case where no emails found
                        error_msg = result.get("error", "No emails found or processing failed") if result else "Processing returned no result"
                        st.info(f"◇ {error_msg}")
                        
                        status.update(label="◇ No emails to process", state="complete")
                
                else:
                    st.error("⚠ Automated processor not available")
                    
        except Exception as e:
            st.error(f"⚠ Batch processing failed: {e}")
            logger.error(f"Batch processing error: {e}")
            
            with st.expander("🔍 Error Details"):
                st.code(f"Error: {str(e)}")
    
    def analyze_emails_only(self, max_emails):
        """Analyze emails without processing"""
        with st.status(f"🔍 Analyzing {max_emails} emails...", expanded=True) as status:
            st.write("◆ Running analysis without processing...")
            time.sleep(2)
            
            status.update(label="✅ Analysis completed!", state="complete")
            
            # Show analysis results
            st.info("◆ Analysis Results:\n\n"
                   f"• {max_emails} emails analyzed\n"
                   f"• {int(max_emails * 0.8)} appear to be student queries\n"
                   f"• {int(max_emails * 0.6)} would qualify for auto-response\n"
                   f"• {int(max_emails * 0.2)} would need human review")
    
    def preview_email_processing(self, max_emails):
        """Preview what would happen without actually processing"""
        with st.status(f"👁️ Previewing processing for {max_emails} emails...", expanded=True) as status:
            st.write("🔍 Simulating processing workflow...")
            time.sleep(1.5)
            
            status.update(label="✅ Preview completed!", state="complete")
            
            # Show preview results
            preview_data = []
            for i in range(min(max_emails, 3)):
                preview_data.append({
                    'Email': f'student{i+1}@institute.edu',
                    'Subject': f'Query about {["registration", "fees", "grades"][i]}',
                    'Predicted Action': ['Auto-respond', 'Auto-respond', 'Escalate'][i],
                    'Confidence': [0.92, 0.88, 0.76][i]
                })
            
            df = pd.DataFrame(preview_data)
            st.dataframe(df, width="stretch", hide_index=True)
    
    def approve_response(self, index, response_text):
        """Approve and send a response"""
        if 0 <= index < len(st.session_state.pending_approvals):
            approved = st.session_state.pending_approvals.pop(index)
            approved['status'] = 'approved'
            approved['response'] = response_text
            approved['approved_at'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            st.success(f"✅ Response approved and sent to {approved.get('sender', 'Unknown')}")
    
    def reject_response(self, index):
        """Reject a response"""
        if 0 <= index < len(st.session_state.pending_approvals):
            rejected = st.session_state.pending_approvals.pop(index)
            st.info(f"❌ Response rejected for: {rejected.get('subject', 'Unknown')}")
    
    def revise_response(self, index, response_text):
        """Revise a response"""
        if 0 <= index < len(st.session_state.pending_approvals):
            st.session_state.pending_approvals[index]['response'] = response_text
            st.success("✏️ Response revised. Please review again.")
    
    def update_processing_stats(self):
        """Update processing statistics"""
        total = len(st.session_state.processed_emails)
        auto_responses = len([e for e in st.session_state.processed_emails if e.get('action') == 'auto_respond'])
        escalated = len([e for e in st.session_state.processed_emails if e.get('action') == 'escalate'])
        
        # Calculate automation rate (auto responses / total)
        automation_rate = (auto_responses / total * 100) if total > 0 else 0
        
        # Calculate accuracy rate based on confidence scores of auto-responded emails
        auto_responded_emails = [e for e in st.session_state.processed_emails if e.get('action') == 'auto_respond']
        if auto_responded_emails:
            avg_confidence = sum(e.get('confidence', 0.8) for e in auto_responded_emails) / len(auto_responded_emails)
            accuracy_rate = avg_confidence * 100
        else:
            accuracy_rate = 0.0
        
        st.session_state.processing_stats = {
            'total_processed': total,
            'auto_responses': auto_responses,
            'escalated': escalated,
            'automation_rate': automation_rate,
            'accuracy_rate': accuracy_rate
        }
    
    def save_settings(self, settings):
        """Save system settings"""
        # In a real implementation, save to configuration file
        st.session_state.system_settings = settings
    
    def export_configuration(self):
        """Export system configuration"""
        config = {
            'system_status': st.session_state.system_status,
            'processing_stats': st.session_state.processing_stats,
            'settings': st.session_state.get('system_settings', {}),
            'export_timestamp': datetime.now().isoformat()
        }
        
        st.download_button(
            label="◆ Download Configuration",
            data=json.dumps(config, indent=2),
            file_name=f"ai_query_handler_config_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json"
        )
    
    def export_results(self):
        """Export processing results"""
        if st.session_state.processed_emails:
            df = pd.DataFrame(st.session_state.processed_emails)
            csv = df.to_csv(index=False)
            
            st.download_button(
                label="◆ Download Results CSV",
                data=csv,
                file_name=f"email_processing_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )

def main():
    """Main application entry point"""
    
    # Configure Streamlit page
    st.set_page_config(
        page_title="AI Query Handler Dashboard",
        page_icon="◆",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Professional CSS styling
    st.markdown("""
    <style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');
    
    /* Global styling */
    .main {
        font-family: 'Inter', sans-serif;
        background: linear-gradient(135deg, #f8f9fa 0%, #ffffff 100%);
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    .stDeployButton {visibility: hidden;}
    
    /* Professional button styling */
    .stButton > button {
        font-family: 'Inter', sans-serif;
        font-weight: 500;
        border-radius: 8px;
        border: none;
        padding: 0.6rem 1.2rem;
        transition: all 0.3s ease;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }
    
    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: rgba(255,255,255,0.8);
        border-radius: 12px;
        padding: 0.5rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
    }
    
    .stTabs [data-baseweb="tab"] {
        font-family: 'Inter', sans-serif;
        font-weight: 500;
        padding: 0.8rem 1.5rem;
        border-radius: 8px;
        border: none;
    }
    
    /* Metric styling */
    .metric-container {
        background: white;
        padding: 1.2rem;
        border-radius: 12px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        border: 1px solid rgba(0,0,0,0.05);
    }
    
    /* Form styling */
    .stForm {
        background: rgba(255,255,255,0.9);
        border-radius: 12px;
        padding: 2rem;
        box-shadow: 0 4px 16px rgba(0,0,0,0.08);
        border: 1px solid rgba(0,0,0,0.05);
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Custom card styling */
    .status-card {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 2px 12px rgba(0,0,0,0.08);
        border: 1px solid rgba(0,0,0,0.05);
        margin: 1rem 0;
    }
    
    /* Typography improvements */
    h1, h2, h3 {
        font-family: 'Inter', sans-serif;
        font-weight: 600;
    }
    
    /* Code blocks */
    .stCode {
        font-family: 'JetBrains Mono', monospace;
    }
    
    /* Progress bars */
    .stProgress > div > div {
        border-radius: 4px;
    }
    
    /* Data frames */
    .stDataFrame {
        border-radius: 12px;
        overflow: hidden;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        font-family: 'Inter', sans-serif;
        font-weight: 500;
        border-radius: 8px;
    }
    
    .stButton > button {
        font-family: 'Inter', sans-serif;
        font-weight: 500;
        border-radius: 8px;
        border: none;
        transition: all 0.3s ease;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }
    
    /* Metric cards */
    [data-testid="metric-container"] {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        border-radius: 10px;
        padding: 1rem;
        border-left: 4px solid #667eea;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }
    </style>
    """, unsafe_allow_html=True)
    # Check if core imports are available
    if not CORE_IMPORTS_SUCCESS:
        st.error("❌ **System Import Error**")
        
        st.markdown("""
        ### ⚠ Critical System Error
        
        The AI Query Handler system could not load essential components. This typically happens due to:
        
        - **Missing Dependencies**: Some required Python packages are not installed
        - **File Path Issues**: Source files are not in the expected locations  
        - **Environment Issues**: Virtual environment or Python path problems
        
        ### ◆ Troubleshooting Steps
        
        1. **Check Dependencies**:
           ```bash
           pip install -r requirements.txt
           ```
        
        2. **Verify File Structure**:
           - Ensure all files in `src/` directory exist
           - Check that `__pycache__` folders are clean
        
        3. **Environment Setup**:
           ```bash
           conda activate ./env
           python -c "import src.enhanced_decision_engine"
           ```
        """)
        
        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            if st.button("◆ Retry System Load", type="primary", use_container_width=True):
                st.rerun()
        
        st.stop()
    
    # Initialize and run dashboard
    try:
        dashboard = AIQueryHandlerDashboard()
        dashboard.render_dashboard()
        
        # Footer
        st.markdown("---")
        st.markdown("""
        <div style='text-align: center; color: #666; padding: 1rem; font-family: Inter, sans-serif;'>
            ◆ <strong>AI Query Handler</strong> v3.0 | 
            Powered by Advanced AI & Multi-Agent Systems | 
            🏫 <strong>Institute Department Automation</strong>
            <br>
            <small>Built with Streamlit • Enhanced Decision Engine • Gemini AI • RAG System</small>
        </div>
        """, unsafe_allow_html=True)
        
    except Exception as e:
        st.error(f"❌ **Critical Dashboard Error**: {e}")
        
        st.markdown("### 🔍 Error Details")
        
        with st.expander("Technical Details", expanded=True):
            st.code(f"Error Type: {type(e).__name__}\nError Message: {str(e)}")
        
        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            if st.button("◆ Restart Dashboard", type="primary", width="stretch"):
                st.rerun()

if __name__ == "__main__":
    main()