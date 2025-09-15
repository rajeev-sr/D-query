from fastapi import APIRouter, HTTPException, status, Depends, Query
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import logging
import json
import os

from api.models import (
    SystemStats, ProcessingMetrics, APIResponse, ProcessingStatus
)
from api.dependencies import (
    get_gmail_client, get_decision_engine, get_automated_processor, get_config
)

router = APIRouter()
logger = logging.getLogger(__name__)

def load_processing_logs() -> List[Dict]:
    """Load processing logs from file"""
    try:
        logs_file = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 
                                'logs', 'processing_history.json')
        
        if os.path.exists(logs_file):
            with open(logs_file, 'r') as f:
                return json.load(f)
        return []
    except Exception as e:
        logger.warning(f"Could not load processing logs: {e}")
        return []

def save_processing_logs(logs: List[Dict]):
    """Save processing logs to file"""
    try:
        logs_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 
                               'logs')
        os.makedirs(logs_dir, exist_ok=True)
        
        logs_file = os.path.join(logs_dir, 'processing_history.json')
        with open(logs_file, 'w') as f:
            json.dump(logs, f, indent=2, default=str)
    except Exception as e:
        logger.error(f"Could not save processing logs: {e}")

@router.get("/dashboard/stats", response_model=SystemStats)
async def get_system_stats(
    gmail_client = Depends(get_gmail_client),
    config = Depends(get_config)
):
    """Get overall system statistics"""
    try:
        # Load processing history
        processing_logs = load_processing_logs()
        
        # Calculate basic stats
        total_processed = len(processing_logs)
        auto_responded = len([log for log in processing_logs if log.get('action') == 'auto_respond'])
        auto_response_rate = (auto_responded / total_processed * 100) if total_processed > 0 else 0
        
        # Calculate average confidence
        confidence_scores = [log.get('confidence_score', 0) for log in processing_logs if log.get('confidence_score')]
        avg_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0
        
        # Get pending review emails count
        try:
            pending_emails = gmail_client.fetch_emails(
                max_results=100,
                query="is:unread -label:AI_PROCESSED"
            )
            pending_count = len(pending_emails)
        except Exception as e:
            logger.warning(f"Could not fetch pending emails: {e}")
            pending_count = 0
        
        # System uptime (simplified - time since last restart)
        uptime = 3600.0  # 1 hour as default
        
        # Last processing run
        last_run = None
        if processing_logs:
            last_log = max(processing_logs, key=lambda x: x.get('timestamp', ''))
            try:
                last_run = datetime.fromisoformat(last_log['timestamp'])
            except:
                pass
        
        return SystemStats(
            total_emails_processed=total_processed,
            auto_response_rate=round(auto_response_rate, 2),
            average_confidence=round(avg_confidence, 3),
            emails_pending_review=pending_count,
            system_uptime=uptime,
            last_processing_run=last_run,
            processing_status=ProcessingStatus.COMPLETED
        )
        
    except Exception as e:
        logger.error(f"Failed to get system stats: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get system stats: {str(e)}"
        )

@router.get("/dashboard/metrics", response_model=ProcessingMetrics)
async def get_processing_metrics(
    days: int = Query(7, ge=1, le=30, description="Number of days to analyze")
):
    """Get detailed processing metrics"""
    try:
        processing_logs = load_processing_logs()
        
        # Filter logs by date range
        cutoff_date = datetime.now() - timedelta(days=days)
        recent_logs = []
        
        for log in processing_logs:
            try:
                log_date = datetime.fromisoformat(log.get('timestamp', ''))
                if log_date >= cutoff_date:
                    recent_logs.append(log)
            except:
                continue
        
        # Daily processing counts
        daily_processed = {}
        for i in range(days):
            date = (datetime.now() - timedelta(days=i)).strftime('%Y-%m-%d')
            daily_processed[date] = 0
        
        for log in recent_logs:
            try:
                log_date = datetime.fromisoformat(log.get('timestamp', '')).strftime('%Y-%m-%d')
                if log_date in daily_processed:
                    daily_processed[log_date] += 1
            except:
                continue
        
        # Weekly processing (group by week)
        weekly_processed = {}
        for log in recent_logs:
            try:
                log_date = datetime.fromisoformat(log.get('timestamp', ''))
                week_start = (log_date - timedelta(days=log_date.weekday())).strftime('%Y-%m-%d')
                weekly_processed[week_start] = weekly_processed.get(week_start, 0) + 1
            except:
                continue
        
        # Confidence distribution
        confidence_distribution = {
            "0.0-0.2": 0,
            "0.2-0.4": 0, 
            "0.4-0.6": 0,
            "0.6-0.8": 0,
            "0.8-1.0": 0
        }
        
        for log in recent_logs:
            confidence = log.get('confidence_score', 0)
            if confidence <= 0.2:
                confidence_distribution["0.0-0.2"] += 1
            elif confidence <= 0.4:
                confidence_distribution["0.2-0.4"] += 1
            elif confidence <= 0.6:
                confidence_distribution["0.4-0.6"] += 1
            elif confidence <= 0.8:
                confidence_distribution["0.6-0.8"] += 1
            else:
                confidence_distribution["0.8-1.0"] += 1
        
        # Action distribution
        action_distribution = {}
        for log in recent_logs:
            action = log.get('action', 'unknown')
            action_distribution[action] = action_distribution.get(action, 0) + 1
        
        # Response times
        response_times = [log.get('processing_time', 0) for log in recent_logs if log.get('processing_time')]
        
        return ProcessingMetrics(
            daily_processed=daily_processed,
            weekly_processed=weekly_processed,
            confidence_distribution=confidence_distribution,
            action_distribution=action_distribution,
            response_times=response_times
        )
        
    except Exception as e:
        logger.error(f"Failed to get processing metrics: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get processing metrics: {str(e)}"
        )

@router.get("/dashboard/real-time", response_model=APIResponse)
async def get_real_time_data(
    gmail_client = Depends(get_gmail_client)
):
    """Get real-time system data"""
    try:
        # Get current inbox stats
        unread_emails = gmail_client.fetch_emails(max_results=1, query="is:unread")
        unread_count = len(unread_emails)
        
        # Get recent activity (last hour)
        one_hour_ago = datetime.now() - timedelta(hours=1)
        processing_logs = load_processing_logs()
        
        recent_activity = []
        for log in processing_logs:
            try:
                log_date = datetime.fromisoformat(log.get('timestamp', ''))
                if log_date >= one_hour_ago:
                    recent_activity.append(log)
            except:
                continue
        
        # Sort by most recent
        recent_activity.sort(key=lambda x: x.get('timestamp', ''), reverse=True)
        recent_activity = recent_activity[:10]  # Last 10 activities
        
        real_time_data = {
            "current_time": datetime.now().isoformat(),
            "unread_emails": unread_count,
            "recent_activity": recent_activity,
            "system_status": "running",
            "last_update": datetime.now().isoformat()
        }
        
        return APIResponse(
            success=True,
            message="Real-time data retrieved successfully",
            data=real_time_data
        )
        
    except Exception as e:
        logger.error(f"Failed to get real-time data: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get real-time data: {str(e)}"
        )

@router.post("/dashboard/log-processing", response_model=APIResponse)
async def log_processing_result(
    email_id: str,
    action: str,
    confidence_score: float,
    processing_time: float,
    ai_response: Optional[str] = None,
    error: Optional[str] = None
):
    """Log a processing result for analytics"""
    try:
        processing_logs = load_processing_logs()
        
        new_log = {
            "timestamp": datetime.now().isoformat(),
            "email_id": email_id,
            "action": action,
            "confidence_score": confidence_score,
            "processing_time": processing_time,
            "ai_response": ai_response,
            "error": error
        }
        
        processing_logs.append(new_log)
        
        # Keep only last 10000 logs to prevent file from growing too large
        if len(processing_logs) > 10000:
            processing_logs = processing_logs[-10000:]
        
        save_processing_logs(processing_logs)
        
        return APIResponse(
            success=True,
            message="Processing result logged successfully"
        )
        
    except Exception as e:
        logger.error(f"Failed to log processing result: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to log processing result: {str(e)}"
        )

@router.get("/dashboard/performance", response_model=APIResponse)
async def get_performance_metrics():
    """Get system performance metrics"""
    try:
        processing_logs = load_processing_logs()
        
        # Calculate performance metrics
        if not processing_logs:
            performance_data = {
                "average_processing_time": 0,
                "success_rate": 100,
                "error_rate": 0,
                "throughput": 0
            }
        else:
            # Processing times
            processing_times = [log.get('processing_time', 0) for log in processing_logs if log.get('processing_time')]
            avg_processing_time = sum(processing_times) / len(processing_times) if processing_times else 0
            
            # Success/error rates
            total_logs = len(processing_logs)
            error_logs = len([log for log in processing_logs if log.get('error')])
            success_rate = ((total_logs - error_logs) / total_logs * 100) if total_logs > 0 else 100
            error_rate = (error_logs / total_logs * 100) if total_logs > 0 else 0
            
            # Throughput (emails per hour in last 24 hours)
            last_24_hours = datetime.now() - timedelta(hours=24)
            recent_logs = []
            for log in processing_logs:
                try:
                    log_date = datetime.fromisoformat(log.get('timestamp', ''))
                    if log_date >= last_24_hours:
                        recent_logs.append(log)
                except:
                    continue
            
            throughput = len(recent_logs)  # emails processed in last 24 hours
            
            performance_data = {
                "average_processing_time": round(avg_processing_time, 2),
                "success_rate": round(success_rate, 2),
                "error_rate": round(error_rate, 2),
                "throughput": throughput,
                "total_processed": total_logs
            }
        
        return APIResponse(
            success=True,
            message="Performance metrics retrieved successfully",
            data=performance_data
        )
        
    except Exception as e:
        logger.error(f"Failed to get performance metrics: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get performance metrics: {str(e)}"
        )