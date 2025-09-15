"""
Email Processing Integration API Router
Connects the existing automated processor with the API endpoints
"""
from fastapi import APIRouter, HTTPException, BackgroundTasks, Depends
from typing import Dict, Any, List
import logging
import asyncio
from datetime import datetime

from api.models import (
    APIResponse, 
    BatchProcessingResult, 
    ProcessingResult, 
    EmailProcessRequest,
    EmailData,
    EmailAction
)
from api.dependencies import get_automated_processor

router = APIRouter()
logger = logging.getLogger(__name__)

@router.post("/process/batch", response_model=BatchProcessingResult)
async def process_email_batch(
    request: EmailProcessRequest = EmailProcessRequest(),
    processor = Depends(get_automated_processor)
):
    """Process a batch of emails using the automated processor"""
    try:
        logger.info(f"Starting batch processing with max_emails: {request.max_emails}")
        
        # Use the existing automated processor
        result = processor.process_emails_batch()
        
        if result.get("status") == "error":
            raise HTTPException(status_code=500, detail=result.get("error"))
        
        # Convert to API response format
        processing_results = []
        for detail in result.get("details", []):
            processing_results.append(
                ProcessingResult(
                    email_id=detail["email_id"],
                    action_taken=EmailAction(detail["action"]),
                    confidence_score=detail.get("ai_decision", {}).get("confidence", 0.0),
                    processing_time=0.0,  # Individual timing not tracked
                    ai_response=detail.get("ai_decision", {}).get("response"),
                    error=detail.get("error")
                )
            )
        
        return BatchProcessingResult(
            total_processed=result["processed"],
            successful=result["processed"] - result["errors"],
            failed=result["errors"],
            auto_responded=result.get("auto_responded", 0),
            needs_review=result.get("review_needed", 0),
            escalated=result.get("escalated", 0),
            results=processing_results,
            processing_time=result.get("processing_time", 0.0)
        )
        
    except Exception as e:
        logger.error(f"Batch processing failed: {e}")
        raise HTTPException(
            status_code=500, 
            detail=f"Email processing failed: {str(e)}"
        )

@router.post("/process/single/{email_id}", response_model=ProcessingResult)
async def process_single_email(
    email_id: str,
    processor = Depends(get_automated_processor)
):
    """Process a single email by ID"""
    try:
        logger.info(f"Processing single email: {email_id}")
        
        # Fetch the specific email
        gmail_client = processor.gmail_client
        email_data = gmail_client.get_message_details(email_id)
        
        if not email_data:
            raise HTTPException(status_code=404, detail="Email not found")
        
        # Process the email
        result = processor._process_single_email(email_data)
        
        return ProcessingResult(
            email_id=result["email_id"],
            action_taken=EmailAction(result["action"]),
            confidence_score=result.get("ai_decision", {}).get("confidence", 0.0),
            processing_time=0.0,
            ai_response=result.get("ai_decision", {}).get("response"),
            error=result.get("error")
        )
        
    except Exception as e:
        logger.error(f"Single email processing failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to process email {email_id}: {str(e)}"
        )

@router.post("/start-automation", response_model=APIResponse)
async def start_automation(
    background_tasks: BackgroundTasks,
    processor = Depends(get_automated_processor)
):
    """Start the automated email processing in background"""
    try:
        # Add automated processing to background tasks
        background_tasks.add_task(run_continuous_processing, processor)
        
        return APIResponse(
            success=True,
            message="Automated email processing started",
            data={"status": "running"}
        )
        
    except Exception as e:
        logger.error(f"Failed to start automation: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to start automation: {str(e)}"
        )

@router.post("/stop-automation", response_model=APIResponse)
async def stop_automation():
    """Stop the automated email processing"""
    try:
        # This would integrate with a process manager
        # For now, just return success
        return APIResponse(
            success=True,
            message="Automated email processing stop requested",
            data={"status": "stopping"}
        )
        
    except Exception as e:
        logger.error(f"Failed to stop automation: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to stop automation: {str(e)}"
        )

@router.get("/processing-status", response_model=APIResponse)
async def get_processing_status(processor = Depends(get_automated_processor)):
    """Get current processing status and statistics including Gemini filtering"""
    try:
        # Get processing statistics from the processor
        processing_stats = processor.get_processing_stats() if hasattr(processor, 'get_processing_stats') else {}
        filtering_stats = processor.get_filtering_stats() if hasattr(processor, 'get_filtering_stats') else {}
        
        stats = {
            "last_processed": processor.last_processed_time.isoformat() if processor.last_processed_time else None,
            "total_processed_today": len([log for log in processor.processing_log 
                                        if log.get("processed_at", "").startswith(datetime.now().strftime("%Y-%m-%d"))]),
            "config": processor.config,
            "is_running": hasattr(processor, '_is_running') and processor._is_running,
            "filtering_stats": filtering_stats,
            "processing_stats": processing_stats,
            "gemini_filter_enabled": getattr(processor, 'gemini_enabled', False) and processor.config.get('gemini_filter_enabled', True),
            "gemini_available": getattr(processor, 'gemini_enabled', False)
        }
        
        return APIResponse(
            success=True,
            message="Processing status retrieved",
            data=stats
        )
        
    except Exception as e:
        logger.error(f"Failed to get processing status: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get processing status: {str(e)}"
        )

@router.get("/review-queue", response_model=List[EmailData])
async def get_review_queue(processor = Depends(get_automated_processor)):
    """Get emails that need human review"""
    try:
        # This would fetch emails marked for review
        # For now, return empty list as placeholder
        review_emails = []
        
        # In a real implementation, this would query emails with review labels
        # review_emails = processor.get_review_queue()
        
        return review_emails
        
    except Exception as e:
        logger.error(f"Failed to get review queue: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get review queue: {str(e)}"
        )

async def run_continuous_processing(processor):
    """Background task for continuous email processing"""
    processor._is_running = True
    
    try:
        while processor._is_running:
            # Check if within working hours
            if processor._is_within_working_hours():
                result = processor.process_emails_batch()
                logger.info(f"Automated batch processing completed: {result}")
            
            # Wait for next processing interval
            interval_minutes = processor.config.get("processing_interval_minutes", 30)
            await asyncio.sleep(interval_minutes * 60)
            
    except Exception as e:
        logger.error(f"Continuous processing error: {e}")
    finally:
        processor._is_running = False

# Add method to check working hours
def _is_within_working_hours(processor) -> bool:
    """Check if current time is within configured working hours"""
    try:
        working_hours = processor.config.get("working_hours", {})
        if not working_hours.get("enabled", True):
            return True
            
        from datetime import datetime
        import pytz
        
        # Get timezone
        tz = pytz.timezone(working_hours.get("timezone", "UTC"))
        current_time = datetime.now(tz)
        
        # Parse working hours
        start_time = datetime.strptime(working_hours.get("start", "09:00"), "%H:%M").time()
        end_time = datetime.strptime(working_hours.get("end", "17:00"), "%H:%M").time()
        
        current_time_only = current_time.time()
        
        return start_time <= current_time_only <= end_time
        
    except Exception as e:
        logger.warning(f"Working hours check failed: {e}, defaulting to True")
        return True

@router.post("/toggle-gemini-filter", response_model=APIResponse)
async def toggle_gemini_filter(
    enabled: bool, 
    processor = Depends(get_automated_processor)
):
    """Enable or disable Gemini query filtering"""
    try:
        if not hasattr(processor, 'toggle_gemini_filter'):
            raise HTTPException(
                status_code=501, 
                detail="Gemini filtering not available in this processor version"
            )
        
        success = processor.toggle_gemini_filter(enabled)
        
        if success:
            return APIResponse(
                success=True,
                message=f"Gemini filtering {'enabled' if enabled else 'disabled'}",
                data={"gemini_filter_enabled": enabled}
            )
        else:
            return APIResponse(
                success=False,
                message="Gemini filter not available (API key missing or initialization failed)",
                data={"gemini_filter_enabled": False}
            )
            
    except Exception as e:
        logger.error(f"Failed to toggle Gemini filter: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to toggle Gemini filter: {str(e)}"
        )

@router.get("/filter-stats", response_model=APIResponse)
async def get_filter_statistics(processor = Depends(get_automated_processor)):
    """Get detailed email filtering statistics"""
    try:
        if hasattr(processor, 'get_filtering_stats'):
            stats = processor.get_filtering_stats()
        else:
            stats = {
                "total_fetched": 0,
                "student_emails": 0,
                "query_emails": 0,
                "non_query_emails": 0,
                "gemini_filter_enabled": False,
                "gemini_available": False
            }
        
        return APIResponse(
            success=True,
            message="Filter statistics retrieved",
            data=stats
        )
        
    except Exception as e:
        logger.error(f"Failed to get filter statistics: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get filter statistics: {str(e)}"
        )

# Monkey patch the method to the processor class
import types

def patch_working_hours_method():
    """Patch working hours method to processor class if available"""
    try:
        from src.automated_processor import AutomatedEmailProcessor
        AutomatedEmailProcessor._is_within_working_hours = types.MethodType(_is_within_working_hours, AutomatedEmailProcessor)
        logger.info("✅ Working hours method patched to AutomatedEmailProcessor")
    except ImportError:
        logger.info("⚠️ AutomatedEmailProcessor not available, skipping patch")
    except Exception as e:
        logger.warning(f"⚠️ Could not patch working hours method: {e}")

# Only patch if the module is importable
try:
    patch_working_hours_method()
except Exception as e:
    logger.warning(f"Failed to patch working hours method: {e}")