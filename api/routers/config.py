from fastapi import APIRouter, HTTPException, status, Depends
from typing import Dict, Any
import logging

from api.models import ConfigResponse, ConfigUpdateRequest, APIResponse
from api.dependencies import get_config, update_config

router = APIRouter()
logger = logging.getLogger(__name__)

@router.get("/config", response_model=ConfigResponse)
async def get_configuration(config = Depends(get_config)):
    """Get current system configuration"""
    try:
        return ConfigResponse(
            processing_interval_minutes=config.get('processing_interval_minutes', 30),
            max_emails_per_batch=config.get('max_emails_per_batch', 10),
            auto_respond_enabled=config.get('auto_respond_enabled', False),
            confidence_thresholds=config.get('confidence_thresholds', {}),
            human_review_required_categories=config.get('human_review_required_categories', []),
            escalation_keywords=config.get('escalation_keywords', []),
            working_hours=config.get('working_hours', {}),
            email_settings=config.get('email_settings', {}),
            safety_settings=config.get('safety_settings', {})
        )
    except Exception as e:
        logger.error(f"Failed to get configuration: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get configuration: {str(e)}"
        )

@router.put("/config", response_model=APIResponse)
async def update_configuration(request: ConfigUpdateRequest):
    """Update system configuration"""
    try:
        # Validate configuration
        config = request.config
        
        # Validate confidence thresholds
        if 'confidence_thresholds' in config:
            thresholds = config['confidence_thresholds']
            if not isinstance(thresholds, dict):
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="confidence_thresholds must be a dictionary"
                )
            
            # Validate threshold values
            for key, value in thresholds.items():
                if not isinstance(value, (int, float)) or not 0 <= value <= 1:
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail=f"Threshold {key} must be a number between 0 and 1"
                    )
        
        # Validate processing interval
        if 'processing_interval_minutes' in config:
            interval = config['processing_interval_minutes']
            if not isinstance(interval, int) or interval < 1:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="processing_interval_minutes must be a positive integer"
                )
        
        # Validate max emails per batch
        if 'max_emails_per_batch' in config:
            max_emails = config['max_emails_per_batch']
            if not isinstance(max_emails, int) or max_emails < 1 or max_emails > 100:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="max_emails_per_batch must be between 1 and 100"
                )
        
        # Validate working hours
        if 'working_hours' in config:
            working_hours = config['working_hours']
            if not isinstance(working_hours, dict):
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="working_hours must be a dictionary"
                )
            
            # Validate time format if provided
            for time_field in ['start', 'end']:
                if time_field in working_hours:
                    time_str = working_hours[time_field]
                    if not isinstance(time_str, str):
                        raise HTTPException(
                            status_code=status.HTTP_400_BAD_REQUEST,
                            detail=f"working_hours.{time_field} must be a string"
                        )
                    
                    # Validate HH:MM format
                    try:
                        parts = time_str.split(':')
                        if len(parts) != 2:
                            raise ValueError()
                        hour, minute = int(parts[0]), int(parts[1])
                        if not (0 <= hour <= 23 and 0 <= minute <= 59):
                            raise ValueError()
                    except ValueError:
                        raise HTTPException(
                            status_code=status.HTTP_400_BAD_REQUEST,
                            detail=f"working_hours.{time_field} must be in HH:MM format"
                        )
        
        # Update the configuration
        update_config(config)
        
        return APIResponse(
            success=True,
            message="Configuration updated successfully",
            data={"updated_fields": list(config.keys())}
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to update configuration: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to update configuration: {str(e)}"
        )

@router.get("/config/thresholds", response_model=APIResponse)
async def get_confidence_thresholds(config = Depends(get_config)):
    """Get confidence thresholds configuration"""
    try:
        thresholds = config.get('confidence_thresholds', {})
        
        return APIResponse(
            success=True,
            message="Confidence thresholds retrieved successfully",
            data={
                "thresholds": thresholds,
                "descriptions": {
                    "auto_respond": "Minimum confidence to automatically respond",
                    "review_needed": "Minimum confidence to require human review",
                    "escalate": "Minimum confidence to escalate (below this value)"
                }
            }
        )
        
    except Exception as e:
        logger.error(f"Failed to get confidence thresholds: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get confidence thresholds: {str(e)}"
        )

@router.put("/config/thresholds", response_model=APIResponse)
async def update_confidence_thresholds(
    auto_respond: float = None,
    review_needed: float = None,
    escalate: float = None,
    config = Depends(get_config)
):
    """Update confidence thresholds"""
    try:
        current_config = dict(config)
        thresholds = current_config.get('confidence_thresholds', {})
        
        # Update provided thresholds
        if auto_respond is not None:
            if not 0 <= auto_respond <= 1:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="auto_respond threshold must be between 0 and 1"
                )
            thresholds['auto_respond'] = auto_respond
        
        if review_needed is not None:
            if not 0 <= review_needed <= 1:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="review_needed threshold must be between 0 and 1"
                )
            thresholds['review_needed'] = review_needed
        
        if escalate is not None:
            if not 0 <= escalate <= 1:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="escalate threshold must be between 0 and 1"
                )
            thresholds['escalate'] = escalate
        
        # Validate threshold ordering
        auto_thresh = thresholds.get('auto_respond', 0.65)
        review_thresh = thresholds.get('review_needed', 0.4)
        escalate_thresh = thresholds.get('escalate', 0.0)
        
        if not (escalate_thresh <= review_thresh <= auto_thresh):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Thresholds must be ordered: escalate <= review_needed <= auto_respond"
            )
        
        current_config['confidence_thresholds'] = thresholds
        
        update_config(current_config)
        
        return APIResponse(
            success=True,
            message="Confidence thresholds updated successfully",
            data={"thresholds": thresholds}
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to update confidence thresholds: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to update confidence thresholds: {str(e)}"
        )

@router.get("/config/working-hours", response_model=APIResponse)
async def get_working_hours(config = Depends(get_config)):
    """Get working hours configuration"""
    try:
        working_hours = config.get('working_hours', {})
        
        return APIResponse(
            success=True,
            message="Working hours retrieved successfully",
            data=working_hours
        )
        
    except Exception as e:
        logger.error(f"Failed to get working hours: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get working hours: {str(e)}"
        )

@router.put("/config/working-hours", response_model=APIResponse)
async def update_working_hours(
    enabled: bool = None,
    start_time: str = None,
    end_time: str = None,
    timezone: str = None,
    weekdays_only: bool = None,
    config = Depends(get_config)
):
    """Update working hours configuration"""
    try:
        current_config = dict(config)
        working_hours = current_config.get('working_hours', {})
        
        # Update provided fields
        if enabled is not None:
            working_hours['enabled'] = enabled
        
        if start_time is not None:
            # Validate time format
            try:
                parts = start_time.split(':')
                if len(parts) != 2:
                    raise ValueError()
                hour, minute = int(parts[0]), int(parts[1])
                if not (0 <= hour <= 23 and 0 <= minute <= 59):
                    raise ValueError()
                working_hours['start'] = start_time
            except ValueError:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="start_time must be in HH:MM format"
                )
        
        if end_time is not None:
            # Validate time format
            try:
                parts = end_time.split(':')
                if len(parts) != 2:
                    raise ValueError()
                hour, minute = int(parts[0]), int(parts[1])
                if not (0 <= hour <= 23 and 0 <= minute <= 59):
                    raise ValueError()
                working_hours['end'] = end_time
            except ValueError:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="end_time must be in HH:MM format"
                )
        
        if timezone is not None:
            working_hours['timezone'] = timezone
        
        if weekdays_only is not None:
            working_hours['weekdays_only'] = weekdays_only
        
        current_config['working_hours'] = working_hours
        
        update_config(current_config)
        
        return APIResponse(
            success=True,
            message="Working hours updated successfully",
            data=working_hours
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to update working hours: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to update working hours: {str(e)}"
        )

@router.get("/config/escalation-keywords", response_model=APIResponse)
async def get_escalation_keywords(config = Depends(get_config)):
    """Get escalation keywords configuration"""
    try:
        keywords = config.get('escalation_keywords', [])
        
        return APIResponse(
            success=True,
            message="Escalation keywords retrieved successfully",
            data={"keywords": keywords}
        )
        
    except Exception as e:
        logger.error(f"Failed to get escalation keywords: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get escalation keywords: {str(e)}"
        )

@router.put("/config/escalation-keywords", response_model=APIResponse)
async def update_escalation_keywords(
    keywords: list,
    config = Depends(get_config)
):
    """Update escalation keywords"""
    try:
        if not isinstance(keywords, list):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Keywords must be a list"
            )
        
        # Validate keywords are strings
        for keyword in keywords:
            if not isinstance(keyword, str):
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="All keywords must be strings"
                )
        
        current_config = dict(config)
        current_config['escalation_keywords'] = [k.lower().strip() for k in keywords if k.strip()]
        
        update_config(current_config)
        
        return APIResponse(
            success=True,
            message="Escalation keywords updated successfully",
            data={"keywords": current_config['escalation_keywords']}
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to update escalation keywords: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to update escalation keywords: {str(e)}"
        )

@router.post("/config/reset", response_model=APIResponse)
async def reset_configuration():
    """Reset configuration to defaults"""
    try:
        default_config = {
            "processing_interval_minutes": 30,
            "max_emails_per_batch": 10,
            "auto_respond_enabled": True,
            "confidence_thresholds": {
                "auto_respond": 0.65,
                "review_needed": 0.4,
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
                "email": None,
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
        
        update_config(default_config)
        
        return APIResponse(
            success=True,
            message="Configuration reset to defaults successfully"
        )
        
    except Exception as e:
        logger.error(f"Failed to reset configuration: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to reset configuration: {str(e)}"
        )